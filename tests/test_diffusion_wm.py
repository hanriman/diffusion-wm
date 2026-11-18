"""Unit tests for SpatioTemporalDiT, DiffusionWorldModel, GradientSolver, and CEMSolver."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

import stable_worldmodel as swm
from diffusion_wm.models import (
    DiffusionWorldModel,
    SpatioTemporalDiT,
    TinyConvVAE,
)


def _make_model(
    *,
    img_size: int = 64,
    patch_size: int = 2,
    action_dim: int = 2,
    frameskip: int = 5,
    use_proprio: bool = False,
    proprio_dim: int = 0,
    rollout_mode: str = 'parallel',
    ode_solver: str = 'euler',
    num_ode_steps: int = 3,
) -> DiffusionWorldModel:
    vae = TinyConvVAE(hidden_dim=16, freeze=True)
    lat_hw = img_size // 8
    dit = SpatioTemporalDiT(
        latent_channels=4,
        latent_height=lat_hw,
        latent_width=lat_hw,
        patch_size=patch_size,
        hidden_dim=64,
        depth=2,
        heads=4,
        dim_head=16,
        mlp_ratio=2.0,
        max_seq_len=16,
        action_embed_dim=64,
        use_proprio=use_proprio,
        proprio_dim=proprio_dim,
    )
    # Initialize final heads with small weights so untrained velocity field is non-trivial for solver tests
    torch.nn.init.normal_(dit.vis_head.linear.weight, std=0.02)
    if use_proprio and dit.proprio_head is not None:
        torch.nn.init.normal_(dit.proprio_head.linear.weight, std=0.02)

    return DiffusionWorldModel(
        vae=vae,
        dit=dit,
        action_dim=action_dim,
        frameskip=frameskip,
        action_embed_dim=64,
        history_size=2,
        use_proprio=use_proprio,
        proprio_dim=proprio_dim,
        num_ode_steps=num_ode_steps,
        ode_solver=ode_solver,
        rollout_mode=rollout_mode,
    )


def test_dit_shapes_and_causal_masking():
    torch.manual_seed(0)
    dit = SpatioTemporalDiT(
        latent_channels=4,
        latent_height=8,
        latent_width=8,
        patch_size=2,
        hidden_dim=64,
        depth=2,
        heads=4,
        dim_head=16,
        action_embed_dim=64,
        use_proprio=True,
        proprio_dim=27,
    ).eval()
    torch.nn.init.normal_(dit.vis_head.linear.weight, std=0.05)
    torch.nn.init.normal_(dit.proprio_head.linear.weight, std=0.05)

    b, t = 2, 5
    z_vis = torch.randn(b, t, 4, 8, 8)
    z_p = torch.randn(b, t, 27)
    tau = torch.rand(b, t)
    act_emb = torch.randn(b, t, 64)

    out1 = dit(z_vis, tau, act_emb, z_proprio=z_p)
    assert out1['v_vis'].shape == (b, t, 4, 8, 8)
    assert out1['v_proprio'].shape == (b, t, 27)

    # Perturb future frames t >= 3 and verify frames t < 3 are completely unchanged
    z_vis_pert = z_vis.clone()
    z_p_pert = z_p.clone()
    act_pert = act_emb.clone()
    z_vis_pert[:, 3:] += 10.0
    z_p_pert[:, 3:] -= 10.0
    act_pert[:, 3:] += 5.0

    out2 = dit(z_vis_pert, tau, act_pert, z_proprio=z_p_pert)
    torch.testing.assert_close(out1['v_vis'][:, :3], out2['v_vis'][:, :3])
    torch.testing.assert_close(out1['v_proprio'][:, :3], out2['v_proprio'][:, :3])


@pytest.mark.parametrize('use_proprio', [False, True])
def test_rectified_flow_loss_decreases(use_proprio: bool):
    torch.manual_seed(42)
    model = _make_model(
        img_size=64,
        patch_size=2,
        action_dim=8 if use_proprio else 2,
        frameskip=5,
        use_proprio=use_proprio,
        proprio_dim=27 if use_proprio else 0,
    )
    model.train()
    # Disable random tau sampling variance by fixing ctx_noise_std=0 and overriding _sample_training_tau
    fixed_tau = torch.tensor([[1.0, 1.0, 0.5, 0.5]] * 4)
    fixed_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 4)
    model._sample_training_tau = lambda b, t, h, dev: (
        fixed_tau.to(dev),
        fixed_mask.to(dev),
    )

    batch = {
        'pixels_level1': torch.randn(4, 4, 3, 64, 64),
        'action_level1': torch.randn(4, 4, (8 if use_proprio else 2) * 5),
    }
    if use_proprio:
        batch['proprio_level1'] = torch.randn(4, 4, 27)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-3
    )

    initial_loss = None
    final_loss = None
    for step in range(15):
        torch.manual_seed(100)  # deterministic noise draw for smooth convergence check
        losses = model.compute_loss(batch)
        loss = losses['loss']
        if step == 0:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()

    assert final_loss < initial_loss * 0.6, (
        f'Expected loss to decrease significantly: {initial_loss:.4f} -> {final_loss:.4f}'
    )


@pytest.mark.parametrize('rollout_mode', ['parallel', 'autoregressive'])
@pytest.mark.parametrize('ode_solver', ['euler', 'midpoint'])
def test_ode_rollouts_and_differentiable_cost(
    rollout_mode: str, ode_solver: str
):
    torch.manual_seed(7)
    model = _make_model(
        img_size=64,
        patch_size=2,
        action_dim=8,
        frameskip=5,
        use_proprio=True,
        proprio_dim=27,
        rollout_mode=rollout_mode,
        ode_solver=ode_solver,
        num_ode_steps=2,
    ).eval()

    b, s, h_ctx, t_plan = 2, 3, 1, 4
    info_dict = {
        'pixels': torch.randn(b, s, h_ctx, 3, 64, 64),
        'goal': torch.randn(b, s, 1, 3, 64, 64),
        'proprio': torch.randn(b, s, h_ctx, 27),
        'goal_proprio': torch.randn(b, s, 1, 27),
        'cost_last_n': 2,
    }
    action_candidates = torch.randn(b, s, t_plan, 40, requires_grad=True)

    costs = model.get_cost(info_dict, action_candidates)
    assert costs.shape == (b, s)
    assert costs.requires_grad

    costs.sum().backward()
    assert action_candidates.grad is not None
    assert torch.isfinite(action_candidates.grad).all()
    assert action_candidates.grad.abs().sum() > 0

    decoded = model.decode_pixels(info_dict['predicted_embed_0'][:, 0])
    assert decoded.shape == (b, h_ctx + t_plan, 3, 64, 64)
    assert decoded.min() >= 0.0 and decoded.max() <= 1.0


def test_gradient_solver_and_cem_solver_reduce_cost():
    torch.manual_seed(123)
    model = _make_model(
        img_size=64,
        patch_size=2,
        action_dim=2,
        frameskip=5,
        use_proprio=False,
        rollout_mode='parallel',
        num_ode_steps=2,
    ).eval()

    n_envs = 2
    horizon = 4
    action_block = 5
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(n_envs, 2), dtype=np.float32
    )
    plan_cfg = swm.PlanConfig(
        horizon=horizon,
        receding_horizon=2,
        action_block=action_block,
        cost_last_n=2,
    )

    info_dict = {
        'pixels': torch.randn(n_envs, 1, 3, 64, 64),
        'goal': torch.randn(n_envs, 1, 3, 64, 64),
    }

    # Baseline cost with zero actions
    with torch.no_grad():
        zero_actions = torch.zeros(n_envs, 1, horizon, 2 * action_block)
        expanded_zero_info = {
            k: v.unsqueeze(1) for k, v in info_dict.items()
        }
        expanded_zero_info['cost_last_n'] = 2
        baseline_cost = model.get_cost(expanded_zero_info, zero_actions).squeeze(1)

    # 1. Test GradientSolver
    gd_solver = swm.solver.GradientSolver(
        model=model,
        n_steps=8,
        min_its=8,
        num_samples=4,
        var_scale=0.5,
        device='cpu',
        seed=42,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={'lr': 0.1},
    )
    gd_solver.configure(
        action_space=action_space, n_envs=n_envs, config=plan_cfg
    )
    gd_out = gd_solver.solve(
        dict(info_dict), planning_horizon=horizon, steps_taken=0, eval_budget=20
    )
    assert gd_out['actions'].shape == (n_envs, horizon, 2 * action_block)
    with torch.no_grad():
        eval_info = {k: v.unsqueeze(1) for k, v in info_dict.items()}
        eval_info['cost_last_n'] = 2
        gd_cost = model.get_cost(
            eval_info, gd_out['actions'].unsqueeze(1)
        ).squeeze(1)
    assert (gd_cost <= baseline_cost + 1e-5).all()

    # 2. Test CEMSolver
    cem_solver = swm.solver.CEMSolver(
        model=model,
        n_steps=4,
        num_samples=16,
        num_elites=4,
        var_scale=1.0,
        device='cpu',
        seed=42,
    )
    cem_solver.configure(
        action_space=action_space, n_envs=n_envs, config=plan_cfg
    )
    cem_out = cem_solver.solve(
        dict(info_dict), planning_horizon=horizon, steps_taken=0, eval_budget=20
    )
    assert cem_out['actions'].shape == (n_envs, horizon, 2 * action_block)
    with torch.no_grad():
        eval_info = {k: v.unsqueeze(1) for k, v in info_dict.items()}
        eval_info['cost_last_n'] = 2
        cem_cost = model.get_cost(
            eval_info, cem_out['actions'].unsqueeze(1)
        ).squeeze(1)
    assert (cem_cost <= baseline_cost + 1e-5).all()


@pytest.mark.parametrize('rollout_mode', ['parallel', 'autoregressive'])
def test_chunked_checkpointed_rollout_matches_exact_gradients(
    rollout_mode: str,
):
    torch.manual_seed(11)
    model = _make_model(
        img_size=64,
        patch_size=2,
        action_dim=2,
        frameskip=5,
        rollout_mode=rollout_mode,
        num_ode_steps=3,
    ).eval()

    b, s, h_ctx, t_plan = 2, 3, 1, 4
    info_base = {
        'pixels': torch.randn(b, s, h_ctx, 3, 64, 64),
        'goal': torch.randn(b, s, 1, 3, 64, 64),
        'cost_last_n': 2,
    }
    acts_ref = torch.randn(b, s, t_plan, 10, requires_grad=True)
    acts_chk = acts_ref.detach().clone().requires_grad_(True)

    # Reference: unchunked, no gradient checkpointing
    model.rollout_batch_size = None
    model.grad_checkpoint_rollout = False
    cost_ref = model.get_cost(dict(info_base), acts_ref)
    cost_ref.sum().backward()

    # Chunked (size=2 over b*s=6) + gradient checkpointing
    model.rollout_batch_size = 2
    model.grad_checkpoint_rollout = True
    cost_chk = model.get_cost(dict(info_base), acts_chk)
    cost_chk.sum().backward()

    torch.testing.assert_close(cost_ref, cost_chk)
    torch.testing.assert_close(acts_ref.grad, acts_chk.grad)

    # Legacy checkpoint compatibility: missing attributes fallback cleanly
    for attr in (
        'rollout_batch_size',
        'grad_checkpoint_rollout',
        'eval_autocast',
    ):
        delattr(model, attr)
    acts_legacy = acts_ref.detach().clone().requires_grad_(True)
    cost_legacy = model.get_cost(dict(info_base), acts_legacy)
    cost_legacy.sum().backward()
    torch.testing.assert_close(cost_ref, cost_legacy)
    torch.testing.assert_close(acts_ref.grad, acts_legacy.grad)
