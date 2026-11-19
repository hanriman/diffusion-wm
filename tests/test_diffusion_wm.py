"""Unit tests for SpatioTemporalDiT, DiffusionWorldModel, GradientSolver, and CEMSolver."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

import stable_worldmodel as swm
from diffusion_wm.data import normalize_uint8_images_on_device
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


@pytest.mark.parametrize('use_action_token', [False, True])
def test_dit_shapes_and_causal_masking(use_action_token: bool):
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
        use_action_token=use_action_token,
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

    # Chunked (size=2 over b*s=6) + single-level ODE-step checkpointing (default)
    model.rollout_batch_size = 2
    model.grad_checkpoint_rollout = True
    model.grad_checkpoint_blocks = False
    cost_chk = model.get_cost(dict(info_base), acts_chk)
    cost_chk.sum().backward()

    torch.testing.assert_close(cost_ref, cost_chk)
    torch.testing.assert_close(acts_ref.grad, acts_chk.grad)

    # Chunked + nested block checkpointing also matches exact gradients
    acts_nested = acts_ref.detach().clone().requires_grad_(True)
    model.grad_checkpoint_blocks = True
    cost_nested = model.get_cost(dict(info_base), acts_nested)
    cost_nested.sum().backward()
    torch.testing.assert_close(cost_ref, cost_nested)
    torch.testing.assert_close(acts_ref.grad, acts_nested.grad)

    # Legacy checkpoint compatibility: missing attributes fallback cleanly
    for attr in (
        'rollout_batch_size',
        'grad_checkpoint_rollout',
        'grad_checkpoint_blocks',
        'eval_autocast',
    ):
        delattr(model, attr)
    acts_legacy = acts_ref.detach().clone().requires_grad_(True)
    cost_legacy = model.get_cost(dict(info_base), acts_legacy)
    cost_legacy.sum().backward()
    torch.testing.assert_close(cost_ref, cost_legacy)
    torch.testing.assert_close(acts_ref.grad, acts_legacy.grad)


def test_normalize_resize_and_tau_covers_hctx_1():
    batch = {'pixels_level1': torch.randint(0, 255, (2, 4, 3, 224, 224), dtype=torch.uint8)}
    out = normalize_uint8_images_on_device(batch, img_size=128)
    assert out['pixels_level1'].shape == (2, 4, 3, 128, 128)
    assert out['pixels_level1'].dtype == torch.float32

    model = _make_model(img_size=64, patch_size=2, action_dim=2)
    tau, mask = model._sample_training_tau(6, 4, 3, torch.device('cpu'))
    assert tau.shape == (6, 4) and mask.shape == (6, 4)
    # Sample 0 uses h_ctx=1 (matching eval start), sample 2 uses h_ctx=3
    assert (mask[0] == torch.tensor([0.0, 1.0, 1.0, 1.0])).all()
    assert (mask[2] == torch.tensor([0.0, 0.0, 0.0, 1.0])).all()


def test_autoregressive_sliding_window_preserves_null_action_at_pos0():
    torch.manual_seed(0)
    model = _make_model(
        img_size=64,
        patch_size=2,
        action_dim=2,
        frameskip=5,
        rollout_mode='autoregressive',
        num_ode_steps=1,
    ).eval()

    recorded_act_embs = []
    orig_dit_forward = model.dit.forward

    def _spy_forward(z_vis, tau, act_emb, z_proprio=None, **kwargs):
        recorded_act_embs.append(act_emb.detach().clone())
        return orig_dit_forward(z_vis, tau, act_emb, z_proprio=z_proprio, **kwargs)

    model.dit.forward = _spy_forward

    ctx_vis = torch.randn(1, 1, 4, 8, 8)
    actions = torch.randn(1, 5, 10)  # 5 future steps > history_size=2
    model.sample_ode_autoregressive(ctx_vis, actions, history_size=2)

    assert len(recorded_act_embs) == 5
    null_emb = model.null_action_emb.detach()[0, 0]
    for step_t, act_emb in enumerate(recorded_act_embs):
        torch.testing.assert_close(
            act_emb[0, 0],
            null_emb,
            msg=f'Slot 0 at autoregressive step {step_t} must use null_action_emb',
        )


def test_multi_context_actions_criterion_and_goal_time_dim():
    torch.manual_seed(1)
    model = _make_model(
        img_size=64,
        patch_size=2,
        action_dim=2,
        frameskip=5,
        rollout_mode='parallel',
        num_ode_steps=2,
    ).eval()

    # Verify ctx_actions affects multi-context rollout (h_ctx=3)
    ctx_vis = torch.randn(2, 3, 4, 8, 8)
    fut_actions = torch.randn(2, 1, 10)
    ctx_acts_a = torch.zeros(2, 2, 10)
    ctx_acts_b = torch.ones(2, 2, 10) * 3.0
    pred_a, _ = model.sample_ode_parallel(ctx_vis, fut_actions, ctx_actions=ctx_acts_a)
    pred_b, _ = model.sample_ode_parallel(ctx_vis, fut_actions, ctx_actions=ctx_acts_b)
    assert (pred_a[:, -1] - pred_b[:, -1]).abs().sum() > 0

    # Verify criterion only scores predicted frames (excludes context frames even if t_plan < cost_last_n)
    info_dict = {
        'pixels': torch.randn(2, 3, 3, 3, 64, 64),  # H_ctx = 3
        'goal': torch.randn(2, 3, 3, 3, 64, 64),    # H_goal = 3
        'cost_last_n': 2,
    }
    acts = torch.randn(2, 3, 1, 10)  # T_plan = 1 < cost_last_n = 2
    costs = model.get_cost(info_dict, acts)
    assert costs.shape == (2, 3)
    assert info_dict['rollout_ctx_len'] == 3
    assert info_dict['goal_embed_0'].shape[2] == 1


@pytest.mark.parametrize('rollout_mode', ['parallel', 'autoregressive'])
def test_rollout_noise_chunked_matches_unchunked(rollout_mode: str):
    torch.manual_seed(5)
    model = _make_model(
        img_size=64,
        patch_size=2,
        action_dim=2,
        frameskip=5,
        rollout_mode=rollout_mode,
        num_ode_steps=2,
    ).eval()
    model.rollout_noise_scale = 0.5

    b, s, t_plan = 2, 3, 4
    info_base = {
        'pixels': torch.randn(b, s, 1, 3, 64, 64),
        'goal': torch.randn(b, s, 1, 3, 64, 64),
        'cost_last_n': 2,
    }
    actions = torch.randn(b, s, t_plan, 10)

    model.rollout_batch_size = None
    cost_ref = model.get_cost(dict(info_base), actions)

    model.rollout_batch_size = 2
    cost_chk = model.get_cost(dict(info_base), actions)

    torch.testing.assert_close(cost_ref, cost_chk)


def test_hdf5_dataset_closes_handle_before_dataloader_fork(tmp_path):
    import h5py
    from torch.utils.data import DataLoader

    h5_path = tmp_path / 'fork_test.h5'
    n_eps, ep_len = 2, 25
    total = n_eps * ep_len
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('pixels', data=np.zeros((total, 16, 16, 3), dtype=np.uint8))
        f.create_dataset('action', data=np.zeros((total, 2), dtype=np.float32))
        f.create_dataset('qpos', data=np.zeros((total, 5), dtype=np.float32))
        f.create_dataset('qvel', data=np.zeros((total, 4), dtype=np.float32))
        f.create_dataset('ep_len', data=np.full((n_eps,), ep_len, dtype=np.int64))
        f.create_dataset('ep_offset', data=np.arange(0, total, ep_len, dtype=np.int64))

    ds = swm.data.HDF5Dataset(
        'fork_test',
        keys_to_load=['pixels', 'action', 'proprio'],
        keys_to_cache=['action'],
        keys_to_merge={'proprio': [{'key': 'qpos', 'start': 2}, 'qvel']},
        cache_dir=tmp_path,
        level1={'frameskip': 5, 'num_steps': 4},
    )
    assert ds.h5_file is None

    loader = DataLoader(ds, batch_size=4, num_workers=2)
    batch = next(iter(loader))
    assert batch['proprio_level1'].shape == (4, 4, 7)


def test_planning_overrides_normalization_and_action_bounds():
    from omegaconf import OmegaConf

    from diffusion_wm.planning_eval import _apply_model_overrides

    model = _make_model(img_size=64, patch_size=2, action_dim=2).eval()
    _apply_model_overrides(
        model,
        OmegaConf.create({'model_overrides': {'rollout_mode': 'AUTOREGRESSIVE', 'ode_solver': 'MIDPOINT'}}),
    )
    assert model.rollout_mode == 'autoregressive'
    assert model.ode_solver == 'midpoint'

    # Verify GradientSolver respects action_bounds passed via set_action_bounds
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 2), dtype=np.float32)
    plan_cfg = swm.PlanConfig(horizon=2, receding_horizon=1, action_block=5, cost_last_n=1)
    gd_solver = swm.solver.GradientSolver(
        model=model,
        n_steps=5,
        min_its=5,
        num_samples=2,
        var_scale=2.0,
        device='cpu',
        seed=0,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={'lr': 1.0},
    )
    gd_solver.configure(action_space=action_space, n_envs=1, config=plan_cfg)
    gd_solver.set_action_bounds(np.full(10, -1.0), np.full(10, 1.0))
    out = gd_solver.solve(
        {'pixels': torch.randn(1, 1, 3, 64, 64), 'goal': torch.randn(1, 1, 3, 64, 64)},
        planning_horizon=2,
        steps_taken=0,
        eval_budget=10,
    )
    assert (out['actions'] >= -1.0 - 1e-6).all() and (out['actions'] <= 1.0 + 1e-6).all()
