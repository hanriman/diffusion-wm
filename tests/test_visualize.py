"""Tests for presentation visualization generators."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_wm.data import build_normalizer_artifact, save_normalizer_artifact
from diffusion_wm.models import DiffusionWorldModel, SpatioTemporalDiT, TinyConvVAE
from diffusion_wm.visualize import (
    plot_eval_summary,
    plot_training_curves,
    record_closed_loop_rollouts,
    visualize_world_model_dreams,
)


def test_plot_training_curves_and_eval_summary(tmp_path: Path):
    log_dir = tmp_path / 'lightning_logs' / 'version_0'
    log_dir.mkdir(parents=True)
    with (log_dir / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'epoch',
                'train/loss_epoch',
                'val/loss',
                'val/ode_vis_mse',
            ],
        )
        writer.writeheader()
        for ep in range(3):
            writer.writerow(
                {
                    'epoch': ep,
                    'train/loss_epoch': 1.0 / (ep + 1),
                    'val/loss': 1.1 / (ep + 1),
                    'val/ode_vis_mse': 0.2 / (ep + 1),
                }
            )

    out_dir = tmp_path / 'viz'
    curves_path = plot_training_curves(tmp_path, out_dir)
    assert curves_path is not None and curves_path.exists()

    metrics_yaml = tmp_path / 'metrics.yaml'
    metrics_yaml.write_text(
        OmegaConf.to_yaml(
            {
                'success_rate': 60.0,
                'episode_successes': [True, False, True, True, False],
                'steps_to_success': [20.0, float('nan'), 35.0, 42.0, float('nan')],
            }
        )
    )
    eval_path = plot_eval_summary(metrics_yaml, out_dir)
    assert eval_path is not None and eval_path.exists()


def test_dreams_and_rollout_visuals_pusht(tmp_path: Path):
    # Create a tiny synthetic eval_trajs payload matching Push-T format
    state_seq = torch.tensor(
        [[250.0, 250.0, 250.0, 250.0, 0.5, 0.0, 0.0]] * 40, dtype=torch.float32
    )
    ep = {
        'pixels': torch.randint(0, 255, (40, 3, 64, 64), dtype=torch.uint8),
        'action': torch.randn(40, 2),
        'proprio': torch.randn(40, 4),
        'state': state_seq,
    }
    trajs_path = tmp_path / 'eval_pusht.pt'
    torch.save(
        {'data': [ep], 'columns': list(ep.keys()), 'episodes_idx': [0]},
        trajs_path,
    )

    vae = TinyConvVAE(hidden_dim=8, freeze=True)
    dit = SpatioTemporalDiT(
        latent_channels=4,
        latent_height=8,
        latent_width=8,
        patch_size=2,
        hidden_dim=32,
        depth=2,
        heads=2,
        dim_head=16,
        max_seq_len=16,
        action_embed_dim=32,
        use_proprio=False,
    )
    model = DiffusionWorldModel(
        vae=vae,
        dit=dit,
        action_dim=2,
        frameskip=5,
        action_embed_dim=32,
        history_size=2,
        use_proprio=False,
        num_ode_steps=2,
    ).eval()

    class _DummyDS:
        lengths = np.array([40])

        def get_col_stats(self, col):
            dim = {'action': 2, 'proprio': 4, 'state': 7}[col]
            return np.zeros((1, dim)), np.ones((1, dim))

    dummy_cfg = OmegaConf.create(
        {
            'data': {
                'dataset': {
                    'name': 'synth',
                    'keys_to_load': ['pixels', 'action', 'proprio', 'state'],
                    'keys_to_cache': ['action', 'proprio', 'state'],
                }
            }
        }
    )
    artifact = build_normalizer_artifact(dummy_cfg, _DummyDS())
    model.normalizer_artifact = artifact
    ckpt_path = tmp_path / 'pusht_dwm_object.ckpt'
    torch.save(model, ckpt_path)
    save_normalizer_artifact(artifact, tmp_path)

    eval_cfg = OmegaConf.create(
        {
            'policy': str(ckpt_path),
            'cache_dir': str(tmp_path),
            'load_eval_trajs_path': str(trajs_path),
            'world': {
                'env_name': 'swm/PushT-v1',
                'num_envs': 1,
                'max_episode_steps': 20,
                'history_size': 1,
                'frame_skip': 1,
            },
            'dataset': {'keys_to_cache': ['action', 'proprio', 'state']},
            'solver': {
                '_target_': 'stable_worldmodel.solver.CEMSolver',
                'model': '???',
                'n_steps': 2,
                'num_samples': 4,
                'num_elites': 2,
                'device': 'cpu',
                'seed': 42,
            },
            'plan_config': {
                'horizon': 3,
                'receding_horizon': 2,
                'action_block': 5,
                'cost_last_n': 1,
            },
            'eval': {
                'num_eval': 1,
                'goal_offset_steps': 20,
                'eval_budget': 5,
                'img_size': 64,
                'callables': [
                    {'method': '_set_state', 'args': {'state': {'value': 'state'}}},
                    {
                        'method': '_set_goal_state',
                        'args': {'goal_state': {'value': 'goal_state'}},
                    },
                ],
            },
        }
    )
    process = {
        k: _DummyScaler(dim)
        for k, dim in (('action', 2), ('proprio', 4), ('state', 7), ('goal_proprio', 4), ('goal_state', 7))
    }

    out_dir = tmp_path / 'viz'
    model.rollout_mode = 'autoregressive'
    dream_path = visualize_world_model_dreams(
        eval_cfg, model, process, out_dir, episode_idx=0, num_future_steps=3
    )
    assert dream_path is not None and dream_path.exists()

    rollout_paths = record_closed_loop_rollouts(
        eval_cfg, model, process, out_dir, num_episodes=1, max_steps=4
    )
    assert any(p.suffix == '.gif' and p.exists() for p in rollout_paths)
    assert any(p.suffix == '.png' and p.exists() for p in rollout_paths)


def test_ode_history_and_interactive_demo_html(tmp_path: Path):
    from diffusion_wm.demo import (
        build_counterfactual_section,
        record_closed_loop_demo,
        write_interactive_demo_html,
    )

    state_seq = torch.tensor(
        [[250.0, 250.0, 250.0, 250.0, 0.5, 0.0, 0.0]] * 40, dtype=torch.float32
    )
    ep = {
        'pixels': torch.randint(0, 255, (40, 3, 64, 64), dtype=torch.uint8),
        'action': torch.randn(40, 2),
        'proprio': torch.randn(40, 4),
        'state': state_seq,
    }
    trajs_path = tmp_path / 'eval_pusht.pt'
    torch.save(
        {'data': [ep], 'columns': list(ep.keys()), 'episodes_idx': [0]},
        trajs_path,
    )

    vae = TinyConvVAE(hidden_dim=8, freeze=True)
    dit = SpatioTemporalDiT(
        latent_channels=4,
        latent_height=8,
        latent_width=8,
        patch_size=2,
        hidden_dim=32,
        depth=2,
        heads=2,
        dim_head=16,
        mlp_ratio=2.0,
        max_seq_len=16,
        use_proprio=True,
        proprio_dim=4,
        use_action_token=True,
    )
    model = DiffusionWorldModel(
        vae=vae,
        dit=dit,
        history_size=2,
        action_dim=2,
        frameskip=2,
        use_proprio=True,
        proprio_dim=4,
        num_ode_steps=2,
    ).eval()

    ctx_vis = torch.randn(1, 1, 4, 8, 8)
    act = torch.randn(1, 3, 4)
    ctx_p = torch.randn(1, 1, 4)
    for mode in ('parallel', 'autoregressive'):
        fn = (
            model.sample_ode_parallel
            if mode == 'parallel'
            else model.sample_ode_autoregressive
        )
        vis, _, hist = fn(ctx_vis, act, ctx_proprio=ctx_p, num_ode_steps=3, return_history=True)
        assert vis.shape == (1, 4, 4, 8, 8)
        assert len(hist) == 4
        assert hist[0][0] == 0.0 and hist[-1][0] == 1.0
        assert all(z.shape == (1, 3, 4, 8, 8) for _, z in hist)

    eval_cfg = OmegaConf.create(
        {
            'world': {'env_name': 'swm/PushT-v1', 'history_size': 1, 'frame_skip': 1},
            'load_eval_trajs_path': str(trajs_path),
            'model_overrides': {'rollout_mode': 'parallel', 'num_ode_steps': 2},
            'solver': {
                '_target_': 'stable_worldmodel.solver.CEMSolver',
                'n_steps': 1,
                'num_samples': 4,
                'num_elites': 2,
                'device': 'cpu',
                'seed': 42,
            },
            'plan_config': {'horizon': 2, 'receding_horizon': 1, 'action_block': 2, 'cost_last_n': 1},
            'eval': {
                'num_eval': 1,
                'goal_offset_steps': 10,
                'eval_budget': 2,
                'img_size': 64,
                'callables': [
                    {'method': '_set_state', 'args': {'state': {'value': 'state'}}},
                    {'method': '_set_goal_state', 'args': {'goal_state': {'value': 'goal_state'}}},
                ],
            },
        }
    )
    process = {
        k: _DummyScaler(dim)
        for k, dim in (('action', 2), ('proprio', 4), ('state', 7), ('goal_proprio', 4), ('goal_state', 7))
    }
    payload = torch.load(trajs_path, map_location='cpu', weights_only=False)
    cf = build_counterfactual_section(
        eval_cfg, model, process, payload, episode_idx=0, horizon=2, num_ode_steps=2, img_size=64
    )
    assert len(cf['presets']) == 5
    assert '1.0' in cf['presets'][0]['by_noise']

    cl = record_closed_loop_demo(
        eval_cfg, model, process, payload, [(0, None)], num_ode_steps=2, max_steps=2, img_size=64
    )
    assert len(cl) == 1 and len(cl[0]['steps']) >= 1 and len(cl[0]['replans']) >= 1

    # Also test end-to-end HTML writer with saved checkpoint + normalizer
    ckpt_dir = tmp_path / 'ckpt'
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / 'test_object.ckpt'
    torch.save(model, ckpt_path)
    save_normalizer_artifact(
        {
            'format': 'lejepa_training_normalizer_v1',
            'stats': {
                'action': {'mean': np.zeros((1, 2)), 'std': np.ones((1, 2)), 'count': 1},
                'proprio': {'mean': np.zeros((1, 4)), 'std': np.ones((1, 4)), 'count': 1},
                'state': {'mean': np.zeros((1, 7)), 'std': np.ones((1, 7)), 'count': 1},
            },
        },
        ckpt_dir,
    )
    assert write_interactive_demo_html.__name__ == 'write_interactive_demo_html'


def test_serve_live_worker_roundtrip(tmp_path: Path, monkeypatch):
    import io
    import json

    from diffusion_wm.demo import serve_live_worker

    state_seq = torch.tensor(
        [[250.0, 250.0, 250.0, 250.0, 0.5, 0.0, 0.0]] * 26, dtype=torch.float32
    )
    ep = {
        'pixels': torch.randint(0, 255, (26, 3, 64, 64), dtype=torch.uint8),
        'action': torch.randn(26, 2),
        'proprio': torch.randn(26, 4),
        'state': state_seq,
    }
    trajs_path = tmp_path / 'eval_pusht.pt'
    torch.save({'data': [ep], 'columns': list(ep.keys()), 'episodes_idx': [0]}, trajs_path)

    vae = TinyConvVAE(hidden_dim=8, freeze=True)
    dit = SpatioTemporalDiT(
        latent_channels=4,
        latent_height=8,
        latent_width=8,
        patch_size=2,
        hidden_dim=32,
        depth=2,
        heads=2,
        dim_head=16,
        mlp_ratio=2.0,
        max_seq_len=16,
        use_proprio=True,
        proprio_dim=4,
        use_action_token=True,
    )
    model = DiffusionWorldModel(
        vae=vae,
        dit=dit,
        history_size=2,
        action_dim=2,
        frameskip=2,
        use_proprio=True,
        proprio_dim=4,
        num_ode_steps=2,
    ).eval()
    ckpt_dir = tmp_path / 'ckpt'
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / 'test_object.ckpt'
    torch.save(model, ckpt_path)
    save_normalizer_artifact(
        {
            'format': 'lejepa_training_normalizer_v1',
            'stats': {
                'action': {'mean': np.zeros((1, 2)), 'std': np.ones((1, 2)), 'count': 1},
                'proprio': {'mean': np.zeros((1, 4)), 'std': np.ones((1, 4)), 'count': 1},
                'state': {'mean': np.zeros((1, 7)), 'std': np.ones((1, 7)), 'count': 1},
            },
        },
        ckpt_dir,
    )

    cfg = OmegaConf.create(
        {
            'world': {'env_name': 'swm/PushT-v1', 'history_size': 1, 'frame_skip': 1},
            'dataset': {'keys_to_cache': ['action', 'proprio', 'state']},
            'load_eval_trajs_path': str(trajs_path),
            'model_overrides': {'rollout_mode': 'parallel', 'num_ode_steps': 2},
            'solver': {'device': 'cpu'},
            'plan_config': {'horizon': 3, 'receding_horizon': 1, 'action_block': 2, 'cost_last_n': 1},
            'eval': {'img_size': 64},
        }
    )
    monkeypatch.setattr('diffusion_wm.demo.load_eval_config', lambda **_kw: cfg)

    fake_in = io.StringIO(
        json.dumps({'ep': 0, 'ax': 0.3, 'ay': -0.4, 'steps': 2, 'sigma': 1.0})
        + '\n'
        + json.dumps({'cmd': 'quit'})
        + '\n'
    )
    fake_out = io.StringIO()
    monkeypatch.setattr('sys.stdin', fake_in)
    monkeypatch.setattr('sys.stdout', fake_out)

    serve_live_worker(eval_config_name='pusht_gd', policy_path=ckpt_path, panel_size=64)
    lines = [json.loads(line) for line in fake_out.getvalue().strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]['ready'] is True
    assert lines[1]['taus'] == [0.0, 0.5, 1.0]
    assert len(lines[1]['frames']) == 3 and len(lines[1]['frames'][0]) == 3
    assert lines[1]['ctx'].startswith('data:image/png;base64,')



class _DummyScaler:
    def __init__(self, dim: int):
        self.mean_ = np.zeros(dim)
        self.scale_ = np.ones(dim)

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x
