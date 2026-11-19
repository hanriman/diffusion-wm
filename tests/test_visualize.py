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


class _DummyScaler:
    def __init__(self, dim: int):
        self.mean_ = np.zeros(dim)
        self.scale_ = np.ones(dim)

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x
