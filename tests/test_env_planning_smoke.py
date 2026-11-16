"""End-to-end smoke tests for training, checkpointing, and closed-loop Push-T & AntMaze planning."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
from diffusion_wm.data import IMAGENET_STATS
from diffusion_wm.planning_eval import _build_eval_process_from_policy_normalizer
from diffusion_wm.train import train_from_config


def _create_synthetic_pusht_h5(path: Path, img_size: int = 64) -> None:
    n_episodes = 2
    ep_len = 25
    total = n_episodes * ep_len
    rng = np.random.default_rng(0)

    with h5py.File(path, 'w') as f:
        f.create_dataset(
            'pixels',
            data=rng.integers(0, 255, size=(total, img_size, img_size, 3), dtype=np.uint8),
        )
        f.create_dataset(
            'action',
            data=rng.uniform(-1.0, 1.0, size=(total, 2)).astype(np.float32),
        )
        f.create_dataset(
            'proprio',
            data=rng.normal(size=(total, 4)).astype(np.float32),
        )
        f.create_dataset(
            'state',
            data=rng.normal(size=(total, 7)).astype(np.float32),
        )
        ep_indices = np.repeat(np.arange(n_episodes, dtype=np.int64), ep_len)
        step_indices = np.tile(np.arange(ep_len, dtype=np.int64), n_episodes)
        f.create_dataset('episode_idx', data=ep_indices)
        f.create_dataset('step_idx', data=step_indices)
        f.create_dataset(
            'ep_len',
            data=np.full((n_episodes,), ep_len, dtype=np.int64),
        )
        f.create_dataset(
            'ep_offset',
            data=np.arange(0, total, ep_len, dtype=np.int64),
        )


def _create_synthetic_ant_h5(path: Path, img_size: int = 64) -> None:
    n_episodes = 2
    ep_len = 25
    total = n_episodes * ep_len
    rng = np.random.default_rng(1)

    with h5py.File(path, 'w') as f:
        f.create_dataset(
            'pixels',
            data=rng.integers(0, 255, size=(total, img_size, img_size, 3), dtype=np.uint8),
        )
        f.create_dataset(
            'action',
            data=rng.uniform(-1.0, 1.0, size=(total, 8)).astype(np.float32),
        )
        f.create_dataset(
            'xy',
            data=rng.normal(size=(total, 2)).astype(np.float32),
        )
        f.create_dataset(
            'qpos',
            data=rng.normal(size=(total, 15)).astype(np.float32),
        )
        f.create_dataset(
            'qvel',
            data=rng.normal(size=(total, 14)).astype(np.float32),
        )
        ep_indices = np.repeat(np.arange(n_episodes, dtype=np.int64), ep_len)
        step_indices = np.tile(np.arange(ep_len, dtype=np.int64), n_episodes)
        f.create_dataset('episode_idx', data=ep_indices)
        f.create_dataset('step_idx', data=step_indices)
        f.create_dataset(
            'ep_len',
            data=np.full((n_episodes,), ep_len, dtype=np.int64),
        )
        f.create_dataset(
            'ep_offset',
            data=np.arange(0, total, ep_len, dtype=np.int64),
        )


def _img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**IMAGENET_STATS),
            transforms.Resize(size=img_size),
        ]
    )


@pytest.mark.parametrize('solver_type', ['gd', 'cem'])
def test_pusht_train_and_closed_loop_planning(tmp_path: Path, solver_type: str):
    h5_file = tmp_path / 'pusht_synth.h5'
    _create_synthetic_pusht_h5(h5_file, img_size=64)

    cfg = OmegaConf.create(
        {
            'cache_dir': str(tmp_path),
            'run_dir': str(tmp_path / 'run_pusht'),
            'subdir': 'pusht/test/seed42',
            'output_model_name': 'pusht_dwm',
            'env': 'pusht',
            'seed': 42,
            'img_size': 64,
            'dump_object': True,
            'save_every_n_epochs': 1,
            'enable_logger': False,
            'data': {
                'dataset': {
                    'name': 'pusht_synth',
                    'val_name': 'pusht_synth',
                    'keys_to_load': ['pixels', 'action', 'proprio', 'state'],
                    'keys_to_cache': ['action', 'proprio', 'state'],
                    'level1': {'frameskip': 5, 'num_steps': 4},
                }
            },
            'trainer': {
                'max_epochs': 1,
                'accelerator': 'cpu',
                'devices': 1,
                'precision': '32-true',
                'num_sanity_val_steps': 0,
            },
            'loader': {
                'batch_size': 4,
                'num_workers': 0,
                'drop_last': False,
                'shuffle': True,
            },
            'optimizer': {'lr': 1e-3, 'weight_decay': 1e-4, 'warmup_ratio': 0.0},
            'vae': {'type': 'tiny', 'hidden_dim': 16, 'freeze': False},
            'wm': {
                'history_size': 3,
                'rollout_n': 1,
                'action_dim': 2,
                'action_embed_dim': 32,
                'use_proprio': False,
                'proprio_dim': 0,
                'num_ode_steps': 2,
                'ode_solver': 'euler',
                'rollout_mode': 'parallel',
            },
            'dit': {
                'patch_size': 2,
                'hidden_dim': 32,
                'depth': 2,
                'heads': 2,
                'dim_head': 16,
                'mlp_ratio': 2.0,
                'max_seq_len': 16,
            },
            'planning_eval': {'enabled': False},
        }
    )

    train_from_config(cfg)
    ckpt_path = tmp_path / 'run_pusht' / 'pusht_dwm_object.ckpt'
    norm_path = tmp_path / 'run_pusht' / 'normalizer.pt'
    assert ckpt_path.exists()
    assert norm_path.exists()

    model = swm.policy.AutoCostModel(str(ckpt_path)).to('cpu')
    eval_cfg = OmegaConf.create(
        {
            'policy': str(ckpt_path),
            'cache_dir': str(tmp_path),
            'dataset': {'keys_to_cache': ['action', 'proprio', 'state']},
        }
    )
    process = _build_eval_process_from_policy_normalizer(eval_cfg, model=model)
    transform = {'pixels': _img_transform(64), 'goal': _img_transform(64)}
    plan_cfg = swm.PlanConfig(
        horizon=3, receding_horizon=1, action_block=5, cost_last_n=1
    )

    if solver_type == 'gd':
        solver = swm.solver.GradientSolver(
            model=model,
            n_steps=2,
            min_its=2,
            num_samples=2,
            device='cpu',
            seed=42,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={'lr': 0.05},
        )
    else:
        solver = swm.solver.CEMSolver(
            model=model,
            n_steps=2,
            num_samples=4,
            num_elites=2,
            device='cpu',
            seed=42,
        )

    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_cfg,
        process=process,
        transform=transform,
        eval_budget=10,
    )

    world = swm.World(
        env_name='swm/PushT-v1',
        num_envs=1,
        image_shape=(64, 64),
        max_episode_steps=20,
        history_size=1,
        frame_skip=1,
    )
    try:
        world.set_policy(policy)
        world.reset(seed=42)
        for _ in range(5):
            world.step()
        assert policy._steps_taken == 5
        assert len(solver.solve_records) >= 1
    finally:
        world.close()


@pytest.mark.parametrize('solver_type', ['gd', 'cem'])
def test_antmaze_train_and_closed_loop_planning(
    tmp_path: Path, solver_type: str
):
    h5_file = tmp_path / 'ant_synth.h5'
    _create_synthetic_ant_h5(h5_file, img_size=64)

    cfg = OmegaConf.create(
        {
            'cache_dir': str(tmp_path),
            'run_dir': str(tmp_path / 'run_ant'),
            'subdir': 'ant/test/seed42',
            'output_model_name': 'ant_dwm',
            'env': 'ant',
            'seed': 42,
            'img_size': 64,
            'dump_object': True,
            'save_every_n_epochs': 1,
            'enable_logger': False,
            'data': {
                'dataset': {
                    'name': 'ant_synth',
                    'val_name': 'ant_synth',
                    'keys_to_load': [
                        'pixels',
                        'action',
                        'xy',
                        'qpos',
                        'qvel',
                        'proprio',
                    ],
                    'keys_to_cache': ['action', 'xy', 'qpos', 'qvel'],
                    'keys_to_merge': {
                        'proprio': [{'key': 'qpos', 'start': 2}, 'qvel']
                    },
                    'level1': {'frameskip': 5, 'num_steps': 4},
                }
            },
            'trainer': {
                'max_epochs': 1,
                'accelerator': 'cpu',
                'devices': 1,
                'precision': '32-true',
                'num_sanity_val_steps': 0,
            },
            'loader': {
                'batch_size': 4,
                'num_workers': 0,
                'drop_last': False,
                'shuffle': True,
            },
            'optimizer': {'lr': 1e-3, 'weight_decay': 1e-4, 'warmup_ratio': 0.0},
            'vae': {'type': 'tiny', 'hidden_dim': 16, 'freeze': False},
            'wm': {
                'history_size': 3,
                'rollout_n': 1,
                'action_dim': 8,
                'action_embed_dim': 32,
                'use_proprio': True,
                'proprio_dim': 27,
                'num_ode_steps': 2,
                'ode_solver': 'euler',
                'rollout_mode': 'parallel',
            },
            'dit': {
                'patch_size': 2,
                'hidden_dim': 32,
                'depth': 2,
                'heads': 2,
                'dim_head': 16,
                'mlp_ratio': 2.0,
                'max_seq_len': 16,
            },
            'planning_eval': {'enabled': False},
        }
    )

    train_from_config(cfg)
    ckpt_path = tmp_path / 'run_ant' / 'ant_dwm_object.ckpt'
    assert ckpt_path.exists()

    model = swm.policy.AutoCostModel(str(ckpt_path)).to('cpu')
    eval_cfg = OmegaConf.create(
        {
            'policy': str(ckpt_path),
            'cache_dir': str(tmp_path),
            'dataset': {
                'keys_to_cache': ['action', 'xy', 'qpos', 'qvel'],
                'keys_to_merge': {
                    'proprio': [{'key': 'qpos', 'start': 2}, 'qvel']
                },
            },
        }
    )
    process = _build_eval_process_from_policy_normalizer(eval_cfg, model=model)
    transform = {'pixels': _img_transform(64), 'goal': _img_transform(64)}
    plan_cfg = swm.PlanConfig(
        horizon=3, receding_horizon=1, action_block=5, cost_last_n=1
    )

    if solver_type == 'gd':
        solver = swm.solver.GradientSolver(
            model=model,
            n_steps=2,
            min_its=2,
            num_samples=2,
            device='cpu',
            seed=42,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={'lr': 0.05},
        )
    else:
        solver = swm.solver.CEMSolver(
            model=model,
            n_steps=2,
            num_samples=4,
            num_elites=2,
            device='cpu',
            seed=42,
        )

    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_cfg,
        process=process,
        transform=transform,
        eval_budget=10,
    )

    world = swm.World(
        env_name='swm/OGBAntMaze-v0',
        num_envs=1,
        image_shape=(64, 64),
        max_episode_steps=20,
        history_size=1,
        frame_skip=1,
        ob_type='pixels',
        width=64,
        height=64,
        camera_name='back',
        maze_type='medium',
        dataset_type='stitch',
        terminate_at_goal=True,
    )
    try:
        world.set_policy(policy)
        world.reset(seed=42)
        for _ in range(5):
            world.step()
        assert policy._steps_taken == 5
        assert len(solver.solve_records) >= 1
    finally:
        world.close()
