"""Planning evaluation pipeline for Diffusion World Model on Push-T and AntMaze."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm

try:
    from .data import (
        IMAGENET_STATS,
        NORMALIZER_ARTIFACT_FILENAME,
        load_normalizer_artifact,
    )
    from .utils import resolve_model_checkpoint_path
except ImportError:
    from data import (
        IMAGENET_STATS,
        NORMALIZER_ARTIFACT_FILENAME,
        load_normalizer_artifact,
    )
    from utils import resolve_model_checkpoint_path


def _default_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**IMAGENET_STATS),
            transforms.Resize(size=int(cfg.eval.img_size)),
        ]
    )


def load_eval_config(
    config_name: str | None = None, config_path: str | None = None
) -> DictConfig:
    if config_path is not None:
        return OmegaConf.load(config_path)
    if config_name is None:
        raise ValueError('Either config_name or config_path must be provided.')
    eval_dir = Path(__file__).parent / 'config' / 'eval'
    return OmegaConf.load(eval_dir / f'{config_name}.yaml')


def _resolve_existing_path(path: str | Path) -> Path:
    raw_path = Path(path).expanduser()
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    candidates = (
        Path.cwd() / raw_path,
        pkg_dir / raw_path,
        repo_root / raw_path,
        repo_root / 'scratch' / 'H-JEPA' / 'h_jepa' / raw_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw_path


def get_episodes_length(dataset, episodes):
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.get('cache_dir') or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        keys_to_merge=cfg.dataset.get('keys_to_merge', None),
        cache_dir=dataset_path,
        level1=cfg.dataset['level1'],
    )


def save_eval_config(results_dir: Path, cfg: DictConfig) -> Path:
    config_path = results_dir / 'eval_config.yaml'
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return config_path


def resolve_results_dir(
    cfg: DictConfig,
    results_dir: str | Path | None = None,
) -> Path:
    if results_dir is not None:
        return Path(results_dir)
    output_dir = cfg.output.get('dir', None)
    if cfg.policy in (None, 'random'):
        return Path(__file__).parent if output_dir is None else Path(output_dir)
    ckpt_path = resolve_model_checkpoint_path(cfg.policy, cfg.get('cache_dir'))
    return ckpt_path.parent / (output_dir or 'planning_eval')


def _required_process_columns(cfg: DictConfig) -> list[str]:
    columns = [
        *cfg.dataset.keys_to_cache,
        *(cfg.dataset.get('keys_to_merge', None) or {}),
    ]
    return [
        col
        for col in dict.fromkeys(columns)
        if col != 'pixels' and not col.startswith('vae_latents')
    ]


def _normalizer_artifact_to_process(
    artifact: dict, required_cols: list[str]
) -> dict:
    stats = artifact['stats']
    missing = [col for col in required_cols if col not in stats]
    if missing:
        available = ', '.join(sorted(stats.keys()))
        raise KeyError(
            f'Training normalizer artifact is missing required columns {missing}. '
            f'Available columns: {available}'
        )

    process = {}
    for col in required_cols:
        col_stats = stats[col]
        mean = np.asarray(col_stats['mean'], dtype=np.float64).reshape(-1)
        scale = np.maximum(
            np.asarray(col_stats['std'], dtype=np.float64).reshape(-1), 1e-6
        )

        processor = preprocessing.StandardScaler()
        processor.mean_ = mean
        processor.scale_ = scale
        processor.var_ = np.square(scale)
        processor.n_features_in_ = int(mean.shape[0])
        processor.n_samples_seen_ = int(col_stats.get('count', 0))
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = processor

    return process


def _load_policy_normalizer_artifact(cfg: DictConfig) -> dict:
    ckpt_path = resolve_model_checkpoint_path(cfg.policy, cfg.get('cache_dir'))
    normalizer_path = ckpt_path.parent / NORMALIZER_ARTIFACT_FILENAME
    if not normalizer_path.exists():
        raise FileNotFoundError(
            f'Training normalizer artifact not found for policy checkpoint. '
            f'Expected: {normalizer_path}.'
        )
    return load_normalizer_artifact(normalizer_path)


def _build_eval_process_from_policy_normalizer(
    cfg: DictConfig,
    model=None,
) -> dict:
    artifact = (
        getattr(model, 'normalizer_artifact', None)
        if model is not None
        else None
    )
    if artifact is None:
        if cfg.get('policy', 'random') in (None, 'random'):
            return {}
        artifact = _load_policy_normalizer_artifact(cfg)

    return _normalizer_artifact_to_process(
        artifact, _required_process_columns(cfg)
    )


def _eval_sdpa_kernel():
    return sdpa_kernel(
        [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    )


def _apply_model_overrides(model, cfg: DictConfig) -> None:
    overrides = cfg.get('model_overrides', None)
    if not overrides:
        return
    for key in (
        'num_ode_steps',
        'ode_solver',
        'rollout_mode',
        'rollout_noise_scale',
        'rollout_batch_size',
        'grad_checkpoint_rollout',
        'eval_autocast',
        'proprio_cost_weight',
    ):
        if key in overrides and overrides[key] is not None:
            setattr(model, key, overrides[key])


def load_model(cfg: DictConfig, model=None):
    """Load `cfg.policy` or extract the `Costable` model and place on target device."""
    target_dev = str(cfg.solver.get('device', _default_device()))
    if target_dev.startswith('cuda') and not torch.cuda.is_available():
        target_dev = 'cpu'

    if model is None:
        model = swm.policy.AutoCostModel(cfg.policy, cfg.get('cache_dir'))
        model = model.to(target_dev)
    else:
        selected_model = next(
            (m for m in model.modules() if hasattr(m, 'get_cost')), None
        )
        if selected_model is None:
            raise RuntimeError(
                "No module with 'get_cost' found in the provided in-memory model."
            )
        model = selected_model.to(target_dev)

    _apply_model_overrides(model, cfg)
    model = model.eval()
    model.interpolate_pos_encoding = True
    return model


def _build_policy_plan_config(cfg: DictConfig) -> swm.PlanConfig:
    plan_cfg = cfg.plan_config
    action_dim = plan_cfg.get('action_dim', None)
    pre_decay = plan_cfg.get('pre_decay_horizon', None)
    return swm.PlanConfig(
        horizon=int(plan_cfg.horizon),
        receding_horizon=int(plan_cfg.receding_horizon),
        action_block=int(plan_cfg.get('action_block', 1)),
        action_dim=None if action_dim is None else int(action_dim),
        pre_decay_horizon=None if pre_decay is None else int(pre_decay),
        cost_last_n=int(plan_cfg.get('cost_last_n', 1)),
        intermediate_cost_weight=float(
            plan_cfg.get('intermediate_cost_weight', 1.0)
        ),
    )


def build_solver(cfg: DictConfig, model):
    solver_cfg = OmegaConf.create(OmegaConf.to_container(cfg.solver, resolve=True))
    if str(solver_cfg.get('device', 'cuda')).startswith('cuda') and not torch.cuda.is_available():
        solver_cfg.device = 'cpu'
    return hydra.utils.instantiate(solver_cfg, model=model)


def _build_policy(
    cfg: DictConfig,
    process: dict,
    transform: dict,
    model=None,
):
    if model is None and cfg.get('policy', 'random') in (None, 'random'):
        return swm.policy.RandomPolicy(seed=int(cfg.get('seed', 42)))

    model = load_model(cfg, model)
    policy_config = _build_policy_plan_config(cfg)
    solver = build_solver(cfg, model)

    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=policy_config,
        process=process,
        transform=transform,
        eval_budget=int(cfg.eval.eval_budget),
    )


def _serialize_metrics(metrics: dict) -> dict:
    metrics_to_save = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_to_save[key] = value.tolist()
        elif isinstance(value, np.generic):
            metrics_to_save[key] = value.item()
        elif torch.is_tensor(value):
            metrics_to_save[key] = value.detach().cpu().tolist()
        else:
            metrics_to_save[key] = value
    return metrics_to_save


def pop_solve_records(solver) -> list:
    if solver is None:
        return []
    solvers = getattr(solver, 'level_solvers', {1: solver})
    records = [
        r for level in sorted(solvers) for r in solvers[level].solve_records
    ]
    for s in solvers.values():
        s.solve_records = []
    return records


def _run_chunked_eval(
    cfg: DictConfig,
    chunk_size: int,
    world_cfg: DictConfig,
    image_shape: tuple,
    load_eval_trajs_path: str,
    process: dict,
    transform: dict,
    model,
    results_dir: Path,
) -> dict:
    payload = torch.load(
        load_eval_trajs_path, map_location='cpu', weights_only=False
    )
    n = int(cfg.eval.num_eval)
    if len(payload['data']) < n:
        raise ValueError(
            f"{load_eval_trajs_path}: {len(payload['data'])} tasks < eval.num_eval={n}"
        )
    ep_idx = payload.get('episodes_idx')
    ep_idx = (
        list(ep_idx)
        if ep_idx is not None and len(ep_idx)
        else list(range(len(payload['data'])))
    )
    callables = OmegaConf.to_container(cfg.eval.get('callables'), resolve=True)
    chunk_dir = results_dir / 'chunks'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = {
        s: chunk_dir / f'tasks_{s:03d}-{min(s + chunk_size, n):03d}.json'
        for s in range(0, n, chunk_size)
    }

    policy, gens = None, []
    for start, path in chunk_paths.items():
        if path.exists():
            continue
        end = min(start + chunk_size, n)
        if policy is None:
            policy = _build_policy(cfg, process, transform, model=model)
            solver = getattr(policy, 'solver', None)
            gens = [
                (s.torch_gen, s.torch_gen.initial_seed())
                for s in getattr(solver, 'level_solvers', {1: solver}).values()
                if hasattr(s, 'torch_gen')
            ]
        for gen, seed in gens:
            gen.manual_seed(seed + start)
        chunk = {
            **payload,
            'data': [dict(ep) for ep in payload['data'][start:end]],
            'episodes_idx': ep_idx[start:end],
        }
        world = swm.World(
            **{**world_cfg, 'num_envs': end - start}, image_shape=image_shape
        )
        try:
            world.set_policy(policy)
            t0 = time.time()
            with torch.enable_grad(), _eval_sdpa_kernel():
                metrics = world.evaluate_from_dataset(
                    None,
                    episodes_idx=None,
                    start_steps=None,
                    goal_offset_steps=cfg.eval.goal_offset_steps,
                    eval_budget=cfg.eval.eval_budget,
                    callables=callables,
                    load_eval_trajs_path=chunk,
                    process=process,
                    start_state_mode=cfg.eval.get(
                        'start_state_mode', 'dataset_full'
                    ),
                    goal_state_mode=cfg.eval.get(
                        'goal_state_mode', 'dataset_full'
                    ),
                )
            metrics['evaluation_time'] = time.time() - t0
        finally:
            world.close()
        metrics['solve_records'] = pop_solve_records(
            getattr(policy, 'solver', None)
        )
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(_serialize_metrics(metrics)))
        os.replace(tmp, path)
        print(
            f"tasks {start}-{end}: success_rate {metrics['success_rate']:.1f}"
        )

    chunks = [json.loads(p.read_text()) for p in chunk_paths.values()]
    successes = np.concatenate(
        [np.asarray(c['episode_successes'], dtype=bool) for c in chunks]
    )
    steps = np.concatenate(
        [np.asarray(c['steps_to_success'], dtype=np.float64) for c in chunks]
    )
    seeds = (
        None
        if chunks[0]['seeds'] is None
        else [s for c in chunks for s in c['seeds']]
    )
    metrics_to_save = {
        'success_rate': float(successes.sum()) / n * 100.0,
        'episode_successes': successes.tolist(),
        'seeds': seeds,
        'wall_clock_to_success': [
            x for c in chunks for x in c['wall_clock_to_success']
        ],
        'steps_to_success': steps.tolist(),
        'loaded_eval_original_lengths': [
            x for c in chunks for x in c['loaded_eval_original_lengths']
        ],
        'steps_to_success_success_only': (
            float(np.mean(steps[np.isfinite(steps)]))
            if np.isfinite(steps).any()
            else float('nan')
        ),
        'evaluation_time': sum(c['evaluation_time'] for c in chunks),
        'chunk_size': chunk_size,
    }
    (results_dir / 'metrics.yaml').write_text(
        OmegaConf.to_yaml(metrics_to_save)
    )
    print(metrics_to_save)
    solve_records = [r for c in chunks for r in c['solve_records']]
    if solve_records:
        (results_dir / 'planning_compute.json').write_text(
            json.dumps(solve_records)
        )
    return metrics_to_save


def run_planning_eval(
    cfg: DictConfig,
    model=None,
    results_dir: str | Path | None = None,
) -> dict:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    resolved_results_dir = resolve_results_dir(cfg, results_dir)
    resolved_results_dir.mkdir(parents=True, exist_ok=True)
    save_eval_config(resolved_results_dir, cfg)

    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))

    transform = {
        'pixels': img_transform(cfg),
        'goal': img_transform(cfg),
    }

    load_eval_trajs_path = cfg.get('load_eval_trajs_path', None)
    if load_eval_trajs_path:
        load_eval_trajs_path = str(_resolve_existing_path(load_eval_trajs_path))
    dump_eval_trajs_path = cfg.get('dump_eval_trajs_path', None)
    dump_eval_only = dump_eval_trajs_path is not None

    image_shape = (int(cfg.eval.img_size),) * 2

    dataset = None
    eval_episodes = None
    eval_start_idx = None
    process = {}

    if not dump_eval_only:
        process = _build_eval_process_from_policy_normalizer(cfg, model=model)

    chunk_size = cfg.eval.get('chunk_size', 1)
    if chunk_size and not dump_eval_only and load_eval_trajs_path:
        return _run_chunked_eval(
            cfg,
            int(chunk_size),
            world_cfg,
            image_shape,
            load_eval_trajs_path,
            process,
            transform,
            model,
            resolved_results_dir,
        )

    world = swm.World(**world_cfg, image_shape=image_shape)
    try:
        policy = _build_policy(cfg, process, transform, model=model)
        world.set_policy(policy)

        start_time = time.time()
        with torch.enable_grad(), _eval_sdpa_kernel():
            callables = OmegaConf.to_container(
                cfg.eval.get('callables'), resolve=True
            )
            start_steps = (
                eval_start_idx.tolist() if eval_start_idx is not None else None
            )
            episodes_idx = (
                eval_episodes.tolist() if eval_episodes is not None else None
            )
            metrics = world.evaluate_from_dataset(
                dataset,
                start_steps=start_steps,
                goal_offset_steps=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=episodes_idx,
                callables=callables,
                dump_eval_trajs_path=dump_eval_trajs_path,
                load_eval_trajs_path=load_eval_trajs_path,
                process=process,
                start_state_mode=cfg.eval.get(
                    'start_state_mode', 'dataset_full'
                ),
                goal_state_mode=cfg.eval.get(
                    'goal_state_mode', 'dataset_full'
                ),
            )
        end_time = time.time()
    finally:
        world.close()

    metrics_to_save = _serialize_metrics(metrics)
    metrics_to_save['evaluation_time'] = end_time - start_time

    metrics_path = resolved_results_dir / 'metrics.yaml'
    metrics_path.write_text(OmegaConf.to_yaml(metrics_to_save))
    print(metrics)

    solve_records = pop_solve_records(getattr(policy, 'solver', None))
    if solve_records:
        (resolved_results_dir / 'planning_compute.json').write_text(
            json.dumps(solve_records)
        )

    return metrics_to_save
