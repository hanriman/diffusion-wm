"""Dataset loading and column normalization utilities for Diffusion World Model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import stable_worldmodel as swm

NORMALIZER_ARTIFACT_FILENAME = 'normalizer.pt'
IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class ColumnTransform:
    """Apply `fn` to `sample[source]` and store the result in `sample[target]`."""

    def __init__(self, fn, source: str, target: str) -> None:
        self.fn, self.source, self.target = fn, source, target

    def __call__(self, sample: dict) -> dict:
        sample[self.target] = self.fn(sample[self.source])
        return sample


class Compose:
    """Apply dict-sample transforms in sequence."""

    def __init__(self, *transforms) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def _select_episodes_for_transition_budget(
    dataset,
    target_transitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid_episodes = np.flatnonzero(
        dataset.lengths >= dataset.last_level['span']
    )
    if valid_episodes.size == 0:
        raise ValueError(f'No valid episodes in {dataset.h5_path}.')

    available_transitions = int(dataset.lengths[valid_episodes].sum())
    if target_transitions > available_transitions:
        raise ValueError(
            f'Requested {target_transitions} transitions from {dataset.h5_path}, '
            f'but only {available_transitions} valid episode transitions are available.'
        )

    shuffled = rng.permutation(valid_episodes)
    chosen = []
    total = 0

    for episode in shuffled:
        length = int(dataset.lengths[episode])
        previous_total = total
        chosen.append(int(episode))
        total += length

        if total >= target_transitions:
            if (
                len(chosen) > 1
                and abs(target_transitions - previous_total)
                < abs(target_transitions - total)
            ):
                chosen.pop()
            break

    if not chosen:
        chosen.append(int(shuffled[0]))

    return np.asarray(sorted(chosen), dtype=np.int64)


def build_hdf5_dataset(dataset_cfg, cache_dir=None):
    if OmegaConf.is_config(dataset_cfg):
        cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    else:
        cfg = dict(dataset_cfg)
    cfg.pop('val_name', None)
    subset_seed = int(cfg.pop('subset_seed', 0))
    total_transitions = cfg.pop('total_transitions', None)
    dataset = swm.data.HDF5Dataset(**cfg, cache_dir=cache_dir)
    if total_transitions is not None:
        episodes = _select_episodes_for_transition_budget(
            dataset, int(total_transitions), np.random.default_rng(subset_seed)
        )
        selected = set(episodes.tolist())
        dataset.clip_indices = [
            (episode, start)
            for episode, start in dataset.clip_indices
            if episode in selected
        ]
    return dataset


def get_column_normalizer(dataset, source: str, target: str):
    """Get z-score normalizer for a specific column in the dataset."""
    if hasattr(dataset, 'get_col_stats'):
        mean_np, std_np = dataset.get_col_stats(source)
        mean = torch.from_numpy(np.array(mean_np)).float()
        std = torch.from_numpy(np.array(std_np)).float().clamp_min(1e-6)
    else:
        col_data = dataset.get_col_data(source)
        data = torch.from_numpy(np.array(col_data))
        data = data[~torch.isnan(data).any(dim=1)]
        mean = data.mean(0, keepdim=True).clone()
        std = data.std(0, keepdim=True).clone().clamp_min(1e-6)

    def norm_fn(x):
        x = (x - mean) / std
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).float()

    return ColumnTransform(norm_fn, source=source, target=target)


def normalizer_columns_from_dataset_cfg(dataset_cfg) -> list[str]:
    """Return non-pixel dataset columns that need z-score normalization."""
    columns = []
    for key in ('keys_to_load', 'keys_to_cache'):
        for col in dataset_cfg.get(key, []) or []:
            if col.startswith(('pixels', 'vae_latents')):
                continue
            if col not in columns:
                columns.append(col)

    keys_to_merge = dataset_cfg.get('keys_to_merge', None)
    if keys_to_merge is not None:
        for col in keys_to_merge:
            if col.startswith(('pixels', 'vae_latents')):
                continue
            if col not in columns:
                columns.append(col)

    return columns


def _column_mean_std(dataset, col: str) -> tuple[np.ndarray, np.ndarray, int]:
    if hasattr(dataset, 'get_col_stats'):
        mean, std = dataset.get_col_stats(col)
        count = int(dataset.lengths.sum())
        return (
            np.asarray(mean, dtype=np.float64),
            np.maximum(np.asarray(std, dtype=np.float64), 1e-6),
            count,
        )

    data = np.asarray(dataset.get_col_data(col))
    flat = data.reshape(data.shape[0], -1)
    valid_mask = ~np.isnan(flat).any(axis=1)
    valid = data[valid_mask].astype(np.float64, copy=False)
    count = int(valid.shape[0])
    if count == 0:
        raise ValueError(f'Column {col!r} has no finite rows.')

    mean = valid.mean(axis=0, keepdims=True)
    std = np.maximum(
        valid.std(axis=0, ddof=1 if count > 1 else 0, keepdims=True), 1e-6
    )
    return mean, std, count


def build_normalizer_artifact(cfg, train_dataset) -> dict:
    """Build serializable training normalizer stats for planning eval."""
    dataset_cfg = cfg.data.dataset
    stats = {}
    for col in normalizer_columns_from_dataset_cfg(dataset_cfg):
        mean, std, count = _column_mean_std(train_dataset, col)
        stats[col] = {
            'mean': mean.astype(np.float32),
            'std': std.astype(np.float32),
            'count': count,
        }

    return {
        'format': 'lejepa_training_normalizer_v1',
        'stats': stats,
        'metadata': {
            'dataset_name': str(dataset_cfg.get('name', '')),
            'columns': list(stats.keys()),
            'std_ddof': 1,
        },
    }


def save_normalizer_artifact(artifact: dict, run_dir: str | Path) -> Path:
    path = Path(run_dir) / NORMALIZER_ARTIFACT_FILENAME
    torch.save(artifact, path)
    return path


def load_normalizer_artifact(path: str | Path) -> dict:
    artifact = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not isinstance(artifact, dict):
        raise TypeError(f'Normalizer artifact must be a dict, got {type(artifact)}.')
    if artifact.get('format') != 'lejepa_training_normalizer_v1':
        raise ValueError(
            f"Unsupported normalizer artifact format: {artifact.get('format')!r}."
        )
    if not isinstance(artifact.get('stats'), dict):
        raise TypeError("Normalizer artifact is missing a 'stats' dictionary.")
    return artifact


def normalize_uint8_images_on_device(
    batch: dict, img_size: int | None = None
) -> dict:
    """Convert uint8 `(..., 3, H, W)` pixel tensors to ImageNet-normalized float32 (optionally resizing)."""
    mean_vals = IMAGENET_STATS['mean']
    std_vals = IMAGENET_STATS['std']
    target_hw = None if img_size is None else (int(img_size), int(img_size))
    for key, value in list(batch.items()):
        if not key.startswith('pixels'):
            continue
        if not torch.is_tensor(value):
            continue
        if value.ndim < 4 or value.size(-3) != 3:
            raise ValueError(
                f"Expected pixel tensor '{key}' to have channel-first RGB "
                f'shape (..., 3, H, W), got {tuple(value.shape)}'
            )
        if value.dtype == torch.uint8:
            view_shape = [1] * value.ndim
            view_shape[-3] = 3
            mean = value.new_tensor(mean_vals, dtype=torch.float32).view(*view_shape)
            std = value.new_tensor(std_vals, dtype=torch.float32).view(*view_shape)
            value = value.float().div(255.0).sub(mean).div(std)
        if target_hw is not None and value.shape[-2:] != target_hw:
            lead = value.shape[:-3]
            flat = value.reshape(-1, 3, value.size(-2), value.size(-1)).float()
            resized = F.interpolate(
                flat, size=target_hw, mode='bilinear', align_corners=False
            )
            value = resized.reshape(*lead, 3, *target_hw)
        batch[key] = value
    return batch
