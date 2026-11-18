"""Create a compact, valid HDF5 subset of an episode dataset (compatible with diffusion_wm and H-JEPA)."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm


def subset_h5(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    num_episodes: int,
    seed: int = 42,
    compression: str = 'zstd',
) -> None:
    src = Path(src_path).expanduser().resolve()
    dst = Path(dst_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f'Source HDF5 not found: {src}')
    dst.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src, 'r') as fin:
        ep_len = fin['ep_len'][:]
        ep_offset = fin['ep_offset'][:]
        total_eps = len(ep_len)
        k = min(int(num_episodes), total_eps)

        rng = np.random.default_rng(seed)
        selected_eps = np.sort(rng.choice(total_eps, size=k, replace=False))
        selected_lens = ep_len[selected_eps]
        selected_offsets = np.zeros(k, dtype=ep_offset.dtype)
        if k > 1:
            selected_offsets[1:] = np.cumsum(selected_lens[:-1], dtype=ep_offset.dtype)
        total_frames = int(selected_lens.sum())

        ep_col_name = (
            'episode_idx'
            if 'episode_idx' in fin
            else ('ep_idx' if 'ep_idx' in fin else None)
        )
        frame_keys = [
            key
            for key in fin
            if key not in ('ep_len', 'ep_offset')
        ]

        blosc_kwargs = dict(
            hdf5plugin.Blosc(
                cname=compression,
                clevel=5,
                shuffle=hdf5plugin.Blosc.SHUFFLE,
            )
        )

        with h5py.File(dst, 'w') as fout:
            fout.create_dataset('ep_len', data=selected_lens)
            fout.create_dataset('ep_offset', data=selected_offsets)

            for key in frame_keys:
                ds_in = fin[key]
                out_shape = (total_frames, *ds_in.shape[1:])
                chunks = (
                    (min(100, total_frames), *ds_in.shape[1:])
                    if ds_in.ndim >= 3
                    else True
                )
                create_kwargs = blosc_kwargs if key == 'pixels' else {}
                ds_out = fout.create_dataset(
                    key,
                    shape=out_shape,
                    dtype=ds_in.dtype,
                    chunks=chunks,
                    **create_kwargs,
                )

                if ds_in.ndim <= 2 and key != ep_col_name:
                    # Load small 1D/2D numeric arrays in memory once for fast slicing
                    full_arr = ds_in[:]
                    slices = [
                        full_arr[int(ep_offset[orig_ep]) : int(ep_offset[orig_ep]) + int(ep_len[orig_ep])]
                        for orig_ep in selected_eps
                    ]
                    ds_out[:] = np.concatenate(slices, axis=0)
                    continue

                if key == ep_col_name:
                    ds_out[:] = np.repeat(
                        np.arange(k, dtype=ds_in.dtype), selected_lens
                    )
                    continue

                write_pos = 0
                for orig_ep in tqdm(
                    selected_eps,
                    desc=f'{src.name} [{key}]',
                    leave=False,
                ):
                    s = int(ep_offset[orig_ep])
                    n = int(ep_len[orig_ep])
                    ds_out[write_pos : write_pos + n] = ds_in[s : s + n]
                    write_pos += n

    size_gb = dst.stat().st_size / (1024**3)
    print(
        f'Wrote {dst} ({k}/{total_eps} episodes, {total_frames:,} frames, {size_gb:.2f} GB)'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Subset an HDF5 dataset by sampling K episodes.'
    )
    parser.add_argument('src', type=str, help='Path to source .h5 file')
    parser.add_argument('dst', type=str, help='Path to output .h5 file')
    parser.add_argument('--num-episodes', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compression', type=str, default='zstd')
    args = parser.parse_args()

    subset_h5(
        args.src,
        args.dst,
        num_episodes=args.num_episodes,
        seed=args.seed,
        compression=args.compression,
    )


if __name__ == '__main__':
    main()
