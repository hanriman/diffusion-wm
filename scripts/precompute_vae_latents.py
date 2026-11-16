"""Pre-encode HDF5 `pixels` with a pretrained SD-VAE into `vae_latents` dataset column."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from diffusion_wm.models.vae import SDVAEWrapper


def precompute_vae_latents(
    h5_path: str | Path,
    model_name: str = 'stabilityai/sd-vae-ft-mse',
    batch_size: int = 64,
    device: str | None = None,
    overwrite: bool = False,
) -> None:
    path = Path(h5_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'HDF5 file not found: {path}')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae = SDVAEWrapper(
        pretrained_model_name_or_path=model_name,
        freeze=True,
        use_posterior_mean=True,
        encode_batch_size=batch_size,
        input_is_imagenet_normalized=False,
    ).to(device)

    with h5py.File(path, 'r+') as f:
        if 'pixels' not in f:
            raise KeyError(f"'pixels' dataset not found in {path}")
        if 'vae_latents' in f:
            if not overwrite:
                print(
                    f"'vae_latents' already exists in {path}. Pass --overwrite to recompute."
                )
                return
            del f['vae_latents']

        pixels_ds = f['pixels']
        n_total = pixels_ds.shape[0]
        # Support both (N, H, W, 3) and (N, 3, H, W)
        if pixels_ds.shape[-1] == 3:
            h, w = pixels_ds.shape[1], pixels_ds.shape[2]
            channel_last = True
        else:
            h, w = pixels_ds.shape[2], pixels_ds.shape[3]
            channel_last = False

        lat_shape = (n_total, 4, h // 8, w // 8)
        lat_ds = f.create_dataset(
            'vae_latents',
            shape=lat_shape,
            dtype='float32',
            chunks=(min(batch_size, n_total), 4, h // 8, w // 8),
        )

        for start in tqdm(
            range(0, n_total, batch_size),
            desc=f'Encoding {path.name} -> vae_latents',
        ):
            end = min(start + batch_size, n_total)
            raw = np.asarray(pixels_ds[start:end])
            t_img = torch.from_numpy(raw).to(device)
            if channel_last:
                t_img = t_img.permute(0, 3, 1, 2).contiguous()
            with torch.no_grad():
                latents = vae.encode(t_img).cpu().numpy()
            lat_ds[start:end] = latents


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Precompute SD-VAE latents into an HDF5 dataset file.'
    )
    parser.add_argument('h5_path', type=str, help='Path to the .h5 dataset file')
    parser.add_argument(
        '--model-name',
        type=str,
        default='stabilityai/sd-vae-ft-mse',
        help='HuggingFace SD-VAE model ID',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    precompute_vae_latents(
        args.h5_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
