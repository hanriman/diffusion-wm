"""Pre-encode HDF5 `pixels` with a pretrained SD-VAE into `vae_latents` dataset column."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffusion_wm.models.vae import SDVAEWrapper


def precompute_vae_latents(
    h5_path: str | Path,
    model_name: str = 'stabilityai/sd-vae-ft-mse',
    batch_size: int = 100,
    device: str | None = None,
    overwrite: bool = False,
    img_size: int | None = None,
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
        pixels_ds = f['pixels']
        n_total = pixels_ds.shape[0]
        # Support both (N, H, W, 3) and (N, 3, H, W)
        if pixels_ds.shape[-1] == 3:
            h, w = pixels_ds.shape[1], pixels_ds.shape[2]
            channel_last = True
        else:
            h, w = pixels_ds.shape[2], pixels_ds.shape[3]
            channel_last = False

        target_h = int(img_size) if img_size is not None else h
        target_w = int(img_size) if img_size is not None else w
        if target_h % 8 != 0 or target_w % 8 != 0:
            raise ValueError(
                f'Target image size ({target_h}, {target_w}) must be divisible by 8 for SD-VAE.'
            )
        lat_shape = (n_total, 4, target_h // 8, target_w // 8)

        if 'vae_latents' in f:
            is_complete = bool(f['vae_latents'].attrs.get('complete', False))
            same_shape = tuple(f['vae_latents'].shape) == lat_shape
            if is_complete and same_shape and not overwrite:
                print(
                    f"'vae_latents' {lat_shape} already exists in {path}. Pass --overwrite to recompute."
                )
                return
            del f['vae_latents']

        lat_ds = f.create_dataset(
            'vae_latents',
            shape=lat_shape,
            dtype='float16',
            chunks=(min(batch_size, n_total), 4, target_h // 8, target_w // 8),
        )
        lat_ds.attrs['complete'] = False

        for start in tqdm(
            range(0, n_total, batch_size),
            desc=f'Encoding {path.name} ({target_h}x{target_w}) -> vae_latents',
        ):
            end = min(start + batch_size, n_total)
            raw = np.asarray(pixels_ds[start:end])
            t_img = torch.from_numpy(raw).to(device)
            if channel_last:
                t_img = t_img.permute(0, 3, 1, 2).contiguous()
            if (h, w) != (target_h, target_w):
                t_img = (
                    F.interpolate(
                        t_img.float(),
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False,
                    )
                    .round()
                    .clamp(0.0, 255.0)
                    .to(torch.uint8)
                )
            amp_ctx = (
                torch.amp.autocast('cuda', dtype=torch.float16)
                if str(device).startswith('cuda')
                else nullcontext()
            )
            with torch.no_grad(), amp_ctx:
                latents = vae.encode(t_img).half().cpu().numpy()
            lat_ds[start:end] = latents

        lat_ds.attrs['complete'] = True


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
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    precompute_vae_latents(
        args.h5_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
        img_size=args.img_size,
    )


if __name__ == '__main__':
    main()
