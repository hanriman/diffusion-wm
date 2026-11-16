"""VAE encoders and decoders for spatial latent world modeling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _imagenet_to_minus_one_one(x: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet-normalized RGB tensor (..., 3, H, W) to [-1, 1]."""
    mean = x.new_tensor(IMAGENET_MEAN).view(*([1] * (x.ndim - 3)), 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(*([1] * (x.ndim - 3)), 3, 1, 1)
    rgb_01 = (x * std + mean).clamp(0.0, 1.0)
    return rgb_01 * 2.0 - 1.0


class SDVAEWrapper(nn.Module):
    """Pretrained Stable Diffusion VAE wrapper (8x spatial downsample, 4 latent channels).

    Encodes `(..., 3, H, W)` images into `(..., 4, H // 8, W // 8)` scaled latents
    and decodes latents back into `[0, 1]` RGB images.
    """

    latent_channels: int = 4
    spatial_downsample: int = 8

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'stabilityai/sd-vae-ft-mse',
        scaling_factor: float = 0.18215,
        freeze: bool = True,
        use_posterior_mean: bool = True,
        encode_batch_size: int = 32,
        input_is_imagenet_normalized: bool = True,
    ) -> None:
        super().__init__()
        from diffusers import AutoencoderKL

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.scaling_factor = float(scaling_factor)
        self.freeze = bool(freeze)
        self.use_posterior_mean = bool(use_posterior_mean)
        self.encode_batch_size = max(1, int(encode_batch_size))
        self.input_is_imagenet_normalized = bool(input_is_imagenet_normalized)

        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)
        if self.freeze:
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.vae.eval()
        return self

    def _prepare_pixels(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            return x.float().div(127.5).sub(1.0)
        x = x.float()
        if self.input_is_imagenet_normalized:
            return _imagenet_to_minus_one_one(x)
        if x.min() >= 0.0 and x.max() <= 1.0:
            return x * 2.0 - 1.0
        return x.clamp(-1.0, 1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode `(..., 3, H, W)` RGB tensor to `(..., 4, H // 8, W // 8)` latents."""
        leading_shape = x.shape[:-3]
        c, h, w = x.shape[-3:]
        assert c == 3, f'Expected 3 RGB channels, got {c}'
        assert h % 8 == 0 and w % 8 == 0, (
            f'Image dimensions ({h}, {w}) must be divisible by 8 for SD-VAE.'
        )

        flat = self._prepare_pixels(x).reshape(-1, 3, h, w)
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        chunks = []
        with context:
            for i in range(0, flat.size(0), self.encode_batch_size):
                batch = flat[i : i + self.encode_batch_size]
                posterior = self.vae.encode(batch).latent_dist
                latents = (
                    posterior.mode()
                    if self.use_posterior_mean
                    else posterior.sample()
                )
                chunks.append(latents * self.scaling_factor)

        out = torch.cat(chunks, dim=0)
        return out.reshape(*leading_shape, 4, h // 8, w // 8)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode `(..., 4, H', W')` latents to `(..., 3, H' * 8, W' * 8)` RGB in [0, 1]."""
        leading_shape = z.shape[:-3]
        c, hp, wp = z.shape[-3:]
        assert c == 4, f'Expected 4 latent channels, got {c}'

        flat = (z.float() / self.scaling_factor).reshape(-1, 4, hp, wp)
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        chunks = []
        with context:
            for i in range(0, flat.size(0), self.encode_batch_size):
                batch = flat[i : i + self.encode_batch_size]
                decoded = self.vae.decode(batch).sample
                rgb_01 = ((decoded + 1.0) * 0.5).clamp(0.0, 1.0)
                chunks.append(rgb_01)

        out = torch.cat(chunks, dim=0)
        return out.reshape(*leading_shape, 3, hp * 8, wp * 8)


class TinyConvVAE(nn.Module):
    """Lightweight 8x downsampling convolutional autoencoder with 4 latent channels.

    Matches `SDVAEWrapper`'s exact input/output shapes `(..., 4, H // 8, W // 8)`
    for fast offline unit testing or lightweight local runs.
    """

    latent_channels: int = 4
    spatial_downsample: int = 8

    def __init__(
        self,
        hidden_dim: int = 32,
        scaling_factor: float = 0.18215,
        freeze: bool = False,
        input_is_imagenet_normalized: bool = True,
    ) -> None:
        super().__init__()
        self.scaling_factor = float(scaling_factor)
        self.freeze = bool(freeze)
        self.input_is_imagenet_normalized = bool(input_is_imagenet_normalized)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, 4, kernel_size=4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(
                hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_dim, 3, kernel_size=4, stride=2, padding=1),
        )
        if self.freeze:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def _prepare_pixels(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            return x.float().div(127.5).sub(1.0)
        x = x.float()
        if self.input_is_imagenet_normalized:
            return _imagenet_to_minus_one_one(x)
        if x.min() >= 0.0 and x.max() <= 1.0:
            return x * 2.0 - 1.0
        return x.clamp(-1.0, 1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-3]
        c, h, w = x.shape[-3:]
        flat = self._prepare_pixels(x).reshape(-1, c, h, w)
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            z = self.encoder(flat) * self.scaling_factor
        return z.reshape(*leading_shape, 4, h // 8, w // 8)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        leading_shape = z.shape[:-3]
        c, hp, wp = z.shape[-3:]
        flat = (z.float() / self.scaling_factor).reshape(-1, c, hp, wp)
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            decoded = self.decoder(flat)
            rgb_01 = ((torch.tanh(decoded) + 1.0) * 0.5).clamp(0.0, 1.0)
        return rgb_01.reshape(*leading_shape, 3, hp * 8, wp * 8)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        target_01 = (self._prepare_pixels(x) + 1.0) * 0.5
        z = self.encode(x)
        recon_01 = self.decode(z)
        return F.mse_loss(recon_01, target_01)


def build_vae(vae_cfg: dict) -> nn.Module:
    vae_type = str(vae_cfg.get('type', 'sd_vae')).lower()
    if vae_type == 'sd_vae':
        return SDVAEWrapper(
            pretrained_model_name_or_path=vae_cfg.get(
                'pretrained_model_name_or_path', 'stabilityai/sd-vae-ft-mse'
            ),
            scaling_factor=float(vae_cfg.get('scaling_factor', 0.18215)),
            freeze=bool(vae_cfg.get('freeze', True)),
            use_posterior_mean=bool(vae_cfg.get('use_posterior_mean', True)),
            encode_batch_size=int(vae_cfg.get('encode_batch_size', 32)),
            input_is_imagenet_normalized=bool(
                vae_cfg.get('input_is_imagenet_normalized', True)
            ),
        )
    if vae_type == 'tiny':
        return TinyConvVAE(
            hidden_dim=int(vae_cfg.get('hidden_dim', 32)),
            scaling_factor=float(vae_cfg.get('scaling_factor', 0.18215)),
            freeze=bool(vae_cfg.get('freeze', False)),
            input_is_imagenet_normalized=bool(
                vae_cfg.get('input_is_imagenet_normalized', True)
            ),
        )
    raise ValueError(f"Unsupported VAE type: {vae_type!r}. Expected 'sd_vae' or 'tiny'.")
