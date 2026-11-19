"""Spatio-Temporal Causal Diffusion Transformer (DiT) for Diffusion Forcing."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint


class SinusoidalTimestepEmbedder(nn.Module):
    """Embeds continuous scalar flow/diffusion timesteps tau in [0, 1] into vectors."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: float = 10000.0
    ) -> torch.Tensor:
        """Create sinusoidal timestep embeddings for `t` of shape `(N,)`."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        # Scale t in [0, 1] by 1000.0 for standard DiT frequency resolution
        args = (t.float() * 1000.0).unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed `t` of shape `(B, T)` into `(B, T, hidden_size)`."""
        shape = t.shape
        t_flat = t.reshape(-1)
        t_freq = self.timestep_embedding(t_flat, self.frequency_embedding_size)
        out = self.mlp(t_freq.to(dtype=self.mlp[0].weight.dtype))
        return out.reshape(*shape, -1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class _SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 6,
        dim_head: int = 64,
        dropout: float = 0.0,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.is_causal = is_causal
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (Batch, Seq, Dim)"""
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.view(b, n, self.heads, self.dim_head).transpose(1, 2) for t in qkv
        )
        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop_p, is_causal=self.is_causal
        )
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class SpatioTemporalDiTBlock(nn.Module):
    """Factorized Spatio-Temporal DiT block with AdaLN-Zero conditioning.

    Each block applies:
      1. Spatial bidirectional self-attention within each frame `t`.
      2. Causal temporal self-attention across frames `0..T-1` per token slot.
      3. Pointwise MLP.
    All three sublayers are modulated by per-frame conditioning `c` of shape `(B, T, D)`.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int = 6,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_spatial = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn_spatial = _SelfAttention(
            hidden_dim, heads=heads, dim_head=dim_head, dropout=dropout, is_causal=False
        )

        self.norm_temporal = nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.attn_temporal = _SelfAttention(
            hidden_dim, heads=heads, dim_head=dim_head, dropout=dropout, is_causal=True
        )

        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # 9 modulation parameters: (shift, scale, gate) x (spatial, temporal, mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim, bias=True),
        )
        nn.init.normal_(self.adaLN_modulation[-1].weight, std=0.02)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token tensor of shape `(B, T, N, D)`.
            c: Per-frame conditioning tensor of shape `(B, T, D)`.
        """
        b, t, n, _d = x.shape
        (
            shift_s,
            scale_s,
            gate_s,
            shift_t,
            scale_t,
            gate_t,
            shift_m,
            scale_m,
            gate_m,
        ) = self.adaLN_modulation(c).unsqueeze(2).chunk(9, dim=-1)

        # 1. Spatial self-attention within each frame (B*T, N, D)
        x_s = _modulate(self.norm_spatial(x), shift_s, scale_s)
        x_s = rearrange(x_s, 'b t n d -> (b t) n d')
        x_s = self.attn_spatial(x_s)
        x_s = rearrange(x_s, '(b t) n d -> b t n d', b=b, t=t)
        x = x + gate_s * x_s

        # 2. Causal temporal self-attention across frames (B*N, T, D)
        x_t = _modulate(self.norm_temporal(x), shift_t, scale_t)
        x_t = rearrange(x_t, 'b t n d -> (b n) t d')
        x_t = self.attn_temporal(x_t)
        x_t = rearrange(x_t, '(b n) t d -> b t n d', b=b, n=n)
        x = x + gate_t * x_t

        # 3. Pointwise MLP
        x_m = _modulate(self.norm_mlp(x), shift_m, scale_m)
        x = x + gate_m * self.mlp(x_m)
        return x


class FinalAdaLNHead(nn.Module):
    """AdaLN-Zero final output head projecting tokens of shape `(B, T, N, D)`."""

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        nn.init.normal_(self.adaLN_modulation[-1].weight, std=0.02)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: `(B, T, N, D)` or `(B, T, 1, D)`
            c: `(B, T, D)`
        """
        shift, scale = self.adaLN_modulation(c).unsqueeze(2).chunk(2, dim=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class SpatioTemporalDiT(nn.Module):
    """Spatio-Temporal Causal DiT backbone for Diffusion Forcing World Models.

    Predicts velocity fields `(v_vis, v_proprio)` from noisy spatial VAE latents
    `z_vis` `(B, T, C, H_lat, W_lat)`, optional noisy `z_proprio` `(B, T, proprio_dim)`,
    per-frame flow timesteps `tau` `(B, T)`, and per-frame action embeddings `act_emb` `(B, T, act_emb_dim)`.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 4,
        latent_height: int = 28,
        latent_width: int = 28,
        patch_size: int = 2,
        hidden_dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 32,
        action_embed_dim: int = 192,
        use_proprio: bool = False,
        proprio_dim: int = 0,
        use_action_token: bool = False,
    ) -> None:
        super().__init__()
        assert latent_height % patch_size == 0 and latent_width % patch_size == 0, (
            f'Latent shape ({latent_height}, {latent_width}) must be divisible by patch_size={patch_size}'
        )
        self.latent_channels = int(latent_channels)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.use_proprio = bool(use_proprio)
        self.proprio_dim = int(proprio_dim) if use_proprio else 0
        self.use_action_token = bool(use_action_token)

        self.grid_h = latent_height // patch_size
        self.grid_w = latent_width // patch_size
        self.num_vis_patches = self.grid_h * self.grid_w
        self.patch_dim = latent_channels * patch_size * patch_size
        self.num_tokens = (
            self.num_vis_patches
            + (1 if self.use_proprio else 0)
            + (1 if self.use_action_token else 0)
        )

        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)
        if self.use_proprio:
            assert self.proprio_dim > 0, 'proprio_dim must be > 0 when use_proprio=True'
            self.proprio_embed = nn.Sequential(
                nn.Linear(self.proprio_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proprio_head = FinalAdaLNHead(hidden_dim, self.proprio_dim)
        else:
            self.proprio_embed = None
            self.proprio_head = None

        if self.use_action_token:
            self.action_token_proj = nn.Sequential(
                nn.Linear(action_embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.action_token_proj = None

        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.num_tokens, hidden_dim) * 0.02
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, 1, hidden_dim) * 0.02
        )

        self.time_embedder = SinusoidalTimestepEmbedder(hidden_dim)
        self.action_cond_proj = nn.Sequential(
            nn.Linear(action_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.vis_head = FinalAdaLNHead(hidden_dim, self.patch_dim)

    def _patchify(self, z_vis: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H', W') -> (B, T, N_patches, C * p * p)"""
        p = self.patch_size
        return rearrange(
            z_vis,
            'b t c (gh p1) (gw p2) -> b t (gh gw) (p1 p2 c)',
            p1=p,
            p2=p,
        )

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, T, N_patches, C * p * p) -> (B, T, C, H', W')"""
        p = self.patch_size
        return rearrange(
            patches,
            'b t (gh gw) (p1 p2 c) -> b t c (gh p1) (gw p2)',
            gh=self.grid_h,
            gw=self.grid_w,
            p1=p,
            p2=p,
            c=self.latent_channels,
        )

    def _get_temporal_pos_embed(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.temporal_pos_embed.size(1):
            return self.temporal_pos_embed[:, :seq_len]
        # Linearly interpolate temporal positional embeddings if rollout exceeds max_seq_len
        pe = rearrange(self.temporal_pos_embed, '1 t 1 d -> 1 d t')
        pe = F.interpolate(pe, size=seq_len, mode='linear', align_corners=False)
        return rearrange(pe, '1 d t -> 1 t 1 d')

    def forward(
        self,
        z_vis: torch.Tensor,
        tau: torch.Tensor,
        act_emb: torch.Tensor,
        z_proprio: torch.Tensor | None = None,
        *,
        checkpoint_blocks: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Predict velocity fields `(v_vis, v_proprio)`.

        Args:
            z_vis: Noisy visual VAE latents `(B, T, C, H', W')`.
            tau: Per-frame flow time in `[0, 1]` of shape `(B, T)`.
            act_emb: Per-frame action embeddings of shape `(B, T, act_emb_dim)`.
            z_proprio: Optional noisy proprioception of shape `(B, T, proprio_dim)`.
            checkpoint_blocks: Recompute transformer blocks during backward to
                reduce activation memory (e.g., during GradientSolver planning).

        Returns:
            Dictionary with `'v_vis'` `(B, T, C, H', W')` and optionally
            `'v_proprio'` `(B, T, proprio_dim)`.
        """
        t_len = z_vis.size(1)
        vis_tokens = self.patch_embed(self._patchify(z_vis))  # (B, T, N_patches, D)

        token_parts = [vis_tokens]
        if self.use_proprio:
            if z_proprio is None:
                raise ValueError(
                    'SpatioTemporalDiT configured with use_proprio=True, '
                    'but z_proprio was not provided.'
                )
            token_parts.append(self.proprio_embed(z_proprio).unsqueeze(2))  # (B, T, 1, D)
        if getattr(self, 'use_action_token', False) and self.action_token_proj is not None:
            token_parts.append(self.action_token_proj(act_emb).unsqueeze(2))  # (B, T, 1, D)
        tokens = torch.cat(token_parts, dim=2) if len(token_parts) > 1 else vis_tokens

        tokens = (
            tokens
            + self.spatial_pos_embed
            + self._get_temporal_pos_embed(t_len)
        )

        c = self.time_embedder(tau) + self.action_cond_proj(act_emb)  # (B, T, D)

        use_ckpt = bool(checkpoint_blocks) and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                tokens = checkpoint(
                    block,
                    tokens,
                    c,
                    use_reentrant=False,
                    preserve_rng_state=self.training,
                )
            else:
                tokens = block(tokens, c)

        vis_out_tokens = tokens[:, :, : self.num_vis_patches]
        v_vis = self._unpatchify(self.vis_head(vis_out_tokens, c))
        if self.use_proprio:
            proprio_out_token = tokens[
                :, :, self.num_vis_patches : self.num_vis_patches + 1
            ]
            v_proprio = self.proprio_head(proprio_out_token, c).squeeze(2)
            return {'v_vis': v_vis, 'v_proprio': v_proprio}

        return {'v_vis': v_vis}
