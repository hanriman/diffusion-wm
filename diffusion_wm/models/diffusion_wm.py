"""Pretrained VAE Latent Diffusion World Model (Rectified Flow + Diffusion Forcing)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .dit import SpatioTemporalDiT
from .vae import build_vae


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class DiffusionWorldModel(nn.Module):
    """Latent Diffusion World Model using Pretrained VAE + Spatio-Temporal Causal DiT.

    Implements the ``stable_worldmodel.solver.Costable`` protocol (`encode`, `rollout`,
    `criterion`, `get_cost`) so it works seamlessly with `WorldModelPolicy`,
    `GradientSolver`, and `CEMSolver`.
    """

    def __init__(
        self,
        *,
        vae: nn.Module,
        dit: SpatioTemporalDiT,
        action_dim: int,
        frameskip: int = 5,
        action_embed_dim: int = 192,
        history_size: int = 3,
        use_proprio: bool = False,
        proprio_dim: int = 0,
        proprio_loss_weight: float = 1.0,
        proprio_cost_weight: float = 1.0,
        ctx_noise_std: float = 0.05,
        num_ode_steps: int = 4,
        ode_solver: str = 'euler',
        rollout_mode: str = 'parallel',
        rollout_noise_scale: float = 0.0,
        action_queue_size: int = 0,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.dit = dit
        self.raw_action_dim = int(action_dim)
        self.frameskip = int(frameskip)
        self.stacked_action_dim = self.raw_action_dim * self.frameskip
        self.action_embed_dim = int(action_embed_dim)
        self.history_size = int(history_size)
        self.use_proprio = bool(use_proprio)
        self.proprio_dim = int(proprio_dim) if use_proprio else 0
        self.proprio_loss_weight = float(proprio_loss_weight)
        self.proprio_cost_weight = float(proprio_cost_weight)
        self.ctx_noise_std = float(ctx_noise_std)
        self.num_ode_steps = max(1, int(num_ode_steps))
        self.ode_solver = str(ode_solver).lower()
        self.rollout_mode = str(rollout_mode).lower()
        self.rollout_noise_scale = float(rollout_noise_scale)
        self.level = 1
        self.num_levels = 1
        self.interpolate_pos_encoding = True

        self.action_embed = nn.Sequential(
            nn.Linear(self.stacked_action_dim, action_embed_dim),
            nn.SiLU(),
            nn.Linear(action_embed_dim, action_embed_dim),
        )
        # Learned embedding for frame 0 (which has no preceding transition action)
        self.null_action_emb = nn.Parameter(
            torch.randn(1, 1, action_embed_dim) * 0.02
        )

        self.configure_action_queue(action_queue_size)

    def configure_action_queue(self, queue_size: int = 0) -> None:
        queue_size = int(queue_size)
        self.action_queue_size = queue_size
        if queue_size <= 0:
            self.latent_action_queue = None
            return
        if 'latent_action_queue' not in self._buffers:
            self.register_buffer(
                'latent_action_queue', torch.empty(0, 0), persistent=True
            )
        elif self.latent_action_queue is None:
            self._buffers['latent_action_queue'] = torch.empty(0, 0)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + 'latent_action_queue'
        if key in state_dict and 'latent_action_queue' in self._buffers:
            saved = state_dict[key]
            self._buffers['latent_action_queue'] = saved.new_empty(saved.shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _update_latent_action_queue(self, actions: torch.Tensor) -> None:
        if not self.training or self.action_queue_size <= 0:
            return
        if actions.ndim != 3:
            return
        samples = actions[:, 0].detach()
        queue = self.latent_action_queue
        if queue is None or queue.numel() == 0:
            updated = samples
        else:
            updated = torch.cat(
                [queue.to(device=samples.device, dtype=samples.dtype), samples],
                dim=0,
            )
        self._buffers['latent_action_queue'] = updated[-self.action_queue_size :]

    def get_latent_action_queue(self) -> torch.Tensor | None:
        queue = getattr(self, 'latent_action_queue', None)
        if not torch.is_tensor(queue) or queue.numel() == 0:
            return None
        return queue

    def _build_shifted_action_embeddings(
        self, actions: torch.Tensor, target_seq_len: int
    ) -> torch.Tensor:
        """Align transition actions with target frames.

        For a state sequence `x_0, x_1, ..., x_{T-1}`, frame `0` gets `null_action_emb`,
        and frame `t >= 1` gets `action_embed(a_{t-1})` (the action that transitions
        `x_{t-1} -> x_t`).
        """
        b = actions.size(0)
        null_emb = self.null_action_emb.expand(b, 1, -1)
        if target_seq_len <= 1:
            return null_emb[:, :target_seq_len]
        needed_actions = actions[:, : target_seq_len - 1].float()
        act_emb = self.action_embed(needed_actions)
        return torch.cat([null_emb, act_emb], dim=1)

    def encode(
        self,
        info: dict,
        key: str = 'pixels',
    ) -> dict:
        """Encode observations (and optional proprio/actions) into latent space."""
        pixels = info[key]
        z_vis = self.vae.encode(pixels)
        info['embed_0'] = z_vis

        if self.use_proprio and 'proprio' in info:
            info['proprio_0'] = info['proprio'].float()

        if (
            'action' in info
            and torch.is_tensor(info['action'])
            and info['action'].shape[-1] == self.stacked_action_dim
        ):
            act = info['action'].float()
            self._update_latent_action_queue(act)
            info['action_0'] = self.action_embed(act)

        return info

    def decode_pixels(self, z_vis: torch.Tensor) -> torch.Tensor:
        """Decode spatial VAE latents `(..., 4, H', W')` back to `[0, 1]` RGB images."""
        return self.vae.decode(z_vis)

    def _sample_training_tau(
        self, batch_size: int, seq_len: int, history_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample Diffusion Forcing per-frame flow times `tau` in `[0, 1]`.

        Combines three regimes across the batch so the model excels at both
        parallel full-horizon ODE rollouts and autoregressive single-step rollouts:
          1. Clean/stabilized history (`t < H_ctx` has `tau ~ U(1 - sigma, 1)`)
             + synchronized future `tau` (`t >= H_ctx` shares one `tau ~ U(0, 1)`).
          2. Clean/stabilized history + independent per-frame future `tau_t ~ U(0, 1)`.
          3. Frame 0 clean (`tau_0 = 1`) + all subsequent frames `t >= 1` independent `tau_t ~ U(0, 1)`.

        Returns:
            tau: `(B, T)` tensor of flow times in `[0, 1]`.
            loss_mask: `(B, T)` boolean/float mask of frames where velocity loss is active.
        """
        h_ctx = min(max(1, history_size), seq_len - 1)
        n_fut = seq_len - h_ctx

        tau = torch.empty((batch_size, seq_len), device=device, dtype=torch.float32)
        loss_mask = torch.zeros(
            (batch_size, seq_len), device=device, dtype=torch.float32
        )

        # Context stabilization noise: tau_ctx in [1 - ctx_noise_std, 1.0]
        if self.ctx_noise_std > 0:
            tau_ctx = 1.0 - self.ctx_noise_std * torch.rand(
                (batch_size, h_ctx), device=device
            )
        else:
            tau_ctx = torch.ones((batch_size, h_ctx), device=device)

        tau[:, :h_ctx] = tau_ctx

        # Split batch across synchronized-future (50%) and independent-future (50%)
        half = batch_size // 2
        if half > 0:
            sync_u = torch.rand((half, 1), device=device).expand(half, n_fut)
            tau[:half, h_ctx:] = sync_u
        if batch_size - half > 0:
            indep_u = torch.rand((batch_size - half, n_fut), device=device)
            tau[half:, h_ctx:] = indep_u

        loss_mask[:, h_ctx:] = 1.0

        # Also supervise intermediate frames t in [1, h_ctx) for a third of the batch
        third = batch_size // 3
        if third > 0 and h_ctx > 1:
            tau[:third, 0] = 1.0
            tau[:third, 1:h_ctx] = torch.rand((third, h_ctx - 1), device=device)
            loss_mask[:third, 1:h_ctx] = 1.0

        return tau, loss_mask

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute Diffusion Forcing Rectified Flow loss on a training/val batch.

        Expects `batch` to contain:
          - `'pixels_level1'` (or `'pixels'`): `(B, T, 3, H, W)`
          - `'action_level1'` (or `'action'`): `(B, T, stacked_action_dim)`
          - `'proprio_level1'` (or `'proprio'`) if `use_proprio=True`: `(B, T, proprio_dim)`
        """
        pixels = batch.get('pixels_level1', batch.get('pixels'))
        actions = batch.get('action_level1', batch.get('action')).float()
        proprio = (
            batch.get('proprio_level1', batch.get('proprio'))
            if self.use_proprio
            else None
        )

        self._update_latent_action_queue(actions)

        # 1. Encode visual latents (frozen by default)
        if 'vae_latents_level1' in batch:
            z_vis_clean = batch['vae_latents_level1'].float()
        else:
            z_vis_clean = self.vae.encode(pixels)

        b, t_len = z_vis_clean.shape[:2]
        device = z_vis_clean.device

        # 2. Sample per-frame flow times tau in [0, 1] and noise epsilon ~ N(0, I)
        tau, loss_mask = self._sample_training_tau(
            b, t_len, self.history_size, device
        )
        eps_vis = torch.randn_like(z_vis_clean)
        tau_vis = tau.view(b, t_len, 1, 1, 1)

        # Rectified Flow linear interpolation: z_tau = (1 - tau) * eps + tau * z_clean
        z_vis_noisy = (1.0 - tau_vis) * eps_vis + tau_vis * z_vis_clean
        target_v_vis = z_vis_clean - eps_vis

        if self.use_proprio:
            if proprio is None:
                raise KeyError(
                    "Model configured with use_proprio=True, but 'proprio' is missing from batch."
                )
            p_clean = proprio.float()
            eps_p = torch.randn_like(p_clean)
            tau_p = tau.unsqueeze(-1)
            z_p_noisy = (1.0 - tau_p) * eps_p + tau_p * p_clean
            target_v_p = p_clean - eps_p
        else:
            p_clean = None
            z_p_noisy = None
            target_v_p = None

        # 3. Shift actions so frame t is conditioned on action a_{t-1}
        act_emb = self._build_shifted_action_embeddings(actions, t_len)

        # 4. Predict velocity with Spatio-Temporal Causal DiT
        pred = self.dit(z_vis_noisy, tau, act_emb, z_proprio=z_p_noisy)
        pred_v_vis = pred['v_vis']

        mask_denom = loss_mask.sum().clamp_min(1.0)
        per_frame_vis_mse = (
            F.mse_loss(pred_v_vis, target_v_vis, reduction='none')
            .mean(dim=(2, 3, 4))
        )
        vis_loss = (per_frame_vis_mse * loss_mask).sum() / mask_denom
        total_loss = vis_loss

        out = {
            'loss': total_loss,
            'rf_vis_loss': vis_loss.detach(),
        }

        if self.use_proprio and target_v_p is not None:
            pred_v_p = pred['v_proprio']
            per_frame_p_mse = F.mse_loss(
                pred_v_p, target_v_p, reduction='none'
            ).mean(dim=-1)
            proprio_loss = (per_frame_p_mse * loss_mask).sum() / mask_denom
            total_loss = total_loss + self.proprio_loss_weight * proprio_loss
            out['loss'] = total_loss
            out['rf_proprio_loss'] = proprio_loss.detach()

        if hasattr(self.vae, 'reconstruction_loss') and not getattr(
            self.vae, 'freeze', True
        ):
            vae_recon_loss = self.vae.reconstruction_loss(pixels)
            out['loss'] = out['loss'] + vae_recon_loss
            out['vae_recon_loss'] = vae_recon_loss.detach()

        return out

    def _ode_step(
        self,
        z_vis_seq: torch.Tensor,
        z_p_seq: torch.Tensor | None,
        tau_seq: torch.Tensor,
        act_emb_seq: torch.Tensor,
        dt: float,
        active_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Perform one Euler or Midpoint ODE integration step on `[:, active_start:]`."""
        pred = self.dit(z_vis_seq, tau_seq, act_emb_seq, z_proprio=z_p_seq)
        v_vis = pred['v_vis'][:, active_start:]
        v_p = pred['v_proprio'][:, active_start:] if self.use_proprio else None

        if self.ode_solver == 'midpoint':
            half_dt = 0.5 * dt
            z_vis_mid = z_vis_seq.clone()
            z_vis_mid[:, active_start:] = (
                z_vis_seq[:, active_start:] + half_dt * v_vis
            )
            tau_mid = tau_seq.clone()
            tau_mid[:, active_start:] = (tau_seq[:, active_start:] + half_dt).clamp(
                0.0, 1.0
            )
            if self.use_proprio and z_p_seq is not None and v_p is not None:
                z_p_mid = z_p_seq.clone()
                z_p_mid[:, active_start:] = (
                    z_p_seq[:, active_start:] + half_dt * v_p
                )
            else:
                z_p_mid = None

            pred_mid = self.dit(z_vis_mid, tau_mid, act_emb_seq, z_proprio=z_p_mid)
            v_vis = pred_mid['v_vis'][:, active_start:]
            v_p = (
                pred_mid['v_proprio'][:, active_start:]
                if self.use_proprio
                else None
            )

        z_vis_next = z_vis_seq.clone()
        z_vis_next[:, active_start:] = z_vis_seq[:, active_start:] + dt * v_vis

        if self.use_proprio and z_p_seq is not None and v_p is not None:
            z_p_next = z_p_seq.clone()
            z_p_next[:, active_start:] = z_p_seq[:, active_start:] + dt * v_p
        else:
            z_p_next = None

        return z_vis_next, z_p_next

    def _init_future_noise(
        self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self.rollout_noise_scale <= 0.0:
            return torch.zeros(shape, device=device, dtype=dtype)
        gen = torch.Generator(device=device).manual_seed(0)
        return (
            torch.randn(shape, generator=gen, device=device, dtype=dtype)
            * self.rollout_noise_scale
        )

    def sample_ode_parallel(
        self,
        ctx_vis: torch.Tensor,
        actions: torch.Tensor,
        ctx_proprio: torch.Tensor | None = None,
        num_ode_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Denoise all `T_plan` future steps in parallel in `K` ODE steps.

        Args:
            ctx_vis: Clean context latents `(N, H_ctx, 4, H', W')`.
            actions: Candidate action sequence `(N, T_plan, stacked_action_dim)`.
            ctx_proprio: Optional clean context proprio `(N, H_ctx, proprio_dim)`.
            num_ode_steps: Optional override for number of Euler/Midpoint steps.

        Returns:
            `(pred_vis, pred_proprio)` containing the full trajectory of length
            `H_ctx + T_plan`.
        """
        k_steps = max(1, int(num_ode_steps or self.num_ode_steps))
        n_batch, h_ctx, c, hp, wp = ctx_vis.shape
        t_plan = actions.size(1)
        total_len = h_ctx + t_plan
        device = ctx_vis.device
        dtype = ctx_vis.dtype

        z_fut_vis = self._init_future_noise(
            (n_batch, t_plan, c, hp, wp), device, dtype
        )
        z_vis_seq = torch.cat([ctx_vis, z_fut_vis], dim=1)

        if self.use_proprio:
            assert ctx_proprio is not None, 'ctx_proprio required when use_proprio=True'
            z_fut_p = self._init_future_noise(
                (n_batch, t_plan, self.proprio_dim), device, ctx_proprio.dtype
            )
            z_p_seq = torch.cat([ctx_proprio, z_fut_p], dim=1)
        else:
            z_p_seq = None

        # Build action conditioning for [0..H_ctx + T_plan - 1]:
        # Frames 0..H_ctx-1 get null_action_emb; frame H_ctx + k gets action[:, k]
        null_ctx = self.null_action_emb.expand(n_batch, h_ctx, -1)
        fut_act_emb = self.action_embed(actions.float())
        act_emb_seq = torch.cat([null_ctx, fut_act_emb], dim=1)

        dt = 1.0 / float(k_steps)
        tau_seq = torch.ones((n_batch, total_len), device=device, dtype=torch.float32)

        for step_idx in range(k_steps):
            t_val = float(step_idx) * dt
            tau_seq[:, h_ctx:] = t_val
            z_vis_seq, z_p_seq = self._ode_step(
                z_vis_seq,
                z_p_seq,
                tau_seq,
                act_emb_seq,
                dt=dt,
                active_start=h_ctx,
            )

        return z_vis_seq, z_p_seq

    def sample_ode_autoregressive(
        self,
        ctx_vis: torch.Tensor,
        actions: torch.Tensor,
        ctx_proprio: torch.Tensor | None = None,
        num_ode_steps: int | None = None,
        history_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Denoise future steps sequentially one frame at a time with sliding context."""
        k_steps = max(1, int(num_ode_steps or self.num_ode_steps))
        hs = max(1, int(history_size or self.history_size))
        n_batch, _, c, hp, wp = ctx_vis.shape
        t_plan = actions.size(1)
        device = ctx_vis.device
        dtype = ctx_vis.dtype
        dt = 1.0 / float(k_steps)

        vis_traj = ctx_vis
        p_traj = ctx_proprio if self.use_proprio else None
        act_emb_all = self.action_embed(actions.float())  # (N, T_plan, A_emb)
        past_act_embs = self.null_action_emb.expand(n_batch, ctx_vis.size(1), -1)

        for step_t in range(t_plan):
            win_vis = vis_traj[:, -hs:]
            win_len = win_vis.size(1)
            z_next_vis = self._init_future_noise(
                (n_batch, 1, c, hp, wp), device, dtype
            )
            cur_vis_seq = torch.cat([win_vis, z_next_vis], dim=1)

            if self.use_proprio and p_traj is not None:
                win_p = p_traj[:, -hs:]
                z_next_p = self._init_future_noise(
                    (n_batch, 1, self.proprio_dim), device, p_traj.dtype
                )
                cur_p_seq = torch.cat([win_p, z_next_p], dim=1)
            else:
                cur_p_seq = None

            win_act_emb = past_act_embs[:, -win_len:]
            next_act_emb = act_emb_all[:, step_t : step_t + 1]
            cur_act_emb = torch.cat([win_act_emb, next_act_emb], dim=1)

            tau_seq = torch.ones(
                (n_batch, win_len + 1), device=device, dtype=torch.float32
            )
            for k in range(k_steps):
                tau_seq[:, -1] = float(k) * dt
                cur_vis_seq, cur_p_seq = self._ode_step(
                    cur_vis_seq,
                    cur_p_seq,
                    tau_seq,
                    cur_act_emb,
                    dt=dt,
                    active_start=win_len,
                )

            vis_traj = torch.cat([vis_traj, cur_vis_seq[:, -1:]], dim=1)
            if self.use_proprio and p_traj is not None and cur_p_seq is not None:
                p_traj = torch.cat([p_traj, cur_p_seq[:, -1:]], dim=1)
            past_act_embs = torch.cat([past_act_embs, next_act_emb], dim=1)

        return vis_traj, p_traj

    def rollout(
        self,
        info: dict,
        action_sequence: torch.Tensor,
        history_size: int | None = None,
    ) -> dict:
        """Roll out the Diffusion World Model from `info` under `action_sequence`.

        Args:
            info: Dictionary containing `'pixels'` `(B, S, H, 3, H_img, W_img)`
                (or pre-encoded `'embed_0'`) and optionally `'proprio'`.
            action_sequence: Candidate actions of shape `(B, S, T_plan, stacked_action_dim)`.
            history_size: Optional sliding window size for autoregressive rollout.

        Returns:
            Updated `info` dictionary with `'predicted_embed_0'` of shape
            `(B, S, H + T_plan, 4, H', W')` and optionally `'predicted_proprio_0'`.
        """
        assert 'pixels' in info or 'embed_0' in info, (
            'pixels or embed_0 must be in info_dict'
        )
        b, s, _ = action_sequence.shape[:3]

        if 'embed_0' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            with torch.no_grad():
                _init = self.encode(_init)
            info['embed_0'] = (
                _init['embed_0']
                .unsqueeze(1)
                .expand(b, s, *_init['embed_0'].shape[1:])
            )
            if self.use_proprio and 'proprio_0' in _init:
                info['proprio_0'] = (
                    _init['proprio_0']
                    .unsqueeze(1)
                    .expand(b, s, *_init['proprio_0'].shape[1:])
                )
        elif self.use_proprio and 'proprio_0' not in info and 'proprio' in info:
            info['proprio_0'] = info['proprio'].float()

        ctx_vis = rearrange(info['embed_0'], 'b s ... -> (b s) ...').clone()
        act_flat = rearrange(action_sequence, 'b s ... -> (b s) ...')
        ctx_p = (
            rearrange(info['proprio_0'], 'b s ... -> (b s) ...').clone()
            if self.use_proprio and 'proprio_0' in info
            else None
        )

        if self.rollout_mode == 'autoregressive':
            pred_vis, pred_p = self.sample_ode_autoregressive(
                ctx_vis,
                act_flat,
                ctx_proprio=ctx_p,
                history_size=history_size or self.history_size,
            )
        else:
            pred_vis, pred_p = self.sample_ode_parallel(
                ctx_vis,
                act_flat,
                ctx_proprio=ctx_p,
            )

        info['predicted_embed_0'] = rearrange(
            pred_vis, '(b s) ... -> b s ...', b=b, s=s
        )
        if self.use_proprio and pred_p is not None:
            info['predicted_proprio_0'] = rearrange(
                pred_p, '(b s) ... -> b s ...', b=b, s=s
            )

        return info

    def criterion(self, info_dict: dict) -> torch.Tensor:
        """Compute planning cost over the last `cost_last_n` predicted steps."""
        pred_vis = info_dict['predicted_embed_0'][:, :, 1:]
        cost_last_n = max(1, int(info_dict.get('cost_last_n', 1)))
        cost_window = min(cost_last_n, pred_vis.shape[2])

        pred_vis_win = pred_vis[:, :, -cost_window:]
        goal_vis = info_dict['goal_embed_0'].expand_as(pred_vis_win).detach()
        vis_cost = F.mse_loss(pred_vis_win, goal_vis, reduction='none').mean(
            dim=(-3, -2, -1)
        )  # (B, S, W)

        total_step_cost = vis_cost

        if (
            self.use_proprio
            and 'predicted_proprio_0' in info_dict
            and 'goal_proprio_0' in info_dict
        ):
            pred_p = info_dict['predicted_proprio_0'][:, :, 1:][:, :, -cost_window:]
            goal_p = info_dict['goal_proprio_0'].expand_as(pred_p).detach()
            p_cost = F.mse_loss(pred_p, goal_p, reduction='none').mean(dim=-1)
            total_step_cost = (
                total_step_cost + self.proprio_cost_weight * p_cost
            )

        intermediate_weight = float(
            info_dict.get('intermediate_cost_weight', 1.0)
        )
        if intermediate_weight != 1.0 and total_step_cost.shape[2] > 1:
            weights = total_step_cost.new_ones(total_step_cost.shape[2])
            weights[:-1] = intermediate_weight
            total_step_cost = total_step_cost * weights.view(1, 1, -1)

        return total_step_cost.sum(dim=2)

    def get_cost(
        self, info_dict: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Compute planning cost for `action_candidates` of shape `(B, S, T_plan, act_dim)`."""
        if 'goal_embed_0' not in info_dict:
            assert 'goal' in info_dict, 'goal not in info_dict'

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        if 'goal_embed_0' not in info_dict:
            _goal = {
                k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)
            }
            _goal['pixels'] = _goal['goal']
            for k in list(info_dict.keys()):
                if k.startswith('goal_') and torch.is_tensor(info_dict[k]):
                    _goal[k[len('goal_') :]] = info_dict[k][:, 0]
            _goal.pop('action', None)
            with torch.no_grad():
                _goal = self.encode(_goal)
            info_dict['goal_embed_0'] = _goal['embed_0'].unsqueeze(1)
            if self.use_proprio and 'proprio_0' in _goal:
                info_dict['goal_proprio_0'] = _goal['proprio_0'].unsqueeze(1)

        info_dict = self.rollout(info_dict, action_candidates.to(device))
        return self.criterion(info_dict)


def build_diffusion_world_model(cfg) -> DiffusionWorldModel:
    """Factory to construct `DiffusionWorldModel` from a Hydra/OmegaConf config."""
    vae = build_vae(cfg.vae)
    img_size = int(cfg.img_size)
    latent_hw = img_size // int(getattr(vae, 'spatial_downsample', 8))
    latent_channels = int(getattr(vae, 'latent_channels', 4))

    wm_cfg = cfg.wm
    dit_cfg = cfg.dit
    frameskip = int(cfg.data.dataset.level1.frameskip)
    use_proprio = bool(wm_cfg.get('use_proprio', False))
    proprio_dim = int(wm_cfg.get('proprio_dim', 0)) if use_proprio else 0
    action_embed_dim = int(wm_cfg.get('action_embed_dim', dit_cfg.hidden_dim))

    dit = SpatioTemporalDiT(
        latent_channels=latent_channels,
        latent_height=latent_hw,
        latent_width=latent_hw,
        patch_size=int(dit_cfg.patch_size),
        hidden_dim=int(dit_cfg.hidden_dim),
        depth=int(dit_cfg.depth),
        heads=int(dit_cfg.heads),
        dim_head=int(dit_cfg.get('dim_head', 64)),
        mlp_ratio=float(dit_cfg.get('mlp_ratio', 4.0)),
        dropout=float(dit_cfg.get('dropout', 0.0)),
        max_seq_len=int(dit_cfg.get('max_seq_len', 32)),
        action_embed_dim=action_embed_dim,
        use_proprio=use_proprio,
        proprio_dim=proprio_dim,
    )

    return DiffusionWorldModel(
        vae=vae,
        dit=dit,
        action_dim=int(wm_cfg.action_dim),
        frameskip=frameskip,
        action_embed_dim=action_embed_dim,
        history_size=int(wm_cfg.history_size),
        use_proprio=use_proprio,
        proprio_dim=proprio_dim,
        proprio_loss_weight=float(wm_cfg.get('proprio_loss_weight', 1.0)),
        proprio_cost_weight=float(wm_cfg.get('proprio_cost_weight', 1.0)),
        ctx_noise_std=float(wm_cfg.get('ctx_noise_std', 0.05)),
        num_ode_steps=int(wm_cfg.get('num_ode_steps', 4)),
        ode_solver=str(wm_cfg.get('ode_solver', 'euler')),
        rollout_mode=str(wm_cfg.get('rollout_mode', 'parallel')),
        rollout_noise_scale=float(wm_cfg.get('rollout_noise_scale', 0.0)),
        action_queue_size=int(wm_cfg.get('action_queue_size', 0)),
    )
