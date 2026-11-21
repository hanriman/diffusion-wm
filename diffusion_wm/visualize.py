"""Presentation-ready visualization utilities for Diffusion World Model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

import stable_worldmodel as swm

try:
    from .planning_eval import (
        _apply_model_overrides,
        _build_eval_process_from_policy_normalizer,
        _build_policy,
        _eval_sdpa_kernel,
        _resolve_existing_path,
        apply_eval_cli_overrides,
        get_solver_generators,
        img_transform,
        load_eval_config,
        seed_solver_for_offset,
    )
    from .utils import resolve_model_checkpoint_path
except ImportError:
    from planning_eval import (
        _apply_model_overrides,
        _build_eval_process_from_policy_normalizer,
        _build_policy,
        _eval_sdpa_kernel,
        _resolve_existing_path,
        apply_eval_cli_overrides,
        get_solver_generators,
        img_transform,
        load_eval_config,
        seed_solver_for_offset,
    )
    from utils import resolve_model_checkpoint_path


def _tensor_01_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert `(..., 3, H, W)` float tensor in [0, 1] to `(..., H, W, 3)` uint8."""
    arr = t.detach().cpu().clamp(0.0, 1.0).numpy()
    arr = np.moveaxis(arr, -3, -1)
    return (arr * 255.0).round().astype(np.uint8)


def _imagenet_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert ImageNet-normalized `(..., 3, H, W)` tensor to `(..., H, W, 3)` uint8."""
    mean = t.new_tensor([0.485, 0.456, 0.406]).view(*([1] * (t.ndim - 3)), 3, 1, 1)
    std = t.new_tensor([0.229, 0.224, 0.225]).view(*([1] * (t.ndim - 3)), 3, 1, 1)
    rgb01 = (t * std + mean).clamp(0.0, 1.0)
    return _tensor_01_to_uint8(rgb01)


def plot_training_curves(run_dir: str | Path, out_dir: str | Path) -> Path | None:
    """Plot training/validation losses from Lightning CSV logs."""
    run_path = Path(run_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(run_path.glob('lightning_logs/version_*/metrics.csv'))
    if not csv_files:
        return None

    metrics_csv = csv_files[-1]
    rows_by_epoch: dict[int, dict[str, float]] = {}
    with metrics_csv.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep_raw = row.get('epoch', '')
            if not ep_raw:
                continue
            ep = int(float(ep_raw))
            entry = rows_by_epoch.setdefault(ep, {})
            for key in (
                'train/loss_epoch',
                'val/loss',
                'val/ode_vis_mse',
                'val/ode_proprio_mse',
            ):
                val_raw = row.get(key, '')
                if val_raw not in ('', None, 'nan'):
                    entry[key] = float(val_raw)

    if not rows_by_epoch:
        return None

    epochs = sorted(rows_by_epoch)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

    for key, label, marker in (
        ('train/loss_epoch', 'Train RF Loss', 'o'),
        ('val/loss', 'Val RF Loss', 's'),
    ):
        xs = [e for e in epochs if key in rows_by_epoch[e]]
        ys = [rows_by_epoch[e][key] for e in xs]
        if xs:
            axes[0].plot(xs, ys, marker=marker, linewidth=2, label=label)
    axes[0].set_title('Rectified Flow Velocity Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for key, label, marker in (
        ('val/ode_vis_mse', 'Val ODE Latent MSE', 'o'),
        ('val/ode_proprio_mse', 'Val ODE Proprio MSE', '^'),
    ):
        xs = [e for e in epochs if key in rows_by_epoch[e]]
        ys = [rows_by_epoch[e][key] for e in xs]
        if xs:
            axes[1].plot(xs, ys, marker=marker, linewidth=2, label=label)
    axes[1].set_title('Validation ODE Rollout Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].grid(True, alpha=0.3)
    if axes[1].has_data():
        axes[1].legend()

    fig.tight_layout()
    save_path = out_path / 'training_curves.png'
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_eval_summary(
    metrics_yaml: str | Path, out_dir: str | Path
) -> Path | None:
    """Plot per-episode planning outcomes and steps-to-success summary."""
    metrics_path = Path(metrics_yaml)
    if not metrics_path.exists():
        return None

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    m = OmegaConf.to_container(OmegaConf.load(metrics_path), resolve=True)
    successes = np.asarray(m.get('episode_successes', []), dtype=bool)
    steps = np.asarray(m.get('steps_to_success', []), dtype=np.float64)
    if successes.size == 0:
        return None

    success_rate = float(m.get('success_rate', successes.mean() * 100.0))
    n_eps = len(successes)
    n_ok = int(successes.sum())
    max_step_display = (
        float(np.nanmax(steps)) * 1.15 if np.isfinite(steps).any() else 100.0
    )
    bar_heights = np.where(np.isfinite(steps), steps, max_step_display)
    colors = ['#2ca02c' if ok else '#d62728' for ok in successes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)

    axes[0].bar(np.arange(n_eps), bar_heights, color=colors, alpha=0.85)
    axes[0].set_title(
        f'Planning Evaluation Outcomes — Success Rate: {success_rate:.1f}% ({n_ok}/{n_eps})'
    )
    axes[0].set_xlabel('Episode Index')
    axes[0].set_ylabel('Steps to Success (Red = Timeout)')
    axes[0].grid(True, axis='y', alpha=0.3)

    valid_steps = steps[np.isfinite(steps)]
    if valid_steps.size > 0:
        axes[1].hist(
            valid_steps,
            bins=min(15, max(5, valid_steps.size)),
            color='#1f77b4',
            alpha=0.85,
            edgecolor='black',
        )
        axes[1].axvline(
            valid_steps.mean(),
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f'Mean = {valid_steps.mean():.1f} steps',
        )
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            'No successful episodes yet\n(increase training epochs or samples)',
            ha='center',
            va='center',
            fontsize=11,
        )
    axes[1].set_title('Steps-to-Success Distribution')
    axes[1].set_xlabel('Environment Steps')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = out_path / 'eval_summary.png'
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return save_path


@torch.no_grad()
def visualize_world_model_dreams(
    cfg,
    model,
    process: dict,
    out_dir: str | Path,
    *,
    episode_idx: int = 0,
    num_future_steps: int = 6,
) -> Path | None:
    """Compare Ground-Truth frames vs. SD-VAE reconstruction vs. Diffusion WM dreams."""
    load_path = cfg.get('load_eval_trajs_path', None)
    if not load_path:
        return None
    trajs_path = _resolve_existing_path(load_path)
    if not trajs_path.exists():
        return None

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    payload = torch.load(trajs_path, map_location='cpu', weights_only=False)
    ep = payload['data'][min(episode_idx, len(payload['data']) - 1)]

    frameskip = int(getattr(model, 'frameskip', 5))
    h_ctx = int(getattr(model, 'history_size', 3))
    total_model_frames = h_ctx + num_future_steps
    needed_raw = (total_model_frames - 1) * frameskip + 1

    raw_pixels = ep['pixels']  # (T_raw, 3, H, W)
    raw_actions = ep['action']  # (T_raw, act_dim) or (T_raw - 1, act_dim)
    if raw_pixels.size(0) < needed_raw or raw_actions.size(0) < (total_model_frames - 1) * frameskip:
        max_frames = min(
            (raw_pixels.size(0) - 1) // frameskip + 1,
            raw_actions.size(0) // frameskip + 1,
        )
        total_model_frames = max(h_ctx + 1, max_frames)
        num_future_steps = total_model_frames - h_ctx

    # Subsample RGB frames at `frameskip`
    frame_indices = [i * frameskip for i in range(total_model_frames)]
    sub_pixels = raw_pixels[frame_indices]  # (T_model, 3, H, W)

    # Apply eval image transform -> (T_model, 3, img_size, img_size)
    tx = img_transform(cfg)
    device = next(model.parameters()).device
    norm_pixels = torch.stack([tx(img) for img in sub_pixels]).to(device)

    # Stack and normalize actions over `frameskip`
    ctx_actions = None
    if h_ctx > 1:
        ctx_act_blocks = []
        for i in range(h_ctx - 1):
            start_a = i * frameskip
            chunk = np.asarray(raw_actions[start_a : start_a + frameskip])
            if 'action' in process:
                chunk = process['action'].transform(chunk)
            ctx_act_blocks.append(chunk.reshape(-1))
        ctx_actions = (
            torch.from_numpy(np.stack(ctx_act_blocks))
            .float()
            .unsqueeze(0)
            .to(device)
        )

    act_blocks = []
    for i in range(num_future_steps):
        start_a = (h_ctx - 1 + i) * frameskip
        chunk = np.asarray(raw_actions[start_a : start_a + frameskip])
        if 'action' in process:
            chunk = process['action'].transform(chunk)
        act_blocks.append(chunk.reshape(-1))
    fut_actions = (
        torch.from_numpy(np.stack(act_blocks))
        .float()
        .unsqueeze(0)
        .to(device)
    )

    ctx_proprio = None
    if getattr(model, 'use_proprio', False) and 'proprio' in ep:
        p_raw = np.asarray(ep['proprio'][frame_indices[:h_ctx]])
        if 'proprio' in process:
            p_raw = process['proprio'].transform(p_raw)
        ctx_proprio = torch.from_numpy(p_raw).float().unsqueeze(0).to(device)

    z_all = model.vae.encode(norm_pixels.unsqueeze(0))  # (1, T_model, 4, H', W')
    vae_recon = model.decode_pixels(z_all)[0]

    ctx_vis = z_all[:, :h_ctx]
    if str(getattr(model, 'rollout_mode', 'parallel')).lower() == 'autoregressive':
        pred_vis, _ = model.sample_ode_autoregressive(
            ctx_vis, fut_actions, ctx_proprio=ctx_proprio, ctx_actions=ctx_actions
        )
    else:
        pred_vis, _ = model.sample_ode_parallel(
            ctx_vis, fut_actions, ctx_proprio=ctx_proprio, ctx_actions=ctx_actions
        )
    dream_rgb = model.decode_pixels(pred_vis)[0]

    gt_u8 = _imagenet_to_uint8(norm_pixels)
    vae_u8 = _tensor_01_to_uint8(vae_recon)
    dream_u8 = _tensor_01_to_uint8(dream_rgb)
    err_u8 = np.abs(gt_u8.astype(np.int16) - dream_u8.astype(np.int16)).mean(
        axis=-1
    )

    fig, axes = plt.subplots(
        4, total_model_frames, figsize=(2.1 * total_model_frames, 8.2), dpi=150
    )
    row_labels = ['Ground Truth', 'SD-VAE Recon', 'Diffusion WM Dream', '|GT - Dream|']
    for t in range(total_model_frames):
        col_title = f'CTX {t}' if t < h_ctx else f'PRED +{t - h_ctx + 1}'
        axes[0, t].set_title(col_title, fontsize=10, fontweight='bold')
        axes[0, t].imshow(gt_u8[t])
        axes[1, t].imshow(vae_u8[t])
        axes[2, t].imshow(dream_u8[t])
        axes[3, t].imshow(err_u8[t], cmap='inferno', vmin=0, vmax=80)
        for r in range(4):
            axes[r, t].set_xticks([])
            axes[r, t].set_yticks([])
            if t == 0:
                axes[r, t].set_ylabel(row_labels[r], fontsize=10, fontweight='bold')

    fig.suptitle(
        f'World Model Open-Loop Rollout (Episode {episode_idx}, {model.num_ode_steps}-Step ODE)',
        fontsize=12,
        fontweight='bold',
    )
    fig.tight_layout()
    save_path = out_path / f'world_model_dreams_ep{episode_idx}.png'
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return save_path


def _annotate_panel(
    img_hwc: np.ndarray, label: str, target_size: int = 224
) -> np.ndarray:
    pil = Image.fromarray(img_hwc).resize(
        (target_size, target_size), Image.Resampling.BILINEAR
    )
    canvas = Image.new('RGB', (target_size, target_size + 28), color=(25, 25, 25))
    canvas.paste(pil, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, fill=(240, 240, 240))
    return np.asarray(canvas)


def _make_recording_step(
    *,
    world,
    orig_step,
    step_counter: dict,
    latest_imagined: dict,
    recorded_frames: list[np.ndarray],
    budget: int,
):
    def _recording_step(*args, **kwargs):
        if not step_counter['done']:
            orig_step(*args, **kwargs)
            obs_img = np.asarray(world.infos['pixels'][0, -1], dtype=np.uint8)
            goal_img = np.asarray(world.infos['goal'][0, -1], dtype=np.uint8)
            step_counter['t'] += 1
            imagined = (
                latest_imagined['rgb']
                if latest_imagined['rgb'] is not None
                else obs_img
            )
            status = (
                'SUCCESS!'
                if bool(world.terminateds[0])
                else f'Step {step_counter["t"]}/{budget}'
            )
            p1 = _annotate_panel(obs_img, f'Env ({status})')
            p2 = _annotate_panel(imagined, 'WM Imagined Plan End')
            p3 = _annotate_panel(goal_img, 'Target Goal')
            recorded_frames.append(np.concatenate([p1, p2, p3], axis=1))
            if bool(world.terminateds[0]):
                step_counter['done'] = True
        else:
            orig_step(*args, **kwargs)

    return _recording_step


def record_closed_loop_rollouts(
    cfg,
    model,
    process: dict,
    out_dir: str | Path,
    *,
    num_episodes: int = 2,
    episodes: list[int] | None = None,
    max_steps: int | None = None,
) -> list[Path]:
    """Run closed-loop MPC episodes and save side-by-side GIFs and storyboards."""
    load_path = cfg.get('load_eval_trajs_path', None)
    if not load_path:
        return []
    trajs_path = _resolve_existing_path(load_path)
    if not trajs_path.exists():
        return []

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    payload = torch.load(trajs_path, map_location='cpu', weights_only=False)
    n_avail = len(payload['data'])
    if episodes is not None:
        target_eps = [i for i in episodes if 0 <= i < n_avail]
    else:
        target_eps = list(range(min(int(num_episodes), n_avail)))
    if not target_eps:
        return []

    cfg_local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg_local.eval.num_eval = 1
    cfg_local.world.num_envs = 1
    budget = int(max_steps or cfg_local.eval.eval_budget)
    cfg_local.eval.eval_budget = budget
    cfg_local.world.max_episode_steps = max(
        2 * budget, 2 * int(cfg_local.eval.goal_offset_steps)
    )

    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg_local.world, resolve=True))
    image_shape = (int(cfg_local.eval.img_size),) * 2
    transform = {
        'pixels': img_transform(cfg_local),
        'goal': img_transform(cfg_local),
    }

    policy = _build_policy(cfg_local, process, transform, model=model)
    gens = get_solver_generators(getattr(policy, 'solver', None))
    latest_imagined: dict[str, np.ndarray | None] = {'rgb': None}

    if hasattr(policy, 'solver'):
        orig_solve = policy.solver.solve

        def _hooked_solve(*args, **kwargs):
            out = orig_solve(*args, **kwargs)
            preds = out.get('predictions', {})
            pred_lat = preds.get('predicted_embed_0')
            if pred_lat is not None:
                with torch.no_grad():
                    # Decode the final predicted step of the planned horizon
                    final_lat = pred_lat[0, -1:].to(next(model.parameters()).device)
                    dec = model.decode_pixels(final_lat)[0]
                    latest_imagined['rgb'] = _tensor_01_to_uint8(dec)
            return out

        policy.solver.solve = _hooked_solve

    saved_paths: list[Path] = []
    callables = OmegaConf.to_container(cfg_local.eval.get('callables'), resolve=True)
    ep_idx_all = payload.get('episodes_idx', list(range(len(payload['data']))))

    for ep_i in target_eps:
        seed_solver_for_offset(gens, ep_i)
        latest_imagined['rgb'] = None
        single_chunk = {
            **payload,
            'data': [dict(payload['data'][ep_i])],
            'episodes_idx': [ep_idx_all[ep_i]],
        }
        recorded_frames: list[np.ndarray] = []
        world = swm.World(**world_cfg, image_shape=image_shape)
        try:
            world.set_policy(policy)
            # Capture frames by stepping through evaluate_from_dataset's initialized env
            orig_step = world.step
            step_counter = {'t': 0, 'done': False}
            world.step = _make_recording_step(
                world=world,
                orig_step=orig_step,
                step_counter=step_counter,
                latest_imagined=latest_imagined,
                recorded_frames=recorded_frames,
                budget=budget,
            )
            with torch.enable_grad(), _eval_sdpa_kernel():
                world.evaluate_from_dataset(
                    None,
                    episodes_idx=None,
                    start_steps=None,
                    goal_offset_steps=cfg_local.eval.goal_offset_steps,
                    eval_budget=budget,
                    callables=callables,
                    load_eval_trajs_path=single_chunk,
                    process=process,
                    start_state_mode=cfg_local.eval.get('start_state_mode', 'dataset_full'),
                    goal_state_mode=cfg_local.eval.get('goal_state_mode', 'dataset_full'),
                )
        finally:
            world.close()

        if recorded_frames:
            gif_path = out_path / f'rollout_ep{ep_i}.gif'
            imageio.mimsave(gif_path, recorded_frames, duration=0.12, loop=0)
            saved_paths.append(gif_path)

            # Also save a 5-frame static storyboard PNG for slides
            pick = np.linspace(0, len(recorded_frames) - 1, min(5, len(recorded_frames)), dtype=int)
            fig, axes = plt.subplots(len(pick), 1, figsize=(7.5, 2.5 * len(pick)), dpi=140)
            if len(pick) == 1:
                axes = [axes]
            for ax, f_idx in zip(axes, pick):
                ax.imshow(recorded_frames[f_idx])
                ax.axis('off')
            fig.suptitle(f'Closed-Loop MPC Rollout Storyboard (Episode {ep_i})', fontweight='bold')
            fig.tight_layout()
            sb_path = out_path / f'rollout_storyboard_ep{ep_i}.png'
            fig.savefig(sb_path, bbox_inches='tight')
            plt.close(fig)
            saved_paths.append(sb_path)

    return saved_paths


def generate_presentation_report(
    *,
    eval_config_name: str,
    policy_path: str | Path,
    run_dir: str | Path,
    eval_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    episodes: list[int] | None = None,
    num_dream_episodes: int = 2,
    num_rollout_videos: int = 2,
    max_rollout_steps: int | None = None,
    num_samples: int | None = None,
    img_size: int | None = None,
    horizon: int | None = None,
    proprio_cost_weight: float | None = None,
    seed: int | None = None,
) -> list[Path]:
    """Generate all training, dream, evaluation summary, and rollout visuals."""
    ckpt_path = resolve_model_checkpoint_path(policy_path, cache_dir)
    run_path = Path(run_dir)
    target_out = Path(out_dir or (run_path / 'visualizations'))
    target_out.mkdir(parents=True, exist_ok=True)

    cfg = load_eval_config(config_name=eval_config_name)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.policy = str(ckpt_path)
    if cache_dir is not None:
        cfg.cache_dir = str(cache_dir)
    apply_eval_cli_overrides(
        cfg,
        num_samples=num_samples,
        img_size=img_size,
        horizon=horizon,
        proprio_cost_weight=proprio_cost_weight,
        seed=seed,
    )

    generated: list[Path] = []

    p_curves = plot_training_curves(run_path, target_out)
    if p_curves is not None:
        generated.append(p_curves)

    if eval_dir is not None:
        p_eval = plot_eval_summary(Path(eval_dir) / 'metrics.yaml', target_out)
        if p_eval is not None:
            generated.append(p_eval)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.policy.AutoCostModel(str(ckpt_path), cache_dir).to(device).eval()
    _apply_model_overrides(model, cfg)
    process = _build_eval_process_from_policy_normalizer(cfg, model=model)

    dream_eps = (
        episodes
        if episodes is not None
        else list(range(max(0, int(num_dream_episodes))))
    )
    for ep_i in dream_eps:
        p_dream = visualize_world_model_dreams(
            cfg, model, process, target_out, episode_idx=ep_i
        )
        if p_dream is not None:
            generated.append(p_dream)

    if num_rollout_videos > 0 or episodes is not None:
        generated.extend(
            record_closed_loop_rollouts(
                cfg,
                model,
                process,
                target_out,
                num_episodes=num_rollout_videos,
                episodes=episodes,
                max_steps=max_rollout_steps,
            )
        )

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate presentation visuals for Diffusion World Model.'
    )
    parser.add_argument('--eval-config', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--eval-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--episodes', type=int, nargs='*', default=None)
    parser.add_argument('--num-dreams', type=int, default=2)
    parser.add_argument('--num-videos', type=int, default=2)
    parser.add_argument('--max-rollout-steps', type=int, default=None)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--proprio-cost-weight', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    paths = generate_presentation_report(
        eval_config_name=args.eval_config,
        policy_path=args.policy,
        run_dir=args.run_dir,
        eval_dir=args.eval_dir,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        episodes=args.episodes,
        num_dream_episodes=args.num_dreams,
        num_rollout_videos=args.num_videos,
        max_rollout_steps=args.max_rollout_steps,
        num_samples=args.num_samples,
        img_size=args.img_size,
        horizon=args.horizon,
        proprio_cost_weight=args.proprio_cost_weight,
        seed=args.seed,
    )
    for p in paths:
        print('Generated:', p)


if __name__ == '__main__':
    main()
