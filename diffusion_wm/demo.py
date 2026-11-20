"""Interactive lecture demo & standalone HTML slider for Diffusion World Model."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

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
        img_transform,
        load_eval_config,
    )
    from .utils import resolve_model_checkpoint_path
    from .visualize import _tensor_01_to_uint8
except ImportError:
    from planning_eval import (
        _apply_model_overrides,
        _build_eval_process_from_policy_normalizer,
        _build_policy,
        _eval_sdpa_kernel,
        _resolve_existing_path,
        img_transform,
        load_eval_config,
    )
    from utils import resolve_model_checkpoint_path
    from visualize import _tensor_01_to_uint8


def _encode_png_b64(img_hwc: np.ndarray, size: int = 192) -> str:
    pil = Image.fromarray(np.asarray(img_hwc, dtype=np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    buf = io.BytesIO()
    pil.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _draw_action_arrow(
    img_hwc: np.ndarray,
    dx: float,
    dy: float,
    size: int = 192,
) -> np.ndarray:
    """Draw a clean action vector arrow on the bottom-left HUD corner of `img_hwc`."""
    pil = Image.fromarray(np.asarray(img_hwc, dtype=np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    draw = ImageDraw.Draw(pil)
    cx, cy = 30, size - 30
    r = 22
    draw.ellipse(
        (cx - r, cy - r, cx + r, cy + r),
        fill=(255, 255, 255),
        outline=(60, 60, 60),
        width=2,
    )
    scale = 18.0
    ex = round(cx + float(np.clip(dx, -1.0, 1.0)) * scale)
    ey = round(cy + float(np.clip(dy, -1.0, 1.0)) * scale)
    draw.line((cx, cy, ex, ey), fill=(220, 40, 40), width=3)
    draw.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(220, 40, 40))
    return np.asarray(pil)


def _decode_ode_history(
    model,
    history: list[tuple[float, torch.Tensor]],
    *,
    batch_idx: int = 0,
    decode_chunk_size: int = 16,
    img_size: int = 192,
) -> tuple[list[float], list[list[str]]]:
    """Decode `(K+1)` ODE snapshots `(N, T_fut, C, H', W')` -> base64 PNG grid `[tau_idx][k_idx]`."""
    taus = [round(float(tau), 3) for tau, _ in history]
    lat_grid = torch.cat(
        [z[batch_idx] for _, z in history], dim=0
    )  # ((K+1)*T_fut, C, H', W')
    t_fut = history[0][1].shape[1]

    decoded_chunks = []
    with torch.no_grad():
        for start in range(0, lat_grid.size(0), decode_chunk_size):
            end = min(start + decode_chunk_size, lat_grid.size(0))
            decoded_chunks.append(model.decode_pixels(lat_grid[start:end]).cpu())
    decoded = torch.cat(decoded_chunks, dim=0)  # ((K+1)*T_fut, 3, H, W)

    frames_b64: list[list[str]] = []
    idx = 0
    for _ in range(len(history)):
        row = []
        for _k in range(t_fut):
            u8 = _tensor_01_to_uint8(decoded[idx])
            row.append(_encode_png_b64(u8, size=img_size))
            idx += 1
        frames_b64.append(row)
    return taus, frames_b64


def _rollout_with_history(
    model,
    ctx_vis: torch.Tensor,
    actions_norm: torch.Tensor,
    ctx_proprio: torch.Tensor | None = None,
    *,
    num_ode_steps: int = 8,
    noise_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[float, torch.Tensor]]]:
    """Run forward ODE rollout capturing `(tau, z_fut_vis)` without mutating model defaults."""
    orig_noise = float(getattr(model, 'rollout_noise_scale', 0.0))
    model.rollout_noise_scale = float(noise_scale)
    try:
        with torch.no_grad():
            if str(getattr(model, 'rollout_mode', 'parallel')).lower() == 'autoregressive':
                return model.sample_ode_autoregressive(
                    ctx_vis,
                    actions_norm,
                    ctx_proprio=ctx_proprio,
                    num_ode_steps=num_ode_steps,
                    return_history=True,
                )
            return model.sample_ode_parallel(
                ctx_vis,
                actions_norm,
                ctx_proprio=ctx_proprio,
                num_ode_steps=num_ode_steps,
                return_history=True,
            )
    finally:
        model.rollout_noise_scale = orig_noise


def _select_demo_episodes(
    payload: dict, eval_dir: str | Path | None, max_eps: int = 2
) -> list[tuple[int, bool | None]]:
    """Pick episodes prioritizing solved episodes first (`metrics.yaml`), then ep 0."""
    n_avail = len(payload['data'])
    if eval_dir is not None:
        metrics_path = Path(eval_dir) / 'metrics.yaml'
        if metrics_path.exists():
            m = OmegaConf.load(metrics_path)
            succ = [bool(x) for x in m.get('episode_successes', [])]
            winners = [i for i, s in enumerate(succ) if s and i < n_avail]
            losers = [i for i, s in enumerate(succ) if not s and i < n_avail]
            ordered = winners + losers
            if ordered:
                chosen = ordered[:max_eps]
                return [(i, succ[i]) for i in chosen]
    return [(i, None) for i in range(min(max_eps, n_avail))]


def build_counterfactual_section(
    cfg,
    model,
    process: dict,
    payload: dict,
    *,
    episode_idx: int = 0,
    horizon: int = 7,
    num_ode_steps: int = 8,
    img_size: int = 192,
) -> dict[str, Any]:
    """Precompute `(noise_scale, action_preset, tau, horizon_k)` dreams for Tab 2."""
    ep = payload['data'][min(episode_idx, len(payload['data']) - 1)]
    frameskip = int(getattr(model, 'frameskip', 5))
    h_steps = min(horizon, max(1, (ep['pixels'].size(0) - 1) // frameskip))
    tx = img_transform(cfg)
    device = next(model.parameters()).device

    raw_act_seq = np.asarray(ep['action'][: h_steps * frameskip])
    raw_px_0 = tx(ep['pixels'][0]).unsqueeze(0).unsqueeze(0).to(device)
    goal_idx = ep['pixels'].size(0) - 1
    goal_px = tx(ep['pixels'][goal_idx]).unsqueeze(0).unsqueeze(0).to(device)

    ctx_p = None
    goal_p = None
    if getattr(model, 'use_proprio', False) and 'proprio' in ep:
        p0 = np.asarray(ep['proprio'][0:1])
        pg = np.asarray(ep['proprio'][goal_idx : goal_idx + 1])
        if 'proprio' in process:
            p0 = process['proprio'].transform(p0)
            pg = process['proprio'].transform(pg)
        ctx_p = torch.from_numpy(p0).float().unsqueeze(0).to(device)
        goal_p = torch.from_numpy(pg).float().unsqueeze(0).to(device)

    with torch.no_grad():
        ctx_vis = model.vae.encode(raw_px_0)
        goal_vis = model.vae.encode(goal_px)
        gt_u8 = _tensor_01_to_uint8(model.decode_pixels(ctx_vis[0])[0])
        goal_u8 = _tensor_01_to_uint8(model.decode_pixels(goal_vis[0])[0])

    presets: list[tuple[str, np.ndarray]] = [
        ('EXPERT', raw_act_seq),
        ('UP', np.tile(np.array([[0.0, -0.6]], dtype=np.float32), (h_steps * frameskip, 1))),
        ('DOWN', np.tile(np.array([[0.0, 0.6]], dtype=np.float32), (h_steps * frameskip, 1))),
        ('LEFT', np.tile(np.array([[-0.6, 0.0]], dtype=np.float32), (h_steps * frameskip, 1))),
        ('RIGHT', np.tile(np.array([[0.6, 0.0]], dtype=np.float32), (h_steps * frameskip, 1))),
    ]

    presets_out = []
    for name, raw_seq in presets:
        act_norm = (
            process['action'].transform(raw_seq)
            if 'action' in process
            else raw_seq
        )
        act_t = (
            torch.from_numpy(act_norm.reshape(1, h_steps, -1))
            .float()
            .to(device)
        )
        mean_xy = raw_seq.reshape(h_steps, frameskip, -1).mean(axis=1)[:, :2].tolist()

        by_noise = {}
        for sigma in (1.0, 0.0):
            pred_vis, pred_p, hist = _rollout_with_history(
                model,
                ctx_vis,
                act_t,
                ctx_proprio=ctx_p,
                num_ode_steps=num_ode_steps,
                noise_scale=sigma,
            )
            taus, frames_b64 = _decode_ode_history(
                model, hist, img_size=img_size
            )
            info = {
                'rollout_ctx_len': 1,
                'predicted_embed_0': pred_vis.unsqueeze(1),
                'goal_embed_0': goal_vis.unsqueeze(1),
                'cost_last_n': min(2, h_steps),
            }
            if pred_p is not None and goal_p is not None:
                info['predicted_proprio_0'] = pred_p.unsqueeze(1)
                info['goal_proprio_0'] = goal_p.unsqueeze(1)
            cost_val = float(model.criterion(info).item())
            by_noise[str(sigma)] = {
                'taus': taus,
                'frames': frames_b64,
                'cost': round(cost_val, 4),
            }

        presets_out.append({
            'name': name,
            'mean_xy': [[round(float(x), 3), round(float(y), 3)] for x, y in mean_xy],
            'by_noise': by_noise,
        })

    return {
        'episode_idx': int(episode_idx),
        'horizon': int(h_steps),
        'ctx_img': _encode_png_b64(gt_u8, size=img_size),
        'goal_img': _encode_png_b64(goal_u8, size=img_size),
        'presets': presets_out,
    }


def _make_demo_step(
    *,
    world,
    orig_step,
    state: dict[str, Any],
    action_block: int,
    img_size: int,
):
    def _rec_step(*args, **kwargs):
        if not state['done']:
            orig_step(*args, **kwargs)
            obs_u8 = np.asarray(world.infos['pixels'][0, -1], dtype=np.uint8)
            goal_u8 = np.asarray(world.infos['goal'][0, -1], dtype=np.uint8)
            replan_idx = max(0, len(state['replans']) - 1)
            rp = state['replans'][replan_idx] if state['replans'] else None
            dx, dy = (0.0, 0.0)
            if rp is not None:
                block_offset = min(
                    max(0, (state['step'] - rp['start_step']) // action_block),
                    len(rp['mean_xy']) - 1,
                )
                dx, dy = rp['mean_xy'][block_offset]
            obs_arrow = _draw_action_arrow(obs_u8, dx, dy, size=img_size)
            term = bool(world.terminateds[0])
            state['steps'].append({
                't': int(state['step'] + 1),
                'env_img': _encode_png_b64(obs_arrow, size=img_size),
                'goal_img': _encode_png_b64(goal_u8, size=img_size),
                'replan_idx': int(replan_idx),
                'dx': round(float(dx), 3),
                'dy': round(float(dy), 3),
                'success': term,
            })
            state['step'] += 1
            if term:
                state['done'] = True
        else:
            orig_step(*args, **kwargs)

    return _rec_step


def record_closed_loop_demo(
    cfg,
    model,
    process: dict,
    payload: dict,
    episodes: list[tuple[int, bool | None]],
    *,
    num_ode_steps: int = 8,
    max_steps: int | None = None,
    img_size: int = 192,
) -> list[dict[str, Any]]:
    """Run closed-loop MPC on `episodes`, capturing env frames, plans, and ODE histories."""
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
    action_block = int(cfg_local.plan_config.get('action_block', 5))
    callables = OmegaConf.to_container(cfg_local.eval.get('callables'), resolve=True)
    ep_idx_all = payload.get('episodes_idx', list(range(len(payload['data']))))
    device = next(model.parameters()).device

    state: dict[str, Any] = {'step': 0, 'done': False, 'replans': [], 'steps': []}
    if hasattr(policy, 'solver'):
        orig_solve = policy.solver.solve

        def _hooked_solve(info_dict, *args, **kwargs):
            out = orig_solve(info_dict, *args, **kwargs)
            best_act = out['actions'][:1].to(device)  # (1, H, act_block*2)
            with torch.no_grad():
                _info = {
                    k: v[:1].to(device)
                    for k, v in info_dict.items()
                    if torch.is_tensor(v)
                }
                if 'embed_0' not in _info:
                    _info = model.encode(_info)
                ctx_vis = _info['embed_0']
                ctx_p = (
                    _info.get('proprio_0')
                    if getattr(model, 'use_proprio', False)
                    else None
                )
                _, _, hist = _rollout_with_history(
                    model,
                    ctx_vis,
                    best_act,
                    ctx_proprio=ctx_p,
                    num_ode_steps=num_ode_steps,
                    noise_scale=0.0,
                )
                taus, frames_b64 = _decode_ode_history(
                    model, hist, img_size=img_size
                )

            act_np = best_act[0].detach().cpu().numpy().reshape(-1, 2)
            if 'action' in process:
                act_np = process['action'].inverse_transform(act_np)
            mean_xy = (
                act_np.reshape(best_act.shape[1], action_block, -1)
                .mean(axis=1)[:, :2]
                .tolist()
            )
            state['replans'].append({
                'start_step': int(state['step']),
                'horizon': int(best_act.shape[1]),
                'taus': taus,
                'frames': frames_b64,
                'mean_xy': [[round(float(x), 3), round(float(y), 3)] for x, y in mean_xy],
            })
            return out

        policy.solver.solve = _hooked_solve

    recorded_eps: list[dict[str, Any]] = []
    for ep_i, known_success in episodes:
        state = {'step': 0, 'done': False, 'replans': [], 'steps': []}
        single_chunk = {
            **payload,
            'data': [dict(payload['data'][ep_i])],
            'episodes_idx': [ep_idx_all[ep_i]],
        }
        world = swm.World(**world_cfg, image_shape=image_shape)
        solved = False
        try:
            world.set_policy(policy)
            world.step = _make_demo_step(
                world=world,
                orig_step=world.step,
                state=state,
                action_block=action_block,
                img_size=img_size,
            )
            with torch.enable_grad(), _eval_sdpa_kernel():
                metrics = world.evaluate_from_dataset(
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
            solved = bool(metrics['episode_successes'][0])
        finally:
            world.close()

        recorded_eps.append({
            'episode_idx': int(ep_i),
            'solved': solved if known_success is None else bool(solved or known_success),
            'steps_taken': len(state['steps']),
            'action_block': int(action_block),
            'replans': state['replans'],
            'steps': state['steps'],
        })

    return recorded_eps


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Diffusion World Model — Interactive Lecture Demo</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }
  .wrap { max-width: 1080px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 6px; }
  .sub { color: #94a3b8; font-size: 13px; margin-bottom: 16px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
  .tab { padding: 8px 16px; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: #cbd5e1; cursor: pointer; font-weight: 600; }
  .tab.active { background: #2563eb; border-color: #3b82f6; color: #fff; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
  .controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; align-items: center; }
  .ctrl label { display: block; font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
  .ctrl input[type=range] { width: 100%; }
  select, button.btn { background: #0f172a; color: #e2e8f0; border: 1px solid #475569; border-radius: 6px; padding: 6px 10px; font-size: 13px; cursor: pointer; }
  button.btn.active { background: #2563eb; border-color: #3b82f6; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; text-align: center; margin-top: 14px; }
  .panel { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px; }
  .panel img { width: 100%; max-width: 260px; border-radius: 6px; background: #fff; }
  .ptitle { font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #f8fafc; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; margin-left: 6px; }
  .ok { background: #16a34a; color: #fff; }
  .fail { background: #dc2626; color: #fff; }
  .meta { font-size: 12px; color: #cbd5e1; margin-top: 8px; }
  .hidden { display: none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Diffusion World Model (Push-T) — Interactive Lecture Explorer</h1>
  <div class="sub">Scrub real closed-loop rollouts, multi-step imagined futures (Horizon k), and Rectified Flow ODE denoising (&tau; = 0 &rarr; 1).</div>
  <div class="tabs">
    <button class="tab active" id="tab-mpc" onclick="switchTab('mpc')">1. Closed-Loop MPC (Plan &amp; Act)</button>
    <button class="tab" id="tab-cf" onclick="switchTab('cf')">2. Diffusion &amp; Counterfactual Dreams</button>
  </div>

  <div id="view-mpc">
    <div class="card">
      <div class="controls">
        <div class="ctrl">
          <label>Episode</label>
          <select id="mpc-ep" onchange="onEpChange()"></select>
          <button class="btn" id="mpc-play" onclick="togglePlay()">Play</button>
        </div>
        <div class="ctrl">
          <label>Real Env Step t: <b id="mpc-t-lbl">1</b></label>
          <input type="range" id="mpc-t" min="0" max="0" value="0" oninput="renderMpc()"/>
        </div>
        <div class="ctrl">
          <label>Imagined Future Step k: <b id="mpc-k-lbl">1</b></label>
          <input type="range" id="mpc-k" min="0" max="0" value="0" oninput="renderMpc()"/>
        </div>
        <div class="ctrl">
          <label>Diffusion ODE Step &tau;: <b id="mpc-tau-lbl">1.00</b></label>
          <input type="range" id="mpc-tau" min="0" max="0" value="0" oninput="renderMpc()"/>
        </div>
      </div>
      <div class="grid">
        <div class="panel">
          <div class="ptitle">Real Environment <span id="mpc-status" class="badge"></span></div>
          <img id="mpc-env-img"/>
          <div class="meta" id="mpc-act-meta"></div>
        </div>
        <div class="panel">
          <div class="ptitle">WM Imagined Future (k=<span id="mpc-k-head"></span>, &tau;=<span id="mpc-tau-head"></span>)</div>
          <img id="mpc-wm-img"/>
          <div class="meta" id="mpc-plan-meta"></div>
        </div>
        <div class="panel">
          <div class="ptitle">Target Goal</div>
          <img id="mpc-goal-img"/>
          <div class="meta">Goal state for episode</div>
        </div>
      </div>
    </div>
  </div>

  <div id="view-cf" class="hidden">
    <div class="card">
      <div class="controls">
        <div class="ctrl">
          <label>Action Candidate</label>
          <div id="cf-presets" style="display:flex;flex-wrap:wrap;gap:6px;"></div>
        </div>
        <div class="ctrl">
          <label>Init Mode</label>
          <select id="cf-noise" onchange="renderCf()">
            <option value="1.0">From Gaussian Noise (&sigma;=1.0)</option>
            <option value="0.0">Deterministic Flow (&sigma;=0.0)</option>
          </select>
          <button class="btn" id="cf-play" onclick="toggleTauPlay()">Animate &tau;</button>
        </div>
        <div class="ctrl">
          <label>Future Horizon k: <b id="cf-k-lbl">1</b></label>
          <input type="range" id="cf-k" min="0" max="0" value="0" oninput="renderCf()"/>
        </div>
        <div class="ctrl">
          <label>Diffusion ODE Step &tau;: <b id="cf-tau-lbl">1.00</b></label>
          <input type="range" id="cf-tau" min="0" max="0" value="0" oninput="renderCf()"/>
        </div>
      </div>
      <div class="grid">
        <div class="panel">
          <div class="ptitle">Initial Observation (t=0)</div>
          <img id="cf-ctx-img"/>
          <div class="meta" id="cf-act-meta"></div>
        </div>
        <div class="panel">
          <div class="ptitle">Diffusion WM Dream (k=<span id="cf-k-head"></span>, &tau;=<span id="cf-tau-head"></span>)</div>
          <img id="cf-dream-img"/>
          <div class="meta" id="cf-cost-meta"></div>
        </div>
        <div class="panel">
          <div class="ptitle">Target Goal</div>
          <img id="cf-goal-img"/>
          <div class="meta">Lower planning cost = closer to goal</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = __DEMO_DATA_JSON__;
let playTimer = null, tauTimer = null, cfPresetIdx = 0;

function switchTab(which) {
  document.getElementById('view-mpc').classList.toggle('hidden', which !== 'mpc');
  document.getElementById('view-cf').classList.toggle('hidden', which !== 'cf');
  document.getElementById('tab-mpc').classList.toggle('active', which === 'mpc');
  document.getElementById('tab-cf').classList.toggle('active', which === 'cf');
}

function initMpc() {
  const sel = document.getElementById('mpc-ep');
  sel.innerHTML = '';
  DATA.closed_loop.forEach((ep, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.text = `Episode ${ep.episode_idx} (${ep.solved ? 'SOLVED in ' + ep.steps_taken + ' steps' : 'Timeout ' + ep.steps_taken + ' steps'})`;
    sel.appendChild(opt);
  });
  onEpChange();
}

function onEpChange() {
  const ep = DATA.closed_loop[parseInt(document.getElementById('mpc-ep').value || '0')];
  const tSlider = document.getElementById('mpc-t');
  tSlider.max = Math.max(0, ep.steps.length - 1);
  tSlider.value = 0;
  const rp = ep.replans[0];
  const kSlider = document.getElementById('mpc-k');
  const tauSlider = document.getElementById('mpc-tau');
  kSlider.max = rp ? Math.max(0, rp.horizon - 1) : 0;
  kSlider.value = rp ? Math.max(0, rp.horizon - 1) : 0;
  tauSlider.max = rp ? Math.max(0, rp.taus.length - 1) : 0;
  tauSlider.value = rp ? Math.max(0, rp.taus.length - 1) : 0;
  renderMpc();
}

function renderMpc() {
  const ep = DATA.closed_loop[parseInt(document.getElementById('mpc-ep').value || '0')];
  if (!ep || !ep.steps.length) return;
  const tIdx = Math.min(parseInt(document.getElementById('mpc-t').value), ep.steps.length - 1);
  const st = ep.steps[tIdx];
  const rp = ep.replans[st.replan_idx];
  const kSlider = document.getElementById('mpc-k');
  kSlider.max = rp ? Math.max(0, rp.horizon - 1) : 0;
  const kIdx = Math.min(parseInt(kSlider.value), Math.max(0, rp.horizon - 1));
  const tauSlider = document.getElementById('mpc-tau');
  const tauIdx = Math.min(parseInt(tauSlider.value), Math.max(0, rp.taus.length - 1));
  const tauVal = rp.taus[tauIdx];

  document.getElementById('mpc-t-lbl').innerText = `${st.t}/${ep.steps.length}`;
  document.getElementById('mpc-k-lbl').innerText = `+${kIdx + 1} (replan@${rp.start_step})`;
  document.getElementById('mpc-tau-lbl').innerText = tauVal.toFixed(2);
  document.getElementById('mpc-k-head').innerText = `+${kIdx + 1}`;
  document.getElementById('mpc-tau-head').innerText = tauVal.toFixed(2);

  document.getElementById('mpc-env-img').src = st.env_img;
  document.getElementById('mpc-goal-img').src = st.goal_img;
  document.getElementById('mpc-wm-img').src = rp.frames[tauIdx][kIdx];

  const badge = document.getElementById('mpc-status');
  badge.innerText = st.success ? 'SUCCESS!' : (ep.solved ? 'SOLVING' : 'STEP ' + st.t);
  badge.className = 'badge ' + (st.success ? 'ok' : (ep.solved ? 'ok' : 'fail'));

  document.getElementById('mpc-act-meta').innerText = `Executed action: dx=${st.dx.toFixed(2)}, dy=${st.dy.toFixed(2)}`;
  const pxy = rp.mean_xy[kIdx] || [0, 0];
  document.getElementById('mpc-plan-meta').innerText = `Planned chunk +${kIdx + 1}: dx=${pxy[0].toFixed(2)}, dy=${pxy[1].toFixed(2)}`;
}

function togglePlay() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; document.getElementById('mpc-play').innerText = 'Play'; return; }
  document.getElementById('mpc-play').innerText = 'Pause';
  playTimer = setInterval(() => {
    const sl = document.getElementById('mpc-t');
    sl.value = (parseInt(sl.value) >= parseInt(sl.max)) ? 0 : parseInt(sl.value) + 1;
    renderMpc();
  }, 160);
}

function initCf() {
  const cf = DATA.counterfactual;
  document.getElementById('cf-ctx-img').src = cf.ctx_img;
  document.getElementById('cf-goal-img').src = cf.goal_img;
  const wrap = document.getElementById('cf-presets');
  wrap.innerHTML = '';
  cf.presets.forEach((p, i) => {
    const b = document.createElement('button');
    b.className = 'btn' + (i === 0 ? ' active' : '');
    b.innerText = p.name;
    b.onclick = () => {
      cfPresetIdx = i;
      Array.from(wrap.children).forEach((el, j) => el.classList.toggle('active', j === i));
      renderCf();
    };
    wrap.appendChild(b);
  });
  const kSl = document.getElementById('cf-k');
  kSl.max = Math.max(0, cf.horizon - 1);
  kSl.value = Math.max(0, cf.horizon - 1);
  const firstNoise = cf.presets[0].by_noise['1.0'];
  const tauSl = document.getElementById('cf-tau');
  tauSl.max = Math.max(0, firstNoise.taus.length - 1);
  tauSl.value = Math.max(0, firstNoise.taus.length - 1);
  renderCf();
}

function renderCf() {
  const cf = DATA.counterfactual;
  const preset = cf.presets[cfPresetIdx];
  const sigma = document.getElementById('cf-noise').value;
  const data = preset.by_noise[sigma];
  const kIdx = Math.min(parseInt(document.getElementById('cf-k').value), cf.horizon - 1);
  const tauIdx = Math.min(parseInt(document.getElementById('cf-tau').value), data.taus.length - 1);
  const tauVal = data.taus[tauIdx];

  document.getElementById('cf-k-lbl').innerText = `+${kIdx + 1} (${(kIdx + 1) * 5} env steps)`;
  document.getElementById('cf-tau-lbl').innerText = tauVal.toFixed(2);
  document.getElementById('cf-k-head').innerText = `+${kIdx + 1}`;
  document.getElementById('cf-tau-head').innerText = tauVal.toFixed(2);
  document.getElementById('cf-dream-img').src = data.frames[tauIdx][kIdx];
  const xy = preset.mean_xy[kIdx] || [0, 0];
  document.getElementById('cf-act-meta').innerText = `${preset.name} chunk +${kIdx + 1}: dx=${xy[0].toFixed(2)}, dy=${xy[1].toFixed(2)}`;
  document.getElementById('cf-cost-meta').innerText = `Planning cost vs goal: ${data.cost.toFixed(4)}`;
}

function toggleTauPlay() {
  if (tauTimer) { clearInterval(tauTimer); tauTimer = null; document.getElementById('cf-play').innerText = 'Animate \\u03c4'; return; }
  document.getElementById('cf-play').innerText = 'Stop';
  tauTimer = setInterval(() => {
    const sl = document.getElementById('cf-tau');
    sl.value = (parseInt(sl.value) >= parseInt(sl.max)) ? 0 : parseInt(sl.value) + 1;
    renderCf();
  }, 260);
}

initMpc();
initCf();
</script>
</body>
</html>
"""


def write_interactive_demo_html(
    *,
    eval_config_name: str,
    policy_path: str | Path,
    out_html_path: str | Path,
    eval_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    episodes: list[int] | None = None,
    num_ode_steps: int = 8,
    max_rollout_steps: int | None = None,
    panel_size: int = 192,
) -> Path:
    """Generate standalone offline HTML slider demo (`Closed-Loop MPC` + `Diffusion & Counterfactuals`)."""
    ckpt_path = resolve_model_checkpoint_path(policy_path, cache_dir)
    cfg = load_eval_config(config_name=eval_config_name)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.policy = str(ckpt_path)
    if cache_dir is not None:
        cfg.cache_dir = str(cache_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.policy.AutoCostModel(str(ckpt_path), cache_dir).to(device).eval()
    _apply_model_overrides(model, cfg)
    process = _build_eval_process_from_policy_normalizer(cfg, model=model)

    trajs_path = _resolve_existing_path(cfg.load_eval_trajs_path)
    payload = torch.load(trajs_path, map_location='cpu', weights_only=False)

    if episodes:
        selected = [(i, None) for i in episodes if 0 <= i < len(payload['data'])]
    else:
        selected = _select_demo_episodes(payload, eval_dir, max_eps=2)

    closed_loop = record_closed_loop_demo(
        cfg,
        model,
        process,
        payload,
        selected,
        num_ode_steps=num_ode_steps,
        max_steps=max_rollout_steps,
        img_size=panel_size,
    )
    cf_ep = selected[0][0] if selected else 0
    counterfactual = build_counterfactual_section(
        cfg,
        model,
        process,
        payload,
        episode_idx=cf_ep,
        horizon=int(cfg.plan_config.horizon),
        num_ode_steps=num_ode_steps,
        img_size=panel_size,
    )

    demo_json = json.dumps({'closed_loop': closed_loop, 'counterfactual': counterfactual})
    html_text = _HTML_TEMPLATE.replace('__DEMO_DATA_JSON__', demo_json)
    out_path = Path(out_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding='utf-8')
    return out_path


def launch_live_demo(
    eval_config_name: str,
    policy_path: str | Path,
    cache_dir: str | Path | None = None,
) -> None:
    """Launch live Jupyter `ipywidgets` sliders for `(action_x, action_y, k, tau, noise)`."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    ckpt_path = resolve_model_checkpoint_path(policy_path, cache_dir)
    cfg = load_eval_config(config_name=eval_config_name)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.policy = str(ckpt_path)
    if cache_dir is not None:
        cfg.cache_dir = str(cache_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.policy.AutoCostModel(str(ckpt_path), cache_dir).to(device).eval()
    _apply_model_overrides(model, cfg)
    process = _build_eval_process_from_policy_normalizer(cfg, model=model)
    payload = torch.load(
        _resolve_existing_path(cfg.load_eval_trajs_path),
        map_location='cpu',
        weights_only=False,
    )
    tx = img_transform(cfg)
    frameskip = int(getattr(model, 'frameskip', 5))
    horizon = int(cfg.plan_config.horizon)

    cache: dict[tuple, dict[str, Any]] = {}
    out = widgets.Output()

    w_ep = widgets.IntSlider(value=0, min=0, max=min(9, len(payload['data']) - 1), description='Episode')
    w_ax = widgets.FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.1, description='Action X')
    w_ay = widgets.FloatSlider(value=-0.5, min=-1.0, max=1.0, step=0.1, description='Action Y')
    w_k = widgets.IntSlider(value=horizon, min=1, max=horizon, description='Horizon k')
    w_steps = widgets.IntSlider(value=8, min=2, max=16, step=2, description='ODE Steps')
    w_tau = widgets.IntSlider(value=8, min=0, max=8, description='Tau Step')
    w_sigma = widgets.FloatSlider(value=1.0, min=0.0, max=1.0, step=0.5, description='Noise Sigma')

    def _sync_tau_max(*_args):
        w_tau.max = w_steps.value
        w_tau.value = min(w_tau.value, w_steps.value)

    w_steps.observe(_sync_tau_max, names='value')

    def _render(*_args):
        key = (w_ep.value, round(w_ax.value, 2), round(w_ay.value, 2), w_steps.value, round(w_sigma.value, 2))
        if key not in cache:
            ep = payload['data'][w_ep.value]
            h = min(horizon, max(1, (ep['pixels'].size(0) - 1) // frameskip))
            px0 = tx(ep['pixels'][0]).unsqueeze(0).unsqueeze(0).to(device)
            g_idx = min(ep['pixels'].size(0) - 1, h * frameskip)
            gpx = tx(ep['pixels'][g_idx]).unsqueeze(0).unsqueeze(0).to(device)
            ctx_p = None
            if getattr(model, 'use_proprio', False) and 'proprio' in ep:
                p0 = np.asarray(ep['proprio'][0:1])
                if 'proprio' in process:
                    p0 = process['proprio'].transform(p0)
                ctx_p = torch.from_numpy(p0).float().unsqueeze(0).to(device)
            raw_act = np.tile(np.array([[w_ax.value, w_ay.value]], dtype=np.float32), (h * frameskip, 1))
            act_norm = process['action'].transform(raw_act) if 'action' in process else raw_act
            act_t = torch.from_numpy(act_norm.reshape(1, h, -1)).float().to(device)
            with torch.no_grad():
                ctx_vis = model.vae.encode(px0)
                goal_vis = model.vae.encode(gpx)
                _, _, hist = _rollout_with_history(
                    model, ctx_vis, act_t, ctx_proprio=ctx_p,
                    num_ode_steps=w_steps.value, noise_scale=w_sigma.value,
                )
                ctx_u8 = _tensor_01_to_uint8(model.decode_pixels(ctx_vis[0])[0])
                goal_u8 = _tensor_01_to_uint8(model.decode_pixels(goal_vis[0])[0])
            cache.clear()
            cache[key] = {'ctx': ctx_u8, 'goal': goal_u8, 'hist': hist, 'h': h, 'decoded': {}}

        entry = cache[key]
        k_idx = min(w_k.value - 1, entry['h'] - 1)
        tau_idx = min(w_tau.value, len(entry['hist']) - 1)
        if (tau_idx, k_idx) not in entry['decoded']:
            with torch.no_grad():
                lat = entry['hist'][tau_idx][1][0, k_idx : k_idx + 1]
                entry['decoded'][(tau_idx, k_idx)] = _tensor_01_to_uint8(model.decode_pixels(lat)[0])

        tau_val = entry['hist'][tau_idx][0]
        ctx_arrow = _draw_action_arrow(entry['ctx'], w_ax.value, w_ay.value, size=192)
        with out:
            out.clear_output(wait=True)
            _fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3), dpi=120)
            axes[0].imshow(ctx_arrow)
            axes[0].set_title(f'Context (Ep {w_ep.value}, dx={w_ax.value:.1f}, dy={w_ay.value:.1f})')
            axes[1].imshow(entry['decoded'][(tau_idx, k_idx)])
            axes[1].set_title(f'WM Dream (k=+{k_idx + 1}, tau={tau_val:.2f})')
            axes[2].imshow(entry['goal'])
            axes[2].set_title('Target Goal')
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

    for w in (w_ep, w_ax, w_ay, w_k, w_steps, w_tau, w_sigma):
        w.observe(_render, names='value')

    controls = widgets.VBox([
        widgets.HBox([w_ep, w_steps, w_sigma]),
        widgets.HBox([w_ax, w_ay]),
        widgets.HBox([w_k, w_tau]),
    ])
    display(controls, out)
    _render()


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate offline HTML interactive Diffusion WM demo.')
    parser.add_argument('--eval-config', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--eval-dir', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--episodes', type=int, nargs='*', default=None)
    parser.add_argument('--num-ode-steps', type=int, default=8)
    parser.add_argument('--max-rollout-steps', type=int, default=None)
    args = parser.parse_args()

    out_file = write_interactive_demo_html(
        eval_config_name=args.eval_config,
        policy_path=args.policy,
        out_html_path=args.out_path,
        eval_dir=args.eval_dir,
        cache_dir=args.cache_dir,
        episodes=args.episodes,
        num_ode_steps=args.num_ode_steps,
        max_rollout_steps=args.max_rollout_steps,
    )
    print('Generated interactive demo:', out_file)


if __name__ == '__main__':
    main()
