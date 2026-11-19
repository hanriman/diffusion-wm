# Project Plan & Roadmap (`diffusion-wm`)

This folder tracks our active design documents and execution plans for the Diffusion World Model project and benchmark comparisons on **Push-T** (`swm/PushT-v1`) and **Visual AntMaze** (`swm/OGBAntMaze-v0`).

---

## Current Status (2026-11-19)
- **Local Pipeline & Verification:** `ruff check` passes, `18/18` `pytest` tests pass, and 1-batch end-to-end `pusht_dwm` (`128×128`) training + FP16 `precompute_vae_latents.py --img-size 128` verified cleanly.
- **HuggingFace Subset Datasets Live:** [`Hanifanrn/pusht-5k`](https://huggingface.co/datasets/Hanifanrn/pusht-5k) (`5,000` episodes, `9.2 GB`) and [`Hanifanrn/antmaze-10k`](https://huggingface.co/datasets/Hanifanrn/antmaze-10k) (`10,000` episodes, `9.5 GB`, `zstd`) wired into `scripts/download_datasets.sh`.
- **Kaggle Push-T Run 1 (`rightfel/pusht`) Findings & Fixes:**
  - *Baseline run (`224×224`, `5` epochs / `5,000` steps, `parallel` `GradientSolver`):* finished `0/10` (`0.0%`) in `~10h` (`eval_gd` took `25,373s` / `7.05h`).
  - *Diagnostics on downloaded checkpoint (`outputs/kaggle_pusht`):* `Identity copy MSE = 0.0328` vs `Model ODE MSE = 0.0253` and `Action sensitivity MSE = 0.000047` — the short `224×224` pixel-only run froze the T-block, dropped the small blue agent dot, and `parallel` `horizon=15` read untrained `temporal_pos_embed[:, 4:]`.
  - *Fixes implemented (ready for Kaggle Run 2):* Push-T `128×128` (`64` patches + FP16 precomputed latents `~1.2 GB`), `use_proprio: true` (`4-D` agent state supervision; `proprio_cost_weight: 0.0` at eval), spatial `use_action_token: true`, `h_ctx ∈ [1..3]` context training in `_sample_training_tau`, `autoregressive` eval + `CEMSolver` default, and single-level `_ode_step` checkpointing + MemEff SDPA.

---

## Phase 1: Core Diffusion World Model — DONE
See [diffusion_wm_plan.md](./diffusion_wm_plan.md).
- Self-contained repo with `uv`, `stable_worldmodel`, `GradientSolver`, and `CEMSolver`.
- `SDVAEWrapper` (`stabilityai/sd-vae-ft-mse`) + `SpatioTemporalDiT` (with spatial `action_token`, `4-D` proprio token on Push-T, and `27-D` proprio token on AntMaze) + `DiffusionWorldModel` (Rectified Flow + Diffusion Forcing).
- Training (`diffusion_wm/train.py`), evaluation (`diffusion_wm/eval.py`), latent precomputation (`scripts/precompute_vae_latents.py`), and presentation visuals (`diffusion_wm/visualize.py`).
- Verified locally with `18` `pytest` unit/smoke/visual tests and `ruff check`.

---

## Phase 2: Kaggle Training, Evaluation & Presentation — IN PROGRESS
1. **Push-T (`notebooks/kaggle_pusht.ipynb`) — READY FOR RUN 2**
   - Precompute `128×128` SD-VAE latents (`~1.2 GB` on `Hanifanrn/pusht-5k`), train `pusht_dwm` (`MAX_EPOCHS=15`, `LIMIT_TRAIN_BATCHES=2000`, `BATCH_SIZE=64`, `lr=2e-4`, `16-mixed`, `use_proprio=true`, `use_action_token=true`).
   - Evaluate with fast `autoregressive` `CEMSolver` (`pusht_cem`, `10` tasks) and optional `pusht_gd`.
   - Inspect `world_model_dreams_ep*.png` (verify moving agent dot + T-block physics), `training_curves.png`, `eval_summary.png`, and `rollout_ep0.gif`.
2. **Visual AntMaze (`notebooks/kaggle_antmaze.ipynb`) — QUEUED AFTER PUSH-T**
   - Train `ant_dwm` on `Hanifanrn/antmaze-10k` (joint visual `4x8x8` + `27-D` proprioception diffusion).
   - Evaluate with `autoregressive` `ant_cem` / `ant_gd`, and collect dream + rollout visuals.

---

## Phase 3: H-JEPA / LeWM Baseline Comparison — PLANNED
See [hjepa_plan.md](../hjepa_plan.md).
Implement **Predictive Latent World Models (LeWM / H-JEPA)** from `scratch/H-JEPA` as an independent package **`hjepa_wm/`** (sharing `wm_common/` data/eval utilities with `diffusion_wm/`):
- **Step 3.0 (Shared `wm_common/` extraction):** Factor dataset, normalizer, checkpoint (`*_object.ckpt` + portable `*_weights.pt`), and `planning_eval` utilities out of `diffusion_wm/` without breaking existing `diffusion_wm` workflows.
- **Step 3.1 (Flat LeWM baseline):** Port 1-level `JEPA` + `ARPredictor` + `SIGReg` (`pusht_lewm` / `ant_lewm`) into `hjepa_wm/`.
- **Step 3.2 (Hierarchical H-JEPA):** Port `HJEPA` + `HierarchicalSolver` (`ant_hjepa_l2`/`l3` and `pusht_hjepa_l2`/`l3`) for multi-timescale planning with both `GradientSolver` and `CEMSolver`.
- **Step 3.3 (Comparison Table):** Compare success rate, training speed, planner compute, and visual interpretability (SD-VAE dreams vs. post-hoc `CLSDecoder`) for the final presentation.

---

## Phase 4: H-JEPA + TEMPO (Temporal Planning Objective) — PLANNED
See [hjepa_tempo_plan.md](../hjepa_tempo_plan.md).
Implement **`hjepa_tempo_wm/`** as an independent wrapper package over frozen `hjepa_wm` checkpoints (zero modifications to `hjepa_wm/`):
- **Step 4.1 (Core TEMPO & Shared Latent Cache):** Port `TemporalMap`, `MultiLevelLatentBuffer`, `MapTrainer`, and `(checkpoint_hash, dataset_name)`-keyed multi-level HDF5 latent caching (`tempo_cache/`).
- **Step 4.2 (Differentiable & CEM Hierarchical TEMPO Objective):** Implement `TempoCostModel` and `TempoLevelCostModel` (defaulting to top-level TEMPO $w_L > 0, w_1 = 0$) supporting `CEMSolver`, `GradientSolver` (`sqdist` smooth gradients), and re-rank mode.
- **Step 4.3 (Offline + Online Closed-Loop Refinement & Evaluation):** Train $F_{\phi, \ell}$ offline on trajectory windows and refine online on closed-loop planner rollouts (`HJEPAOnlineRecorder`), then evaluate across Push-T and Visual AntMaze benchmarks.
