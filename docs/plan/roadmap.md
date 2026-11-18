# Project Plan & Roadmap (`diffusion-wm`)

This folder tracks our active design documents and execution plans for the Diffusion World Model project and benchmark comparisons on **Push-T** (`swm/PushT-v1`) and **Visual AntMaze** (`swm/OGBAntMaze-v0`).

---

## Phase 1: Core Diffusion World Model — DONE
See [diffusion_wm_plan.md](./diffusion_wm_plan.md).
- Self-contained repo with `uv`, `stable_worldmodel`, `GradientSolver`, and `CEMSolver`.
- `SDVAEWrapper` (`stabilityai/sd-vae-ft-mse`) + `SpatioTemporalDiT` (with 27-D proprio token for AntMaze) + `DiffusionWorldModel` (Rectified Flow + Diffusion Forcing).
- Training (`diffusion_wm/train.py`), evaluation (`diffusion_wm/eval.py`), and presentation visuals (`diffusion_wm/visualize.py`).
- Verified locally with 14 `pytest` unit/smoke tests and `ruff check`.

---

## Phase 2: Kaggle Training, Evaluation & Presentation — IN PROGRESS
1. **Push-T (`notebooks/kaggle_pusht.ipynb`)**
   - Train `pusht_dwm` (`MAX_EPOCHS=5`, `LIMIT_TRAIN_BATCHES=1000`, `16-mixed` precision).
   - Evaluate with `GradientSolver` (`pusht_gd`) and `CEMSolver` (`pusht_cem`).
   - Collect `training_curves.png`, `world_model_dreams_ep*.png`, `eval_summary.png`, and `rollout_ep*.gif` for presentation.
2. **Visual AntMaze (`notebooks/kaggle_antmaze.ipynb`)**
   - Train `ant_dwm` (joint visual `4x8x8` + 27-D proprioception diffusion).
   - Evaluate with `ant_gd` and `ant_cem`, and collect dream + rollout visuals.

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
