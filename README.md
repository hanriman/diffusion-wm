# Diffusion World Model (`diffusion-wm`)

A self-contained **Pretrained VAE Latent Diffusion World Model** for visual model-based planning on **Push-T** (`swm/PushT-v1`) and **Visual AntMaze** (`swm/OGBAntMaze-v0`), built on `stable_worldmodel` and evaluated with the H-JEPA / LeWM benchmark protocols.

## Architecture

- **Visual Encoder / Decoder (`SDVAEWrapper`)**: Frozen pretrained Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`, $8\times$ spatial downsample) mapping RGB frames to spatial latents (`4×16×16` for Push-T `128×128`, `4×8×8` for Visual AntMaze `64×64`) and decoding predicted latents back to RGB images.
- **Joint Proprioception & Spatial Action Token**: When `use_proprio=true` (`4-D` agent state on Push-T with `proprio_cost_weight=0` at eval; `27-D` `qpos[2:] + qvel` on Visual AntMaze), proprioception is projected to a dedicated token and jointly denoised. Optional `use_action_token=true` (enabled on Push-T) appends a per-frame action token directly to spatial self-attention alongside AdaLN-Zero modulation.
- **Dynamics (`SpatioTemporalDiT` + Rectified Flow + Diffusion Forcing)**: Factorized Spatio-Temporal Causal Diffusion Transformer trained across randomized context lengths (`1..history_size`) and mixed synchronized/independent flow times $\tau_t \in [0, 1]$, supporting both **autoregressive** (default for `num_steps=4` models) and **parallel** ODE rollouts.
- **Planning Solvers**: Supports sampling-based **CEMSolver / MPPI** (`stable_worldmodel.solver.CEMSolver`, default for Kaggle notebooks) and differentiable **GradientSolver** (`stable_worldmodel.solver.GradientSolver`, with chunked rollouts, CUDA AMP autocast, and single-level ODE-step checkpointing).

## Current Status

- **Datasets:** 5k Push-T (`Hanifanrn/pusht-5k`, `9.2 GB`) and 10k Visual AntMaze (`Hanifanrn/antmaze-10k`, `9.5 GB`) episode subsets are hosted on HuggingFace and wired into `scripts/download_datasets.sh`.
- **Push-T:** Run 1 (`224×224`, `5k` steps, `parallel` GD) got `0/10` (`0.0%`) due to eval OOM/slowdown and weak action conditioning (`Action sensitivity MSE = 0.000047`). Pipeline has been upgraded (`128×128` + precomputed FP16 latents, `4-D` proprio supervision, spatial `action_token`, randomized `h_ctx` training, `autoregressive` `CEMSolver` default, `15×2000` training schedule) and verified locally (`18/18` pytest pass, `ruff check` clean). Ready for Kaggle Run 2.
- **Visual AntMaze:** Local smoke tests pass; Kaggle run queued after Push-T.

---

## 1. Installation

```bash
uv venv --python 3.11
uv pip install -e ".[env]"
uv pip install pytest ruff
export HJEPA_HOME=/path/to/data  # defaults to ~/.stable_worldmodel
```

## 2. Download Evaluation Tasks & Datasets

```bash
# Download the 50 fixed evaluation tasks for Push-T and Visual AntMaze (~15 MB)
bash scripts/download_eval_trajs.sh

# Download HDF5 training/validation datasets (pusht | ant | all)
bash scripts/download_datasets.sh all
```

*(Recommended for Push-T `128×128`)* Pre-encode `pixels` into FP16 `vae_latents` inside an HDF5 file (`~1.2 GB` on `pusht-5k`) for fast multi-epoch DiT training:

```bash
uv run python scripts/precompute_vae_latents.py "$HJEPA_HOME/pusht_expert_train.h5" --batch-size 100 --img-size 128
```

## 3. Training

```bash
# Train on Push-T
uv run python -m diffusion_wm.train --config-name pusht_dwm seed=42

# Train on Visual AntMaze
uv run python -m diffusion_wm.train --config-name ant_dwm seed=42
```

## 4. Standalone Planning Evaluation

```bash
# Evaluate Push-T with GradientSolver or CEMSolver
uv run python -m diffusion_wm.eval --config-name pusht_gd \
    policy="$HJEPA_HOME/ckpts/pusht/pusht_dwm/seed42/pusht_dwm_object.ckpt"

uv run python -m diffusion_wm.eval --config-name pusht_cem \
    policy="$HJEPA_HOME/ckpts/pusht/pusht_dwm/seed42/pusht_dwm_object.ckpt"

# Evaluate Visual AntMaze with GradientSolver or CEMSolver
uv run python -m diffusion_wm.eval --config-name ant_gd \
    policy="$HJEPA_HOME/ckpts/ant/ant_dwm/seed42/ant_dwm_object.ckpt"

uv run python -m diffusion_wm.eval --config-name ant_cem \
    policy="$HJEPA_HOME/ckpts/ant/ant_dwm/seed42/ant_dwm_object.ckpt"
```

## 5. Testing

```bash
uv run pytest tests/ -v
```
