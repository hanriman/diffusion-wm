# Diffusion World Model (`diffusion-wm`)

A self-contained **Pretrained VAE Latent Diffusion World Model** for visual model-based planning on **Push-T** (`swm/PushT-v1`) and **Visual AntMaze** (`swm/OGBAntMaze-v0`), built on `stable_worldmodel` and evaluated with the H-JEPA / LeWM benchmark protocols.

## Architecture

- **Visual Encoder / Decoder (`SDVAEWrapper`)**: Frozen pretrained Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`, $8\times$ spatial downsample) mapping RGB frames to spatial latents (`4×28×28` for Push-T `224×224`, `4×8×8` for Visual AntMaze `64×64`) and decoding predicted latents back to RGB images.
- **Joint Proprioception Diffusion**: When `use_proprio=true` (Visual AntMaze, 27-D `qpos[2:]` + `qvel`), proprioception is projected to a dedicated token alongside the spatial patch tokens and jointly denoised so both visual and proprioceptive predictions guide planning.
- **Dynamics (`SpatioTemporalDiT` + Rectified Flow + Diffusion Forcing)**: Factorized Spatio-Temporal Causal Diffusion Transformer with AdaLN-Zero conditioning on per-frame flow time $\tau_t \in [0, 1]$ and stacked actions $a_t$. Supports both fast **parallel full-horizon ODE rollouts** and **sequential autoregressive ODE rollouts**.
- **Planning Solvers**: Supports both differentiable **GradientSolver** (`stable_worldmodel.solver.GradientSolver`) and sampling-based **CEMSolver / MPPI** (`stable_worldmodel.solver.CEMSolver`).

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

*(Optional)* Pre-encode `pixels` into `vae_latents` inside an HDF5 file for faster multi-epoch DiT training:

```bash
uv run python scripts/precompute_vae_latents.py "$HJEPA_HOME/pusht_expert_train.h5"
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
