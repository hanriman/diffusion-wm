#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/diffusion_wm/assets"

mkdir -p "${TARGET_DIR}"

echo "Downloading Push-T and Visual AntMaze evaluation trajectories to ${TARGET_DIR}/eval_trajs ..."
uv run hf download jepa-world-models/h-jepa \
    --repo-type dataset \
    --local-dir "${TARGET_DIR}" \
    --include "eval_trajs/pusht/*" \
    --include "eval_trajs/ant/*" \
    --include "eval_trajs/SHA256SUMS"

if [[ -f "${TARGET_DIR}/eval_trajs/SHA256SUMS" ]]; then
    (cd "${TARGET_DIR}/eval_trajs" && sha256sum -c --ignore-missing SHA256SUMS)
fi

echo "Done! Evaluation tasks are ready in ${TARGET_DIR}/eval_trajs."
