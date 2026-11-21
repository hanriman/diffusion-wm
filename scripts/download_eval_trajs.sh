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

PUSHT_75="${TARGET_DIR}/eval_trajs/pusht/goal_offset_75_val.pt"
PUSHT_40="${TARGET_DIR}/eval_trajs/pusht/goal_offset_40_val.pt"
if [[ -f "${PUSHT_75}" && ! -f "${PUSHT_40}" ]]; then
    echo "Generating easier 40-step Push-T eval slice -> ${PUSHT_40} ..."
    uv run python - "${PUSHT_75}" "${PUSHT_40}" <<'EOF'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
payload = torch.load(src, map_location="cpu", weights_only=False)
sliced_data = []
for ep in payload["data"]:
    sliced_ep = {}
    for k, v in ep.items():
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and len(v) > 41:
            sliced_ep[k] = v[:41].clone() if torch.is_tensor(v) else v[:41]
        else:
            sliced_ep[k] = v
    sliced_data.append(sliced_ep)
payload["data"] = sliced_data
payload["goal_offset_steps"] = 40
payload["eval_budget"] = 60
torch.save(payload, dst)
print(f"Saved {len(sliced_data)} episodes (len={len(sliced_data[0]['pixels'])}) to {dst}")
EOF
fi

PUSHT_25="${TARGET_DIR}/eval_trajs/pusht/goal_offset_25_midpush_val.pt"
if [[ -f "${PUSHT_75}" && ! -f "${PUSHT_25}" ]]; then
    echo "Generating 25-step mid-push Push-T eval slice -> ${PUSHT_25} ..."
    uv run python - "${PUSHT_75}" "${PUSHT_25}" <<'EOF'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
payload = torch.load(src, map_location="cpu", weights_only=False)
sliced_data = []
for ep in payload["data"]:
    sliced_ep = {}
    for k, v in ep.items():
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and len(v) > 36:
            sliced_ep[k] = v[10:36].clone() if torch.is_tensor(v) else v[10:36]
        else:
            sliced_ep[k] = v
    sliced_data.append(sliced_ep)
payload["data"] = sliced_data
payload["goal_offset_steps"] = 25
payload["eval_budget"] = 45
torch.save(payload, dst)
print(f"Saved {len(sliced_data)} episodes (len={len(sliced_data[0]['pixels'])}) to {dst}")
EOF
fi

echo "Done! Evaluation tasks are ready in ${TARGET_DIR}/eval_trajs."
