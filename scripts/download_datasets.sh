#!/usr/bin/env bash
set -euo pipefail

export HJEPA_HOME="${HJEPA_HOME:-$HOME/.stable_worldmodel}"
mkdir -p "${HJEPA_HOME}"

TARGET="${1:-all}"  # pusht | ant | all
PUSHT_HF_REPO="${PUSHT_HF_REPO:-Hanifanrn/pusht-5k}"
ANT_HF_REPO="${ANT_HF_REPO:-Hanifanrn/antmaze-10k}"

if [[ "${TARGET}" == "pusht" || "${TARGET}" == "all" ]]; then
    echo "Downloading Push-T train/val HDF5 datasets from ${PUSHT_HF_REPO} into ${HJEPA_HOME} ..."
    uv run hf download "${PUSHT_HF_REPO}" \
        --repo-type dataset \
        --local-dir "${HJEPA_HOME}" \
        --include "pusht_expert_train.h5" \
        --include "pusht_expert_val.h5" \
        --include "SHA256SUMS"
    if [[ -f "${HJEPA_HOME}/SHA256SUMS" ]]; then
        (cd "${HJEPA_HOME}" && sha256sum -c --ignore-missing SHA256SUMS)
    fi
fi

if [[ "${TARGET}" == "ant" || "${TARGET}" == "all" ]]; then
    echo "Downloading Visual AntMaze train/val HDF5 datasets from ${ANT_HF_REPO} into ${HJEPA_HOME} ..."
    uv run hf download "${ANT_HF_REPO}" \
        --repo-type dataset \
        --local-dir "${HJEPA_HOME}" \
        --include "visual_antmaze_medium_explore_stitch_train.h5" \
        --include "visual_antmaze_medium_stitch_val.h5" \
        --include "SHA256SUMS"
    if [[ -f "${HJEPA_HOME}/SHA256SUMS" ]]; then
        (cd "${HJEPA_HOME}" && sha256sum -c --ignore-missing SHA256SUMS)
    fi
fi

echo "Datasets ready in ${HJEPA_HOME}."
