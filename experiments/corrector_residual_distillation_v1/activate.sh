#!/usr/bin/env bash
# Activate the only environment used by this experiment and keep mutable
# download/temporary caches on the project volume.
MATTERGEN_V1_ROOT="/mnt/lis-wam-data/dxl/mattergen_v1"
export MATTERGEN_V1_ROOT
export PIP_CACHE_DIR="${MATTERGEN_V1_ROOT}/.cache/pip"
export HF_HOME="${MATTERGEN_V1_ROOT}/.cache/huggingface"
export TORCH_HOME="${MATTERGEN_V1_ROOT}/.cache/torch"
export TMPDIR="${MATTERGEN_V1_ROOT}/.cache/tmp"
export PATH="${MATTERGEN_V1_ROOT}/.tools/git-lfs/usr/bin:${PATH}"
mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"
# shellcheck disable=SC1091
source "${MATTERGEN_V1_ROOT}/.venv/bin/activate"
