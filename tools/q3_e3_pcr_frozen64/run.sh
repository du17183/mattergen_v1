#!/usr/bin/env bash
set -euo pipefail
source /data/dxl/env.sh
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /data/dxl/mattergen_v1
exec /data/dxl/envs/mattergen_py310/bin/python \
  -m research.q3_frozen64 pipeline \
  >> /data/dxl/logs/q3_e3_pcr/frozen64/q3_frozen64.log 2>&1
