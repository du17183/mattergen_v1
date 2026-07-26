#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/data/dxl/mattergen_v1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /data/dxl/mattergen_v1
exec /data/dxl/envs/mattergen_py310/bin/python -m research.fn_pra.run_phase1 --resume
