#!/usr/bin/env bash
set -euo pipefail
source /data/dxl/env.sh
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate /data/dxl/envs/mattergen_py310
cd /data/dxl/mattergen_v1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export MATTERGEN_BUDGET_LAUNCHER=1
mkdir -p /data/dxl/results/budget_aware_gating/progress
exec 9>/data/dxl/results/budget_aware_gating/progress/launcher.lock
if ! flock -n 9; then
  echo 'A budget-aware launcher already holds the lock.' >&2
  exit 75
fi
/data/dxl/envs/mattergen_py310/bin/python /data/dxl/tools/budget_aware_gating/run_probe_smoke.py
/data/dxl/envs/mattergen_py310/bin/python /data/dxl/tools/budget_aware_gating/run_budget_validation.py
