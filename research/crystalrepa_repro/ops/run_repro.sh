#!/usr/bin/env bash
set -euo pipefail
source /data/dxl/env.sh
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate /data/dxl/envs/mattergen_py310
cd /data/dxl/mattergen_v1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -f /data/dxl/results/crystalrepa_repro/training/r1/training_summary_1000.json ]]; then
  python -m research.crystalrepa_repro.train_repro --max-steps 1000 --devices 8
fi
python -m research.crystalrepa_repro.gate_smoke
if [[ ! -f /data/dxl/results/crystalrepa_repro/training/r1/training_summary_10000.json ]]; then
  python -m research.crystalrepa_repro.train_repro --max-steps 10000 --devices 8
fi
if [[ -f research/crystalrepa_repro/run_evaluation_pipeline.py ]]; then
  python -m research.crystalrepa_repro.run_evaluation_pipeline
fi
