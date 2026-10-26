#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU START_SEED END_SEED" >&2
  exit 2
fi

physical_gpu="$1"
start_seed="$2"
end_seed="$3"
experiment_root="/mnt/datasets-livsyn/dxl/mattergen_v1_fppc/experiments/fp_pc_p0"
python_bin="/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"
mkdir -p "${experiment_root}/logs"

for seed in $(seq "${start_seed}" "${end_seed}"); do
  log_path="${experiment_root}/logs/diagnostic_seed_${seed}_gpu_${physical_gpu}.log"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  FPPC_PHYSICAL_GPU_ID="${physical_gpu}" \
  PYTHONPATH="/mnt/datasets-livsyn/dxl/mattergen_v1_fppc" \
    "${python_bin}" "${experiment_root}/generate_diagnostic_one.py" --seed "${seed}" \
    >"${log_path}" 2>&1
  echo "completed seed=${seed} physical_gpu=${physical_gpu}"
done
