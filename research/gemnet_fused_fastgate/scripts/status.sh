#!/usr/bin/env bash
set -euo pipefail

progress="/data/dxl/results/gemnet_fused_fastgate/progress/master_progress.json"
if [[ -f "$progress" ]]; then
  jq '{current_stage,overall_status,termination_state,stages}' "$progress"
else
  echo "progress not initialized"
fi
tmux ls 2>&1 | grep mattergen_gemnet_fused_fastgate || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
