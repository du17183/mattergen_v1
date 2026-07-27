#!/usr/bin/env bash
set -euo pipefail

progress="/data/dxl/results/spg_fastgate/progress/master_progress.json"
if [[ -f "$progress" ]]; then
  /data/dxl/envs/mattergen_py310/bin/python -m json.tool "$progress"
else
  echo "progress file not created"
fi
tmux ls 2>/dev/null | grep 'mattergen_spg_fastgate' || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader || true
