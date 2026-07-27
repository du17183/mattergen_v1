#!/usr/bin/env bash
set -euo pipefail

session="mattergen_mps_fastgate"
progress="/data/dxl/results/mps_fastgate/progress/master_progress.json"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "TMUX_STATUS=running"
  tmux list-panes -t "$session" -F 'PANE_PID=#{pane_pid} COMMAND=#{pane_current_command}'
else
  echo "TMUX_STATUS=stopped"
fi

if [[ -f "$progress" ]]; then
  python -m json.tool "$progress"
fi

nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
