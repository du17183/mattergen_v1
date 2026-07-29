#!/usr/bin/env bash
set -euo pipefail
source /data/dxl/env.sh
cd /data/dxl/mattergen_v1
session=mattergen_a0_e3g_compat64
if tmux has-session -t "$session" 2>/dev/null; then
  echo "$session already exists"
  exit 0
fi
tmux new-session -d -s "$session" \
  "/data/dxl/envs/mattergen_py310/bin/python -m research.a0_e3g_compat64 pipeline >> /data/dxl/logs/a0_e3g_compat64/a0_e3g_compat64.log 2>&1"
tmux display-message -p -t "$session" '#{pane_pid}'
