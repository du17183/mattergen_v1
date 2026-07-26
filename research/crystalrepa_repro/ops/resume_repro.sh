#!/usr/bin/env bash
set -euo pipefail
session=mattergen_crystalrepa_repro
log=/data/dxl/logs/crystalrepa_repro/crystalrepa_repro.log
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists: $session"
  exit 0
fi
tmux new-session -d -s "$session" "exec /data/dxl/reports/crystalrepa_repro/run_repro.sh >> '$log' 2>&1"
tmux display-message -p -t "$session" '#{pane_pid}'
