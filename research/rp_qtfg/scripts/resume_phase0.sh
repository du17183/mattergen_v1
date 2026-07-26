#!/usr/bin/env bash
set -euo pipefail
if tmux has-session -t mattergen_rp_qtfg_phase0 2>/dev/null; then
  echo "mattergen_rp_qtfg_phase0 already exists"
  exit 0
fi
rm -f /data/dxl/results/rp_qtfg/phase0/progress/STOP_REQUESTED
tmux new-session -d -s mattergen_rp_qtfg_phase0 \
  "/data/dxl/reports/rp_qtfg/phase0/run_phase0.sh >> /data/dxl/logs/rp_qtfg/phase0/rp_qtfg_phase0.log 2>&1"
tmux list-panes -t mattergen_rp_qtfg_phase0 -F '#{pane_pid} #{pane_current_command}'
