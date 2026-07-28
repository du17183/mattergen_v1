#!/usr/bin/env bash
set -euo pipefail
if tmux has-session -t mattergen_q3_frozen64 2>/dev/null; then
  tmux list-panes -t mattergen_q3_frozen64 \
    -F '#{pane_pid} #{pane_current_command} #{pane_dead}'
  exit 0
fi
tmux new-session -d -s mattergen_q3_frozen64 \
  /data/dxl/tools/q3_e3_pcr/frozen64/run.sh
tmux list-panes -t mattergen_q3_frozen64 \
  -F '#{pane_pid} #{pane_current_command} #{pane_dead}'
