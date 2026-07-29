#!/usr/bin/env bash
set -euo pipefail
if ! tmux has-session -t mattergen_q3_frozen64 2>/dev/null; then
  echo 'mattergen_q3_frozen64 is not running'
  exit 0
fi
tmux list-panes -t mattergen_q3_frozen64 \
  -F '#{pane_pid} #{pane_current_path} #{pane_current_command}'
tmux send-keys -t mattergen_q3_frozen64 C-c
echo 'SIGINT sent only to mattergen_q3_frozen64'
