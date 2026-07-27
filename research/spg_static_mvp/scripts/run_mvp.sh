#!/usr/bin/env bash
set -euo pipefail

session="mattergen_spg_static_mvp"
project="/data/dxl/mattergen_v1"
python_bin="/data/dxl/envs/mattergen_py310/bin/python"
log="/data/dxl/logs/spg_static_mvp/spg_static_mvp.log"

mkdir -p "$(dirname "$log")"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session"
  exit 0
fi

tmux new-session -d -s "$session" \
  "cd '$project' && exec '$python_bin' -m research.spg_static_mvp.run_shape_audit >>'$log' 2>&1"
tmux display-message -p -t "$session" '#{session_name}:#{session_attached}:#{pane_pid}'
