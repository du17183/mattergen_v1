#!/usr/bin/env bash
set -euo pipefail

session="mattergen_a0_e3g_formal256"
log="/data/dxl/logs/a0_e3g_formal256/a0_e3g_formal256.log"
mkdir -p "$(dirname "$log")"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "$session already running"
  exit 0
fi

tmux new-session -d -s "$session" \
  "cd /data/dxl/mattergen_v1 && exec /data/dxl/envs/mattergen_py310/bin/python -m research.a0_e3g_formal256 audit >> '$log' 2>&1"
echo "started $session"
