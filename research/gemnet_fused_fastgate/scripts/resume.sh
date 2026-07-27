#!/usr/bin/env bash
set -euo pipefail

session="mattergen_gemnet_fused_fastgate"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "session already running: $session"
else
  "$(dirname "$0")/run.sh"
fi
