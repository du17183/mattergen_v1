#!/usr/bin/env bash
set -euo pipefail

session="mattergen_gemnet_fused_fastgate"
if tmux has-session -t "$session" 2>/dev/null; then
  tmux send-keys -t "$session" C-c
  echo "sent SIGINT-equivalent Ctrl-C to $session"
else
  echo "session not running"
fi
