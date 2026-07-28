#!/usr/bin/env bash
set -euo pipefail

session="mattergen_a0_e3g_formal256"
if tmux has-session -t "$session" 2>/dev/null; then
  tmux send-keys -t "$session" C-c
  echo "sent SIGINT through tmux to $session"
else
  echo "$session is not running"
fi
