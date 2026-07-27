#!/usr/bin/env bash
set -euo pipefail

session="mattergen_mps_fastgate"

if ! tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session is not running: $session"
  exit 0
fi

tmux send-keys -t "$session" C-c
echo "SIGINT sent to project tmux session: $session"
echo "Wait for workers and the project MPS server to exit cooperatively."
