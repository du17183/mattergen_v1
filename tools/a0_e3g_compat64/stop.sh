#!/usr/bin/env bash
set -euo pipefail
session=mattergen_a0_e3g_compat64
if tmux has-session -t "$session" 2>/dev/null; then
  tmux send-keys -t "$session" C-c
fi
