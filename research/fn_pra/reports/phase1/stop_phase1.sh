#!/usr/bin/env bash
set -euo pipefail

SESSION=mattergen_fn_pra_phase1
PROJECT=/data/dxl/mattergen_v1

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "No active tmux session: $SESSION"
  exit 0
fi

PID=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
if [[ ! "$PID" =~ ^[0-9]+$ ]] || [[ ! -d "/proc/$PID" ]]; then
  echo "Refusing stop: invalid pane PID '$PID'" >&2
  exit 2
fi

OWNER=$(stat -c '%U' "/proc/$PID")
CWD=$(readlink -f "/proc/$PID/cwd")
EXE=$(readlink -f "/proc/$PID/exe")
CMD=$(tr '\0' ' ' < "/proc/$PID/cmdline")
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')

echo "PID=$PID PGID=$PGID OWNER=$OWNER CWD=$CWD EXE=$EXE CMD=$CMD"
if [[ "$OWNER" != "ubuntu" ]] || [[ "$CWD" != "$PROJECT" ]] || [[ "$EXE" != */bash ]] || [[ "$CMD" != *"mattergen_fn_pra_phase1"* && "$CMD" != *"research.fn_pra"* ]]; then
  echo "Refusing stop: process identity verification failed" >&2
  exit 3
fi

tmux send-keys -t "$SESSION" C-c
echo "Sent SIGINT through tmux to verified FN-PRA session; no SIGKILL was used."
