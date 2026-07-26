#!/usr/bin/env bash
set -euo pipefail
session=mattergen_crystalrepa_repro
if ! tmux has-session -t "$session" 2>/dev/null; then
  echo "No project session found. Nothing stopped."
  exit 0
fi
pane_pid=$(tmux list-panes -t "$session" -F '#{pane_pid}' | head -1)
pgid=$(ps -o pgid= -p "$pane_pid" | tr -d ' ')
args=$(ps -o args= -p "$pane_pid")
if [[ "$args" != *crystalrepa_repro* && "$args" != *run_repro.sh* ]]; then
  echo "Refusing SIGINT: pane is not verified as this project: $args" >&2
  exit 2
fi
kill -INT -- "-$pgid"
echo "SIGINT sent to verified project process group $pgid"
