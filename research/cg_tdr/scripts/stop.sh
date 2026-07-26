#!/usr/bin/env bash
set -euo pipefail
session=mattergen_cg_tdr_phase0
pid="$(tmux list-panes -t "$session" -F '#{pane_pid}' 2>/dev/null | head -n1 || true)"
if [[ -z "$pid" ]]; then
  echo "No active $session pane."
  exit 0
fi
cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
if [[ "$cwd" != "/data/dxl/mattergen_v1" ]] || [[ "$cmd" != *"launch_teacher_generation.py"* ]]; then
  echo "Refusing SIGINT: pane ownership check failed." >&2
  echo "pid=$pid cwd=$cwd cmd=$cmd" >&2
  exit 2
fi
kill -INT "$pid"
echo "SIGINT sent to verified CG-TDR launcher pid=$pid"
