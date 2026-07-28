#!/usr/bin/env bash
set -euo pipefail
cd /data/dxl/mattergen_v1
/data/dxl/envs/mattergen_py310/bin/python -m research.q3_formal256 status
tmux list-panes -t mattergen_q3_formal256 \
  -F '#{pane_pid} #{pane_current_command} #{pane_dead}' 2>/dev/null || true
ps -eo pid,ppid,pgid,user,etime,comm,args \
  | grep -E 'q3_formal256|run_sample.py|relax-worker' \
  | grep -v grep || true
nvidia-smi
