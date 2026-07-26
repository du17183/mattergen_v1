#!/usr/bin/env bash
set -euo pipefail
session=mattergen_crystalrepa_repro
if tmux has-session -t "$session" 2>/dev/null; then
  tmux list-panes -t "$session" -F 'session=#{session_name} pane_pid=#{pane_pid} dead=#{pane_dead} command=#{pane_current_command}'
else
  echo "tmux_session=absent"
fi
python -m json.tool /data/dxl/results/crystalrepa_repro/progress/master_progress.json
ps -eo pid,ppid,pgid,sid,user,etime,comm,args | grep -E 'crystalrepa|train_repro|run_repro' | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
