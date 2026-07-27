#!/usr/bin/env bash
set -euo pipefail

cat /data/dxl/results/cg_tdr/phase0/progress/master_progress.json
echo
ps -eo pid,ppid,pgid,sid,user,etime,comm,args \
  | grep -E 'research.cg_tdr|research/cg_tdr' \
  | grep -v grep || true
echo
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
