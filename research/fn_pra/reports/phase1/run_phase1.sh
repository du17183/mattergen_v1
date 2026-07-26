#!/usr/bin/env bash
set -euo pipefail

SESSION=mattergen_fn_pra_phase1
REPORT=/data/dxl/reports/fn_pra/phase1
LOG=/data/dxl/logs/fn_pra/phase1/fn_pra_phase1.log

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash -lc '
  set -o pipefail
  export PYTHONPATH=/data/dxl/mattergen_v1
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export NUMEXPR_NUM_THREADS=2
  export TOKENIZERS_PARALLELISM=false
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  cd /data/dxl/mattergen_v1
  /data/dxl/envs/mattergen_py310/bin/python -m research.fn_pra.run_phase1 --resume \
    2>&1 | tee -a /data/dxl/logs/fn_pra/phase1/fn_pra_phase1.log
  exec bash
'"

tmux display-message -p -t "$SESSION" '#{pane_pid}' > "$REPORT/launcher.pid"
echo "started $SESSION pid=$(<"$REPORT/launcher.pid") log=$LOG"
