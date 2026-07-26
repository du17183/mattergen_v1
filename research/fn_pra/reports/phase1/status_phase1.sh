#!/usr/bin/env bash
set -euo pipefail

SESSION=mattergen_fn_pra_phase1
PROGRESS=/data/dxl/results/fn_pra/phase1/progress/master_progress.json

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "TMUX_STATUS=running"
  tmux display-message -p -t "$SESSION" 'PANE_PID=#{pane_pid} PANE_CURRENT_COMMAND=#{pane_current_command} PANE_DEAD=#{pane_dead}'
else
  echo "TMUX_STATUS=absent"
fi

/data/dxl/envs/mattergen_py310/bin/python - "$PROGRESS" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1])
if not path.exists():
    print('PROGRESS=missing')
    raise SystemExit
state=json.loads(path.read_text())
print(f"CURRENT_STAGE={state['current_stage']}")
print(f"OVERALL_STATUS={state['overall_status']}")
for name, item in state['stages'].items():
    if item['status'] != 'pending':
        print(f"{name}={item['status']} | {item['detail']}")
PY

nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
