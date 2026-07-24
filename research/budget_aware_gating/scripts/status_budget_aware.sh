#!/usr/bin/env bash
set -euo pipefail
/data/dxl/envs/mattergen_py310/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path
master=Path('/data/dxl/results/budget_aware_gating/progress/master_progress.json')
print(json.dumps(json.loads(master.read_text()), indent=2, ensure_ascii=False))
for name in ('threshold_probe_tasks.json','eight_seed_tasks.json','thirty_two_generation_tasks.json','sixty_four_generation_tasks.json'):
    path=master.parent/name
    if path.exists():
        data=json.loads(path.read_text())
        print(name, dict(Counter(item['status'] for item in data.get('tasks',[]))))

dev=Path('/data/dxl/results/budget_aware_gating/development/progress')
for name in ('generation_progress.json','relax_progress.json'):
    path=dev/name
    if path.exists():
        data=json.loads(path.read_text())
        print('development/'+name, dict(Counter(item['status'] for item in data.get('tasks',[]))))
final=Path('/data/dxl/reports/budget_aware_gating/final/budget_aware_final_report.json')
if final.exists(): print(json.dumps(json.loads(final.read_text()), indent=2, ensure_ascii=False))
PY
