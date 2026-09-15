"""Start fresh stages only when the frozen offline decision is explicitly GO."""

import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
DECISION = ROOT / "offline/decision_summary.json"

while not DECISION.exists():
    live = subprocess.run(["tmux", "has-session", "-t", "risk_cfg_pipeline"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not live:
        raise RuntimeError("dataset/offline pipeline exited without a decision")
    time.sleep(30)

decision = json.loads(DECISION.read_text())
if decision["OFFLINE_PREDICTABILITY"] != "GO":
    print(json.dumps({"event": "STOP", "reason": decision["OFFLINE_PREDICTABILITY"]}))
    raise SystemExit(0)

result = subprocess.run(
    ["/mnt/datasets-livsyn/dxl/alm/.venv/bin/python", str(ROOT / "run_fresh_pipeline.py")],
    cwd=ROOT.parents[1],
)
raise SystemExit(result.returncode)
