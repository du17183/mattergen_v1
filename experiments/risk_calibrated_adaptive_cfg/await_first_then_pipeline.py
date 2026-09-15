"""Execution-only bridge from the retained first trajectory to full pipeline."""

from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "dataset/generation/1155501844/run_summary.json"

while not SUMMARY.exists():
    live = subprocess.run(
        ["tmux", "has-session", "-t", "risk_cfg_first"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not live:
        raise RuntimeError("first trajectory exited without an atomic success summary")
    time.sleep(20)

result = subprocess.run(
    ["/mnt/datasets-livsyn/dxl/alm/.venv/bin/python", str(ROOT / "run_dataset_pipeline.py")],
    cwd=ROOT.parents[1],
)
raise SystemExit(result.returncode)
