"""Wait for all 128 structures, then run frozen evaluation and paired analysis."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "timing_formal32"
ALM_PYTHON = Path("/mnt/datasets-livsyn/dxl/alm/.venv/bin/python")


def write_status(value: dict) -> None:
    path = OUT / "pipeline_status.json"
    temporary = OUT / "pipeline_status.json.tmp"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    if Path(sys.executable).resolve() != ALM_PYTHON.resolve():
        raise RuntimeError("ALM interpreter required")
    OUT.mkdir(exist_ok=True)
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    started = time.time()
    status_path = OUT / "generation_status_attempt3.json"
    while True:
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            if status["status"] == "FAIL":
                write_status({"status": "BLOCKED_GENERATION_FAILED", "generation_status": str(status_path)})
                raise RuntimeError("Timing generation failed; evaluation prohibited")
            if status["status"] == "COMPLETE":
                break
        if time.time() - started > 12 * 3600:
            write_status({"status": "BLOCKED_GENERATION_TIMEOUT", "generation_status": str(status_path)})
            raise TimeoutError("Timing generation did not finish within 12 hours")
        time.sleep(15)
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    env.update(PYTHONPATH=str(PROJECT), TMPDIR=str(PROJECT / ".tmp"),
               XDG_CACHE_HOME=str(PROJECT / ".cache"),
               HF_HOME=str(PROJECT / ".cache/huggingface"),
               TORCH_HOME=str(PROJECT / ".cache/torch"),
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    stages = [
        ("prepare", ["-m", "experiments.rcnfgd_timing_ablation_20260929.evaluate_timing", "--stage", "prepare"]),
        ("properties", ["-m", "experiments.rcnfgd_timing_ablation_20260929.evaluate_timing", "--stage", "properties"]),
        ("relax", ["-m", "experiments.rcnfgd_timing_ablation_20260929.evaluate_timing", "--stage", "relax"]),
        ("quality", ["-m", "experiments.rcnfgd_timing_ablation_20260929.evaluate_timing", "--stage", "quality"]),
        ("analysis", ["-m", "experiments.rcnfgd_timing_ablation_20260929.analyze_timing"]),
    ]
    completed = []
    for stage, command in stages:
        log = logs / f"pipeline_{stage}.log"
        if log.exists():
            raise FileExistsError(f"stage log already exists: {log}")
        write_status({"status": "RUNNING", "stage": stage, "completed_stages": completed,
                      "generation_status": str(status_path), "updated_epoch": time.time()})
        print(json.dumps({"event": "stage_start", "stage": stage, "log": str(log)}), flush=True)
        stage_env = env | {"CUDA_VISIBLE_DEVICES": "2" if stage == "properties" else ""}
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, *command], cwd=PROJECT,
                                    env=stage_env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            write_status({"status": "FAILED", "stage": stage, "return_code": result.returncode,
                          "completed_stages": completed, "log": str(log), "updated_epoch": time.time()})
            raise RuntimeError(f"Timing {stage} stage failed; see {log}")
        completed.append(stage)
        print(json.dumps({"event": "stage_complete", "stage": stage}), flush=True)
    write_status({"status": "COMPLETE", "completed_stages": completed,
                  "generation_status": str(status_path), "updated_epoch": time.time()})
    print(json.dumps({"event": "timing_pipeline_complete", "stages": completed}), flush=True)


if __name__ == "__main__":
    main()
