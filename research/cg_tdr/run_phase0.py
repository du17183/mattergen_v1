#!/usr/bin/env python3
"""Strict serial Phase-B orchestrator.

It waits for the active Teacher launcher, then performs labeling, training,
tests, the 8-seed gate, and only conditionally the 32-seed gate.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = Path("/data/dxl/mattergen_v1")
RESULTS = Path("/data/dxl/results/cg_tdr/phase0")
REPORTS = Path("/data/dxl/reports/cg_tdr/phase0")
LOG = Path("/data/dxl/logs/cg_tdr/phase0/cg_tdr_pipeline.log")
PROGRESS = RESULTS / "progress/master_progress.json"
EVENTS = RESULTS / "progress/events.jsonl"
MATTERGEN_PYTHON = Path("/data/dxl/envs/mattergen_py310/bin/python")
TEACHER_PYTHON = Path("/data/dxl/envs/fn_pra_teacher/bin/python")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def event(name: str, **payload) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as stream:
        stream.write(json.dumps({"time": now(), "event": name, **payload}) + "\n")


def set_stage(stage: str, status: str, **payload) -> None:
    current = {}
    try:
        current = json.loads(PROGRESS.read_text())
    except Exception:
        pass
    current.update(
        {
            "stage": stage,
            "overall_status": status,
            "pipeline_pid": os.getpid(),
            "eight_seed_started": stage.startswith("eight_")
            or bool(current.get("eight_seed_started")),
            "thirty_two_seed_started": stage.startswith("thirty_two")
            or bool(current.get("thirty_two_seed_started")),
            "sixty_four_seed_started": False,
            "formal_seeds_started": False,
            "updated_at": now(),
            **payload,
        }
    )
    atomic_json(PROGRESS, current)
    event("stage", stage=stage, status=status, **payload)


def run(command: list[str], stage: str, environment: dict[str, str] | None = None) -> None:
    set_stage(stage, "running", command=" ".join(command))
    merged = os.environ.copy()
    merged.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if environment:
        merged.update(environment)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as stream:
        stream.write(f"\n[{now()}] START {stage}: {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=REPOSITORY,
            env=merged,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        stream.write(f"[{now()}] END {stage}: rc={result.returncode}\n")
        stream.flush()
    if result.returncode != 0:
        set_stage(stage, "failed", return_code=result.returncode)
        raise RuntimeError(f"{stage} failed with return code {result.returncode}")
    set_stage(stage, "completed")


def wait_for_teacher() -> None:
    set_stage("teacher_data_generation", "waiting_for_active_launcher")
    while True:
        progress = json.loads(PROGRESS.read_text())
        status = progress.get("overall_status")
        if status == "completed":
            if int(progress.get("completed_tasks", 0)) != 512:
                raise RuntimeError("Teacher launcher claimed completion without 512 tasks")
            return
        if status in {"failed", "stopped"}:
            raise RuntimeError(f"Teacher generation ended with status={status}")
        time.sleep(30)


def main() -> int:
    lock_path = RESULTS / "progress/pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another CG-TDR Phase-B pipeline holds the lock.", file=sys.stderr)
        return 2
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    try:
        wait_for_teacher()
        run(
            [str(TEACHER_PYTHON), "research/cg_tdr/build_teacher_labels.py"],
            "teacher_data_validation",
            {"CUDA_VISIBLE_DEVICES": "0"},
        )
        run(
            [str(MATTERGEN_PYTHON), "research/cg_tdr/train.py"],
            "training_full",
            {"CUDA_VISIBLE_DEVICES": "0"},
        )
        run(
            [
                str(MATTERGEN_PYTHON),
                "-m",
                "pytest",
                "mattergen/diffusion/tests/test_guidance_schedule.py",
                "research/cg_tdr/tests",
                "-q",
            ],
            "unit_tests",
            {"CUDA_VISIBLE_DEVICES": ""},
        )
        run(
            [str(MATTERGEN_PYTHON), "-m", "pytest", "-q"],
            "full_tests",
            {"CUDA_VISIBLE_DEVICES": ""},
        )
        run(
            [
                str(MATTERGEN_PYTHON),
                "-m",
                "research.cg_tdr.experiment_generation",
                "launch",
                "--mode",
                "eight",
            ],
            "eight_seed_generation",
        )
        run(
            [
                str(MATTERGEN_PYTHON),
                "-m",
                "research.cg_tdr.experiment_relax",
                "launch",
                "--mode",
                "eight",
                "--workers",
                "16",
            ],
            "eight_seed_relax",
        )
        run(
            [
                str(MATTERGEN_PYTHON),
                "-m",
                "research.cg_tdr.analyze",
                "--mode",
                "eight",
            ],
            "eight_seed_review",
            {"CUDA_VISIBLE_DEVICES": ""},
        )
        eight = json.loads((REPORTS / "eight_seed/analysis_report.json").read_text())
        if not eight["CG_TDR_EIGHT_SEED_GO"]:
            set_stage(
                "stop_for_review",
                "completed",
                CG_TDR_EIGHT_SEED_GO=False,
                CG_TDR_MVP_GO=False,
                CG_TDR_MVP_NO_GO=True,
                stop_reason="eight_seed_gate_no_go",
            )
            return 0
        run(
            [
                str(MATTERGEN_PYTHON),
                "-m",
                "research.cg_tdr.experiment_generation",
                "launch",
                "--mode",
                "thirtytwo",
            ],
            "thirty_two_generation",
        )
        run(
            [
                str(MATTERGEN_PYTHON),
                "-m",
                "research.cg_tdr.experiment_relax",
                "launch",
                "--mode",
                "thirtytwo",
                "--workers",
                "16",
            ],
            "thirty_two_relax",
        )
        run(
            [
                str(MATTERGEN_PYTHON),
                "-m",
                "research.cg_tdr.analyze",
                "--mode",
                "thirtytwo",
            ],
            "metrics",
            {"CUDA_VISIBLE_DEVICES": ""},
        )
        final = json.loads((REPORTS / "thirty_two_seed/analysis_report.json").read_text())
        set_stage(
            "stop_for_review",
            "completed",
            CG_TDR_EIGHT_SEED_GO=True,
            CG_TDR_MVP_GO=bool(final["CG_TDR_MVP_GO"]),
            CG_TDR_MVP_NO_GO=bool(final["CG_TDR_MVP_NO_GO"]),
        )
        return 0
    except BaseException as error:
        set_stage("pipeline", "failed", error=repr(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
