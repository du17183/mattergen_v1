from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path("/data/dxl")
PROJECT = ROOT / "mattergen_v1"
RESULTS = ROOT / "results/fn_pra/phase1"
REPORTS = ROOT / "reports/fn_pra/phase1"
LOGS = ROOT / "logs/fn_pra/phase1"
CACHE = ROOT / "data/fn_pra_teacher_cache"
PROGRESS = RESULTS / "progress/master_progress.json"
EVENTS = RESULTS / "progress/events.jsonl"
TZ = ZoneInfo("Asia/Shanghai")

STAGES = [
    "environment_audit",
    "data_audit",
    "teacher_candidate_audit",
    "teacher_probe",
    "teacher_selection",
    "online_teacher_validation",
    "teacher_cache",
    "fn_pra_implementation",
    "fn_pra_tests",
    "v1_smoke_training",
    "v1_decision_training",
    "batch_benchmark",
    "eight_seed_generation",
    "thirty_two_seed_generation",
    "mattersim_relaxation",
    "phase1_analysis",
    "github_publish",
    "stop_for_review",
]


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def append_event(stage: str, status: str, detail: str, metrics: dict[str, Any] | None = None) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "time": now(),
        "stage": stage,
        "status": status,
        "detail": detail,
        "metrics": metrics or {},
    }
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def initialize_progress() -> dict[str, Any]:
    state = read_json(PROGRESS)
    if state is not None:
        return state
    state = {
        "schema_version": 1,
        "created_at": now(),
        "updated_at": now(),
        "base_branch": "main",
        "base_commit": "9bc6747a3ddfd26db6d931bcdb6df5d299844544",
        "branch": "feature/fn-pra",
        "tmux_session": "mattergen_fn_pra_phase1",
        "current_stage": "environment_audit",
        "overall_status": "running",
        "formal_validation_started": False,
        "sixty_four_seed_started": False,
        "stages": {
            name: {"status": "pending", "detail": "", "metrics": {}} for name in STAGES
        },
    }
    atomic_json(PROGRESS, state)
    return state


def set_stage(
    stage: str,
    status: str,
    detail: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    state = initialize_progress()
    state["updated_at"] = now()
    state["current_stage"] = stage
    state["stages"][stage] = {
        "status": status,
        "detail": detail,
        "metrics": metrics or {},
        "updated_at": now(),
    }
    if status in {"blocked", "failed", "stop_for_review"}:
        state["overall_status"] = status
    else:
        state["overall_status"] = "running"
    atomic_json(PROGRESS, state)
    append_event(stage, status, detail, metrics)
