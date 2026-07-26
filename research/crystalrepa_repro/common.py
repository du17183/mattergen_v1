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
RESULTS = ROOT / "results/crystalrepa_repro"
REPORTS = ROOT / "reports/crystalrepa_repro"
LOGS = ROOT / "logs/crystalrepa_repro"
TOOLS = ROOT / "tools/crystalrepa_repro"
PROGRESS = RESULTS / "progress/master_progress.json"
EVENTS = RESULTS / "progress/events.jsonl"
TZ = ZoneInfo("Asia/Shanghai")

STAGES = [
    "state_audit",
    "paper_config_verification",
    "branch_creation",
    "checkpoint_audit",
    "cache_reuse_audit",
    "repro_implementation",
    "tests",
    "training_smoke",
    "training_decision",
    "eight_seed_smoke",
    "sixty_four_generation",
    "sixty_four_relax",
    "metrics",
    "paired_statistics",
    "repro_go_no_go",
    "github_archive",
    "stop_for_review",
]


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_progress() -> dict[str, Any]:
    existing = read_json(PROGRESS)
    if existing is not None:
        return existing
    timestamp = now()
    state = {
        "schema_version": 1,
        "created_at": timestamp,
        "updated_at": timestamp,
        "base_branch": "main",
        "base_commit": "9bc6747a3ddfd26db6d931bcdb6df5d299844544",
        "branch": "feature/crystalrepa-repro",
        "tmux_session": "mattergen_crystalrepa_repro",
        "current_stage": "state_audit",
        "overall_status": "running",
        "conditional_fn_pra_started": False,
        "formal_seeds_started": False,
        "stages": {
            stage: {"status": "pending", "detail": "", "metrics": {}}
            for stage in STAGES
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
    timestamp = now()
    state["updated_at"] = timestamp
    state["current_stage"] = stage
    state["overall_status"] = (
        status if status in {"blocked", "failed", "stop_for_review"} else "running"
    )
    state["stages"][stage] = {
        "status": status,
        "detail": detail,
        "metrics": metrics or {},
        "updated_at": timestamp,
    }
    atomic_json(PROGRESS, state)
    event = {
        "time": timestamp,
        "stage": stage,
        "status": status,
        "detail": detail,
        "metrics": metrics or {},
    }
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
