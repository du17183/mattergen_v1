from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo


ROOT = Path("/data/dxl")
PROJECT = ROOT / "mattergen_v1"
RESULTS = ROOT / "results/rp_qtfg/phase0"
REPORTS = ROOT / "reports/rp_qtfg/phase0"
LOGS = ROOT / "logs/rp_qtfg/phase0"
TOOLS = ROOT / "tools/rp_qtfg"
PROGRESS_DIR = RESULTS / "progress"
PROGRESS = PROGRESS_DIR / "master_progress.json"
EVENTS = PROGRESS_DIR / "events.jsonl"
LOCK = PROGRESS_DIR / "phase0.lock"
STOP_MARKER = PROGRESS_DIR / "STOP_REQUESTED"
TZ = ZoneInfo("Asia/Shanghai")

STAGES = (
    "state_audit",
    "branch_creation",
    "literature_verification",
    "mattergen_code_map",
    "mag_oracle_validation",
    "offline_direction_probe",
    "direction_go_no_go",
    "implementation",
    "tests",
    "eight_seed_generation",
    "eight_seed_review",
    "thirty_two_generation",
    "thirty_two_relax",
    "metrics",
    "mvp_go_no_go",
    "github_archive",
    "stop_for_review",
)


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
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
    )


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def exclusive_lock(path: Path = LOCK) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def initial_progress() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": now(),
        "updated_at": now(),
        "base_branch": "main",
        "base_commit": "9bc6747a3ddfd26db6d931bcdb6df5d299844544",
        "branch": "feature/rp-qtfg",
        "tmux_session": "mattergen_rp_qtfg_phase0",
        "current_stage": "state_audit",
        "overall_status": "running",
        "eight_seed_started": False,
        "thirty_two_seed_started": False,
        "sixty_four_seed_started": False,
        "formal_seed_started": False,
        "other_processes_terminated": False,
        "sigkill_used": False,
        "stages": {
            stage: {"status": "pending", "detail": "", "metrics": {}}
            for stage in STAGES
        },
    }


def initialize_progress() -> dict[str, Any]:
    with exclusive_lock():
        state = read_json(PROGRESS)
        if state is None:
            state = initial_progress()
            atomic_json(PROGRESS, state)
        return state


def append_event(
    stage: str,
    status: str,
    detail: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": now(),
        "stage": stage,
        "status": status,
        "detail": detail,
        "metrics": metrics or {},
    }
    with EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        )


def set_stage(
    stage: str,
    status: str,
    detail: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown RP-QTFG stage: {stage}")
    with exclusive_lock():
        state = read_json(PROGRESS, initial_progress())
        state.pop("eighty_seed_started", None)
        state.setdefault("eight_seed_started", False)
        state["updated_at"] = now()
        state["current_stage"] = stage
        state["stages"][stage] = {
            "status": status,
            "detail": detail,
            "metrics": metrics or {},
            "updated_at": now(),
        }
        if status in {"failed", "blocked", "stop_for_review"}:
            state["overall_status"] = status
        else:
            state["overall_status"] = "running"
        atomic_json(PROGRESS, state)
        append_event(stage, status, detail, metrics)


def update_progress(**fields: Any) -> None:
    with exclusive_lock():
        state = read_json(PROGRESS, initial_progress())
        state.pop("eighty_seed_started", None)
        state.setdefault("eight_seed_started", False)
        state.update(fields)
        state["updated_at"] = now()
        atomic_json(PROGRESS, state)


def stop_requested() -> bool:
    return STOP_MARKER.exists()
