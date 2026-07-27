from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo


ROOT = Path("/data/dxl")
PROJECT = ROOT / "mattergen_v1"
RESULTS = ROOT / "results/mps_fastgate"
REPORTS = ROOT / "reports/mps_fastgate"
LOGS = ROOT / "logs/mps_fastgate"
TOOLS = ROOT / "tools/mps_fastgate"
PROGRESS = RESULTS / "progress/master_progress.json"
EVENTS = RESULTS / "progress/events.jsonl"
MPS_ROOT = RESULTS / "mps_runtime"
MPS_PIPE = MPS_ROOT / "pipe"
MPS_LOG = MPS_ROOT / "log"
TZ = ZoneInfo("Asia/Shanghai")


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


@contextmanager
def locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initialize_progress() -> dict[str, Any]:
    for path in (RESULTS, REPORTS, LOGS, TOOLS, PROGRESS.parent):
        path.mkdir(parents=True, exist_ok=True)
    with locked(PROGRESS.parent / ".lock"):
        if PROGRESS.is_file():
            return json.loads(PROGRESS.read_text(encoding="utf-8"))
        state = {
            "created_at": now(),
            "updated_at": now(),
            "base_branch": "main",
            "base_commit": "9bc6747a3ddfd26db6d931bcdb6df5d299844544",
            "branch": "feature/mps-runtime-fastgate",
            "current_stage": "state_audit",
            "overall_status": "running",
            "final_state": None,
            "other_processes_terminated": False,
            "sigkill_used": False,
            "stages": {},
        }
        atomic_json(PROGRESS, state)
        return state


def set_stage(stage: str, status: str, detail: str, metrics: dict[str, Any] | None = None) -> None:
    initialize_progress()
    event = {
        "time": now(),
        "stage": stage,
        "status": status,
        "detail": detail,
        "metrics": metrics or {},
    }
    with locked(PROGRESS.parent / ".lock"):
        state = json.loads(PROGRESS.read_text(encoding="utf-8"))
        state["updated_at"] = now()
        state["current_stage"] = stage
        state["stages"][stage] = event
        state["overall_status"] = status if status in {"blocked", "stop_for_review"} else "running"
        atomic_json(PROGRESS, state)
    with locked(PROGRESS.parent / ".events.lock"):
        with EVENTS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def set_final_state(final_state: str) -> None:
    if final_state not in {"MPS_PAPER_GO", "MPS_ENGINEERING_ONLY", "MPS_NO_GO"}:
        raise ValueError(final_state)
    initialize_progress()
    with locked(PROGRESS.parent / ".lock"):
        state = json.loads(PROGRESS.read_text(encoding="utf-8"))
        state["updated_at"] = now()
        state["final_state"] = final_state
        atomic_json(PROGRESS, state)


def configure_environment() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
