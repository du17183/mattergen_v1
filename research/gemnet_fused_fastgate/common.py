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
RESULTS = ROOT / "results/gemnet_fused_fastgate"
REPORTS = ROOT / "reports/gemnet_fused_fastgate"
LOGS = ROOT / "logs/gemnet_fused_fastgate"
TOOLS = ROOT / "tools/gemnet_fused_fastgate"
PROGRESS_DIR = RESULTS / "progress"
PROGRESS = PROGRESS_DIR / "master_progress.json"
EVENTS = PROGRESS_DIR / "events.jsonl"
CHECKPOINT_ROOT = ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
CHECKPOINT = CHECKPOINT_ROOT / "checkpoints/last.ckpt"
CHECKPOINT_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
PYTHON = ROOT / "envs/mattergen_py310/bin/python"
TZ = ZoneInfo("Asia/Shanghai")

STAGES = (
    "state_audit",
    "hotspot_profile",
    "fusion_candidate_selection",
    "fusion_implementation",
    "numerical_validation",
    "chain_microbenchmark",
    "forward_microbenchmark",
    "fusion_go_no_go",
    "eight_seed_generation",
    "eight_seed_quality",
    "runtime_fallback",
    "final_decision",
    "github_archive",
    "stop_for_review",
)


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initial_progress() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": now(),
        "updated_at": now(),
        "base_branch": "main",
        "base_commit": "9bc6747a3ddfd26db6d931bcdb6df5d299844544",
        "branch": "feature/gemnet-fused-inference-fastgate",
        "tmux_session": "mattergen_gemnet_fused_fastgate",
        "current_stage": "state_audit",
        "overall_status": "running",
        "termination_state": None,
        "other_processes_terminated": False,
        "sigkill_used": False,
        "stages": {
            stage: {"status": "pending", "detail": "", "metrics": {}}
            for stage in STAGES
        },
    }


def initialize() -> dict[str, Any]:
    for path in (RESULTS, REPORTS, LOGS, TOOLS, PROGRESS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    with file_lock(PROGRESS_DIR / ".master_progress.lock"):
        state = read_json(PROGRESS)
        if state is None:
            state = initial_progress()
            atomic_json(PROGRESS, state)
    return state


def set_stage(
    stage: str,
    status: str,
    detail: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    initialize()
    payload = metrics or {}
    with file_lock(PROGRESS_DIR / ".master_progress.lock"):
        state = read_json(PROGRESS)
        assert state is not None
        state["updated_at"] = now()
        state["current_stage"] = stage
        state["stages"][stage] = {
            "status": status,
            "detail": detail,
            "metrics": payload,
            "updated_at": now(),
        }
        state["overall_status"] = (
            status if status in {"failed", "blocked", "stop_for_review"} else "running"
        )
        atomic_json(PROGRESS, state)
    event = {
        "time": now(),
        "stage": stage,
        "status": status,
        "detail": detail,
        "metrics": payload,
    }
    with file_lock(PROGRESS_DIR / ".events.lock"):
        with EVENTS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def set_termination_state(state_name: str) -> None:
    allowed = {
        "FUSED_KERNEL_GO",
        "FUSED_KERNEL_NO_GO_RUNTIME_PASS",
        "GPU_ACCELERATION_NO_GO",
        "HARD_BLOCKED",
    }
    if state_name not in allowed:
        raise ValueError(f"invalid termination state: {state_name}")
    initialize()
    with file_lock(PROGRESS_DIR / ".master_progress.lock"):
        state = read_json(PROGRESS)
        assert state is not None
        state["termination_state"] = state_name
        state["updated_at"] = now()
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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
