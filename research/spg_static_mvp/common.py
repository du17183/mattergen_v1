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
RESULTS = ROOT / "results/spg_static_mvp"
REPORTS = ROOT / "reports/spg_static_mvp"
LOGS = ROOT / "logs/spg_static_mvp"
TOOLS = ROOT / "tools/spg_static_mvp"
PROGRESS_DIR = RESULTS / "progress"
PROGRESS = PROGRESS_DIR / "master_progress.json"
EVENTS = PROGRESS_DIR / "events.jsonl"
STOP_MARKER = PROGRESS_DIR / "STOP_REQUESTED"
CHECKPOINT_ROOT = ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
CHECKPOINT = CHECKPOINT_ROOT / "checkpoints/last.ckpt"
CHECKPOINT_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
MATTERSIM = ROOT / "mattersim_weights/mattersim-v1.0.0-5M.pth"
MATTERSIM_SHA256 = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
PYTHON = ROOT / "envs/mattergen_py310/bin/python"
TZ = ZoneInfo("Asia/Shanghai")

STAGES = (
    "state_audit",
    "branch_creation",
    "shape_distribution",
    "semantic_code_map",
    "static_builder_design",
    "static_builder_implementation",
    "unit_tests",
    "state_equivalence_10000",
    "gemnet_numerical_equivalence",
    "builder_microbenchmark",
    "forward_microbenchmark",
    "mvp_gate",
    "eight_seed_generation",
    "eight_seed_quality",
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
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


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
        "branch": "feature/spg-static-periodic-graph-mvp",
        "tmux_session": "mattergen_spg_static_mvp",
        "current_stage": "state_audit",
        "overall_status": "running",
        "termination_state": None,
        "compile_started": False,
        "cuda_graph_started": False,
        "a0_started": False,
        "sixty_four_seed_started": False,
        "formal_256_started": False,
        "other_processes_terminated": False,
        "sigkill_used": False,
        "stages": {
            name: {"status": "pending", "detail": "", "metrics": {}} for name in STAGES
        },
    }


def initialize_progress() -> dict[str, Any]:
    with file_lock(PROGRESS_DIR / ".master_progress.lock"):
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
    event = {
        "time": now(),
        "stage": stage,
        "status": status,
        "detail": detail,
        "metrics": metrics or {},
    }
    with file_lock(PROGRESS_DIR / ".events.lock"):
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def set_stage(
    stage: str,
    status: str,
    detail: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown SPG static MVP stage: {stage}")
    initialize_progress()
    with file_lock(PROGRESS_DIR / ".master_progress.lock"):
        state = read_json(PROGRESS)
        assert state is not None
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


def set_termination_state(state_name: str) -> None:
    if state_name not in {
        "MVP_PASS",
        "MVP_NO_GO",
        "SINGLE_BUCKET_STRONG_PASS",
        "SINGLE_BUCKET_TECHNICAL_PASS",
        "SINGLE_BUCKET_NO_GO",
        "HARD_BLOCKED",
    }:
        raise ValueError(f"invalid termination state: {state_name}")
    initialize_progress()
    with file_lock(PROGRESS_DIR / ".master_progress.lock"):
        state = read_json(PROGRESS)
        assert state is not None
        state["termination_state"] = state_name
        state["updated_at"] = now()
        atomic_json(PROGRESS, state)


def verify_artifacts() -> dict[str, str]:
    checkpoint_hash = sha256_file(CHECKPOINT)
    mattersim_hash = sha256_file(MATTERSIM)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise RuntimeError("MatterGen checkpoint SHA256 mismatch")
    if mattersim_hash != MATTERSIM_SHA256:
        raise RuntimeError("MatterSim checkpoint SHA256 mismatch")
    return {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "mattersim": str(MATTERSIM),
        "mattersim_sha256": mattersim_hash,
    }


def base_environment(gpu: int | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment
