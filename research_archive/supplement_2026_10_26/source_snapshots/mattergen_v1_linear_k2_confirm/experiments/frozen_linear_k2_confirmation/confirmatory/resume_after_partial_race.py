"""Operational recovery from a monitor-only `.partial` directory race.

This wrapper does not alter any frozen experimental file.  It invokes the
hash-verified frozen generator for the still-unfinished registered C1 seeds,
then hands control back to the frozen pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


PROJECT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1_linear_k2_confirm")
ROOT = PROJECT / "experiments/frozen_linear_k2_confirmation"
FIELD_PROJECT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1_field_cfg")
OLD_PROJECT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1")
PYTHON = Path("/mnt/datasets-livsyn/dxl/alm/.venv/bin/python")
GENERATOR = ROOT / "run_generation.py"
PIPELINE = ROOT / "run_pipeline.py"
EXECUTION_MANIFEST = ROOT / "protocol/frozen_execution_manifest.json"
SEED_MANIFEST = ROOT / "protocol/seed_manifest_256.json"
STATUS = ROOT / "pipeline_status.json"
RECOVERY_STATUS = ROOT / "confirmatory/recovery_status.json"
RECOVERY_AUDIT = ROOT / "confirmatory/recovery_audit.json"
PHASE_ROOT = ROOT / "c1_128"
GPUS = (2, 4)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def verify_frozen_files() -> None:
    manifest = json.loads(EXECUTION_MANIFEST.read_text())
    for relative, expected in manifest["execution_files_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"frozen execution file changed: {relative}")
    for path, expected in manifest["external_source_sha256"].items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"frozen external source changed: {path}")


def completed() -> set[int]:
    result: set[int] = set()
    generation = PHASE_ROOT / "generation"
    for child in generation.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        summary = child / "run_summary.json"
        if summary.exists() and json.loads(summary.read_text()).get("success") is True:
            result.add(int(child.name))
    return result


def assert_idle() -> dict[int, dict[str, int]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    snapshot: dict[int, dict[str, int]] = {}
    for line in output.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        snapshot[index] = {"memory_used_mib": memory, "utilization_percent": utilization}
    for gpu in GPUS:
        values = snapshot[gpu]
        if values["memory_used_mib"] > 64 or values["utilization_percent"] != 0:
            raise RuntimeError(f"GPU {gpu} is no longer idle: {values}")
    return snapshot


def environment(gpu: int | None = None) -> dict[str, str]:
    result = dict(os.environ)
    result.update(
        PYTHONPATH=f"{FIELD_PROJECT}:{PROJECT}:{OLD_PROJECT}",
        PYTHONPYCACHEPREFIX=str(ROOT / ".pycache_runtime"),
        XDG_CACHE_HOME=str(OLD_PROJECT / ".cache"),
        HF_HOME=str(OLD_PROJECT / ".cache/huggingface"),
        TORCH_HOME=str(OLD_PROJECT / ".cache/torch"),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
    )
    if gpu is not None:
        temporary = PHASE_ROOT / "runtime_tmp" / f"gpu_{gpu}"
        temporary.mkdir(parents=True, exist_ok=True)
        result.update(CUDA_VISIBLE_DEVICES=str(gpu), LINEAR_K2_PHYSICAL_GPU=str(gpu), TMPDIR=str(temporary))
    else:
        result["TMPDIR"] = str(ROOT / "runtime_tmp")
        result.pop("CUDA_VISIBLE_DEVICES", None)
    return result


def main() -> None:
    verify_frozen_files()
    partials = list((PHASE_ROOT / "generation").glob("*.partial"))
    if partials:
        raise RuntimeError(f"unarchived partial directories remain: {partials}")
    registered = tuple(map(int, json.loads(SEED_MANIFEST.read_text())["c1_128"]))
    done = completed()
    if registered[0] not in done:
        raise RuntimeError("the formal canary is not complete")
    original_post_canary = registered[1:]
    assignments = {
        gpu: tuple(seed for index, seed in enumerate(original_post_canary) if index % 2 == rank and seed not in done)
        for rank, gpu in enumerate(GPUS)
    }
    if set(assignments[2]) & set(assignments[4]):
        raise RuntimeError("recovery assignments overlap")
    if set(assignments[2]) | set(assignments[4]) != set(registered) - done:
        raise RuntimeError("recovery assignments do not cover exactly the unfinished seeds")
    snapshot = assert_idle()
    old_status = json.loads(STATUS.read_text())
    audit: dict[str, Any] = {
        "schema_version": 1,
        "reason": "monitor parsed a transient numeric_seed.partial directory while run_generation atomically committed a complete seed",
        "experimental_failure": False,
        "protocol_or_method_change": False,
        "frozen_files_verified": True,
        "recovery_wrapper_sha256": sha256(Path(__file__)),
        "detected_complete_seed_count": len(done),
        "unfinished_seed_count": sum(map(len, assignments.values())),
        "assignments": {str(gpu): values for gpu, values in assignments.items()},
        "archived_empty_partials": ["1034872534.partial", "1154755706.partial"],
        "gpu_snapshot_before_resume": snapshot,
        "started_utc": now(),
        "status": "RUNNING",
    }
    atomic_json(RECOVERY_AUDIT, audit)
    recovered_failure = {
        "error_type": old_status.get("error_type"),
        "error": old_status.get("error"),
        "failed_utc": old_status.get("failed_utc"),
    }
    old_status.update(
        status="RECOVERING_MONITOR_RACE",
        initial_started_utc=old_status.get("started_utc"),
        recovered_failure=recovered_failure,
        recovery_audit=str(RECOVERY_AUDIT),
        completed_seeds=len(done),
        updated_utc=now(),
    )
    for key in ("error_type", "error", "failed_utc"):
        old_status.pop(key, None)
    atomic_json(STATUS, old_status)

    workers: list[tuple[int, subprocess.Popen[Any], Any]] = []
    for gpu in GPUS:
        seeds = assignments[gpu]
        if not seeds:
            continue
        log = (ROOT / "logs" / f"recovery_generation_c1_gpu_{gpu}.log").open("a", encoding="utf-8")
        command = [str(PYTHON), str(GENERATOR), "--phase", "c1_128", "--seeds", ",".join(map(str, seeds))]
        workers.append((gpu, subprocess.Popen(command, cwd=FIELD_PROJECT, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT), log))
    while True:
        states = {gpu: process.poll() for gpu, process, _ in workers}
        current = len(completed())
        atomic_json(RECOVERY_STATUS, {"status": "RUNNING", "updated_utc": now(), "completed_seeds": current, "total_seeds": 128, "worker_returncodes": states})
        failure = {gpu: state for gpu, state in states.items() if state not in (None, 0)}
        if failure:
            for _, process, _ in workers:
                if process.poll() is None:
                    process.terminate()
            raise RuntimeError(f"recovery generation worker failed: {failure}")
        if all(state == 0 for state in states.values()):
            break
        time.sleep(30)
    for _, process, log in workers:
        process.wait()
        log.close()
    if completed() != set(registered):
        raise RuntimeError("recovery generation did not complete all 128 registered C1 seeds")
    audit.update(status="GENERATION_COMPLETE", completed_utc=now(), final_complete_seed_count=128)
    atomic_json(RECOVERY_AUDIT, audit)
    atomic_json(RECOVERY_STATUS, {"status": "HANDING_BACK_TO_FROZEN_PIPELINE", "updated_utc": now(), "completed_seeds": 128, "total_seeds": 128})
    with (ROOT / "logs/pipeline_driver_recovery.log").open("a", encoding="utf-8") as log:
        result = subprocess.run([str(PYTHON), str(PIPELINE), "--gpus", "2,4"], cwd=FIELD_PROJECT, env=environment(), stdout=log, stderr=subprocess.STDOUT)
    audit.update(status="FROZEN_PIPELINE_EXITED", pipeline_returncode=result.returncode, pipeline_exit_utc=now())
    atomic_json(RECOVERY_AUDIT, audit)
    if result.returncode:
        raise RuntimeError(f"frozen pipeline returned {result.returncode}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        atomic_json(RECOVERY_STATUS, {"status": "FAILED", "updated_utc": now(), "error_type": type(error).__name__, "error": str(error)})
        raise
