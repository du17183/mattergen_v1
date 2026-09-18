"""Resumable, gated confirmatory pipeline for the frozen Linear-K2 allocator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/frozen_linear_k2_confirmation"
FIELD_PROJECT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1_field_cfg")
OLD_PROJECT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1")
PYTHON = Path("/mnt/datasets-livsyn/dxl/alm/.venv/bin/python")
STATUS = ROOT / "pipeline_status.json"
EXECUTION_MANIFEST = ROOT / "protocol/frozen_execution_manifest.json"
LOG_ROOT = ROOT / "logs"
IDLE_MEMORY_MIB_MAX = 64
POLL_SECONDS = 30


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def update_status(**changes: Any) -> None:
    payload: dict[str, Any] = {}
    if STATUS.exists():
        payload = json.loads(STATUS.read_text())
    payload.update(changes)
    payload["updated_utc"] = utc_now()
    atomic_json(STATUS, payload)


def verify_execution_freeze() -> dict[str, Any]:
    if not EXECUTION_MANIFEST.exists():
        raise FileNotFoundError("missing frozen execution manifest")
    manifest = json.loads(EXECUTION_MANIFEST.read_text())
    if manifest.get("status") != "FROZEN_BEFORE_CONFIRMATORY_GENERATION":
        raise RuntimeError("execution manifest is not frozen")
    for relative, expected in manifest["execution_files_sha256"].items():
        observed = sha256(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"frozen execution file changed: {relative}")
    for path, expected in manifest["external_source_sha256"].items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"frozen external source changed: {path}")
    return manifest


def gpu_snapshot() -> dict[int, dict[str, int | str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    snapshot: dict[int, dict[str, int | str]] = {}
    for line in output.splitlines():
        index, uuid, used, total, utilization = (value.strip() for value in line.split(","))
        snapshot[int(index)] = {
            "uuid": uuid,
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
            "utilization_percent": int(utilization),
        }
    return snapshot


def assert_idle(gpus: tuple[int, ...], stage: str) -> None:
    snapshot = gpu_snapshot()
    problems = []
    for gpu in gpus:
        values = snapshot.get(gpu)
        if values is None:
            problems.append(f"GPU {gpu} absent")
        elif int(values["memory_used_mib"]) > IDLE_MEMORY_MIB_MAX or int(values["utilization_percent"]) != 0:
            problems.append(f"GPU {gpu} is not idle: {values}")
    audit_path = ROOT / "confirmatory" / f"gpu_audit_{stage}_{int(time.time())}.json"
    atomic_json(audit_path, {"stage": stage, "time_utc": utc_now(), "requested_gpus": gpus, "snapshot": snapshot, "problems": problems})
    if problems:
        raise RuntimeError("; ".join(problems))


def base_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=f"{FIELD_PROJECT}:{PROJECT}:{OLD_PROJECT}",
        XDG_CACHE_HOME=str(OLD_PROJECT / ".cache"),
        HF_HOME=str(OLD_PROJECT / ".cache/huggingface"),
        TORCH_HOME=str(OLD_PROJECT / ".cache/torch"),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
    )
    return environment


def run_logged(label: str, command: list[str], *, environment: dict[str, str], cwd: Path) -> None:
    log_path = LOG_ROOT / f"{label}.log"
    update_status(stage=label, active_command=command, active_log=str(log_path))
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{utc_now()}] START {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"[{utc_now()}] EXIT {result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"stage failed ({label}); see {log_path}")


def registered_seeds(phase: str) -> tuple[int, ...]:
    manifest = json.loads((ROOT / "protocol/seed_manifest_256.json").read_text())
    return tuple(map(int, manifest[phase]))


def completed_seeds(phase: str) -> set[int]:
    generation = ROOT / phase / "generation"
    if not generation.exists():
        return set()
    return {
        int(path.parent.name)
        for path in generation.glob("*/run_summary.json")
        if json.loads(path.read_text()).get("success")
    }


def generation_workers(phase: str, seeds: tuple[int, ...], gpus: tuple[int, ...], label: str) -> None:
    if not seeds:
        return
    assert_idle(gpus, label)
    assignments = {
        gpu: tuple(seed for index, seed in enumerate(seeds) if index % len(gpus) == rank)
        for rank, gpu in enumerate(gpus)
    }
    workers: list[tuple[int, subprocess.Popen[Any], Any, Path]] = []
    for gpu, assigned in assignments.items():
        if not assigned:
            continue
        environment = base_environment()
        worker_tmp = ROOT / phase / "runtime_tmp" / f"gpu_{gpu}"
        worker_tmp.mkdir(parents=True, exist_ok=True)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            LINEAR_K2_PHYSICAL_GPU=str(gpu),
            TMPDIR=str(worker_tmp),
            PYTHONPYCACHEPREFIX=str(ROOT / ".pycache_runtime"),
        )
        log_path = LOG_ROOT / f"generation_{phase}_gpu_{gpu}.log"
        stream = log_path.open("a", encoding="utf-8")
        command = [str(PYTHON), str(ROOT / "run_generation.py"), "--phase", phase, "--seeds", ",".join(map(str, assigned))]
        stream.write(f"\n[{utc_now()}] START {' '.join(command)}\n"); stream.flush()
        workers.append((gpu, subprocess.Popen(command, cwd=FIELD_PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT), stream, log_path))
    total = len(registered_seeds(phase)); started = time.monotonic()
    while True:
        failures = [(gpu, process.returncode, log) for gpu, process, _, log in workers if process.poll() not in (None, 0)]
        complete = len(completed_seeds(phase))
        elapsed = time.monotonic() - started
        rate = complete / elapsed if elapsed > 0 else 0.0
        eta = (total - complete) / rate if rate > 0 else None
        update_status(stage=label, phase=phase, completed_seeds=complete, total_seeds=total, active_gpus=list(gpus), stage_elapsed_seconds=elapsed, estimated_remaining_seconds=eta)
        if failures:
            for _, process, _, _ in workers:
                if process.poll() is None:
                    process.terminate()
            raise RuntimeError(f"generation worker failure: {failures}")
        if all(process.poll() == 0 for _, process, _, _ in workers):
            break
        time.sleep(POLL_SECONDS)
    for _, process, stream, _ in workers:
        process.wait(); stream.write(f"[{utc_now()}] EXIT {process.returncode}\n"); stream.close()


def generate_phase(phase: str, gpus: tuple[int, ...]) -> None:
    seeds = registered_seeds(phase)
    phase_root = ROOT / phase
    (phase_root / "generation").mkdir(parents=True, exist_ok=True)
    current = completed_seeds(phase)
    if not current:
        generation_workers(phase, (seeds[0],), (gpus[0],), f"{phase}_generation_canary")
        summary = json.loads((phase_root / "generation" / str(seeds[0]) / "run_summary.json").read_text())
        if not summary["reference_reproduction"]["success"] or summary["mattergen_score_calls_acquisition"] != 11200:
            raise RuntimeError("formal canary determinism/compute audit failed")
    remaining = tuple(seed for seed in seeds if seed not in completed_seeds(phase))
    generation_workers(phase, remaining, gpus, f"{phase}_generation")
    missing = set(seeds) - completed_seeds(phase)
    if missing:
        raise RuntimeError(f"generation incomplete for {phase}: {sorted(missing)}")


def phase_pipeline(phase: str, gpus: tuple[int, ...]) -> dict[str, Any]:
    generate_phase(phase, gpus)
    phase_root = ROOT / phase
    if not (phase_root / "property_metrics_raw.csv").exists():
        assert_idle((gpus[0],), f"{phase}_property")
        environment = base_environment(); environment.update(CUDA_VISIBLE_DEVICES=str(gpus[0]), TMPDIR=str(phase_root / "runtime_tmp"))
        run_logged(f"{phase}_prepare_property", [str(PYTHON), str(ROOT / "prepare_phase.py"), "--phase", phase], environment=environment, cwd=PROJECT)
    assert_idle(gpus, f"{phase}_branch_quality")
    environment = base_environment(); environment.pop("CUDA_VISIBLE_DEVICES", None); environment["TMPDIR"] = str(phase_root / "runtime_tmp")
    run_logged(f"{phase}_quality_branches", [str(PYTHON), str(ROOT / "run_quality.py"), "--phase", phase, "--mode", "branches", "--gpus", ",".join(map(str, gpus))], environment=environment, cwd=PROJECT)
    if not (phase_root / "selected_outcomes.csv").exists():
        analysis_environment = base_environment(); analysis_environment.update(CUDA_VISIBLE_DEVICES="", TMPDIR=str(phase_root / "runtime_tmp"), MPLCONFIGDIR=str(ROOT / ".matplotlib"))
        run_logged(f"{phase}_selection", [str(PYTHON), str(ROOT / "analyze_phase.py"), "--phase", phase, "--stage", "select"], environment=analysis_environment, cwd=PROJECT)
    assert_idle(gpus, f"{phase}_mixed_quality")
    run_logged(f"{phase}_quality_mixed", [str(PYTHON), str(ROOT / "run_quality.py"), "--phase", phase, "--mode", "mixed", "--gpus", ",".join(map(str, gpus))], environment=environment, cwd=PROJECT)
    decision_path = phase_root / ("continuation_decision.json" if phase == "c1_128" else "decision_summary.json")
    if not decision_path.exists():
        analysis_environment = base_environment(); analysis_environment.update(CUDA_VISIBLE_DEVICES="", TMPDIR=str(phase_root / "runtime_tmp"), MPLCONFIGDIR=str(ROOT / ".matplotlib"))
        run_logged(f"{phase}_finalize", [str(PYTHON), str(ROOT / "analyze_phase.py"), "--phase", phase, "--stage", "finalize"], environment=analysis_environment, cwd=PROJECT)
    return json.loads(decision_path.read_text())


def pooled_pipeline(gpus: tuple[int, ...]) -> dict[str, Any]:
    pooled = ROOT / "pooled256"
    analysis_environment = base_environment(); analysis_environment.update(CUDA_VISIBLE_DEVICES="", TMPDIR=str(ROOT / "runtime_tmp"), MPLCONFIGDIR=str(ROOT / ".matplotlib"))
    if not (pooled / "selected_outcomes.csv").exists():
        run_logged("pooled256_assemble", [str(PYTHON), str(ROOT / "analyze_phase.py"), "--phase", "pooled256", "--stage", "pool"], environment=analysis_environment, cwd=PROJECT)
    (pooled / "runtime_tmp").mkdir(exist_ok=True)
    assert_idle(gpus, "pooled256_mixed_quality")
    quality_environment = base_environment(); quality_environment.pop("CUDA_VISIBLE_DEVICES", None); quality_environment["TMPDIR"] = str(pooled / "runtime_tmp")
    run_logged("pooled256_quality_mixed", [str(PYTHON), str(ROOT / "run_quality.py"), "--phase", "pooled256", "--mode", "mixed", "--gpus", ",".join(map(str, gpus))], environment=quality_environment, cwd=PROJECT)
    if not (pooled / "decision_summary.json").exists():
        run_logged("pooled256_finalize", [str(PYTHON), str(ROOT / "analyze_phase.py"), "--phase", "pooled256", "--stage", "finalize"], environment=analysis_environment, cwd=PROJECT)
    return json.loads((pooled / "decision_summary.json").read_text())


def main(gpus: tuple[int, ...]) -> None:
    LOG_ROOT.mkdir(exist_ok=True); (ROOT / "confirmatory").mkdir(exist_ok=True); (ROOT / "runtime_tmp").mkdir(exist_ok=True)
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU list must be non-empty and unique")
    execution = verify_execution_freeze()
    update_status(
        study="Frozen Allocator Confirmatory Study",
        status="RUNNING",
        started_utc=utc_now(),
        execution_manifest_sha256=sha256(EXECUTION_MANIFEST),
        execution_git_base=execution["git_base_head"],
        requested_gpus=list(gpus),
        historical_phase_b_decision="ADAPTIVE_ALLOCATION_VALUE=NOT_SUPPORTED",
        historical_phase_b_decision_immutable=True,
        repeated_c0="NOT_APPLICABLE",
        dft_verified=False,
    )
    try:
        c1 = phase_pipeline("c1_128", gpus)
        if c1["C1_CONFIRMATION"] != "CONTINUE":
            update_status(status="PROTOCOL_COMPLETE_C1_FAIL", stage="STOPPED_BEFORE_C2", c1_decision=c1, c2_started=False, completed_utc=utc_now())
            return
        c2 = phase_pipeline("c2_128", gpus)
        pooled = pooled_pipeline(gpus)
        update_status(status="COMPLETE", stage="FINAL_REPORT_COMPLETE", c1_decision=c1, c2_decision=c2, pooled_decision=pooled, completed_utc=utc_now())
    except BaseException as error:
        update_status(status="FAILED", error_type=type(error).__name__, error=str(error), failed_utc=utc_now())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--gpus", required=True); arguments = parser.parse_args()
    main(tuple(int(value) for value in arguments.gpus.split(",") if value))
