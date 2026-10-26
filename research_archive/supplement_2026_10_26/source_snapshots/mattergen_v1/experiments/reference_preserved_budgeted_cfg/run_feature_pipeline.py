"""Resumable gate-enforcing supervisor for the complete Phase-B study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
FIELD_PROJECT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1_field_cfg")
ROOT = PROJECT / "experiments/reference_preserved_budgeted_cfg"
FEATURE_ROOT = ROOT / "feature_study"
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"
FROZEN_FILES = (
    "feature_study_protocol.yaml",
    "register_feature_study_seeds.py",
    "phase_b_sampler.py",
    "run_feature_generation.py",
    "prepare_feature_study.py",
    "run_feature_quality.py",
    "analyze_feature_study.py",
    "run_feature_pipeline.py",
    "feature_study/feature_study_seed_manifest.json",
    "feature_study/feature_study_seed_audit.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen() -> dict[str, Any]:
    manifest = json.loads((FEATURE_ROOT / "frozen_manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise RuntimeError(f"frozen Phase-B file changed: {relative}")
    return manifest


def update_status(**updates: Any) -> None:
    path = ROOT / "pipeline_status.json"
    values = json.loads(path.read_text())
    values.update(updates)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    temporary.replace(path)


def audit_selected_gpus(gpus: tuple[int, ...]) -> None:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    snapshot = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        index = int(fields[0])
        snapshot[index] = {
            "index": index, "name": fields[1], "memory_total_mib": int(fields[2]),
            "memory_used_mib": int(fields[3]), "utilization_percent": int(fields[4]),
        }
    missing = sorted(set(gpus) - set(snapshot))
    busy = [snapshot[gpu] for gpu in gpus if snapshot[gpu]["memory_used_mib"] > 1024 or snapshot[gpu]["utilization_percent"] > 10]
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name", "--format=csv,noheader"],
        check=False, capture_output=True, text=True,
    ).stdout.splitlines()
    audit = {
        "selected_gpus": list(gpus), "idle_rule": "memory_used_mib <= 1024 and utilization_percent <= 10",
        "snapshot": [snapshot[index] for index in sorted(snapshot)], "compute_processes": processes,
        "selected_missing": missing, "selected_busy": busy, "launch_allowed": not missing and not busy,
    }
    (FEATURE_ROOT / "launch_gpu_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if missing or busy:
        raise RuntimeError(f"selected GPU audit failed: missing={missing}, busy={busy}")


def environment(gpu: int | None = None) -> dict[str, str]:
    result = dict(os.environ)
    result.update(
        PYTHONPATH=f"{FIELD_PROJECT}:{PROJECT}", PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        MPLCONFIGDIR=str(ROOT / ".matplotlib"), XDG_CACHE_HOME=str(PROJECT / ".cache"),
        HF_HOME=str(PROJECT / ".cache/huggingface"), TORCH_HOME=str(PROJECT / ".cache/torch"),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    )
    if gpu is not None:
        result.update(CUDA_VISIBLE_DEVICES=str(gpu), RPB_CFG_PHYSICAL_GPU=str(gpu))
    return result


def next_log(name: str) -> Path:
    base = ROOT / "logs" / name
    if not base.exists():
        return base
    ordinal = 2
    while base.with_name(f"{base.stem}_attempt{ordinal}{base.suffix}").exists():
        ordinal += 1
    return base.with_name(f"{base.stem}_attempt{ordinal}{base.suffix}")


def run_logged(command: list[str], name: str, gpu: int | None = None) -> None:
    path = next_log(name)
    with path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=FIELD_PROJECT, env=environment(gpu), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed {result.returncode}: {path}")


def generation(gpus: tuple[int, ...]) -> None:
    seed_manifest = json.loads((FEATURE_ROOT / "feature_study_seed_manifest.json").read_text())
    seeds = list(map(int, seed_manifest["seeds"]))
    pending = [seed for seed in seeds if not (FEATURE_ROOT / "generation" / str(seed) / "run_summary.json").exists()]
    partitions = {gpu: pending[index::len(gpus)] for index, gpu in enumerate(gpus)}
    processes = []
    for gpu, assigned in partitions.items():
        if not assigned:
            continue
        log = next_log(f"generation_feature_study_gpu_{gpu}.log")
        stream = log.open("x", encoding="utf-8")
        command = [PYTHON, str(ROOT / "run_feature_generation.py"), "--seeds", ",".join(map(str, assigned))]
        process = subprocess.Popen(command, cwd=FIELD_PROJECT, env=environment(gpu), stdout=stream, stderr=subprocess.STDOUT)
        processes.append((gpu, process, stream, log))
    while processes:
        running = []
        failure = None
        for gpu, process, stream, log in processes:
            code = process.poll()
            if code is None:
                running.append((gpu, process, stream, log))
            else:
                stream.close()
                if code:
                    failure = (gpu, code, str(log))
                    break
        if failure:
            for _, process, stream, _ in running:
                if process.poll() is None:
                    process.terminate()
                stream.close()
            raise RuntimeError(f"generation worker failure: {failure}")
        processes = running
        complete = sum((FEATURE_ROOT / "generation" / str(seed) / "run_summary.json").exists() for seed in seeds)
        update_status(stage="phase_b_generation", active_phase="feature_study", phase_complete=complete, phase_total=len(seeds))
        print(json.dumps({"stage": "phase_b_generation", "complete": complete, "total": len(seeds)}), flush=True)
        if processes:
            time.sleep(30)
    complete = sum((FEATURE_ROOT / "generation" / str(seed) / "run_summary.json").exists() for seed in seeds)
    if complete != len(seeds):
        raise RuntimeError(f"Phase-B generation incomplete: {complete}/{len(seeds)}")


def main(gpus: tuple[int, ...]) -> None:
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("invalid GPU list")
    assert_frozen()
    audit_selected_gpus(gpus)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / ".matplotlib").mkdir(exist_ok=True)
    update_status(stage="phase_b_generation", active_phase="feature_study", PREBRANCH_FEATURE_DATASET="RUNNING", ALLOCATOR_TEST="NOT_RUN", ADAPTIVE_ALLOCATION_VALUE="NOT_EVALUABLE", STOP_REASON=None)
    generation(gpus)
    if not (FEATURE_ROOT / "property_summary.json").exists():
        update_status(stage="phase_b_property_evaluation")
        run_logged([PYTHON, str(ROOT / "prepare_feature_study.py")], "prepare_feature_study.log", gpus[0])
    if not all((FEATURE_ROOT / "quality_policies" / policy / "official_detailed.json.gz").exists() for policy in ("C0", "GPulse", "APulse", "PPulse", "CPulse")):
        update_status(stage="phase_b_policy_quality_evaluation")
        run_logged([PYTHON, str(ROOT / "run_feature_quality.py"), "--mode", "policies", "--gpus", ",".join(map(str, gpus))], "quality_feature_policies_driver.log")
    if not (FEATURE_ROOT / "allocator_selection.json").exists():
        update_status(stage="phase_b_train_validation_and_locked_test")
        run_logged([PYTHON, str(ROOT / "analyze_feature_study.py"), "--stage", "select"], "analyze_feature_select.log")
    mixed_groups = [path.stem for path in (FEATURE_ROOT / "mixed_evaluation_structures").glob("*.extxyz")]
    if not mixed_groups or not all((FEATURE_ROOT / "quality_mixed" / group / "official_detailed.json.gz").exists() for group in mixed_groups):
        update_status(stage="phase_b_mixed_batch_quality_evaluation")
        run_logged([PYTHON, str(ROOT / "run_feature_quality.py"), "--mode", "mixed", "--gpus", ",".join(map(str, gpus))], "quality_feature_mixed_driver.log")
    if not (FEATURE_ROOT / "phase_b_decision_summary.json").exists():
        update_status(stage="phase_b_statistics_bootstrap_plots_report")
        run_logged([PYTHON, str(ROOT / "analyze_feature_study.py"), "--stage", "finalize"], "analyze_feature_finalize.log")
    decision = json.loads((FEATURE_ROOT / "phase_b_decision_summary.json").read_text())
    update_status(**decision, active_phase=None, stage="phase_b_complete_supported" if decision["ADAPTIVE_ALLOCATION_VALUE"] == "SUPPORTED" else "stopped_after_phase_b_not_supported")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    arguments = parser.parse_args()
    main(tuple(int(value) for value in arguments.gpus.split(",") if value))
