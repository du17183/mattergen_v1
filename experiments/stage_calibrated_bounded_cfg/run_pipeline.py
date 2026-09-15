"""Gate-enforcing supervisor from calibration generation through final report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/stage_calibrated_bounded_cfg"
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen() -> None:
    manifest = json.loads((ROOT / "frozen_manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        if sha256(PROJECT / relative) != expected:
            raise RuntimeError(f"frozen file changed: {relative}")


def update_status(**updates) -> None:
    path = ROOT / "pipeline_status.json"
    values = json.loads(path.read_text())
    values.update(updates)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    os.replace(temporary, path)


def audit_selected_gpus(gpus: tuple[int, ...]) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    snapshot: dict[int, dict[str, object]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        index = int(fields[0])
        snapshot[index] = {
            "index": index,
            "name": fields[1],
            "memory_total_mib": int(fields[2]),
            "memory_used_mib": int(fields[3]),
            "utilization_percent": int(fields[4]),
        }
    missing = sorted(set(gpus) - set(snapshot))
    busy = [
        snapshot[gpu]
        for gpu in gpus
        if snapshot[gpu]["memory_used_mib"] > 1024
        or snapshot[gpu]["utilization_percent"] > 10
    ]
    audit = {
        "selected_gpus": list(gpus),
        "idle_rule": "memory_used_mib <= 1024 and utilization_percent <= 10",
        "snapshot": [snapshot[index] for index in sorted(snapshot)],
        "selected_missing": missing,
        "selected_busy": busy,
        "launch_allowed": not missing and not busy,
    }
    (ROOT / "launch_gpu_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if missing or busy:
        raise RuntimeError(f"selected GPU audit failed: missing={missing}, busy={busy}")


def environment(gpu: int | None = None) -> dict[str, str]:
    result = dict(os.environ)
    result.update(
        PYTHONPATH=str(PROJECT),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MPLCONFIGDIR=str(ROOT / ".matplotlib"),
    )
    if gpu is not None:
        result.update(CUDA_VISIBLE_DEVICES=str(gpu), STAGE_CFG_PHYSICAL_GPU=str(gpu))
    return result


def run_logged(command: list[str], log_name: str, gpu: int | None = None) -> None:
    log_path = ROOT / "logs" / log_name
    if log_path.exists():
        raise FileExistsError(f"refusing to replace retained log: {log_path}")
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=PROJECT, env=environment(gpu), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def generation(phase: str, gpus: tuple[int, ...]) -> None:
    phase_root = ROOT / phase
    seeds = list(map(int, json.loads((phase_root / "seeds.json").read_text())["seeds"]))
    pending = [seed for seed in seeds if not (phase_root / "generation" / str(seed) / "run_summary.json").exists()]
    partitions = {gpu: pending[index::len(gpus)] for index, gpu in enumerate(gpus)}
    processes = []
    for gpu, assigned in partitions.items():
        if not assigned:
            continue
        log_path = ROOT / "logs" / f"generation_{phase}_gpu_{gpu}.log"
        if log_path.exists():
            raise FileExistsError(f"refusing to replace retained log: {log_path}")
        stream = log_path.open("x", encoding="utf-8")
        command = [PYTHON, str(ROOT / "run_generation.py"), "--phase", phase, "--seeds", ",".join(map(str, assigned))]
        process = subprocess.Popen(command, cwd=PROJECT, env=environment(gpu), stdout=stream, stderr=subprocess.STDOUT)
        processes.append((gpu, process, stream, log_path))
    while processes:
        still_running = []
        failed = None
        for gpu, process, stream, log_path in processes:
            code = process.poll()
            if code is None:
                still_running.append((gpu, process, stream, log_path))
            else:
                stream.close()
                if code:
                    failed = (gpu, code, log_path)
                    break
        if failed:
            for _, process, stream, _ in still_running:
                if process.poll() is None:
                    process.terminate()
                stream.close()
            raise RuntimeError(f"generation worker failed: gpu={failed[0]} code={failed[1]} log={failed[2]}")
        processes = still_running
        completed = sum((phase_root / "generation" / str(seed) / "run_summary.json").exists() for seed in seeds)
        update_status(stage=f"{phase}_generation", active_phase=phase, phase_complete=completed, phase_total=len(seeds))
        print(json.dumps({"stage": f"{phase}_generation", "complete": completed, "total": len(seeds)}), flush=True)
        if processes:
            time.sleep(30)
    completed = sum((phase_root / "generation" / str(seed) / "run_summary.json").exists() for seed in seeds)
    if completed != len(seeds):
        raise RuntimeError(f"{phase} generation incomplete: {completed}/{len(seeds)}")


def materialize_seeds(phase: str) -> None:
    path = ROOT / phase / "seeds.json"
    if path.exists():
        return
    reservations = json.loads((ROOT / "future_seed_reservations.json").read_text())
    key = "p0_seeds" if phase == "p0" else "formal256_seeds"
    seeds = list(map(int, reservations[key]))
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(json.dumps({"schema_version": 1, "purpose": phase, "materialized_only_after_required_gate": True, "count": len(seeds), "seeds": seeds, "historical_overlap": 0}, indent=2) + "\n")


def run_phase(phase: str, gpus: tuple[int, ...]) -> dict:
    generation(phase, gpus)
    update_status(stage=f"{phase}_property_evaluation")
    run_logged([PYTHON, str(ROOT / "prepare_phase.py"), "--phase", phase], f"prepare_{phase}.log", gpus[0])
    update_status(stage=f"{phase}_quality_evaluation")
    run_logged([PYTHON, str(ROOT / "run_quality.py"), "--phase", phase, "--gpus", ",".join(map(str, gpus))], f"quality_driver_{phase}.log")
    update_status(stage=f"{phase}_paired_statistics_bootstrap_plots_report")
    run_logged([PYTHON, str(ROOT / "analyze_phase.py"), "--phase", phase], f"analysis_{phase}.log")
    return json.loads((ROOT / phase / "decision_summary.json").read_text())


def main(gpus: tuple[int, ...]) -> None:
    assert_frozen()
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU list must be nonempty and unique")
    audit_selected_gpus(gpus)
    (ROOT / "logs").mkdir(exist_ok=True)
    calibration = run_phase("calibration", gpus)
    update_status(**calibration)
    if calibration["STAGE_SCHEDULE_CALIBRATION"] != "GO":
        update_status(stage="stopped_after_calibration_fail", active_phase=None)
        return
    materialize_seeds("p0")
    p0 = run_phase("p0", gpus)
    update_status(**p0)
    if p0["STAGE_CFG_P0"] != "GO":
        update_status(stage="stopped_after_p0_fail", active_phase=None)
        return
    materialize_seeds("formal256")
    formal = run_phase("formal256", gpus)
    update_status(**formal, stage="complete", active_phase=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU IDs confirmed idle at launch")
    arguments = parser.parse_args()
    main(tuple(int(value) for value in arguments.gpus.split(",") if value))
