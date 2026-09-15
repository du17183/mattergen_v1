"""Supervise all authorized stages through reports, statistics, plots, and hard stops."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/calibrated_safe_selection_cfg"
DATASET = ROOT / "dataset"
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
    temporary.write_text(json.dumps(values, indent=2, default=float) + "\n")
    os.replace(temporary, path)


def environment(gpu: int | None = None) -> dict:
    values = dict(os.environ)
    values.update(
        PYTHONPATH=str(PROJECT), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        MPLCONFIGDIR=str(ROOT / ".matplotlib"),
    )
    if gpu is not None:
        values.update(CUDA_VISIBLE_DEVICES=str(gpu), SAFE_CFG_PHYSICAL_GPU=str(gpu))
    return values


def run_logged(command: list[str], log_name: str, env: dict) -> None:
    log_path = ROOT / "logs" / log_name
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def parallel_generation(cohort: str, seeds: list[int], gpus: tuple[int, ...]) -> None:
    cohort_root = DATASET if cohort == "dataset" else ROOT / cohort
    pending = [
        seed for seed in seeds
        if not (cohort_root / "generation" / str(seed) / "run_summary.json").exists()
    ]
    partitions = {gpu: pending[index::len(gpus)] for index, gpu in enumerate(gpus)}
    processes = []
    for gpu, assigned in partitions.items():
        if not assigned:
            continue
        log_path = ROOT / "logs" / f"{cohort}_generation_gpu_{gpu}.log"
        stream = log_path.open("x", encoding="utf-8")
        if cohort == "dataset":
            command = [PYTHON, str(ROOT / "run_dataset_generation.py"), "--seeds", ",".join(map(str, assigned))]
        else:
            command = [
                PYTHON, str(ROOT / "run_fresh_generation.py"), "--cohort", cohort,
                "--seeds", ",".join(map(str, assigned)),
            ]
        process = subprocess.Popen(command, cwd=PROJECT, env=environment(gpu), stdout=stream, stderr=subprocess.STDOUT)
        processes.append((gpu, process, stream, log_path))
    while processes:
        still_running = []
        failure = None
        for gpu, process, stream, log_path in processes:
            code = process.poll()
            if code is None:
                still_running.append((gpu, process, stream, log_path))
            else:
                stream.close()
                if code:
                    failure = RuntimeError(f"{cohort} generation worker GPU {gpu} failed; see {log_path}")
                    break
        if failure is not None:
            for _, process, stream, _ in processes:
                if process.poll() is None:
                    process.terminate()
                if not stream.closed:
                    stream.close()
            raise failure
        processes = still_running
        completed = sum(
            (cohort_root / "generation" / str(seed) / "run_summary.json").exists()
            for seed in seeds
        )
        update_status(stage=f"{cohort}_generation", cohort=cohort, cohort_complete=completed, cohort_total=len(seeds))
        print(json.dumps({"stage": f"{cohort}_generation", "complete": completed, "total": len(seeds)}), flush=True)
        if processes:
            time.sleep(30)
    completed = sum(
        (cohort_root / "generation" / str(seed) / "run_summary.json").exists()
        for seed in seeds
    )
    if completed != len(seeds):
        raise RuntimeError(f"{cohort} generation ended with {completed}/{len(seeds)} complete")


def fresh_stage(cohort: str, gpus: tuple[int, ...]) -> dict:
    reservations = json.loads((ROOT / "future_seed_reservations.json").read_text())
    seeds = list(map(int, reservations["p0_seeds" if cohort == "p0" else "formal256_seeds"]))
    parallel_generation(cohort, seeds, gpus)
    update_status(stage=f"{cohort}_property_evaluation")
    run_logged(
        [PYTHON, str(ROOT / "prepare_fresh.py"), "--cohort", cohort],
        f"prepare_{cohort}.log", environment(gpus[0]),
    )
    update_status(stage=f"{cohort}_quality_evaluation")
    run_logged(
        [PYTHON, str(ROOT / "run_fresh_quality.py"), "--cohort", cohort, "--gpus", ",".join(map(str, gpus))],
        f"quality_{cohort}_driver.log", environment(),
    )
    update_status(stage=f"{cohort}_paired_statistics_bootstrap_plots_report")
    run_logged(
        [PYTHON, str(ROOT / "analyze_fresh.py"), "--cohort", cohort],
        f"analyze_{cohort}.log", environment(),
    )
    status = json.loads((ROOT / "final_status.json").read_text())
    update_status(**status, stage=f"{cohort}_complete")
    return status


def main(gpus: tuple[int, ...]) -> None:
    if not gpus or any(gpu == 2 for gpu in gpus):
        raise ValueError("non-empty GPU set excluding GPU2 required")
    assert_frozen()
    dataset_seeds = list(map(int, json.loads((DATASET / "seeds.json").read_text())["seeds"]))
    parallel_generation("dataset", dataset_seeds, gpus)
    update_status(stage="dataset_property_evaluation")
    run_logged([PYTHON, str(ROOT / "prepare_dataset.py")], "prepare_dataset.log", environment(gpus[0]))
    update_status(stage="dataset_quality_evaluation")
    run_logged(
        [PYTHON, str(ROOT / "run_quality.py"), "--gpus", ",".join(map(str, gpus))],
        "quality_dataset_driver.log", environment(),
    )
    update_status(stage="offline_train_calibration_validation_test")
    run_logged([PYTHON, str(ROOT / "run_offline_analysis.py")], "offline_analysis.log", environment())
    status = json.loads((ROOT / "final_status.json").read_text())
    update_status(**status)
    if status["OFFLINE_SAFE_SELECTION"] != "GO":
        update_status(stage="stopped_offline_gate")
        print(json.dumps(status, indent=2), flush=True)
        return
    status = fresh_stage("p0", gpus)
    if status["SAFE_SELECTION_P0"] != "GO":
        update_status(stage="stopped_p0_gate")
        print(json.dumps(status, indent=2), flush=True)
        return
    status = fresh_stage("formal256", gpus)
    update_status(stage="complete")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args()
    selected = tuple(int(value) for value in args.gpus.split(",") if value)
    try:
        main(selected)
    except BaseException as error:
        update_status(stage="ERROR", error=f"{type(error).__name__}: {error}")
        raise
