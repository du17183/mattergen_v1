"""Execution-only supervisor for generation, quality, and frozen offline gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/risk_calibrated_adaptive_cfg"
DATASET = ROOT / "dataset"
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"
GPUS = (0, 1, 3, 4, 5, 6, 7)


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


def run_logged(command: list[str], log_name: str, environment: dict | None = None) -> None:
    log_path = ROOT / "logs" / log_name
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command, cwd=PROJECT, env=environment, stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def generation() -> None:
    seeds = list(map(int, json.loads((DATASET / "seeds.json").read_text())["seeds"]))
    pending = [
        seed for seed in seeds
        if not (DATASET / "generation" / str(seed) / "run_summary.json").exists()
    ]
    partitions = {gpu: pending[index::len(GPUS)] for index, gpu in enumerate(GPUS)}
    processes = []
    for gpu, assigned in partitions.items():
        if not assigned:
            continue
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu), RISK_CFG_PHYSICAL_GPU=str(gpu),
            PYTHONPATH=str(PROJECT), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
            MPLCONFIGDIR=str(ROOT / ".matplotlib"),
        )
        log_path = ROOT / "logs" / f"dataset_generation_gpu_{gpu}.log"
        stream = log_path.open("x", encoding="utf-8")
        command = [
            PYTHON, str(ROOT / "run_dataset_generation.py"),
            "--seeds", ",".join(map(str, assigned)),
        ]
        process = subprocess.Popen(
            command, cwd=PROJECT, env=environment, stdout=stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((gpu, process, stream, log_path))
    while processes:
        still_running = []
        for gpu, process, stream, log_path in processes:
            code = process.poll()
            if code is None:
                still_running.append((gpu, process, stream, log_path))
            else:
                stream.close()
                if code:
                    for _, other, other_stream, _ in still_running + processes:
                        if other.poll() is None: other.terminate()
                        other_stream.close()
                    raise RuntimeError(f"generation worker GPU {gpu} failed; see {log_path}")
        processes = still_running
        completed = sum(
            (DATASET / "generation" / str(seed) / "run_summary.json").exists()
            for seed in seeds
        )
        update_status(generation_complete=completed, stage="dataset_generation")
        print(json.dumps({"stage": "generation", "complete": completed, "total": len(seeds)}), flush=True)
        if processes:
            time.sleep(30)
    completed = sum(
        (DATASET / "generation" / str(seed) / "run_summary.json").exists()
        for seed in seeds
    )
    if completed != len(seeds):
        raise RuntimeError(f"generation ended with only {completed}/{len(seeds)} complete")


def main() -> None:
    assert_frozen()
    (ROOT / "logs").mkdir(exist_ok=True)
    generation()
    update_status(stage="property_evaluation", generation_complete=96)
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="0", RISK_CFG_PHYSICAL_GPU="0", PYTHONPATH=str(PROJECT),
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MPLCONFIGDIR=str(ROOT / ".matplotlib"),
    )
    run_logged([PYTHON, str(ROOT / "prepare_dataset.py")], "prepare_dataset.log", environment)
    update_status(stage="quality_evaluation")
    run_logged([PYTHON, str(ROOT / "run_quality.py")], "quality_driver.log", dict(os.environ, PYTHONPATH=str(PROJECT), PYTHONUNBUFFERED="1"))
    update_status(stage="offline_analysis")
    run_logged(
        [PYTHON, str(ROOT / "run_offline_analysis.py")], "offline_analysis.log",
        dict(os.environ, PYTHONPATH=str(PROJECT), PYTHONUNBUFFERED="1", MPLCONFIGDIR=str(ROOT / ".matplotlib")),
    )
    decision = json.loads((ROOT / "offline/decision_summary.json").read_text())
    update_status(**decision, stage="fresh_p0_ready" if decision["OFFLINE_PREDICTABILITY"] == "GO" else "stopped_offline")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
