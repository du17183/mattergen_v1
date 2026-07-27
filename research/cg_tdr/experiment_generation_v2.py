#!/usr/bin/env python3
"""Resume-safe V2P/V2C generation on the frozen eight development seeds."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path("/data/dxl/results/cg_tdr/phase0")
GENERATION = ROOT / "generation"
CHECKPOINT = ROOT / "training_v2/checkpoints/best.pt"
TRAINING_SUMMARY = ROOT / "training_v2/training_summary.json"
RUNNER = Path("/data/dxl/mattergen_v1/research/cg_tdr/run_eval_sample.py")
PYTHON = Path("/data/dxl/envs/mattergen_py310/bin/python")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def successful(method: str, seed: int) -> bool:
    try:
        summary = json.loads(
            (GENERATION / method / str(seed) / "run_summary.json").read_text()
        )
        return bool(summary.get("success"))
    except Exception:
        return False


def tasks() -> list[dict[str, Any]]:
    training = json.loads(TRAINING_SUMMARY.read_text())
    methods = (
        ("V2P", "V2C")
        if training.get("CG_TDR_GATE_V2_VALID")
        else ("V2P",)
    )
    return [
        {"method": method, "seed": seed}
        for method in methods
        for seed in range(23000, 23008)
    ]


def worker(gpu: int, assigned: list[dict[str, Any]]) -> int:
    failures = 0
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    all_tasks = tasks()
    for task in assigned:
        method, seed = task["method"], task["seed"]
        if successful(method, seed):
            continue
        output = GENERATION / method / str(seed)
        output.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON),
            str(RUNNER),
            "--method",
            method,
            "--seed",
            str(seed),
            "--physical-gpu",
            str(gpu),
            "--output-dir",
            str(output),
            "--cg-tdr-checkpoint",
            str(CHECKPOINT),
        ]
        with (output / "console.log").open("a") as stream:
            result = subprocess.run(
                command,
                cwd="/data/dxl/mattergen_v1",
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        failures += result.returncode != 0 or not successful(method, seed)
        success = sum(successful(item["method"], item["seed"]) for item in all_tasks)
        atomic_json(
            ROOT / "progress/generation_eight_v2.json",
            {
                "stage": "v2_eight_seed_generation",
                "total": len(all_tasks),
                "success": success,
                "remaining": len(all_tasks) - success,
                "failed_seen": failures,
            },
        )
    return failures


def main() -> int:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    payload = __import__("torch").load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    if int(payload.get("training_seed", -1)) != 3101:
        raise ValueError("Refusing non-V2 checkpoint")
    all_tasks = tasks()
    partitions = [all_tasks[gpu::8] for gpu in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        failures = sum(
            future.result()
            for future in as_completed(
                [
                    executor.submit(worker, gpu, partitions[gpu])
                    for gpu in range(8)
                ]
            )
        )
    success = sum(successful(item["method"], item["seed"]) for item in all_tasks)
    atomic_json(
        ROOT / "progress/generation_eight_v2.json",
        {
            "stage": "v2_eight_seed_generation",
            "status": "success" if success == len(all_tasks) else "failed",
            "total": len(all_tasks),
            "success": success,
            "remaining": len(all_tasks) - success,
            "failures": failures,
        },
    )
    return 0 if success == len(all_tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
