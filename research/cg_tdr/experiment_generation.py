#!/usr/bin/env python3
"""Resume-safe A0/T1/T2 generation launcher for CG-TDR gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path("/data/dxl/results/cg_tdr/phase0")
GENERATION = ROOT / "generation"
REPORTS = Path("/data/dxl/reports/cg_tdr/phase0")
CHECKPOINT = ROOT / "training/checkpoints/best.pt"
RUNNER = Path("/data/dxl/mattergen_v1/research/cg_tdr/run_eval_sample.py")
PYTHON = Path("/data/dxl/envs/mattergen_py310/bin/python")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def successful(method: str, seed: int) -> bool:
    path = GENERATION / method / str(seed) / "run_summary.json"
    try:
        return bool(json.loads(path.read_text()).get("success"))
    except Exception:
        return False


def selected_method() -> str:
    return str(json.loads((REPORTS / "eight_seed/selected_candidate.json").read_text())["selected_config"])


def tasks(mode: str) -> list[dict[str, Any]]:
    if mode == "eight":
        methods, seeds = ("A0", "T1", "T2"), range(23000, 23008)
    elif mode == "thirtytwo":
        methods, seeds = ("A0", selected_method()), range(23000, 23032)
    else:
        raise ValueError(mode)
    return [{"method": method, "seed": seed} for method in methods for seed in seeds]


def worker(gpu: int, assigned: list[dict[str, Any]], mode: str) -> int:
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
        ]
        if method != "A0":
            command += ["--cg-tdr-checkpoint", str(CHECKPOINT)]
        with (output / "console.log").open("a") as stream:
            result = subprocess.run(
                command,
                cwd="/data/dxl/mattergen_v1",
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        failures += result.returncode != 0 or not successful(method, seed)
        all_tasks = tasks(mode)
        success_count = sum(successful(item["method"], item["seed"]) for item in all_tasks)
        atomic_json(
            ROOT / "progress" / f"generation_{mode}.json",
            {
                "stage": f"{mode}_seed_generation",
                "total": len(all_tasks),
                "success": success_count,
                "remaining": len(all_tasks) - success_count,
                "failed_seen": failures,
            },
        )
    return failures


def launch(mode: str) -> int:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    all_tasks = tasks(mode)
    partitions = [all_tasks[gpu::8] for gpu in range(8)]
    failures = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, gpu, partitions[gpu], mode) for gpu in range(8)]
        for future in as_completed(futures):
            failures += future.result()
    success_count = sum(successful(item["method"], item["seed"]) for item in all_tasks)
    atomic_json(
        ROOT / "progress" / f"generation_{mode}.json",
        {
            "stage": f"{mode}_seed_generation",
            "status": "success" if success_count == len(all_tasks) else "failed",
            "total": len(all_tasks),
            "success": success_count,
            "remaining": len(all_tasks) - success_count,
            "failures": failures,
        },
    )
    return 0 if success_count == len(all_tasks) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "status"))
    parser.add_argument("--mode", choices=("eight", "thirtytwo"), required=True)
    args = parser.parse_args()
    if args.command == "launch":
        return launch(args.mode)
    all_tasks = tasks(args.mode)
    success_count = sum(successful(item["method"], item["seed"]) for item in all_tasks)
    print(json.dumps({"total": len(all_tasks), "success": success_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
