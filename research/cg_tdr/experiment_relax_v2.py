#!/usr/bin/env python3
"""Independent resume-safe MatterSim evaluation for Gate V2 structures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from research.cg_tdr.experiment_relax import atomic_json, run_task, safe_result


ROOT = Path("/data/dxl/results/cg_tdr/phase0")
GENERATION = ROOT / "generation"
OUT = ROOT / "relax"
PYTHON = Path("/data/dxl/envs/mattergen_py310/bin/python")
RELAX_TOOLS = Path("/data/dxl/tools/guidance_stage7_eval")


def methods() -> tuple[str, ...]:
    summary = json.loads(
        (ROOT / "training_v2/training_summary.json").read_text()
    )
    return ("V2P", "V2C") if summary["CG_TDR_GATE_V2_VALID"] else ("V2P",)


def tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": f"{method}_{seed}",
            "method": method,
            "seed": seed,
            "input_path": str(
                GENERATION / method / str(seed) / "generated_crystals.extxyz"
            ),
            "output_dir": str(OUT / method / str(seed)),
        }
        for method in methods()
        for seed in range(23000, 23008)
    ]


def worker(rank: int, workers: int) -> int:
    import sys

    sys.path.insert(0, str(RELAX_TOOLS))
    from relax_common import load_potential

    potential = load_potential("cuda")
    failures = 0
    for task in tasks()[rank::workers]:
        failures += run_task(potential, task, rank).get("status") != "success"
    return 1 if failures else 0


def launch(workers: int) -> int:
    processes = []
    streams = []
    worker_root = OUT / "workers/eight_v2"
    worker_root.mkdir(parents=True, exist_ok=True)
    for rank in range(workers):
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(rank % 8),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "NUMEXPR_NUM_THREADS": "2",
            }
        )
        stream = (worker_root / f"rank_{rank}.log").open("a")
        streams.append(stream)
        processes.append(
            subprocess.Popen(
                [
                    str(PYTHON),
                    "-m",
                    "research.cg_tdr.experiment_relax_v2",
                    "worker",
                    "--rank",
                    str(rank),
                    "--workers",
                    str(workers),
                ],
                cwd="/data/dxl/mattergen_v1",
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        )
    return_codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    all_tasks = tasks()
    success = sum(
        safe_result(Path(task["output_dir"]) / "result.json") is not None
        for task in all_tasks
    )
    atomic_json(
        ROOT / "progress/relax_eight_v2.json",
        {
            "status": "success" if success == len(all_tasks) else "failed",
            "total": len(all_tasks),
            "success": success,
            "remaining": len(all_tasks) - success,
            "return_codes": return_codes,
        },
    )
    return 0 if success == len(all_tasks) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--workers", type=int, default=16)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--rank", type=int, required=True)
    worker_parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.command == "launch":
        return launch(args.workers)
    return worker(args.rank, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
