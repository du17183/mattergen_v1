#!/usr/bin/env python3
"""Independent MatterSim-5M evaluation for CG-TDR gate structures."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import ase.io
import numpy as np


ROOT = Path("/data/dxl/results/cg_tdr/phase0")
GENERATION = ROOT / "generation"
OUT = ROOT / "relax"
REPORTS = Path("/data/dxl/reports/cg_tdr/phase0")
PYTHON = Path("/data/dxl/envs/mattergen_py310/bin/python")
RELAX_TOOLS = Path("/data/dxl/tools/guidance_stage7_eval")
WORKERS = 16


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def selected_method() -> str:
    return str(json.loads((REPORTS / "eight_seed/selected_candidate.json").read_text())["selected_config"])


def tasks(mode: str) -> list[dict[str, Any]]:
    if mode == "eight":
        methods, seeds = ("A0", "T1", "T2"), range(23000, 23008)
    elif mode == "thirtytwo":
        methods, seeds = ("A0", selected_method()), range(23000, 23032)
    else:
        raise ValueError(mode)
    return [
        {
            "task_id": f"{method}_{seed}",
            "method": method,
            "seed": seed,
            "input_path": str(GENERATION / method / str(seed) / "generated_crystals.extxyz"),
            "output_dir": str(OUT / method / str(seed)),
        }
        for method in methods
        for seed in seeds
    ]


def safe_result(path: Path) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text())
        if result.get("status") == "success" and Path(result["output_path"]).exists():
            return result
    except Exception:
        pass
    return None


def singlepoint(potential: Any, atoms: Any) -> tuple[float, np.ndarray, np.ndarray]:
    from mattersim.forcefield import MatterSimCalculator

    calculator = MatterSimCalculator.from_potential(potential=potential, device="cuda")
    probe = atoms.copy()
    probe.calc = calculator
    return (
        float(probe.get_potential_energy()),
        np.asarray(probe.get_forces(), dtype=float),
        np.asarray(probe.get_stress(voigt=False), dtype=float),
    )


def run_task(potential: Any, task: dict[str, Any], rank: int) -> dict[str, Any]:
    sys.path.insert(0, str(RELAX_TOOLS))
    from relax_common import relax_group, structure_hash

    output = Path(task["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    with (output / "task.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = safe_result(result_path)
        if existing:
            return existing
        started = time.monotonic()
        result = {**task, "status": "running", "rank": rank, "pid": os.getpid()}
        atomic_json(result_path, result)
        try:
            atoms = ase.io.read(task["input_path"])
            energy0, forces0, stress0 = singlepoint(potential, atoms)
            relaxed_result = relax_group(potential, [atoms])[0]
            relaxed = relaxed_result["atoms"]
            output_path = output / "relaxed_structure.extxyz"
            ase.io.write(output_path, relaxed, format="extxyz")
            arrays = (forces0, stress0, relaxed.positions, relaxed.cell.array)
            scalars = (
                energy0,
                relaxed_result["energy_ev"],
                relaxed_result["max_force_ev_ang"],
            )
            if not all(np.isfinite(value).all() for value in arrays):
                raise RuntimeError("non-finite MatterSim array")
            if not all(math.isfinite(float(value)) for value in scalars):
                raise RuntimeError("non-finite MatterSim scalar")
            result.update(
                {
                    "status": "success",
                    "elapsed_seconds": time.monotonic() - started,
                    "input_hash": structure_hash(atoms),
                    "initial_energy_ev": energy0,
                    "initial_energy_per_atom_ev": energy0 / len(atoms),
                    "initial_max_force_ev_ang": float(np.linalg.norm(forces0, axis=1).max()),
                    "initial_stress_frobenius_ev_ang3": float(np.linalg.norm(stress0)),
                    "relaxed_energy_ev": float(relaxed_result["energy_ev"]),
                    "relaxed_energy_per_atom_ev": float(relaxed_result["energy_ev"]) / len(relaxed),
                    "relaxed_max_force_ev_ang": float(relaxed_result["max_force_ev_ang"]),
                    "relax_steps": int(relaxed_result["steps"]),
                    "converged": bool(relaxed_result["converged"]),
                    "output_path": str(output_path),
                    "atomic_numbers_unchanged": bool(
                        np.array_equal(atoms.numbers, relaxed.numbers)
                    ),
                }
            )
        except BaseException:
            result.update(
                {
                    "status": "failed",
                    "elapsed_seconds": time.monotonic() - started,
                    "error": traceback.format_exc(),
                }
            )
        atomic_json(result_path, result)
        return result


def worker(mode: str, rank: int, workers: int) -> int:
    sys.path.insert(0, str(RELAX_TOOLS))
    from relax_common import load_potential

    potential = load_potential("cuda")
    failures = 0
    for task in tasks(mode)[rank::workers]:
        failures += run_task(potential, task, rank).get("status") != "success"
    return 1 if failures else 0


def launch(mode: str, workers: int) -> int:
    processes = []
    streams = []
    worker_root = OUT / "workers" / mode
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
                    "research.cg_tdr.experiment_relax",
                    "worker",
                    "--mode",
                    mode,
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
    all_tasks = tasks(mode)
    success = sum(
        safe_result(Path(task["output_dir"]) / "result.json") is not None
        for task in all_tasks
    )
    atomic_json(
        ROOT / "progress" / f"relax_{mode}.json",
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
    launch_parser.add_argument("--mode", choices=("eight", "thirtytwo"), required=True)
    launch_parser.add_argument("--workers", type=int, default=WORKERS)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--mode", choices=("eight", "thirtytwo"), required=True)
    worker_parser.add_argument("--rank", type=int, required=True)
    worker_parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.command == "launch":
        return launch(args.mode, args.workers)
    return worker(args.mode, args.rank, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
