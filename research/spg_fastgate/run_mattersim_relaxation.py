"""Persistent MatterSim single-point and relaxation evaluation for Fast Gate."""

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

from research.spg_fastgate.common import (
    LOGS,
    MATTERSIM,
    MATTERSIM_SHA256,
    PROJECT,
    PYTHON,
    REPORTS,
    RESULTS,
    STOP_MARKER,
    atomic_json,
    base_environment,
    now,
    read_json,
    set_stage,
    sha256_file,
)


WORKERS = 16
RELAX_TOOLS = Path("/data/dxl/tools/guidance_stage7_eval")
OUT = RESULTS / "relaxed"
WORKER_DIR = OUT / "workers"
PROGRESS = RESULTS / "progress/mattersim_relaxation.json"


def generation_tasks() -> list[dict[str, Any]]:
    tasks = []
    for method in ("C0", "A0"):
        for batch_size in (1, 4):
            config = f"{method}_B{batch_size}"
            for seed in range(24064, 24128):
                tasks.append(
                    {
                        "task_id": f"quality_{config}_{seed}",
                        "family": "quality",
                        "config": config,
                        "method": method,
                        "batch_size": batch_size,
                        "precision": "FP32",
                        "seed": seed,
                        "input_path": str(
                            RESULTS
                            / "quality_generation"
                            / method
                            / f"B{batch_size}"
                            / f"seed_{seed}"
                            / "generated_crystals.extxyz"
                        ),
                        "generation_summary": str(
                            RESULTS
                            / "quality_generation"
                            / method
                            / f"B{batch_size}"
                            / f"seed_{seed}"
                            / "summary.json"
                        ),
                        "output_dir": str(OUT / "quality" / config / f"seed_{seed}"),
                    }
                )
    state_probe = read_json(RESULTS / "bf16_state_probe.json", {})
    if state_probe.get("FIELD_SAFE_BF16_STATE_GO"):
        for method in ("C0", "A0"):
            for precision in ("FP32", "FIELD_SAFE_BF16"):
                config = f"{method}_{precision}"
                for seed in range(24128, 24136):
                    tasks.append(
                        {
                            "task_id": f"bf16_{config}_{seed}",
                            "family": "bf16",
                            "config": config,
                            "method": method,
                            "batch_size": 1,
                            "precision": precision,
                            "seed": seed,
                            "input_path": str(
                                RESULTS
                                / "bf16_generation"
                                / precision
                                / method
                                / "B1"
                                / f"seed_{seed}"
                                / "generated_crystals.extxyz"
                            ),
                            "generation_summary": str(
                                RESULTS
                                / "bf16_generation"
                                / precision
                                / method
                                / "B1"
                                / f"seed_{seed}"
                                / "summary.json"
                            ),
                            "output_dir": str(OUT / "bf16" / config / f"seed_{seed}"),
                        }
                    )
    return tasks


def safe_result(path: Path) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "success":
            return None
        if not Path(result["output_path"]).is_file():
            return None
        return result
    except Exception:
        return None


def singlepoint(potential: Any, atoms: Any) -> tuple[float, np.ndarray, np.ndarray]:
    from mattersim.forcefield import MatterSimCalculator

    calculator = MatterSimCalculator.from_potential(
        potential=potential,
        device="cuda",
    )
    probe = atoms.copy()
    probe.calc = calculator
    energy = float(probe.get_potential_energy())
    forces = np.asarray(probe.get_forces(), dtype=float)
    stress = np.asarray(probe.get_stress(voigt=False), dtype=float)
    return energy, forces, stress


def run_task(potential: Any, task: dict[str, Any], rank: int) -> dict[str, Any]:
    sys.path.insert(0, str(RELAX_TOOLS))
    from relax_common import relax_group, structure_hash

    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    with (output_dir / "task.lock").open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        existing = safe_result(result_path)
        if existing is not None:
            return existing
        started = time.monotonic()
        result: dict[str, Any] = {
            **task,
            "status": "running",
            "rank": rank,
            "pid": os.getpid(),
            "started_at": now(),
        }
        atomic_json(result_path, result)
        try:
            input_path = Path(task["input_path"])
            generation_summary = read_json(Path(task["generation_summary"]))
            if not input_path.is_file() or generation_summary.get("success") is not True:
                raise RuntimeError(f"generation prerequisite invalid: {task['task_id']}")
            atoms = ase.io.read(input_path)
            input_hash = structure_hash(atoms)
            energy0, forces0, stress0 = singlepoint(potential, atoms)
            relaxed_result = relax_group(potential, [atoms])[0]
            relaxed = relaxed_result["atoms"]
            output_path = output_dir / "relaxed_structure.extxyz"
            ase.io.write(output_path, relaxed, format="extxyz")
            arrays = (
                forces0,
                stress0,
                relaxed.positions,
                relaxed.cell.array,
            )
            scalars = (
                energy0,
                float(relaxed_result["energy_ev"]),
                float(relaxed_result["max_force_ev_ang"]),
            )
            if not all(np.isfinite(value).all() for value in arrays):
                raise RuntimeError("non-finite MatterSim array")
            if not all(math.isfinite(value) for value in scalars):
                raise RuntimeError("non-finite MatterSim scalar")
            result.update(
                {
                    "status": "success",
                    "finished_at": now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "input_hash": input_hash,
                    "initial_energy_ev": energy0,
                    "initial_energy_per_atom_ev": energy0 / len(atoms),
                    "initial_max_force_ev_ang": float(
                        np.linalg.norm(forces0, axis=1).max()
                    ),
                    "initial_stress_frobenius_ev_ang3": float(
                        np.linalg.norm(stress0)
                    ),
                    "relaxed_energy_ev": float(relaxed_result["energy_ev"]),
                    "relaxed_energy_per_atom_ev": float(
                        relaxed_result["energy_per_atom_ev"]
                    ),
                    "relaxed_max_force_ev_ang": float(
                        relaxed_result["max_force_ev_ang"]
                    ),
                    "relax_steps": int(relaxed_result["steps"]),
                    "converged": bool(relaxed_result["converged"]),
                    "output_path": str(output_path),
                    "output_hash": structure_hash(relaxed),
                    "atomic_numbers_unchanged": bool(
                        np.array_equal(atoms.numbers, relaxed.numbers)
                    ),
                }
            )
        except BaseException:
            result.update(
                {
                    "status": "failed",
                    "finished_at": now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "error": traceback.format_exc(),
                }
            )
        atomic_json(result_path, result)
        return result


def worker(rank: int, workers: int) -> int:
    sys.path.insert(0, str(RELAX_TOOLS))
    from relax_common import load_potential

    potential = load_potential("cuda")
    tasks = generation_tasks()[rank::workers]
    failures = 0
    for completed, task in enumerate(tasks, start=1):
        if STOP_MARKER.exists():
            return 130
        result = run_task(potential, task, rank)
        failures += result.get("status") != "success"
        atomic_json(
            WORKER_DIR / f"rank_{rank}.json",
            {
                "rank": rank,
                "pid": os.getpid(),
                "completed": completed,
                "assigned": len(tasks),
                "failures": failures,
                "updated_at": now(),
            },
        )
    return 1 if failures else 0


def freeze_config() -> dict:
    if sha256_file(MATTERSIM) != MATTERSIM_SHA256:
        raise RuntimeError("MatterSim-5M checkpoint SHA256 mismatch")
    source = read_json(
        Path("/data/dxl/reports/guidance_stage7_eval/relax_config_frozen.json"),
        {},
    )
    source.update(
        {
            "checkpoint": str(MATTERSIM),
            "checkpoint_sha256": MATTERSIM_SHA256,
            "optimizer": "FIRE",
            "cell_filter": "EXPCELLFILTER",
            "fmax_ev_ang": 0.05,
            "max_steps": 500,
            "workers_per_gpu": 2,
            "total_concurrency": WORKERS,
            "frozen_at": now(),
        }
    )
    atomic_json(REPORTS / "mattersim_config.json", source)
    return source


def launch() -> int:
    tasks = generation_tasks()
    freeze_config()
    missing = [
        task["task_id"]
        for task in tasks
        if not Path(task["input_path"]).is_file()
    ]
    if missing:
        raise RuntimeError(f"missing generation prerequisites: {missing[:10]}")
    set_stage(
        "b4_relaxation",
        "running",
        f"Running {len(tasks)} MatterSim single-point and relax tasks with 16 workers.",
        {"tasks": len(tasks), "workers": WORKERS},
    )
    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    for rank in range(WORKERS):
        gpu = rank % 8
        log = LOGS / "mattersim" / f"rank_{rank}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a", encoding="utf-8")
        streams.append(stream)
        processes.append(
            subprocess.Popen(
                [
                    str(PYTHON),
                    "-m",
                    "research.spg_fastgate.run_mattersim_relaxation",
                    "worker",
                    "--rank",
                    str(rank),
                    "--workers",
                    str(WORKERS),
                ],
                cwd=PROJECT,
                env=base_environment(gpu),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        )
    return_codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    results = [
        safe_result(Path(task["output_dir"]) / "result.json")
        for task in tasks
    ]
    success = sum(result is not None for result in results)
    state = {
        "updated_at": now(),
        "total": len(tasks),
        "success": success,
        "failed": len(tasks) - success,
        "return_codes": return_codes,
    }
    atomic_json(PROGRESS, state)
    if success != len(tasks):
        set_stage(
            "b4_relaxation",
            "failed",
            f"MatterSim relaxation incomplete: {success}/{len(tasks)}.",
            state,
        )
        return 1
    set_stage(
        "b4_relaxation",
        "success",
        f"MatterSim single-point and relaxation complete: {success}/{len(tasks)}.",
        state,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "worker"), nargs="?", default="launch")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--workers", type=int, default=WORKERS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "worker":
        if args.rank is None:
            raise ValueError("worker requires --rank")
        return worker(args.rank, args.workers)
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
