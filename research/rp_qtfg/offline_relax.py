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

from research.rp_qtfg.common import RESULTS, atomic_json, now, set_stage, stop_requested


SEEDS = tuple(range(20000, 20064))
METHODS = ("baseline", "pos_1", "pos_3", "pos_5", "poscell_1", "poscell_3")
PROBE = RESULTS / "offline_probe/structures"
OUT = RESULTS / "offline_relax"
WORKERS = 16
MATTERSIM_ENV = Path("/data/dxl/envs/mattergen_py310/bin/python")
RELAX_TOOLS = Path("/data/dxl/tools/guidance_stage7_eval")
FROZEN_A0 = Path("/data/dxl/results/formal_256/relaxed/A0")


def _tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": f"{method}_{seed}",
            "method": method,
            "seed": seed,
            "input_path": str(PROBE / method / f"{seed}.extxyz"),
            "output_dir": str(OUT / method / str(seed)),
        }
        for method in METHODS
        for seed in SEEDS
    ]


def _safe_result(path: Path) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text())
        if result.get("status") != "success":
            return None
        if not Path(result["output_path"]).is_file():
            return None
        return result
    except Exception:
        return None


def _singlepoint(
    potential: Any,
    atoms: Any,
) -> tuple[float, np.ndarray, np.ndarray]:
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


def _run_task(
    potential: Any,
    task: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(RELAX_TOOLS))
    from relax_common import relax_group, structure_hash

    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    lock_path = output_dir / "task.lock"
    with lock_path.open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        existing = _safe_result(result_path)
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
            atoms = ase.io.read(task["input_path"])
            input_hash = structure_hash(atoms)
            energy0, forces0, stress0 = _singlepoint(potential, atoms)
            if task["method"] == "baseline":
                frozen_dir = FROZEN_A0 / str(task["seed"])
                frozen_summary = json.loads(
                    (frozen_dir / "relax_summary.json").read_text()
                )
                if input_hash != frozen_summary["input_hash"]:
                    raise RuntimeError(
                        f"frozen A0 input hash mismatch for seed {task['seed']}"
                    )
                relaxed = ase.io.read(frozen_dir / "relaxed_structure.extxyz")
                relaxed_energy = float(frozen_summary["energy_ev"])
                relaxed_max_force = float(
                    frozen_summary["maximum_force_ev_ang"]
                )
                steps = int(frozen_summary["steps"])
                converged = bool(frozen_summary["converged"])
                output_path = frozen_dir / "relaxed_structure.extxyz"
                reused = True
            else:
                relaxed_result = relax_group(potential, [atoms])[0]
                relaxed = relaxed_result["atoms"]
                relaxed_energy = float(relaxed_result["energy_ev"])
                relaxed_max_force = float(
                    relaxed_result["max_force_ev_ang"]
                )
                steps = int(relaxed_result["steps"])
                converged = bool(relaxed_result["converged"])
                output_path = output_dir / "relaxed_structure.extxyz"
                ase.io.write(output_path, relaxed, format="extxyz")
                reused = False
            arrays = (forces0, stress0, relaxed.positions, relaxed.cell.array)
            if not all(np.isfinite(value).all() for value in arrays):
                raise RuntimeError("non-finite MatterSim result")
            if not all(
                math.isfinite(value)
                for value in (energy0, relaxed_energy, relaxed_max_force)
            ):
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
                    "relaxed_energy_ev": relaxed_energy,
                    "relaxed_energy_per_atom_ev": relaxed_energy / len(relaxed),
                    "relaxed_max_force_ev_ang": relaxed_max_force,
                    "relax_steps": steps,
                    "converged": converged,
                    "output_path": str(output_path),
                    "frozen_a0_relax_reused": reused,
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
    tasks = _tasks()[rank::workers]
    failures = 0
    for completed, task in enumerate(tasks, start=1):
        if stop_requested():
            return 130
        result = _run_task(potential, task, rank)
        failures += result.get("status") != "success"
        atomic_json(
            OUT / "workers" / f"rank_{rank}.json",
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


def launch(workers: int = WORKERS) -> int:
    set_stage(
        "offline_direction_probe",
        "running",
        "Running independent MatterSim single-point and relaxation evaluation for Gate 0B.",
        {
            "tasks": len(_tasks()),
            "workers": workers,
            "workers_per_gpu": workers // 8,
        },
    )
    (OUT / "workers").mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    for rank in range(workers):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(rank % 8)
        command = [
            str(MATTERSIM_ENV),
            "-m",
            "research.rp_qtfg.offline_relax",
            "worker",
            "--rank",
            str(rank),
            "--workers",
            str(workers),
        ]
        stdout = (OUT / "workers" / f"rank_{rank}.stdout.log").open("a")
        stderr = (OUT / "workers" / f"rank_{rank}.stderr.log").open("a")
        streams.extend((stdout, stderr))
        processes.append(
            subprocess.Popen(
                command,
                cwd="/data/dxl/mattergen_v1",
                env=environment,
                start_new_session=False,
                stdout=stdout,
                stderr=stderr,
            )
        )
    return_codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    results = [
        _safe_result(Path(task["output_dir"]) / "result.json")
        for task in _tasks()
    ]
    success = sum(result is not None for result in results)
    atomic_json(
        OUT / "relax_progress.json",
        {
            "status": "success" if success == len(results) else "failed",
            "updated_at": now(),
            "total": len(results),
            "success": success,
            "failed": len(results) - success,
            "return_codes": return_codes,
        },
    )
    if success != len(results):
        raise RuntimeError(
            f"Gate 0B MatterSim incomplete: success={success}/{len(results)}"
        )
    return 0


def status() -> dict[str, Any]:
    tasks = _tasks()
    successes = sum(
        _safe_result(Path(task["output_dir"]) / "result.json") is not None
        for task in tasks
    )
    return {
        "total": len(tasks),
        "success": successes,
        "remaining": len(tasks) - successes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--rank", type=int, required=True)
    worker_parser.add_argument("--workers", type=int, default=WORKERS)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--workers", type=int, default=WORKERS)
    subparsers.add_parser("status")
    args = parser.parse_args()
    (OUT / "workers").mkdir(parents=True, exist_ok=True)
    if args.command == "worker":
        return worker(args.rank, args.workers)
    if args.command == "launch":
        return launch(args.workers)
    print(json.dumps(status(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
