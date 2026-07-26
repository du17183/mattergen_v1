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

from research.rp_qtfg.common import REPORTS, RESULTS, atomic_json, now, set_stage, stop_requested
from research.rp_qtfg.experiment_config import EIGHT_SEED_CONFIGS, EIGHT_SEEDS, THIRTY_TWO_SEEDS


GENERATION = RESULTS / "generation"
OUT = RESULTS / "relax"
SELECTED = Path("/data/dxl/reports/rp_qtfg/phase0/eight_seed/selected_candidate.json")
LAUNCHER = REPORTS / "launcher.json"
WORKERS = 16
MATTERSIM_ENV = Path("/data/dxl/envs/mattergen_py310/bin/python")
RELAX_TOOLS = Path("/data/dxl/tools/guidance_stage7_eval")


def _selected_config() -> str:
    return str(json.loads(SELECTED.read_text())["selected_config"])


def _tasks(mode: str) -> list[dict[str, Any]]:
    if mode == "eight":
        configs, seeds = EIGHT_SEED_CONFIGS, EIGHT_SEEDS
    elif mode == "thirtytwo":
        configs, seeds = ("A0", _selected_config()), THIRTY_TWO_SEEDS
    else:
        raise ValueError(mode)
    return [
        {
            "task_id": f"{config_id}_{seed}",
            "config_id": config_id,
            "seed": seed,
            "input_path": str(GENERATION / config_id / str(seed) / "generated_crystals.extxyz"),
            "output_dir": str(OUT / config_id / str(seed)),
        }
        for config_id in configs
        for seed in seeds
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
            relaxed_result = relax_group(potential, [atoms])[0]
            relaxed = relaxed_result["atoms"]
            relaxed_energy = float(relaxed_result["energy_ev"])
            relaxed_max_force = float(relaxed_result["max_force_ev_ang"])
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


def worker(mode: str, rank: int, workers: int) -> int:
    sys.path.insert(0, str(RELAX_TOOLS))
    from relax_common import load_potential

    potential = load_potential("cuda")
    tasks = _tasks(mode)[rank::workers]
    failures = 0
    for completed, task in enumerate(tasks, start=1):
        if stop_requested():
            return 130
        result = _run_task(potential, task, rank)
        failures += result.get("status") != "success"
        atomic_json(
            OUT / "workers" / mode / f"rank_{rank}.json",
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


def launch(mode: str, workers: int = WORKERS) -> int:
    atomic_json(
        LAUNCHER,
        {
            "pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgid(0),
            "sid": os.getsid(0), "user": os.environ.get("USER", ""),
            "cwd": os.getcwd(), "exe": os.readlink(f"/proc/{os.getpid()}/exe"),
            "command": " ".join(sys.argv), "mode": mode, "started_at": now(),
        },
    )
    set_stage(
        "eight_seed_review" if mode == "eight" else "thirty_two_relax",
        "running",
        f"Running independent MatterSim evaluation for {mode}.",
        {
            "tasks": len(_tasks(mode)),
            "workers": workers,
            "workers_per_gpu": workers // 8,
        },
    )
    (OUT / "workers" / mode).mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    for rank in range(workers):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(rank % 8)
        command = [
            str(MATTERSIM_ENV),
            "-m",
            "research.rp_qtfg.experiment_relax",
            "worker",
            "--mode",
            mode,
            "--rank",
            str(rank),
            "--workers",
            str(workers),
        ]
        stdout = (OUT / "workers" / mode / f"rank_{rank}.stdout.log").open("a")
        stderr = (OUT / "workers" / mode / f"rank_{rank}.stderr.log").open("a")
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
        for task in _tasks(mode)
    ]
    success = sum(result is not None for result in results)
    atomic_json(
        OUT / f"relax_progress_{mode}.json",
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
            f"{mode} MatterSim incomplete: success={success}/{len(results)}"
        )
    return 0


def status(mode: str) -> dict[str, Any]:
    tasks = _tasks(mode)
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
    worker_parser.add_argument("--mode", choices=("eight", "thirtytwo"), required=True)
    worker_parser.add_argument("--rank", type=int, required=True)
    worker_parser.add_argument("--workers", type=int, default=WORKERS)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--mode", choices=("eight", "thirtytwo"), required=True)
    launch_parser.add_argument("--workers", type=int, default=WORKERS)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--mode", choices=("eight", "thirtytwo"), required=True)
    args = parser.parse_args()
    (OUT / "workers" / args.mode).mkdir(parents=True, exist_ok=True)
    if args.command == "worker":
        return worker(args.mode, args.rank, args.workers)
    if args.command == "launch":
        return launch(args.mode, args.workers)
    print(json.dumps(status(args.mode), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
