from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from research.rp_qtfg.common import (
    REPORTS,
    RESULTS,
    atomic_json,
    now,
    set_stage,
    stop_requested,
    update_progress,
)
from research.rp_qtfg.experiment_config import (
    EIGHT_SEED_CONFIGS,
    EIGHT_SEEDS,
    THIRTY_TWO_SEEDS,
)


OUT = RESULTS / "generation"
WORKERS = 8
PYTHON = Path("/data/dxl/envs/mattergen_py310/bin/python")
PROJECT = Path("/data/dxl/mattergen_v1")
SELECTED = Path(
    "/data/dxl/reports/rp_qtfg/phase0/eight_seed/selected_candidate.json"
)
LAUNCHER = REPORTS / "launcher.json"


def _selected_config() -> str:
    selected = json.loads(SELECTED.read_text())["selected_config"]
    if not selected:
        raise RuntimeError("8-seed review did not select a candidate")
    return str(selected)


def _task_specs(mode: str) -> list[dict[str, Any]]:
    if mode == "eight":
        configs = EIGHT_SEED_CONFIGS
        seeds = EIGHT_SEEDS
    elif mode == "thirtytwo":
        configs = ("A0", _selected_config())
        seeds = THIRTY_TWO_SEEDS
    else:
        raise ValueError(mode)
    return [
        {
            "task_id": f"{config_id}_{seed}",
            "config_id": config_id,
            "seed": seed,
            "output_dir": str(OUT / config_id / str(seed)),
        }
        for config_id in configs
        for seed in seeds
    ]


def _success(task: dict[str, Any]) -> bool:
    output = Path(task["output_dir"])
    try:
        summary = json.loads((output / "run_summary.json").read_text())
        hashes = json.loads((output / "structure_hashes.json").read_text())
        return bool(
            summary.get("success")
            and summary.get("basic_structure_valid")
            and (output / "generated_crystals.extxyz").is_file()
            and hashes.get("initial_state_hash")
        )
    except Exception:
        return False


def worker(mode: str, rank: int, workers: int) -> int:
    tasks = _task_specs(mode)[rank::workers]
    failures = 0
    for completed, task in enumerate(tasks, start=1):
        if stop_requested():
            return 130
        if _success(task):
            continue
        output = Path(task["output_dir"])
        output.parent.mkdir(parents=True, exist_ok=True)
        lock_path = output.parent / f".{output.name}.lock"
        with lock_path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            if _success(task):
                continue
            if output.exists():
                failures += 1
                continue
            command = [
                str(PYTHON),
                "-m",
                "research.rp_qtfg.run_sample",
                "--output-dir",
                str(output),
                "--seed",
                str(task["seed"]),
                "--config-id",
                task["config_id"],
                "--physical-gpu",
                str(rank % 8),
            ]
            stdout_path = output.parent / f"{task['seed']}.stdout.log"
            stderr_path = output.parent / f"{task['seed']}.stderr.log"
            with stdout_path.open("a") as stdout, stderr_path.open("a") as stderr:
                result = subprocess.run(
                    command,
                    cwd=PROJECT,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if result.returncode != 0 or not _success(task):
                failures += 1
        atomic_json(
            RESULTS
            / "progress"
            / f"generation_{mode}_worker_{rank}.json",
            {
                "mode": mode,
                "rank": rank,
                "pid": os.getpid(),
                "completed": completed,
                "assigned": len(tasks),
                "failures": failures,
                "updated_at": now(),
            },
        )
    return 1 if failures else 0


def status(mode: str) -> dict[str, Any]:
    tasks = _task_specs(mode)
    success = sum(_success(task) for task in tasks)
    return {
        "mode": mode,
        "total": len(tasks),
        "success": success,
        "remaining": len(tasks) - success,
    }


def launch(mode: str, workers: int = WORKERS) -> int:
    stage = "eight_seed_generation" if mode == "eight" else "thirty_two_generation"
    atomic_json(
        LAUNCHER,
        {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "pgid": os.getpgid(0),
            "sid": os.getsid(0),
            "user": os.environ.get("USER", ""),
            "cwd": os.getcwd(),
            "exe": os.readlink(f"/proc/{os.getpid()}/exe"),
            "command": " ".join(sys.argv),
            "mode": mode,
            "started_at": now(),
        },
    )
    update_progress(**({"eight_seed_started": True} if mode == "eight" else {"eight_seed_started": True, "thirty_two_seed_started": True}))
    set_stage(
        stage,
        "running",
        f"Launching {len(_task_specs(mode))} deterministic generation tasks.",
        {"workers": workers, "gpu_count": 8, "workers_per_gpu": 1},
    )
    processes = []
    streams = []
    log_dir = RESULTS / "generation_worker_logs" / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(workers):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(rank % 8)
        environment["OMP_NUM_THREADS"] = "2"
        environment["MKL_NUM_THREADS"] = "2"
        environment["OPENBLAS_NUM_THREADS"] = "2"
        environment["NUMEXPR_NUM_THREADS"] = "2"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        command = [
            str(PYTHON),
            "-m",
            "research.rp_qtfg.run_generation",
            "worker",
            "--mode",
            mode,
            "--rank",
            str(rank),
            "--workers",
            str(workers),
        ]
        stdout = (log_dir / f"rank_{rank}.stdout.log").open("a")
        stderr = (log_dir / f"rank_{rank}.stderr.log").open("a")
        streams.extend((stdout, stderr))
        processes.append(
            subprocess.Popen(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
            )
        )
    return_codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    current = status(mode)
    atomic_json(
        RESULTS / "progress" / f"generation_{mode}.json",
        {**current, "return_codes": return_codes, "updated_at": now()},
    )
    if current["remaining"]:
        set_stage(
            stage,
            "failed",
            f"Generation incomplete: {current}",
            current,
        )
        return 1
    set_stage(
        stage,
        "success",
        f"Generation complete: {current['success']}/{current['total']}.",
        current,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("launch", "worker", "status"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--mode", choices=("eight", "thirtytwo"), required=True)
        if command in {"launch", "worker"}:
            subparser.add_argument("--workers", type=int, default=WORKERS)
        if command == "worker":
            subparser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        return worker(args.mode, args.rank, args.workers)
    if args.command == "launch":
        return launch(args.mode, args.workers)
    print(json.dumps(status(args.mode), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
