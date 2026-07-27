"""Launch the 256-structure C0/A0 B1/B4 quality generation matrix."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from research.spg_fastgate.common import (
    LOGS,
    PROJECT,
    PYTHON,
    RESULTS,
    STOP_MARKER,
    atomic_json,
    base_environment,
    now,
    set_stage,
)


CONFIGS = (("C0", 1), ("C0", 4), ("A0", 1), ("A0", 4))
SEEDS = tuple(range(24064, 24128))
OUTPUT = RESULTS / "quality_generation"
PROGRESS = RESULTS / "progress/quality_generation.json"


def read_status(method: str, batch_size: int, seed: int) -> str:
    path = OUTPUT / method / f"B{batch_size}" / f"seed_{seed}" / "status.json"
    if not path.is_file():
        return "pending"
    return "success" if json.loads(path.read_text(encoding="utf-8")).get("success") else "failed"


def save_progress(active: str | None = None) -> dict:
    tasks = [
        {
            "method": method,
            "batch_size": batch_size,
            "seed": seed,
            "gpu": (seed - SEEDS[0]) % 8,
            "status": read_status(method, batch_size, seed),
        }
        for method, batch_size in CONFIGS
        for seed in SEEDS
    ]
    state = {
        "updated_at": now(),
        "active_config": active,
        "total": len(tasks),
        "success": sum(task["status"] == "success" for task in tasks),
        "failed": sum(task["status"] == "failed" for task in tasks),
        "pending": sum(task["status"] == "pending" for task in tasks),
        "tasks": tasks,
    }
    atomic_json(PROGRESS, state)
    return state


def run_config(method: str, batch_size: int) -> None:
    label = f"{method}-B{batch_size}"
    save_progress(label)
    processes: list[tuple[subprocess.Popen, object]] = []
    for gpu in range(8):
        log = LOGS / "quality_generation" / f"{method.lower()}_b{batch_size}_gpu{gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [
                str(PYTHON),
                "-m",
                "research.spg_fastgate.run_quality_worker",
                "--method",
                method,
                "--batch-size",
                str(batch_size),
                "--physical-gpu",
                str(gpu),
            ],
            cwd=PROJECT,
            env=base_environment(gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((process, stream))
    return_codes = []
    for process, stream in processes:
        return_codes.append(process.wait())
        stream.close()
    save_progress(label)
    if any(return_code != 0 for return_code in return_codes):
        raise RuntimeError(f"{label} worker failure: {return_codes}")
    missing = [seed for seed in SEEDS if read_status(method, batch_size, seed) != "success"]
    if missing:
        raise RuntimeError(f"{label} incomplete seeds: {missing}")


def main() -> int:
    set_stage(
        "b4_generation",
        "running",
        "Running the C0/A0 B1/B4 64-seed quality generation matrix.",
        {"seeds": list(SEEDS), "tasks": 256, "gpu_count": 8},
    )
    save_progress()
    try:
        for method, batch_size in CONFIGS:
            if STOP_MARKER.exists():
                raise RuntimeError("STOP_REQUESTED observed before starting next generation config")
            if all(read_status(method, batch_size, seed) == "success" for seed in SEEDS):
                continue
            run_config(method, batch_size)
    except BaseException as error:
        set_stage(
            "b4_generation",
            "failed",
            f"Quality generation stopped: {error}",
            save_progress(),
        )
        return 1
    state = save_progress()
    set_stage(
        "b4_generation",
        "success",
        "C0/A0 B1/B4 quality generation completed 256/256.",
        state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
