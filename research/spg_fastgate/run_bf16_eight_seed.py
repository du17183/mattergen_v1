"""Run the 32-task C0/A0 FP32/field-safe-BF16 endpoint generation matrix."""

from __future__ import annotations

import json
import subprocess

from research.spg_fastgate.common import (
    LOGS,
    PROJECT,
    PYTHON,
    RESULTS,
    atomic_json,
    base_environment,
    now,
    read_json,
    set_stage,
)


CONFIGS = (
    ("C0", "FP32"),
    ("C0", "FIELD_SAFE_BF16"),
    ("A0", "FP32"),
    ("A0", "FIELD_SAFE_BF16"),
)
SEEDS = tuple(range(24128, 24136))
PROGRESS = RESULTS / "progress/bf16_eight_seed.json"


def task_status(method: str, precision: str, seed: int) -> str:
    path = (
        RESULTS
        / "bf16_generation"
        / precision
        / method
        / "B1"
        / f"seed_{seed}"
        / "status.json"
    )
    if not path.is_file():
        return "pending"
    return "success" if json.loads(path.read_text(encoding="utf-8")).get("success") else "failed"


def save_progress(active: str | None = None) -> dict:
    tasks = [
        {
            "method": method,
            "precision": precision,
            "seed": seed,
            "gpu": seed - SEEDS[0],
            "status": task_status(method, precision, seed),
        }
        for method, precision in CONFIGS
        for seed in SEEDS
    ]
    state = {
        "updated_at": now(),
        "active_config": active,
        "total": len(tasks),
        "success": sum(task["status"] == "success" for task in tasks),
        "failed": sum(task["status"] == "failed" for task in tasks),
        "tasks": tasks,
    }
    atomic_json(PROGRESS, state)
    return state


def run_config(method: str, precision: str) -> None:
    processes = []
    for gpu in range(8):
        seed = SEEDS[gpu]
        if task_status(method, precision, seed) == "success":
            continue
        log = LOGS / "bf16_eight_seed" / f"{method.lower()}_{precision.lower()}_{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [
                str(PYTHON),
                "-m",
                "research.spg_fastgate.run_bf16_worker",
                "--method",
                method,
                "--precision",
                precision,
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
    codes = []
    for process, stream in processes:
        codes.append(process.wait())
        stream.close()
    if any(code != 0 for code in codes):
        raise RuntimeError(f"{method}/{precision} BF16 endpoint workers failed: {codes}")


def main() -> int:
    state_probe = read_json(RESULTS / "bf16_state_probe.json", {})
    if not state_probe.get("FIELD_SAFE_BF16_STATE_GO"):
        set_stage(
            "bf16_eight_seed",
            "success",
            "Skipped endpoint BF16 expansion because the frozen-state gate failed.",
            {"skipped": True, "FIELD_SAFE_BF16_GO": False},
        )
        return 0
    set_stage(
        "bf16_eight_seed",
        "running",
        "Running C0/A0 FP32 and field-safe-BF16 on seeds 24128-24135.",
        {"tasks": 32},
    )
    for method, precision in CONFIGS:
        run_config(method, precision)
        save_progress(f"{method}/{precision}")
    state = save_progress()
    if state["success"] != 32:
        raise RuntimeError(f"BF16 endpoint generation incomplete: {state}")
    # MatterSim evaluation is performed together with the B4 quality matrix.
    set_stage(
        "bf16_eight_seed",
        "running",
        "BF16 generation 32/32 complete; MatterSim endpoint evaluation pending.",
        state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
