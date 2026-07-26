"""Resume-safe completion of the paired 32-seed A0/P1 generation set."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median

from research.fn_pra.phase1_common import (
    LOGS,
    PROJECT,
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
    set_stage,
)


PYTHON = "/data/dxl/envs/mattergen_py310/bin/python"
EIGHT_ROOT = RESULTS / "generation/eight_seed"
ROOT = RESULTS / "generation/thirty_two_seed"
PROGRESS = RESULTS / "progress/thirty_two_seed_generation.json"
ALL_SEEDS = tuple(range(15000, 15032))
NEW_SEEDS = tuple(range(15008, 15032))
METHODS = ("A0", "P1")
lock = threading.Lock()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def root_for_seed(seed: int) -> Path:
    return EIGHT_ROOT if seed < 15008 else ROOT


def task_dir(method: str, seed: int) -> Path:
    return root_for_seed(seed) / method / f"seed_{seed}"


def task_success(method: str, seed: int) -> bool:
    path = task_dir(method, seed) / "status.json"
    return path.is_file() and read_json(path).get("success") is True


def gpu_for_seed(seed: int) -> int:
    return (seed - 15008) % 8


def save_progress() -> None:
    rows = []
    for seed in ALL_SEEDS:
        for method in METHODS:
            path = task_dir(method, seed)
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "gpu": gpu_for_seed(seed) if seed in NEW_SEEDS else 4 + (seed - 15000) % 4,
                    "reused_from_eight_seed": seed < 15008,
                    "status": (
                        "success"
                        if task_success(method, seed)
                        else "failed"
                        if (path / "status.json").exists()
                        else "pending"
                    ),
                    "output": str(path),
                }
            )
    atomic_json(
        PROGRESS,
        {
            "updated_at": now(),
            "all_seed_count": len(ALL_SEEDS),
            "new_seed_count": len(NEW_SEEDS),
            "total": len(rows),
            "success": sum(row["status"] == "success" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "tasks": rows,
        },
    )


def run_task(method: str, seed: int) -> None:
    if task_success(method, seed):
        return
    output = task_dir(method, seed)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite incomplete task directory: {output}")
    gpu = gpu_for_seed(seed)
    log = LOGS / f"thirty_two_seed_{method.lower()}_{seed}.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    command = [
        PYTHON,
        "-m",
        "research.fn_pra.run_paired_sample",
        "--output-dir",
        str(output),
        "--method",
        method,
        "--seed",
        str(seed),
        "--physical-gpu",
        str(gpu),
        "--sampling-steps",
        "1000",
    ]
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0 or not task_success(method, seed):
        raise RuntimeError(f"{method} seed {seed} failed; see {log}")
    with lock:
        save_progress()


def paired_lane(seed: int) -> None:
    for method in METHODS:
        run_task(method, seed)


def analyze() -> dict:
    rows = {
        method: {
            seed: read_json(task_dir(method, seed) / "summary.json") for seed in ALL_SEEDS
        }
        for method in METHODS
    }
    initial_exact = {
        str(seed): rows["A0"][seed]["initial_hashes"] == rows["P1"][seed]["initial_hashes"]
        for seed in ALL_SEEDS
    }
    rng_exact = {
        str(seed): rows["A0"][seed]["rng_before_prior_hash"]
        == rows["P1"][seed]["rng_before_prior_hash"]
        for seed in ALL_SEEDS
    }
    times = {
        method: [rows[method][seed]["generation_seconds"] for seed in ALL_SEEDS]
        for method in METHODS
    }
    metrics = {
        "created_at": now(),
        "seeds": list(ALL_SEEDS),
        "new_seeds_only": list(NEW_SEEDS),
        "successful_eight_seed_tasks_reused": 16,
        "A0_success": sum(rows["A0"][seed]["success"] for seed in ALL_SEEDS),
        "P1_success": sum(rows["P1"][seed]["success"] for seed in ALL_SEEDS),
        "A0_structure_valid": sum(
            rows["A0"][seed]["structure_valid"] for seed in ALL_SEEDS
        ),
        "P1_structure_valid": sum(
            rows["P1"][seed]["structure_valid"] for seed in ALL_SEEDS
        ),
        "A0_composition_valid": sum(
            rows["A0"][seed]["composition_valid"] for seed in ALL_SEEDS
        ),
        "P1_composition_valid": sum(
            rows["P1"][seed]["composition_valid"] for seed in ALL_SEEDS
        ),
        "initial_hash_exact_all": all(initial_exact.values()),
        "rng_hash_exact_all": all(rng_exact.values()),
        "A0_median_generation_seconds": median(times["A0"]),
        "P1_median_generation_seconds": median(times["P1"]),
        "P1_median_overhead_fraction": median(times["P1"]) / median(times["A0"]) - 1.0,
        "teacher_used_at_p1_inference": any(
            rows["P1"][seed]["teacher_used_at_inference"] for seed in ALL_SEEDS
        ),
        "projection_heads_loaded_at_p1_inference": any(
            rows["P1"][seed]["projection_heads_loaded_at_inference"] for seed in ALL_SEEDS
        ),
    }
    metrics["passed_generation_gate"] = bool(
        metrics["A0_success"] == 32
        and metrics["P1_success"] == 32
        and metrics["initial_hash_exact_all"]
        and metrics["rng_hash_exact_all"]
        and not metrics["teacher_used_at_p1_inference"]
        and not metrics["projection_heads_loaded_at_p1_inference"]
    )
    atomic_json(REPORTS / "thirty_two_seed_generation.json", metrics)
    atomic_text(
        REPORTS / "thirty_two_seed_generation.md",
        f"""# FN-PRA 32-seed paired generation

- New seeds generated: 15008-15031; successful 15000-15007 tasks were reused.
- A0/P1 success: {metrics["A0_success"]}/32 and {metrics["P1_success"]}/32.
- A0/P1 structure validity: {metrics["A0_structure_valid"]}/32 and {metrics["P1_structure_valid"]}/32.
- A0/P1 composition validity: {metrics["A0_composition_valid"]}/32 and {metrics["P1_composition_valid"]}/32.
- Initial state and RNG pairing exact: {metrics["initial_hash_exact_all"]} / {metrics["rng_hash_exact_all"]}.
- P1 median generation overhead: {100 * metrics["P1_median_overhead_fraction"]:.3f}%.
- Generation gate passed: {metrics["passed_generation_gate"]}.
""",
    )
    return metrics


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not all(task_success(method, seed) for seed in range(15000, 15008) for method in METHODS):
        raise RuntimeError("eight-seed tasks are incomplete; refusing to start 32-seed completion")
    set_stage(
        "thirty_two_seed_generation",
        "running",
        "Generating only new seeds 15008-15031 with paired A0/P1 GPU mapping.",
        {"new_seeds": list(NEW_SEEDS), "max_workers": 24, "max_workers_per_gpu": 3},
    )
    save_progress()
    errors = []
    with ThreadPoolExecutor(max_workers=len(NEW_SEEDS)) as executor:
        futures = [executor.submit(paired_lane, seed) for seed in NEW_SEEDS]
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                errors.append(str(error))
    if errors:
        set_stage(
            "thirty_two_seed_generation",
            "failed",
            f"32-seed completion failed in {len(errors)} paired lanes.",
            {"errors": errors},
        )
        return 1
    metrics = analyze()
    set_stage(
        "thirty_two_seed_generation",
        "success" if metrics["passed_generation_gate"] else "failed",
        f"Paired 32-seed generation completed; gate passed={metrics['passed_generation_gate']}.",
        metrics,
    )
    return 0 if metrics["passed_generation_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
