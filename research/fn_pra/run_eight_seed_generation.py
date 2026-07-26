"""Resume-safe paired A0/P1 generation on the eight Phase-1 smoke seeds."""

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
ROOT = RESULTS / "generation/eight_seed"
PROGRESS = RESULTS / "progress/eight_seed_generation.json"
SEEDS = tuple(range(15000, 15008))
METHODS = ("A0", "P1")
GPU_BY_OFFSET = (4, 5, 6, 7)
lock = threading.Lock()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def task_dir(method: str, seed: int, repeat: bool = False) -> Path:
    suffix = "_repeat" if repeat else ""
    return ROOT / method / f"seed_{seed}{suffix}"


def task_success(method: str, seed: int, repeat: bool = False) -> bool:
    path = task_dir(method, seed, repeat) / "status.json"
    return path.is_file() and read_json(path).get("success") is True


def save_progress() -> None:
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            path = task_dir(method, seed)
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "gpu": GPU_BY_OFFSET[(seed - SEEDS[0]) % len(GPU_BY_OFFSET)],
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
            "total": len(rows),
            "success": sum(row["status"] == "success" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "tasks": rows,
        },
    )


def run_task(method: str, seed: int, gpu: int, repeat: bool = False) -> None:
    if task_success(method, seed, repeat):
        return
    output = task_dir(method, seed, repeat)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite incomplete task directory: {output}")
    log = LOGS / f"eight_seed_{method.lower()}_{seed}{'_repeat' if repeat else ''}.log"
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
        "--repeat-index",
        "2" if repeat else "1",
    ]
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0 or not task_success(method, seed, repeat):
        raise RuntimeError(f"{method} seed {seed} failed; see {log}")
    with lock:
        save_progress()


def gpu_lane(offset: int) -> None:
    gpu = GPU_BY_OFFSET[offset]
    for seed in SEEDS[offset:: len(GPU_BY_OFFSET)]:
        for method in METHODS:
            run_task(method, seed, gpu)


def analyze() -> dict:
    rows = {
        method: {
            seed: read_json(task_dir(method, seed) / "summary.json") for seed in SEEDS
        }
        for method in METHODS
    }
    initial_exact = {
        str(seed): rows["A0"][seed]["initial_hashes"] == rows["P1"][seed]["initial_hashes"]
        for seed in SEEDS
    }
    rng_exact = {
        str(seed): rows["A0"][seed]["rng_before_prior_hash"]
        == rows["P1"][seed]["rng_before_prior_hash"]
        for seed in SEEDS
    }
    generation_times = {
        method: [rows[method][seed]["generation_seconds"] for seed in SEEDS]
        for method in METHODS
    }
    repeat_checks = {}
    for method in METHODS:
        original = rows[method][SEEDS[0]]
        repeated = read_json(task_dir(method, SEEDS[0], True) / "summary.json")
        repeat_checks[method] = {
            "rng_level1": original["rng_before_prior_hash"]
            == repeated["rng_before_prior_hash"],
            "initial_level1": original["initial_hashes"] == repeated["initial_hashes"],
            "final_level1": original["final_structure_hash"]
            == repeated["final_structure_hash"],
        }
    p1_overhead = median(generation_times["P1"]) / median(generation_times["A0"]) - 1.0
    metrics = {
        "created_at": now(),
        "seeds": list(SEEDS),
        "A0_success": sum(rows["A0"][seed]["success"] for seed in SEEDS),
        "P1_success": sum(rows["P1"][seed]["success"] for seed in SEEDS),
        "A0_structure_valid": sum(rows["A0"][seed]["structure_valid"] for seed in SEEDS),
        "P1_structure_valid": sum(rows["P1"][seed]["structure_valid"] for seed in SEEDS),
        "A0_composition_valid": sum(rows["A0"][seed]["composition_valid"] for seed in SEEDS),
        "P1_composition_valid": sum(rows["P1"][seed]["composition_valid"] for seed in SEEDS),
        "initial_hash_exact_all": all(initial_exact.values()),
        "initial_hash_exact_by_seed": initial_exact,
        "rng_hash_exact_all": all(rng_exact.values()),
        "rng_hash_exact_by_seed": rng_exact,
        "determinism_repeat": repeat_checks,
        "A0_median_generation_seconds": median(generation_times["A0"]),
        "P1_median_generation_seconds": median(generation_times["P1"]),
        "P1_median_overhead_fraction": p1_overhead,
        "teacher_used_at_p1_inference": any(
            rows["P1"][seed]["teacher_used_at_inference"] for seed in SEEDS
        ),
        "projection_heads_loaded_at_p1_inference": any(
            rows["P1"][seed]["projection_heads_loaded_at_inference"] for seed in SEEDS
        ),
    }
    metrics["passed"] = bool(
        metrics["A0_success"] == 8
        and metrics["P1_success"] == 8
        and metrics["A0_structure_valid"] == 8
        and metrics["P1_structure_valid"] == 8
        and metrics["initial_hash_exact_all"]
        and metrics["rng_hash_exact_all"]
        and all(
            all(check.values()) for check in metrics["determinism_repeat"].values()
        )
        and metrics["P1_median_overhead_fraction"] <= 0.05
        and not metrics["teacher_used_at_p1_inference"]
        and not metrics["projection_heads_loaded_at_p1_inference"]
    )
    atomic_json(REPORTS / "eight_seed_generation.json", metrics)
    atomic_text(
        REPORTS / "eight_seed_generation.md",
        f"""# FN-PRA eight-seed smoke

- A0/P1 generation success: {metrics["A0_success"]}/8 and {metrics["P1_success"]}/8.
- A0/P1 structure validity: {metrics["A0_structure_valid"]}/8 and {metrics["P1_structure_valid"]}/8.
- A0/P1 composition validity: {metrics["A0_composition_valid"]}/8 and {metrics["P1_composition_valid"]}/8.
- Initial state and RNG hashes paired exactly: {metrics["initial_hash_exact_all"]} / {metrics["rng_hash_exact_all"]}.
- A0/P1 median generation time: {metrics["A0_median_generation_seconds"]:.3f}s / {metrics["P1_median_generation_seconds"]:.3f}s.
- P1 median overhead: {100 * metrics["P1_median_overhead_fraction"]:.3f}%.
- Teacher or projection heads present in P1 inference: {metrics["teacher_used_at_p1_inference"]} / {metrics["projection_heads_loaded_at_p1_inference"]}.
- Smoke gate passed: {metrics["passed"]}.
""",
    )
    return metrics


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    set_stage(
        "eight_seed_generation",
        "running",
        "Running paired A0/P1 generation for seeds 15000-15007 on GPUs 4-7.",
        {"seeds": list(SEEDS), "gpu_mapping": list(GPU_BY_OFFSET)},
    )
    save_progress()
    errors = []
    with ThreadPoolExecutor(max_workers=len(GPU_BY_OFFSET)) as executor:
        futures = [executor.submit(gpu_lane, offset) for offset in range(len(GPU_BY_OFFSET))]
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                errors.append(str(error))
    if errors:
        set_stage(
            "eight_seed_generation",
            "failed",
            f"Eight-seed generation failed in {len(errors)} GPU lanes.",
            {"errors": errors},
        )
        return 1
    for method in METHODS:
        run_task(method, SEEDS[0], GPU_BY_OFFSET[0], repeat=True)
    metrics = analyze()
    set_stage(
        "eight_seed_generation",
        "success" if metrics["passed"] else "failed",
        f"Eight-seed A0/P1 smoke completed; gate passed={metrics['passed']}.",
        metrics,
    )
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
