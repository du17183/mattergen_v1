from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from research.crystalrepa_repro.common import LOGS, REPORTS, RESULTS, atomic_json, now, set_stage

PYTHON = Path("/data/dxl/envs/mattergen_py310/bin/python")
PROJECT = Path("/data/dxl/mattergen_v1")
GENERATION = RESULTS / "generation"
SEEDS = tuple(range(17000, 17064))
METHODS = ("U0", "R1")


def output_dir(method: str, seed: int, repeat: bool = False) -> Path:
    root = RESULTS / "determinism_repeats" if repeat else GENERATION
    return root / method / f"seed_{seed}"


def valid(path: Path) -> bool:
    try:
        status = json.loads((path / "status.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        return status.get("success") is True and summary.get("success") is True and (path / "generated_crystals.extxyz").is_file()
    except Exception:
        return False


def launch_task(method: str, seed: int, repeat: bool = False) -> dict:
    destination = output_dir(method, seed, repeat)
    if valid(destination):
        return {"method": method, "seed": seed, "success": True, "skipped_resume": True}
    gpu = (seed - 17000) % 8
    destination.mkdir(parents=True, exist_ok=True)
    log_dir = LOGS / ("determinism" if repeat else "generation")
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    command = [
        str(PYTHON), "-m", "research.crystalrepa_repro.run_paired_sample",
        "--output-dir", str(destination), "--method", method, "--seed", str(seed),
        "--physical-gpu", str(gpu), "--repeat-index", "2" if repeat else "1",
    ]
    started = time.monotonic()
    with (log_dir / f"{method}_{seed}{'_repeat' if repeat else ''}.log").open("a") as stream:
        result = subprocess.run(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return {
        "method": method, "seed": seed, "gpu": gpu, "return_code": result.returncode,
        "elapsed_seconds": time.monotonic() - started, "success": result.returncode == 0 and valid(destination),
        "skipped_resume": False,
    }


def run_tasks(seeds: tuple[int, ...], workers: int = 32) -> list[dict]:
    tasks = [(method, seed) for method in METHODS for seed in seeds]
    records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(launch_task, method, seed): (method, seed) for method, seed in tasks}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            if not record["success"]:
                raise RuntimeError(f"Generation failed: {record}")
    return records


def paired_audit(seeds: tuple[int, ...]) -> dict:
    rows = []
    for seed in seeds:
        u0 = json.loads((output_dir("U0", seed) / "summary.json").read_text())
        r1 = json.loads((output_dir("R1", seed) / "summary.json").read_text())
        rows.append({
            "seed": seed,
            "initial_hash_match": u0["initial_hashes"]["combined"] == r1["initial_hashes"]["combined"],
            "rng_hash_match": u0["rng_before_prior_hash"] == r1["rng_before_prior_hash"],
            "gpu_match": u0["physical_gpu"] == r1["physical_gpu"],
        })
    passed = all(all(value for key, value in row.items() if key != "seed") for row in rows)
    return {"rows": rows, "passed": passed}


def run_eight() -> None:
    seeds = SEEDS[:8]
    set_stage("eight_seed_smoke", "running", "Launching paired U0/R1 8-seed smoke at fixed GPU mapping.", {"seeds": list(seeds)})
    records = run_tasks(seeds, workers=16)
    pairing = paired_audit(seeds)
    repeats = [launch_task(method, 17000, repeat=True) for method in METHODS]
    determinism = {}
    for method in METHODS:
        original = json.loads((output_dir(method, 17000) / "summary.json").read_text())
        repeated = json.loads((output_dir(method, 17000, True) / "summary.json").read_text())
        determinism[method] = {
            "initial_hash": original["initial_hashes"]["combined"] == repeated["initial_hashes"]["combined"],
            "final_hash": original["final_structure_hash"] == repeated["final_structure_hash"],
            "extxyz_hash": original["extxyz_sha256"] == repeated["extxyz_sha256"],
        }
    passed = pairing["passed"] and all(all(values.values()) for values in determinism.values())
    report = {"created_at": now(), "tasks": records, "pairing": pairing, "repeat_tasks": repeats, "determinism": determinism, "passed": passed}
    atomic_json(REPORTS / "eight_seed_generation.json", report)
    set_stage("eight_seed_smoke", "success" if passed else "failed", f"8-seed smoke passed={passed}.", report)
    if not passed:
        raise RuntimeError("8-seed pairing/determinism failed")


def run_full() -> None:
    set_stage("sixty_four_generation", "running", "Launching/resuming 128 paired generation tasks with 4 workers/GPU.", {"seeds": [17000, 17063], "tasks": 128})
    started = time.monotonic()
    records = run_tasks(SEEDS, workers=32)
    pairing = paired_audit(SEEDS)
    summaries = {method: [json.loads((output_dir(method, seed) / "summary.json").read_text()) for seed in SEEDS] for method in METHODS}
    report = {
        "created_at": now(), "elapsed_seconds": time.monotonic() - started,
        "tasks": records, "pairing": pairing,
        "success": {method: sum(item["success"] for item in rows) for method, rows in summaries.items()},
        "composition_valid": {method: sum(item["composition_valid"] for item in rows) for method, rows in summaries.items()},
        "structure_valid": {method: sum(item["structure_valid"] for item in rows) for method, rows in summaries.items()},
        "teacher_used_at_inference": any(item["teacher_used_at_inference"] for rows in summaries.values() for item in rows),
        "projection_loaded_at_inference": any(item["projection_loaded_at_inference"] for rows in summaries.values() for item in rows),
        "passed": pairing["passed"] and all(len(rows) == 64 and all(item["success"] for item in rows) for rows in summaries.values()),
    }
    atomic_json(REPORTS / "sixty_four_generation.json", report)
    set_stage("sixty_four_generation", "success" if report["passed"] else "failed", f"Paired generation complete; passed={report['passed']}.", report)
    if not report["passed"]:
        raise RuntimeError("64-seed generation audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("eight", "full"))
    args = parser.parse_args()
    run_eight() if args.stage == "eight" else run_full()


if __name__ == "__main__":
    main()
