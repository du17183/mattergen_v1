"""Run a small configured benchmark matrix with seed-level GPU parallelism."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--only-labels", nargs="*")
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def load_matrix(path: Path) -> list[dict[str, Any]]:
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
        payload = json.load(stream)
    methods = payload.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("matrix must contain a non-empty methods list")
    labels = [str(item["label"]) for item in methods]
    if len(labels) != len(set(labels)):
        raise ValueError("matrix labels must be unique")
    if any(str(item.get("method")) == "A0+Adapter+Fallback" for item in methods):
        raise ValueError("V2 explicitly prohibits A0 + Adapter combinations")
    return methods


def command_for(
    entry: dict[str, Any],
    *,
    checkpoint_root: Path,
    output_root: Path,
    seed: int,
) -> list[str]:
    method = str(entry["method"])
    command = [
        sys.executable,
        "-m",
        "research.corrector_distillation.benchmark_sampler",
        "--checkpoint-root",
        str(checkpoint_root),
        "--output-root",
        str(output_root),
        "--method",
        method,
        "--label",
        str(entry["label"]),
        "--seed",
        str(seed),
    ]
    optional = {
        "adapter_checkpoint": "--adapter-checkpoint",
        "coverage": "--coverage",
        "risk_mode": "--risk-mode",
        "risk_fields": "--risk-fields",
        "early_reuse_end": "--early-reuse-end",
        "late_exact_start": "--late-exact-start",
    }
    for key, flag in optional.items():
        value = entry.get(key)
        if value is None:
            continue
        if key == "risk_fields" and isinstance(value, list):
            value = ",".join(str(item) for item in value)
        command.extend((flag, str(value)))
    return command


def main() -> None:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("seed range is empty")
    if args.num_gpus <= 0:
        raise ValueError("num-gpus must be positive")
    project_root = Path(__file__).resolve().parents[2]
    output_root = args.output_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    methods = load_matrix(args.matrix)
    if args.only_labels:
        selected = set(args.only_labels)
        methods = [item for item in methods if str(item["label"]) in selected]
        missing = selected - {str(item["label"]) for item in methods}
        if missing:
            raise ValueError(f"unknown labels: {sorted(missing)}")
    logs_dir = output_root / "logs" / "generation"
    runtime_tmp = output_root / "runtime_tmp"
    incomplete_root = output_root / "generation" / "_incomplete"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    incomplete_root.mkdir(parents=True, exist_ok=True)
    method_summaries = []
    all_failures = []
    matrix_started = time.perf_counter()

    # Complete one method before starting the next so per-method wall throughput
    # is not contaminated by another arm competing for GPU or filesystem I/O.
    for entry in methods:
        label = str(entry["label"])
        jobs = []
        preexisting = 0
        for seed in range(args.seed_start, args.seed_end + 1):
            run_dir = output_root / "generation" / label / str(seed)
            if (run_dir / "run_summary.json").is_file():
                preexisting += 1
                continue
            if run_dir.exists():
                destination = incomplete_root / label / f"{seed}-{time.time_ns()}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(run_dir, destination)
            jobs.append(seed)
        print(
            json.dumps(
                {"event": "method_start", "label": label, "jobs": len(jobs),
                 "preexisting": preexisting},
                sort_keys=True,
            ),
            flush=True,
        )
        running: dict[int, tuple[subprocess.Popen, object, int, float, Path]] = {}
        failures = []
        started = time.perf_counter()
        while jobs or running:
            for gpu in range(args.num_gpus):
                if gpu in running or not jobs:
                    continue
                seed = jobs.pop(0)
                command = command_for(
                    entry,
                    checkpoint_root=checkpoint_root,
                    output_root=output_root,
                    seed=seed,
                )
                log_path = logs_dir / f"{label.replace('+', '_plus_')}__{seed}.log"
                log_stream = log_path.open("w", encoding="utf-8")
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                environment["TMPDIR"] = str(runtime_tmp)
                process = subprocess.Popen(
                    command,
                    cwd=project_root,
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                )
                running[gpu] = (
                    process, log_stream, seed, time.perf_counter(), log_path
                )
            time.sleep(1.0)
            for gpu, value in list(running.items()):
                process, log_stream, seed, job_started, log_path = value
                return_code = process.poll()
                if return_code is None:
                    continue
                log_stream.close()
                del running[gpu]
                event = {
                    "event": "job_complete",
                    "gpu": gpu,
                    "label": label,
                    "seed": seed,
                    "return_code": return_code,
                    "remaining": len(jobs) + len(running),
                }
                print(json.dumps(event, sort_keys=True), flush=True)
                if return_code != 0:
                    failure = {
                        **event,
                        "success": False,
                        "process_elapsed_seconds": time.perf_counter() - job_started,
                        "log_path": str(log_path.resolve()),
                    }
                    failure_path = (
                        output_root / "generation" / label / str(seed)
                        / "failure_summary.json"
                    )
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    with failure_path.open("x", encoding="utf-8") as stream:
                        json.dump(failure, stream, indent=2, sort_keys=True)
                        stream.write("\n")
                    failures.append(failure)
        wall_seconds = time.perf_counter() - started
        completed = sum(
            (output_root / "generation" / label / str(seed) / "run_summary.json").is_file()
            for seed in range(args.seed_start, args.seed_end + 1)
        )
        newly_completed = completed - preexisting
        method_summary = {
            "label": label,
            "method": entry["method"],
            "seed_start": args.seed_start,
            "seed_end": args.seed_end,
            "preexisting_runs": preexisting,
            "newly_completed_runs": newly_completed,
            "completed_runs": completed,
            "failed_runs": len(failures),
            "wall_seconds_for_new_runs": wall_seconds,
            "new_samples_per_hour_8gpu": (
                3600.0 * newly_completed / wall_seconds if newly_completed else None
            ),
        }
        method_summaries.append(method_summary)
        all_failures.extend(failures)
        print(json.dumps({"event": "method_end", **method_summary}, sort_keys=True), flush=True)
        if failures:
            raise RuntimeError(f"generation failures for {label}: {failures}")

    summary = {
        "schema_version": 1,
        "matrix": str(args.matrix.expanduser().resolve()),
        "checkpoint_root": str(checkpoint_root),
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "num_gpus": args.num_gpus,
        "elapsed_seconds": time.perf_counter() - matrix_started,
        "methods": method_summaries,
        "failures": all_failures,
    }
    summary_path = (
        args.summary.expanduser().resolve()
        if args.summary is not None
        else output_root / "matrix_throughput.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
