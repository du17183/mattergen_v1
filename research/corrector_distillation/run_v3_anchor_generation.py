"""Run V3 smoke, Stage-B/Stage-C, and strict single-H20 generation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from research.corrector_distillation.v3_anchor_protocol import (
    EXPERIMENT_ROOT,
    benchmark_command,
    fixed_shards,
    methods_for,
    output_root_for,
    seeds_for,
    validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("smoke", "stage-b", "single-h20-b", "stage-c", "single-h20-c"),
        required=True,
    )
    return parser.parse_args()


def run_wave(
    *,
    output_root: Path,
    label: str,
    assignments: list[tuple[int, int]],
    logs_dir: Path,
) -> tuple[float, list[dict[str, object]]]:
    running: list[tuple[int, int, subprocess.Popen, object, Path, float]] = []
    events: list[dict[str, object]] = []
    wave_started = time.perf_counter()
    for gpu, seed in assignments:
        run_dir = output_root / "generation" / label / str(seed)
        summary_path = run_dir / "run_summary.json"
        if summary_path.is_file():
            events.append(
                {
                    "gpu": gpu,
                    "seed": seed,
                    "label": label,
                    "preexisting": True,
                    "return_code": 0,
                }
            )
            continue
        if run_dir.exists():
            raise RuntimeError(f"incomplete run exists at {run_dir}; inspect before retry")
        command = benchmark_command(output_root=output_root, label=label, seed=seed)
        log_path = logs_dir / f"{label}__{seed}.log"
        log_stream = log_path.open("x", encoding="utf-8")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["TMPDIR"] = str(EXPERIMENT_ROOT.parent.parent / ".tmp")
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        running.append((gpu, seed, process, log_stream, log_path, started))
    for gpu, seed, process, log_stream, log_path, started in running:
        return_code = process.wait()
        log_stream.close()
        events.append(
            {
                "gpu": gpu,
                "seed": seed,
                "label": label,
                "preexisting": False,
                "return_code": return_code,
                "process_wall_seconds": time.perf_counter() - started,
                "log_path": str(log_path.resolve()),
            }
        )
    wall = time.perf_counter() - wave_started
    failures = [event for event in events if int(event["return_code"]) != 0]
    print(
        json.dumps(
            {
                "event": "wave_complete",
                "label": label,
                "wall_seconds": wall,
                "assignments": assignments,
                "failures": failures,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if failures:
        raise RuntimeError(f"generation failures: {failures}")
    return wall, events


def main() -> None:
    args = parse_args()
    preflight = validate_protocol()
    if not preflight["passed"]:
        raise RuntimeError(f"V3 preflight failed: {preflight['failures']}")
    methods = methods_for(args.stage)
    seeds = seeds_for(args.stage)
    output_root = output_root_for(args.stage)
    logs_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "throughput_raw.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    if args.stage == "smoke":
        waves = [[(gpu, seed) for gpu, seed in enumerate(seeds)]]
        gpu_count = len(seeds)
    elif args.stage.startswith("single-h20"):
        waves = [[(0, seed)] for seed in seeds]
        gpu_count = 1
    else:
        shards = fixed_shards(seeds)
        waves = [
            [(gpu, shards[gpu][offset]) for gpu in range(8)]
            for offset in range(len(shards[0]))
        ]
        gpu_count = 8
    started = time.perf_counter()
    method_wall = {method: 0.0 for method in methods}
    events: list[dict[str, object]] = []
    for wave_index, assignments in enumerate(waves):
        for method in methods:
            wall, wave_events = run_wave(
                output_root=output_root,
                label=method,
                assignments=assignments,
                logs_dir=logs_dir,
            )
            method_wall[method] += wall
            events.extend(
                {"wave_index": wave_index, **event} for event in wave_events
            )
    new_counts = {
        method: sum(
            event["label"] == method and not event["preexisting"]
            for event in events
        )
        for method in methods
    }
    summary = {
        "schema_version": 1,
        "stage": args.stage,
        "protocol": "paired global waves; methods run consecutively for each seed wave",
        "gpu_count": gpu_count,
        "batch_size": 1,
        "seeds": list(seeds),
        "methods": list(methods),
        "seed_count_per_method": len(seeds),
        "total_wall_seconds": time.perf_counter() - started,
        "per_method_active_wall_seconds_new_work": method_wall,
        "per_method_new_samples": new_counts,
        "per_method_samples_per_hour_on_new_work": {
            method: (
                3600.0 * new_counts[method] / method_wall[method]
                if new_counts[method] else None
            )
            for method in methods
        },
        "algorithmic_speedup_warning": (
            "Only single-h20 stages estimate algorithmic speedup."
            if gpu_count != 1
            else "Use paired per-seed elapsed-time ratios."
        ),
        "events": events,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
