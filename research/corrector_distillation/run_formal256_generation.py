"""Run immutable C0/V2 pairs in fixed H20 seed shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from research.corrector_distillation.formal256_protocol import (
    ASSET_PATHS,
    EXPERIMENT_ROOT,
    METHODS,
    SINGLE_H20_SEEDS,
    benchmark_command,
    fixed_shards,
    validate_frozen_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("main", "single-h20"), default="main"
    )
    return parser.parse_args()


def run_wave(
    *, output_root: Path, label: str, assignments: list[tuple[int, int]], logs_dir: Path
) -> tuple[float, list[dict[str, object]]]:
    running: list[tuple[int, int, subprocess.Popen, object, Path, float]] = []
    events: list[dict[str, object]] = []
    wave_started = time.perf_counter()
    for gpu, seed in assignments:
        run_dir = output_root / "generation" / label / str(seed)
        summary_path = run_dir / "run_summary.json"
        if summary_path.is_file():
            events.append(
                {"gpu": gpu, "seed": seed, "label": label, "preexisting": True, "return_code": 0}
            )
            continue
        if run_dir.exists():
            raise RuntimeError(
                f"incomplete formal run exists at {run_dir}; preserve and inspect it before retry"
            )
        command = benchmark_command(
            checkpoint_root=ASSET_PATHS["mattergen_checkpoint"].parents[1],
            output_root=output_root,
            label=label,
            seed=seed,
        )
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
        event = {
            "gpu": gpu,
            "seed": seed,
            "label": label,
            "preexisting": False,
            "return_code": return_code,
            "process_wall_seconds": time.perf_counter() - started,
            "log_path": str(log_path.resolve()),
        }
        events.append(event)
    wall = time.perf_counter() - wave_started
    failures = [event for event in events if int(event["return_code"]) != 0]
    print(
        json.dumps(
            {
                "event": "paired_wave_complete",
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
        raise RuntimeError(f"formal generation failures: {failures}")
    return wall, events


def main() -> None:
    args = parse_args()
    preflight = validate_frozen_protocol()
    if not preflight["passed"]:
        raise RuntimeError(f"formal preflight failed: {preflight['failures']}")
    if args.mode == "main":
        output_root = EXPERIMENT_ROOT
        logs_dir = output_root / "logs/generation"
        waves = [
            [(gpu, fixed_shards()[gpu][offset]) for gpu in range(8)]
            for offset in range(32)
        ]
        summary_path = output_root / "eight_h20_throughput.json"
        expected_gpu_count = 8
    else:
        output_root = EXPERIMENT_ROOT / "single_h20"
        logs_dir = EXPERIMENT_ROOT / "logs/single_h20"
        waves = [[(0, seed)] for seed in SINGLE_H20_SEEDS]
        summary_path = output_root / "throughput_raw.json"
        expected_gpu_count = 1
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        raise FileExistsError(summary_path)

    started = time.perf_counter()
    arm_wall = {label: 0.0 for label in METHODS}
    events: list[dict[str, object]] = []
    # Every fixed-shard seed runs C0 immediately followed by V2 in the next
    # global wave. No other arm is permitted.
    for wave_index, assignments in enumerate(waves):
        for label in METHODS:
            wall, wave_events = run_wave(
                output_root=output_root,
                label=label,
                assignments=assignments,
                logs_dir=logs_dir,
            )
            arm_wall[label] += wall
            events.extend({"wave_index": wave_index, **event} for event in wave_events)
    n = 256 if args.mode == "main" else 16
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "protocol": "paired global waves: each fixed-GPU seed C0 then V2",
        "gpu_count": expected_gpu_count,
        "batch_size": 1,
        "seed_count_per_method": n,
        "total_wall_seconds": time.perf_counter() - started,
        "per_method_active_wall_seconds": arm_wall,
        "per_method_samples_per_hour": {
            label: 3600.0 * n / arm_wall[label] for label in METHODS
        },
        "algorithmic_speedup_warning": (
            "8-GPU throughput is a system measure and is not the algorithmic speedup."
            if args.mode == "main"
            else "Use paired per-seed time ratios for algorithmic speedup."
        ),
        "events": events,
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
