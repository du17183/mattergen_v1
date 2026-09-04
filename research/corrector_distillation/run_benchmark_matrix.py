"""Run the fixed Stage-C benchmark matrix with one deterministic job per GPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


MATRIX = (
    ("C0", "C0", None),
    ("A0", "A0", None),
    ("Skip", "Skip", None),
    ("Reuse", "Reuse", None),
    ("Adapter", "Adapter", None),
    ("Adapter+Fallback", "Adapter+Fallback@25", 0.25),
    ("Adapter+Fallback", "Adapter+Fallback@50", 0.50),
    ("Adapter+Fallback", "Adapter+Fallback@75", 0.75),
    ("Adapter+Fallback", "Adapter+Fallback@90", 0.90),
    ("A0+Adapter+Fallback", "A0+Adapter+Fallback@75", 0.75),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    logs_dir = output_root / "logs" / "stage_c_generation"
    logs_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root / "relaxation" / "runtime_tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    incomplete_root = output_root / "generation" / "_incomplete"
    jobs = []
    for method, label, coverage in MATRIX:
        for seed in range(args.seed_start, args.seed_end + 1):
            summary = output_root / "generation" / label / str(seed) / "run_summary.json"
            failure_summary = summary.with_name("failure_summary.json")
            if summary.is_file() or failure_summary.is_file():
                continue
            run_dir = summary.parent
            if run_dir.exists():
                destination = incomplete_root / label / str(seed)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    destination = destination.with_name(
                        f"{destination.name}_{time.time_ns()}"
                    )
                shutil.move(run_dir, destination)
            jobs.append((method, label, coverage, seed))
    print(json.dumps({"jobs": len(jobs), "matrix": MATRIX}), flush=True)
    running: dict[
        int,
        tuple[subprocess.Popen, object, tuple[str, str, float | None, int], float, Path],
    ] = {}
    failures = []
    started = time.perf_counter()
    while jobs or running:
        for gpu in range(args.num_gpus):
            if gpu in running or not jobs:
                continue
            method, label, coverage, seed = jobs.pop(0)
            command = [
                sys.executable,
                "-m",
                "research.corrector_distillation.benchmark_sampler",
                "--checkpoint-root",
                str(args.checkpoint_root.expanduser().resolve()),
                "--output-root",
                str(output_root),
                "--method",
                method,
                "--label",
                label,
                "--seed",
                str(seed),
            ]
            if "Adapter" in method:
                command.extend(
                    (
                        "--adapter-checkpoint",
                        str(args.adapter_checkpoint.expanduser().resolve()),
                    )
                )
            if coverage is not None:
                command.extend(("--coverage", str(coverage)))
            log_path = logs_dir / f"{label.replace('+', '_plus_')}__{seed}.log"
            log_stream = log_path.open("w", encoding="utf-8")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["TMPDIR"] = str(temporary_dir)
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (
                process,
                log_stream,
                (method, label, coverage, seed),
                time.perf_counter(),
                log_path,
            )
        time.sleep(1.0)
        for gpu, (process, log_stream, job, job_started, log_path) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_stream.close()
            del running[gpu]
            method, label, coverage, seed = job
            event = {
                "gpu": gpu,
                "method": method,
                "label": label,
                "coverage": coverage,
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
                    output_root / "generation" / label / str(seed) / "failure_summary.json"
                )
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                with failure_path.open("w", encoding="utf-8") as stream:
                    json.dump(failure, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                failures.append(failure)
    print(
        json.dumps(
            {
                "completed": True,
                "elapsed_seconds": time.perf_counter() - started,
                "failed_jobs": failures,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
