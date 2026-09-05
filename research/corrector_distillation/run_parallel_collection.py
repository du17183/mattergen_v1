"""Run exact-teacher or DAgger collection with one seed per GPU worker."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("teacher", "dagger"), required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument("--generation-label")
    parser.add_argument("--num-gpus", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "dagger" and args.adapter_checkpoint is None:
        raise ValueError("dagger mode requires --adapter-checkpoint")
    output_root = args.output_root.expanduser().resolve()
    logs_dir = output_root / "logs" / f"{args.mode}_{args.split}"
    runtime_tmp = output_root / "runtime_tmp"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    if args.mode == "dagger" and args.generation_label:
        run_root = output_root / "generation" / args.generation_label
    else:
        run_kind = "teacher_runs" if args.mode == "teacher" else "dagger_runs"
        run_root = output_root / run_kind / args.split
    jobs = [
        seed
        for seed in range(args.seed_start, args.seed_end + 1)
        if not (run_root / str(seed) / "run_summary.json").is_file()
    ]
    incomplete_root = output_root / "_incomplete" / args.mode / args.split
    for seed in jobs:
        run_dir = run_root / str(seed)
        data_kind = "teacher_data" if args.mode == "teacher" else "dagger_data"
        data_dir = output_root / data_kind / args.split / str(seed)
        for source, suffix in ((run_dir, "run"), (data_dir, "data")):
            if not source.exists():
                continue
            destination = incomplete_root / f"{seed}-{suffix}-{time.time_ns()}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, destination)
    total_jobs = len(jobs)
    print(json.dumps({"jobs": total_jobs, "mode": args.mode, "split": args.split}), flush=True)
    running: dict[int, tuple[subprocess.Popen, object, int, float, Path]] = {}
    failures = []
    started = time.perf_counter()
    while jobs or running:
        for gpu in range(args.num_gpus):
            if gpu in running or not jobs:
                continue
            seed = jobs.pop(0)
            module = (
                "research.corrector_distillation.collect_teacher"
                if args.mode == "teacher"
                else "research.corrector_distillation.collect_dagger"
            )
            command = [
                sys.executable,
                "-m",
                module,
                "--checkpoint-root",
                str(args.checkpoint_root.expanduser().resolve()),
                "--output-root",
                str(output_root),
                "--seed",
                str(seed),
                "--split",
                args.split,
            ]
            if args.mode == "dagger":
                command.extend(
                    (
                        "--adapter-checkpoint",
                        str(args.adapter_checkpoint.expanduser().resolve()),
                        "--coverage",
                        str(args.coverage),
                    )
                )
                if args.generation_label:
                    command.extend(("--generation-label", args.generation_label))
            log_path = logs_dir / f"{args.mode}__{args.split}__{seed}.log"
            log_stream = log_path.open("w", encoding="utf-8")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["TMPDIR"] = str(runtime_tmp)
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (process, log_stream, seed, time.perf_counter(), log_path)
        time.sleep(1.0)
        for gpu, (process, log_stream, seed, job_started, log_path) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_stream.close()
            del running[gpu]
            event = {
                "gpu": gpu,
                "seed": seed,
                "return_code": return_code,
                "remaining": len(jobs) + len(running),
            }
            print(json.dumps(event, sort_keys=True), flush=True)
            if return_code != 0:
                failures.append(
                    {
                        **event,
                        "elapsed_seconds": time.perf_counter() - job_started,
                        "log_path": str(log_path.resolve()),
                    }
                )
    elapsed = time.perf_counter() - started
    summary = {
        "completed": not failures,
        "mode": args.mode,
        "split": args.split,
        "jobs": total_jobs,
        "elapsed_seconds": elapsed,
        "throughput_samples_per_hour": 0.0 if elapsed == 0 else total_jobs * 3600.0 / elapsed,
        "failures": failures,
    }
    summary_path = logs_dir / f"collection_summary_{args.seed_start}_{args.seed_end}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    if failures:
        raise RuntimeError(f"collection failures: {failures}")


if __name__ == "__main__":
    main()
