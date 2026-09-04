"""Run Stage-C MatterSim relaxation and required metrics with one job per GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


METHOD_LABELS = (
    "C0",
    "A0",
    "Skip",
    "Reuse",
    "Adapter",
    "Adapter+Fallback@25",
    "Adapter+Fallback@50",
    "Adapter+Fallback@75",
    "Adapter+Fallback@90",
    "A0+Adapter+Fallback@75",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures-dir", type=Path, required=True)
    parser.add_argument("--potential-path", type=Path, required=True)
    parser.add_argument("--reference-lmdb-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structures_dir = args.structures_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    logs_dir = args.logs_dir.expanduser().resolve()
    runtime_tmp = output_root.parent / "runtime_tmp"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    jobs = []
    for label in METHOD_LABELS:
        filename = f"{label.replace('+', '_plus_')}_generated.extxyz"
        structures_path = structures_dir / filename
        if not structures_path.is_file():
            raise FileNotFoundError(structures_path)
        output_dir = output_root / label.replace("+", "_plus_")
        if (output_dir / "quality_summary.json").is_file():
            continue
        if output_dir.exists():
            incomplete_root = output_root / "_incomplete"
            incomplete_root.mkdir(parents=True, exist_ok=True)
            output_dir.rename(
                incomplete_root / f"{output_dir.name}-{time.time_ns()}"
            )
        jobs.append((label, structures_path, output_dir))
    print(json.dumps({"quality_jobs": len(jobs)}), flush=True)
    running: dict[int, tuple[subprocess.Popen, object, tuple[str, Path, Path]]] = {}
    failures = []
    started = time.perf_counter()
    while jobs or running:
        for gpu in range(args.num_gpus):
            if gpu in running or not jobs:
                continue
            label, structures_path, output_dir = jobs.pop(0)
            command = [
                sys.executable,
                "-m",
                "research.corrector_distillation.evaluate_quality",
                "--method",
                label,
                "--structures-path",
                str(structures_path),
                "--potential-path",
                str(args.potential_path.expanduser().resolve()),
                "--reference-lmdb-path",
                str(args.reference_lmdb_path.expanduser().resolve()),
                "--reference-is-ordered",
                "--output-dir",
                str(output_dir),
                "--temporary-dir",
                str(runtime_tmp),
                "--device",
                "cuda",
            ]
            log_path = logs_dir / f"{label.replace('+', '_plus_')}.log"
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
            running[gpu] = (process, log_stream, (label, structures_path, output_dir))
        time.sleep(1.0)
        for gpu, (process, log_stream, job) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_stream.close()
            del running[gpu]
            label, _structures_path, _output_dir = job
            event = {"gpu": gpu, "method": label, "return_code": return_code}
            print(json.dumps(event, sort_keys=True), flush=True)
            if return_code != 0:
                failures.append(event)
        if failures:
            for process, log_stream, _job in running.values():
                process.terminate()
                log_stream.close()
            raise RuntimeError(f"Stage-C quality failures: {failures}")
    print(
        json.dumps(
            {"completed": True, "elapsed_seconds": time.perf_counter() - started},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
