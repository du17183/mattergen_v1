"""One-GPU C0/A0 native batch performance benchmark worker."""

from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path
from statistics import median

import numpy as np
import torch

from research.spg_fastgate.common import RESULTS, atomic_json, now
from research.spg_fastgate.generation import (
    build_generator,
    configure_determinism,
    run_group,
)


PERFORMANCE_SEEDS = tuple(range(24000, 24016))
ROOT = RESULTS / "performance/raw"


class Telemetry:
    def __init__(self, physical_gpu: int) -> None:
        self.physical_gpu = physical_gpu
        self.rows: list[dict] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in result.stdout.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    if int(values[0]) == self.physical_gpu:
                        self.rows.append(
                            {
                                "time": time.time(),
                                "memory_used_mib": int(values[1]),
                                "utilization_gpu_percent": int(values[2]),
                                "power_draw_w": float(values[3]),
                            }
                        )
                        break
            except Exception:
                pass
            self.stop_event.wait(0.5)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def summary(self) -> dict:
        if not self.rows:
            return {"samples": 0}
        return {
            "samples": len(self.rows),
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in self.rows),
            "mean_utilization_percent": float(
                np.mean([row["utilization_gpu_percent"] for row in self.rows])
            ),
            "max_utilization_percent": max(
                row["utilization_gpu_percent"] for row in self.rows
            ),
            "mean_power_w": float(np.mean([row["power_draw_w"] for row in self.rows])),
            "max_power_w": max(row["power_draw_w"] for row in self.rows),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("C0", "A0"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=range(8), required=True)
    return parser.parse_args()


def seeds_for_iteration(gpu: int, batch_size: int, iteration: int) -> tuple[int, ...]:
    start = (gpu * batch_size + iteration * batch_size) % len(PERFORMANCE_SEEDS)
    return tuple(
        PERFORMANCE_SEEDS[(start + offset) % len(PERFORMANCE_SEEDS)]
        for offset in range(batch_size)
    )


def main() -> int:
    args = parse_args()
    output = ROOT / f"{args.method}_B{args.batch_size}" / f"gpu{args.physical_gpu}"
    status = output / "status.json"
    if status.is_file():
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".gpu{args.physical_gpu}.{os.getpid()}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    configure_determinism()
    warmups = 2 if args.batch_size == 1 else 1
    formal_repeats = 3
    telemetry = Telemetry(args.physical_gpu)
    telemetry.start()
    try:
        generator = build_generator(
            args.method,
            batch_size=args.batch_size,
            sampling_steps=1000,
        )
        rows = []
        for iteration in range(warmups + formal_repeats):
            seeds = seeds_for_iteration(args.physical_gpu, args.batch_size, iteration)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            cpu_started = time.process_time()
            start_event.record()
            result = run_group(
                generator=generator,
                method=args.method,
                seeds=seeds,
                physical_gpu=args.physical_gpu,
                save_outputs=False,
            )
            end_event.record()
            torch.cuda.synchronize()
            cuda_seconds = start_event.elapsed_time(end_event) / 1000.0
            cpu_seconds = time.process_time() - cpu_started
            rows.append(
                {
                    "iteration": iteration,
                    "warmup": iteration < warmups,
                    "seeds": list(seeds),
                    "wall_seconds": result["elapsed_seconds"],
                    "cuda_event_seconds": cuda_seconds,
                    "cpu_seconds": cpu_seconds,
                    "samples_per_hour": result["samples_per_hour"],
                    "sample_latency_seconds": result["elapsed_seconds"] / args.batch_size,
                    "peak_allocated_bytes": max(
                        row["peak_allocated_bytes"] for row in result["rows"]
                    ),
                    "peak_reserved_bytes": max(
                        row["peak_reserved_bytes"] for row in result["rows"]
                    ),
                }
            )
        formal = [row for row in rows if not row["warmup"]]
        summary = {
            "created_at": now(),
            "method": args.method,
            "batch_size": args.batch_size,
            "physical_gpu": args.physical_gpu,
            "warmups": warmups,
            "formal_repeats": formal_repeats,
            "median_batch_wall_seconds": median(row["wall_seconds"] for row in formal),
            "median_batch_cuda_seconds": median(
                row["cuda_event_seconds"] for row in formal
            ),
            "median_sample_latency_seconds": median(
                row["sample_latency_seconds"] for row in formal
            ),
            "median_samples_per_hour": median(
                row["samples_per_hour"] for row in formal
            ),
            "rows": rows,
        }
        telemetry.stop()
        summary["telemetry"] = telemetry.summary()
        atomic_json(temporary / "summary.json", summary)
        atomic_json(
            temporary / "telemetry.json",
            {"summary": telemetry.summary(), "samples": telemetry.rows},
        )
        atomic_json(
            temporary / "status.json",
            {"success": True, "finished_at": now()},
        )
        os.replace(temporary, output)
        return 0
    except BaseException:
        telemetry.stop()
        atomic_json(
            temporary / "status.json",
            {
                "success": False,
                "finished_at": now(),
                "error": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
