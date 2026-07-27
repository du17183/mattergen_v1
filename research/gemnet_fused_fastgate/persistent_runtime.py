from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import statistics
import subprocess
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.gemnet_fused_fastgate.common import (
    LOGS,
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    configure_environment,
    now,
    set_stage,
    set_termination_state,
)


SEEDS = tuple(range(27000, 27032))
WORKER_LEVELS = (1, 2, 4)
GPU_COUNT = 8
MAX_RETRIES = 1


def tensor_digest(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode())
    digest.update(json.dumps(list(cpu.shape)).encode())
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def serialize_sample(sample) -> dict[str, Any]:
    graph = sample.to_data_list()[0]
    fields = {
        "atomic_numbers": graph.atomic_numbers.detach().contiguous().cpu(),
        "pos": graph.pos.detach().contiguous().cpu(),
        "cell": graph.cell.detach().contiguous().cpu(),
    }
    hashes = {name: tensor_digest(value) for name, value in fields.items()}
    combined = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    return {
        "hashes": hashes | {"combined": combined},
        "atomic_numbers": fields["atomic_numbers"].tolist(),
        "pos": fields["pos"].tolist(),
        "cell": fields["cell"].tolist(),
        "num_atoms": int(fields["atomic_numbers"].numel()),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def persistent_worker(
    physical_gpu: int,
    slot: int,
    workers_per_gpu: int,
    task_queue,
    result_queue,
    ready_queue,
) -> None:
    worker_id = f"gpu{physical_gpu}_slot{slot}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    configure_environment()
    log_path = LOGS / "runtime" / f"w{workers_per_gpu}_{worker_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.set_float32_matmul_precision("highest")
            torch.cuda.set_device(0)
            from research.gemnet_fused_fastgate.harness import (
                build_c0_generator,
                build_sampler,
                load_states,
                prepare_joint_states,
                run_joint_score,
            )

            load_started = time.monotonic()
            generator = build_c0_generator()
            sampler = build_sampler(generator)
            sampling_config = generator.load_sampling_config(
                batch_size=1,
                num_batches=1,
            )
            # A real-state score warmup initializes kernels and the CUDA caching
            # allocator before the measured task window. Every actual task reseeds.
            warm_state = prepare_joint_states(
                load_states(1), sampler, torch.device("cuda:0")
            )[0]
            with torch.inference_mode():
                run_joint_score(generator.model.diffusion_module, sampler, warm_state)
            torch.cuda.synchronize()
            ready_queue.put(
                {
                    "ready": True,
                    "worker_id": worker_id,
                    "gpu": physical_gpu,
                    "slot": slot,
                    "model_load_warmup_seconds": time.monotonic() - load_started,
                }
            )
        except BaseException:
            ready_queue.put(
                {
                    "ready": False,
                    "worker_id": worker_id,
                    "gpu": physical_gpu,
                    "slot": slot,
                    "error": traceback.format_exc(),
                }
            )
            return

        while True:
            task = task_queue.get()
            if task is None:
                return
            seed = int(task["seed"])
            attempt = int(task["attempt"])
            started = time.monotonic()
            try:
                seed_everything(seed)
                if hasattr(sampler, "_sample_seed"):
                    sampler._sample_seed = seed
                if hasattr(sampler, "_run_id"):
                    sampler._run_id = f"runtime_seed_{seed}"
                condition_loader = generator.get_condition_loader(sampling_config)
                conditioning_data, mask = next(iter(condition_loader))
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    _sample, mean_sample = sampler.sample(conditioning_data, mask)
                torch.cuda.synchronize()
                payload = serialize_sample(mean_sample)
                result_queue.put(
                    {
                        "success": True,
                        "seed": seed,
                        "attempt": attempt,
                        "worker_id": worker_id,
                        "gpu": physical_gpu,
                        "slot": slot,
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                        "sample": payload,
                    }
                )
            except BaseException:
                result_queue.put(
                    {
                        "success": False,
                        "seed": seed,
                        "attempt": attempt,
                        "worker_id": worker_id,
                        "gpu": physical_gpu,
                        "slot": slot,
                        "elapsed_seconds": time.monotonic() - started,
                        "error": traceback.format_exc(),
                    }
                )


class Telemetry:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.rows: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                process = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                timestamp = time.time()
                for line in process.stdout.splitlines():
                    gpu, utilization, memory, power = [part.strip() for part in line.split(",")]
                    if int(gpu) < GPU_COUNT:
                        self.rows.append(
                            {
                                "time": timestamp,
                                "gpu": int(gpu),
                                "utilization_percent": float(utilization),
                                "memory_used_mib": float(memory),
                                "power_w": float(power),
                            }
                        )
            except Exception:
                pass
            self.stop_event.wait(1.0)

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            return {"samples": 0}
        return {
            "samples": len(self.rows),
            "utilization_mean_percent": statistics.mean(
                row["utilization_percent"] for row in self.rows
            ),
            "utilization_max_percent": max(
                row["utilization_percent"] for row in self.rows
            ),
            "memory_peak_mib": max(row["memory_used_mib"] for row in self.rows),
            "power_mean_w": statistics.mean(row["power_w"] for row in self.rows),
        }


def stop_workers(processes: list[mp.Process], task_queue) -> None:
    for _ in processes:
        task_queue.put(None)
    for process in processes:
        process.join(timeout=180)
    lingering = [process.pid for process in processes if process.is_alive()]
    if lingering:
        raise RuntimeError(
            f"persistent workers did not exit after cooperative sentinel: {lingering}"
        )


def run_level(workers_per_gpu: int) -> dict[str, Any]:
    level_dir = RESULTS / "runtime" / f"workers_{workers_per_gpu}"
    summary_path = level_dir / "summary.json"
    if summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("success_count") == len(SEEDS):
            return cached
    level_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    ready_queue = ctx.Queue()
    processes = []
    for gpu in range(GPU_COUNT):
        for slot in range(workers_per_gpu):
            process = ctx.Process(
                target=persistent_worker,
                args=(gpu, slot, workers_per_gpu, task_queue, result_queue, ready_queue),
                name=f"gemnet_runtime_w{workers_per_gpu}_g{gpu}_s{slot}",
            )
            process.start()
            processes.append(process)
    ready_rows = []
    for _ in processes:
        ready_rows.append(ready_queue.get(timeout=900))
    readiness_failures = [row for row in ready_rows if not row["ready"]]
    if readiness_failures:
        stop_workers(processes, task_queue)
        raise RuntimeError(f"worker readiness failure: {readiness_failures}")

    telemetry = Telemetry()
    telemetry.start()
    attempts = Counter()
    for seed in SEEDS:
        task_queue.put({"seed": seed, "attempt": 0})
    measured_started = time.monotonic()
    successful: dict[int, dict[str, Any]] = {}
    failed_attempts = []
    try:
        while len(successful) < len(SEEDS):
            row = result_queue.get(timeout=900)
            seed = int(row["seed"])
            if row["success"]:
                successful[seed] = row
                atomic_json(level_dir / f"seed_{seed}.json", row)
                continue
            failed_attempts.append(row)
            attempts[seed] += 1
            if attempts[seed] <= MAX_RETRIES:
                task_queue.put({"seed": seed, "attempt": attempts[seed]})
            else:
                raise RuntimeError(
                    f"seed {seed} failed after retry: {row.get('error')}"
                )
        measured_elapsed = time.monotonic() - measured_started
    finally:
        telemetry.stop()
        stop_workers(processes, task_queue)

    rows = [successful[seed] for seed in SEEDS]
    task_counts = Counter(row["worker_id"] for row in rows)
    summary = {
        "completed_at": now(),
        "workers_per_gpu": workers_per_gpu,
        "total_workers": workers_per_gpu * GPU_COUNT,
        "seeds": list(SEEDS),
        "success_count": len(rows),
        "failure_attempts": len(failed_attempts),
        "measured_wall_seconds": measured_elapsed,
        "total_throughput_samples_per_hour": len(rows) * 3600.0 / measured_elapsed,
        "single_gpu_throughput_samples_per_hour": len(rows) * 3600.0 / measured_elapsed / GPU_COUNT,
        "latency_median_seconds": statistics.median(row["elapsed_seconds"] for row in rows),
        "latency_p95_seconds": float(np.percentile([row["elapsed_seconds"] for row in rows], 95)),
        "worker_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
        "worker_peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in rows),
        "task_counts_per_worker": dict(sorted(task_counts.items())),
        "fairness_min_tasks": min(task_counts.values()),
        "fairness_max_tasks": max(task_counts.values()),
        "ready": ready_rows,
        "telemetry": telemetry.summary(),
        "rows": [
            {
                key: row[key]
                for key in (
                    "seed",
                    "worker_id",
                    "gpu",
                    "slot",
                    "elapsed_seconds",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                )
            }
            | {"combined_hash": row["sample"]["hashes"]["combined"]}
            for row in rows
        ],
    }
    atomic_json(summary_path, summary)
    return summary


def main() -> int:
    configure_environment()
    set_stage(
        "eight_seed_generation",
        "skipped",
        "Skipped because fusion numerical, chain, and full-forward gates failed.",
    )
    set_stage(
        "eight_seed_quality",
        "skipped",
        "No fused E2E samples were authorized after pre-E2E No-Go.",
    )
    set_stage(
        "runtime_fallback",
        "running",
        "Benchmarking 1/2/4 persistent independent B1 workers per GPU on 32 seeds.",
        {"seeds": list(SEEDS), "worker_levels": list(WORKER_LEVELS)},
    )
    levels = [run_level(workers) for workers in WORKER_LEVELS]
    baseline = next(row for row in levels if row["workers_per_gpu"] == 1)
    baseline_hashes = {
        row["seed"]: row["combined_hash"] for row in baseline["rows"]
    }
    for level in levels:
        level["same_seed_bitwise_equivalent"] = all(
            row["combined_hash"] == baseline_hashes[row["seed"]]
            for row in level["rows"]
        )
        level["throughput_speedup_vs_1worker"] = (
            level["total_throughput_samples_per_hour"]
            / baseline["total_throughput_samples_per_hour"]
        )
    best = max(levels[1:], key=lambda row: row["total_throughput_samples_per_hour"])
    runtime_speedup = best["throughput_speedup_vs_1worker"]
    bitwise = all(row["same_seed_bitwise_equivalent"] for row in levels)
    quality_safe = bitwise and all(row["success_count"] == len(SEEDS) for row in levels)
    runtime_pass = runtime_speedup >= 1.25 and quality_safe
    payload = {
        "completed_at": now(),
        "levels": levels,
        "selected_workers_per_gpu": best["workers_per_gpu"],
        "base_total_throughput": baseline["total_throughput_samples_per_hour"],
        "runtime_total_throughput": best["total_throughput_samples_per_hour"],
        "runtime_speedup": runtime_speedup,
        "same_seed_bitwise_equivalent": bitwise,
        "runtime_quality_safe": quality_safe,
        "runtime_pass": runtime_pass,
        "throughput_gate": 1.25,
    }
    atomic_json(RESULTS / "runtime/runtime_decision.json", payload)
    set_stage(
        "runtime_fallback",
        "success" if runtime_pass else "failed",
        "Persistent runtime passed throughput and bitwise gates."
        if runtime_pass
        else "Persistent runtime failed throughput and/or bitwise gate.",
        payload,
    )
    final_state = (
        "FUSED_KERNEL_NO_GO_RUNTIME_PASS"
        if runtime_pass
        else "GPU_ACCELERATION_NO_GO"
    )
    set_termination_state(final_state)
    set_stage(
        "final_decision",
        "success",
        f"Final fast-gate state: {final_state}",
        payload,
    )
    report = [
        "# Persistent multi-trajectory B1 runtime",
        "",
        "| Workers/GPU | 8-GPU samples/hour | Median latency (s) | GPU util mean | Peak memory MiB | Speedup | Bitwise |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for level in levels:
        report.append(
            f"| {level['workers_per_gpu']} | {level['total_throughput_samples_per_hour']:.3f} | "
            f"{level['latency_median_seconds']:.3f} | {level['telemetry'].get('utilization_mean_percent', float('nan')):.2f} | "
            f"{level['telemetry'].get('memory_peak_mib', float('nan')):.0f} | "
            f"{level['throughput_speedup_vs_1worker']:.4f}x | {level['same_seed_bitwise_equivalent']} |"
        )
    report.extend(
        [
            "",
            f"Selected workers/GPU: `{best['workers_per_gpu']}`",
            f"Runtime speedup: `{runtime_speedup:.4f}x`",
            f"Final state: `{final_state}`",
            "",
            "Each task remains batch_size=1, FP32, full Predictor/Corrector, original dynamic graph,",
            "with an independently seeded process/CUDA context. Only residency and scheduling change.",
            "",
        ]
    )
    atomic_text(REPORTS / "runtime_report.md", "\n".join(report))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
