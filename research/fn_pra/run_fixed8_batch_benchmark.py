"""Run the fixed eight-GPU A0 batch throughput benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from research.fn_pra.phase1_common import (
    LOGS,
    PROJECT,
    REPORTS,
    RESULTS,
    atomic_json,
    now,
)


PYTHON = "/data/dxl/envs/mattergen_py310/bin/python"
ROOT = RESULTS / "batch_benchmark/fixed8"
BATCH_REPEATS = {1: 40, 2: 20, 4: 10, 8: 5}
WARMUPS = 3
GPU_COUNT = 8


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_gpu(batch_size: int, gpu: int) -> dict:
    output = ROOT / f"b{batch_size}" / f"gpu{gpu}"
    status = output / "status.json"
    if status.is_file() and read_json(status).get("success") is True:
        return read_json(output / "summary.json")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite incomplete fixed8 task: {output}")
    module = (
        "research.fn_pra.run_batch_trial_native_b1"
        if batch_size == 1
        else "research.fn_pra.run_batch_trial"
    )
    log = LOGS / f"fixed8_batch_b{batch_size}_gpu{gpu}.log"
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
        module,
        "--output-dir",
        str(output),
        "--batch-size",
        str(batch_size),
        "--seed-start",
        "15000",
        "--sampling-steps",
        "1000",
        "--warmups",
        str(WARMUPS),
        "--repeats",
        str(BATCH_REPEATS[batch_size]),
        "--physical-gpu",
        str(gpu),
    ]
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0 or not status.is_file() or not read_json(status).get("success"):
        raise RuntimeError(f"fixed8 task failed: batch={batch_size}, gpu={gpu}")
    return read_json(output / "summary.json")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch_size, repeats in BATCH_REPEATS.items():
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
            futures = [executor.submit(run_gpu, batch_size, gpu) for gpu in range(GPU_COUNT)]
            summaries = [future.result() for future in as_completed(futures)]
        wall = time.monotonic() - started
        measured_seconds_by_gpu = [
            sum(row["elapsed_seconds"] for row in summary["repeats"])
            for summary in summaries
        ]
        measured_samples = GPU_COUNT * repeats * batch_size
        steady_state_wall = max(measured_seconds_by_gpu)
        row = {
            "batch_size_per_gpu": batch_size,
            "gpu_count": GPU_COUNT,
            "warmups_per_gpu": WARMUPS,
            "repeats_per_gpu": repeats,
            "measured_samples": measured_samples,
            "process_wall_seconds_including_warmup_and_load": wall,
            "steady_state_wall_seconds": steady_state_wall,
            "fixed8_samples_per_hour": measured_samples * 3600.0 / steady_state_wall,
            "generation_success": sum(
                summary["structure_count"] == batch_size for summary in summaries
            ),
            "all_deterministic_repeats_level1": all(
                summary["deterministic_repeats_level1"] for summary in summaries
            ),
            "max_peak_allocated_bytes": max(
                max(repeat["peak_allocated_bytes"] for repeat in summary["repeats"])
                for summary in summaries
            ),
            "max_peak_reserved_bytes": max(
                max(repeat["peak_reserved_bytes"] for repeat in summary["repeats"])
                for summary in summaries
            ),
        }
        rows.append(row)
        atomic_json(
            REPORTS / "fixed8_batch_benchmark.partial.json",
            {"updated_at": now(), "rows": rows},
        )
    baseline = rows[0]["fixed8_samples_per_hour"]
    for row in rows:
        row["throughput_gain_vs_batch1"] = (
            row["fixed8_samples_per_hour"] / baseline - 1.0
        )
    report = {
        "created_at": now(),
        "gpu_count": GPU_COUNT,
        "rows": rows,
        "all_success": all(row["generation_success"] == GPU_COUNT for row in rows),
    }
    atomic_json(REPORTS / "fixed8_batch_benchmark.json", report)
    return 0 if report["all_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
