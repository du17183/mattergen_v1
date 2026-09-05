"""Paired algorithmic speed statistics for the strict single-H20 rerun."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from research.corrector_distillation.formal256_protocol import (
    EXPERIMENT_ROOT,
    METHODS,
    SINGLE_H20_SEEDS,
)


BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260905


def sample_std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def bootstrap_mean(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = []
    remaining = BOOTSTRAP_SAMPLES
    while remaining:
        current = min(remaining, 2_000)
        indices = rng.integers(0, values.size, size=(current, values.size))
        estimates.append(values[indices].mean(axis=1))
        remaining -= current
    return tuple(float(value) for value in np.quantile(np.concatenate(estimates), (0.025, 0.975)))


def main() -> None:
    raw_path = EXPERIMENT_ROOT / "single_h20_benchmark.csv"
    with raw_path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    by_key = {(row["method"], int(row["seed"])): row for row in raw}
    expected = {(method, seed) for method in METHODS for seed in SINGLE_H20_SEEDS}
    if set(by_key) != expected or not all(row["success"].lower() in ("true", "1") for row in raw):
        raise ValueError("strict single-H20 benchmark is incomplete")
    rows = []
    for seed in SINGLE_H20_SEEDS:
        c0 = by_key[("C0", seed)]
        v2 = by_key[("V2_Frozen_Atomic75_Late30", seed)]
        c0_time = float(c0["generation_time_seconds"])
        v2_time = float(v2["generation_time_seconds"])
        rows.append(
            {
                "seed": seed,
                "c0_time_seconds": c0_time,
                "v2_time_seconds": v2_time,
                "paired_speedup_c0_over_v2": c0_time / v2_time,
                "v2_logical_score_calls": v2["logical_score_calls"],
                "v2_second_forward_calls": v2["second_forward_calls"],
                "v2_adapter_calls": v2["adapter_calls"],
                "v2_fallback_calls": v2["fallback_calls"],
                "v2_coverage": v2["coverage"],
                "v2_forward_reduction": v2["forward_reduction"],
                "c0_peak_gpu_memory_bytes": c0["peak_gpu_memory_bytes"],
                "v2_peak_gpu_memory_bytes": v2["peak_gpu_memory_bytes"],
            }
        )
    per_seed_path = EXPERIMENT_ROOT / "single_h20_per_seed.csv"
    if per_seed_path.exists():
        raise FileExistsError(per_seed_path)
    with per_seed_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    speedups = np.asarray([row["paired_speedup_c0_over_v2"] for row in rows], dtype=float)
    ci_low, ci_high = bootstrap_mean(speedups)
    numeric = lambda name: np.asarray([float(row[name]) for row in rows], dtype=float)
    summary = {
        "n_paired": len(rows),
        "gpu": "physical GPU0, NVIDIA H20",
        "batch_size": 1,
        "seed_start": SINGLE_H20_SEEDS[0],
        "seed_end": SINGLE_H20_SEEDS[-1],
        "c0_time_per_sample_mean": float(numeric("c0_time_seconds").mean()),
        "v2_time_per_sample_mean": float(numeric("v2_time_seconds").mean()),
        "c0_samples_per_hour_from_mean_time": float(3600 / numeric("c0_time_seconds").mean()),
        "v2_samples_per_hour_from_mean_time": float(3600 / numeric("v2_time_seconds").mean()),
        "paired_speedup_mean": float(speedups.mean()),
        "paired_speedup_median": float(np.median(speedups)),
        "paired_speedup_std": sample_std(speedups),
        "paired_speedup_bootstrap95_low": ci_low,
        "paired_speedup_bootstrap95_high": ci_high,
        "v2_logical_score_calls_mean": float(numeric("v2_logical_score_calls").mean()),
        "v2_second_forward_calls_mean": float(numeric("v2_second_forward_calls").mean()),
        "v2_adapter_calls_mean": float(numeric("v2_adapter_calls").mean()),
        "v2_fallback_calls_mean": float(numeric("v2_fallback_calls").mean()),
        "v2_coverage_mean": float(numeric("v2_coverage").mean()),
        "v2_forward_reduction_mean": float(numeric("v2_forward_reduction").mean()),
        "c0_peak_gpu_memory_bytes_mean": float(numeric("c0_peak_gpu_memory_bytes").mean()),
        "v2_peak_gpu_memory_bytes_mean": float(numeric("v2_peak_gpu_memory_bytes").mean()),
        "v2_peak_gpu_memory_bytes_max": float(numeric("v2_peak_gpu_memory_bytes").max()),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    summary_path = EXPERIMENT_ROOT / "single_h20_metrics.csv"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    with summary_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
