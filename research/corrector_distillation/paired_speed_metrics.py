"""Summarize same-seed timing with paired bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline", default="C0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-seed-output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    return parser.parse_args()


def sample_std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def bootstrap_mean_ci(
    values: np.ndarray, *, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    estimates = []
    remaining = samples
    while remaining:
        current = min(remaining, 2_000)
        indices = rng.integers(0, values.size, size=(current, values.size))
        estimates.append(values[indices].mean(axis=1))
        remaining -= current
    low, high = np.quantile(np.concatenate(estimates), (0.025, 0.975))
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    with args.benchmark.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        input_rows = list(csv.DictReader(stream))
    grouped: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in input_rows:
        if row.get("success", "").lower() not in ("true", "1"):
            continue
        method = row["method"]
        seed = int(row["seed"])
        if seed in grouped[method]:
            raise ValueError(f"duplicate timing row for {method} seed {seed}")
        grouped[method][seed] = row
    if args.baseline not in grouped:
        raise ValueError(f"baseline {args.baseline!r} is missing")

    rng = np.random.default_rng(args.bootstrap_seed)
    baseline = grouped[args.baseline]
    summary_rows = []
    per_seed_rows = []
    for method in sorted(grouped):
        common = sorted(set(baseline) & set(grouped[method]))
        if not common:
            raise ValueError(f"no paired seeds for {method}")
        times = np.asarray(
            [float(grouped[method][seed]["time_per_sample"]) for seed in common]
        )
        baseline_times = np.asarray(
            [float(baseline[seed]["time_per_sample"]) for seed in common]
        )
        time_deltas = times - baseline_times
        speedups = baseline_times / times
        time_ci = bootstrap_mean_ci(
            time_deltas, samples=args.bootstrap_samples, rng=rng
        )
        speedup_ci = bootstrap_mean_ci(
            speedups, samples=args.bootstrap_samples, rng=rng
        )
        rows = [grouped[method][seed] for seed in common]
        numeric = lambda name: np.asarray([float(row[name]) for row in rows])
        logical_calls = numeric("mattergen_score_calls")
        second_forward_calls = logical_calls - 1_000.0
        for seed, value, baseline_value, delta, speedup in zip(
            common, times, baseline_times, time_deltas, speedups
        ):
            row = grouped[method][seed]
            per_seed_rows.append(
                {
                    "method": method,
                    "baseline": args.baseline,
                    "seed": seed,
                    "time_per_sample": value,
                    "baseline_time_per_sample": baseline_value,
                    "paired_time_delta": delta,
                    "paired_speedup": speedup,
                    "logical_score_calls": row["mattergen_score_calls"],
                    "second_forward_calls": float(row["mattergen_score_calls"]) - 1_000.0,
                    "adapter_calls": row["adapter_calls"],
                    "fallback_calls": row["fallback_calls"],
                    "adapter_acceptance_overall": row["adapter_acceptance_overall"],
                    "second_forward_avoidance": row["second_forward_avoidance"],
                    "forward_reduction": row["forward_reduction"],
                    "peak_allocated_bytes": row["peak_allocated_bytes"],
                }
            )
        summary_rows.append(
            {
                "method": method,
                "baseline": args.baseline,
                "n_paired": len(common),
                "time_per_sample_mean": float(times.mean()),
                "time_per_sample_median": float(np.median(times)),
                "time_per_sample_std": sample_std(times),
                "samples_per_hour_from_mean_time": float(3_600.0 / times.mean()),
                "paired_time_delta_mean": float(time_deltas.mean()),
                "paired_time_delta_median": float(np.median(time_deltas)),
                "paired_time_delta_std": sample_std(time_deltas),
                "paired_time_delta_bootstrap95_low": time_ci[0],
                "paired_time_delta_bootstrap95_high": time_ci[1],
                "paired_speedup_mean": float(speedups.mean()),
                "paired_speedup_median": float(np.median(speedups)),
                "paired_speedup_std": sample_std(speedups),
                "paired_speedup_bootstrap95_low": speedup_ci[0],
                "paired_speedup_bootstrap95_high": speedup_ci[1],
                "logical_score_calls_mean": float(logical_calls.mean()),
                "second_forward_calls_mean": float(second_forward_calls.mean()),
                "adapter_calls_mean": float(numeric("adapter_calls").mean()),
                "fallback_calls_mean": float(numeric("fallback_calls").mean()),
                "adapter_acceptance_overall_mean": float(
                    numeric("adapter_acceptance_overall").mean()
                ),
                "second_forward_avoidance_mean": float(
                    numeric("second_forward_avoidance").mean()
                ),
                "forward_reduction_mean": float(numeric("forward_reduction").mean()),
                "peak_allocated_bytes_mean": float(numeric("peak_allocated_bytes").mean()),
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
            }
        )

    output = args.output.expanduser().resolve()
    per_seed_output = args.per_seed_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    per_seed_output.parent.mkdir(parents=True, exist_ok=True)
    for path in (output, per_seed_output):
        if path.exists():
            raise FileExistsError(path)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with per_seed_output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_seed_rows[0]))
        writer.writeheader()
        writer.writerows(per_seed_rows)
    print(
        json.dumps(
            {
                "summary": str(output),
                "per_seed": str(per_seed_output),
                "methods": sorted(grouped),
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
