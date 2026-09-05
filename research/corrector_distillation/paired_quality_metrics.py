"""Compute paired per-seed quality deltas and bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


DETAIL_FIELDS = {
    "energy_above_hull_per_atom": "e_hull",
    "stable": "stable",
    "novel_unique_stable": "nus",
    "novel": "novel",
    "unique": "unique",
    "rmsd_from_relaxation": "rmsd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--baseline", default="C0")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-seed-output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    return parser.parse_args()


def method_dir(root: Path, method: str) -> Path:
    return root / method.replace("+", "_plus_")


def load_method(root: Path, method: str) -> dict[int, dict[str, float]]:
    directory = method_dir(root, method)
    with gzip.open(directory / "official_detailed.json.gz", "rt", encoding="utf-8") as stream:
        detailed = json.load(stream)
    with (directory / "per_structure.csv").open(newline="", encoding="utf-8") as stream:
        structure_rows = list(csv.DictReader(stream))
    lengths = {key: len(detailed[key]) for key in DETAIL_FIELDS}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) != len(structure_rows):
        raise ValueError(f"quality arrays do not align for {method}: {lengths}")
    result = {}
    for index, structure_row in enumerate(structure_rows):
        if structure_row.get("sample_seed", "") == "":
            raise ValueError(f"{method} quality data lacks sample_seed provenance")
        seed = int(structure_row["sample_seed"])
        if seed in result:
            raise ValueError(f"{method} has duplicate seed {seed}")
        row = {
            output_name: float(detailed[input_name][index])
            for input_name, output_name in DETAIL_FIELDS.items()
        }
        row["pre_relaxation_max_force"] = float(
            structure_row["pre_relaxation_max_force"]
        )
        result[seed] = row
    return result


def sample_std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def bootstrap_mean_ci(
    values: np.ndarray, *, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    batch = max(1, min(samples, 2000))
    estimates = []
    remaining = samples
    while remaining:
        current = min(batch, remaining)
        selected = rng.integers(0, values.size, size=(current, values.size))
        estimates.append(values[selected].mean(axis=1))
        remaining -= current
    distribution = np.concatenate(estimates)
    low, high = np.quantile(distribution, (0.025, 0.975))
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    root = args.quality_root.expanduser().resolve()
    methods = list(dict.fromkeys([args.baseline, *args.methods]))
    data = {method: load_method(root, method) for method in methods}
    baseline = data[args.baseline]
    metrics = list(next(iter(baseline.values())))
    per_seed_rows = []
    summary_rows = []
    rng = np.random.default_rng(args.bootstrap_seed)
    for method in methods:
        common_seeds = sorted(set(baseline) & set(data[method]))
        if not common_seeds:
            raise ValueError(f"no paired seeds for {method}")
        for seed in common_seeds:
            for metric in metrics:
                per_seed_rows.append(
                    {
                        "method": method,
                        "baseline": args.baseline,
                        "seed": seed,
                        "metric": metric,
                        "value": data[method][seed][metric],
                        "baseline_value": baseline[seed][metric],
                        "paired_delta": (
                            data[method][seed][metric] - baseline[seed][metric]
                        ),
                    }
                )
        for metric in metrics:
            values = np.asarray(
                [data[method][seed][metric] for seed in common_seeds], dtype=float
            )
            baseline_values = np.asarray(
                [baseline[seed][metric] for seed in common_seeds], dtype=float
            )
            deltas = values - baseline_values
            ci_low, ci_high = bootstrap_mean_ci(
                deltas, samples=args.bootstrap_samples, rng=rng
            )
            summary_rows.append(
                {
                    "method": method,
                    "baseline": args.baseline,
                    "metric": metric,
                    "n_paired": len(common_seeds),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": sample_std(values),
                    "baseline_mean": float(baseline_values.mean()),
                    "paired_delta_mean": float(deltas.mean()),
                    "paired_delta_median": float(np.median(deltas)),
                    "paired_delta_std": sample_std(deltas),
                    "paired_delta_bootstrap95_low": ci_low,
                    "paired_delta_bootstrap95_high": ci_high,
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
                "methods": methods,
                "metrics": metrics,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
