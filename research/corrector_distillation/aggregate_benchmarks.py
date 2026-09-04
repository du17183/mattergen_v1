"""Aggregate paired sampler runs and assemble per-method structure files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from ase.io import read, write


NUMERIC_FIELDS = (
    "elapsed_seconds",
    "time_per_sample",
    "samples_per_hour",
    "mattergen_score_calls",
    "saved_score_calls",
    "adapter_calls",
    "fallback_calls",
    "adapter_coverage",
    "forward_reduction",
    "peak_allocated_bytes",
    "adapter_seconds",
    "exact_fallback_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--output-benchmark", type=Path, required=True)
    parser.add_argument("--output-ablation", type=Path, required=True)
    parser.add_argument("--structures-dir", type=Path, required=True)
    return parser.parse_args()


def _load_run(summary_path: Path, method: str) -> dict[str, Any]:
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    seed = int(summary["seed"])
    structure_path = summary_path.parent / "generated_crystals.extxyz"
    if not structure_path.is_file():
        raise FileNotFoundError(structure_path)
    row = {
        "method": method,
        "seed": seed,
        "success": bool(summary["success"]),
        "formula": summary["formula"],
        "num_atoms": int(summary["num_atoms"]),
        "guidance_schedule": summary["guidance_schedule"],
        "coverage_target": float(summary.get("coverage_target", 0.0)),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "adapter_checkpoint_sha256": summary.get("adapter_checkpoint_sha256"),
        "timing_includes_teacher_recording": False,
        "run_summary_path": str(summary_path.resolve()),
        "structure_path": str(structure_path.resolve()),
    }
    for field in NUMERIC_FIELDS:
        default = summary["elapsed_seconds"] if field == "time_per_sample" else 0.0
        row[field] = float(summary.get(field, default))
    return row


def collect_runs(root: Path, seed_start: int, seed_end: int) -> list[dict[str, Any]]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    generation_root = root / "generation"
    for summary_path in sorted(generation_root.glob("*/*/run_summary.json")):
        method = summary_path.parent.parent.name
        row = _load_run(summary_path, method)
        if seed_start <= row["seed"] <= seed_end:
            selected[(method, row["seed"])] = row
    # Teacher test runs are an exact structural C0 fallback, but their timing
    # includes recorder I/O and must never replace a dedicated pure C0 run.
    for summary_path in sorted((root / "teacher_runs" / "test").glob("*/run_summary.json")):
        row = _load_run(summary_path, "C0")
        if seed_start <= row["seed"] <= seed_end:
            key = ("C0", row["seed"])
            if key not in selected:
                row["timing_includes_teacher_recording"] = True
                selected[key] = row
    if not selected:
        raise FileNotFoundError(f"no runs in seed range {seed_start}-{seed_end}")
    return [selected[key] for key in sorted(selected)]


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    root = args.experiment_root.expanduser().resolve()
    rows = collect_runs(root, args.seed_start, args.seed_end)
    benchmark_path = args.output_benchmark.expanduser().resolve()
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_key = {(row["method"], row["seed"]): row for row in rows}
    for row in rows:
        grouped[row["method"]].append(row)
    ablation_rows = []
    for method, method_rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "method": method,
            "n": len(method_rows),
            "seed_start": min(row["seed"] for row in method_rows),
            "seed_end": max(row["seed"] for row in method_rows),
            "generation_success_rate": statistics.fmean(
                float(row["success"]) for row in method_rows
            ),
            "elapsed_seconds_total": sum(
                float(row["elapsed_seconds"]) for row in method_rows
            ),
        }
        for field in NUMERIC_FIELDS:
            mean, std = mean_std([float(row[field]) for row in method_rows])
            aggregate[f"{field}_mean"] = mean
            aggregate[f"{field}_std"] = std
        paired_speedups = []
        for row in method_rows:
            baseline = by_key.get(("C0", row["seed"]))
            if baseline is not None:
                paired_speedups.append(
                    float(baseline["elapsed_seconds"]) / float(row["elapsed_seconds"])
                )
        speedup_mean, speedup_std = mean_std(paired_speedups)
        aggregate["paired_speedup_mean"] = speedup_mean
        aggregate["paired_speedup_std"] = speedup_std
        aggregate["paired_speedup_n"] = len(paired_speedups)
        ablation_rows.append(aggregate)
    ablation_path = args.output_ablation.expanduser().resolve()
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    with ablation_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ablation_rows[0]))
        writer.writeheader()
        writer.writerows(ablation_rows)

    structures_dir = args.structures_dir.expanduser().resolve()
    structures_dir.mkdir(parents=True, exist_ok=True)
    for method, method_rows in sorted(grouped.items()):
        atoms = []
        for row in sorted(method_rows, key=lambda item: item["seed"]):
            atoms.extend(read(row["structure_path"], index=":"))
        write(structures_dir / f"{method.replace('+', '_plus_')}_generated.extxyz", atoms)
    print(
        json.dumps(
            {
                "benchmark": str(benchmark_path),
                "ablation": str(ablation_path),
                "methods": {method: len(items) for method, items in sorted(grouped.items())},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
