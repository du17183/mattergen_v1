"""Aggregate and fail-closed audit formal C0/V2 generation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ase.io import read, write

from research.corrector_distillation.formal256_protocol import (
    EXPERIMENT_ROOT,
    FORMAL_SEEDS,
    METHODS,
    SINGLE_H20_SEEDS,
    fixed_shards,
    sha256,
    validate_frozen_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("main", "single-h20"), default="main")
    return parser.parse_args()


def reason_counts(trace_path: Path) -> Counter[str]:
    if not trace_path.is_file():
        return Counter()
    with trace_path.open(newline="", encoding="utf-8") as stream:
        return Counter(row["fallback_reason"] for row in csv.DictReader(stream) if row["fallback_reason"])


def row_for(root: Path, label: str, seed: int) -> dict[str, Any]:
    run_dir = root / "generation" / label / str(seed)
    summary_path = run_dir / "run_summary.json"
    failure_path = run_dir / "failure_summary.json"
    if not summary_path.is_file():
        return {
            "method": label,
            "seed": seed,
            "success": False,
            "failure_summary_path": str(failure_path.resolve()) if failure_path.exists() else "",
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    structure_path = run_dir / "generated_crystals.extxyz"
    counts = reason_counts(run_dir / "residual_trace.csv")
    return {
        "method": label,
        "seed": seed,
        "success": summary.get("success") is True and structure_path.is_file(),
        "generation_time_seconds": summary["elapsed_seconds"],
        "samples_per_hour": summary["samples_per_hour"],
        "logical_score_calls": summary["mattergen_score_calls"],
        "second_forward_calls": int(summary["mattergen_score_calls"]) - 1000,
        "adapter_calls": summary.get("adapter_calls", 0),
        "adapter_accept_calls": summary.get("adapter_accept_calls", 0),
        "fallback_calls": summary.get("fallback_calls", 0),
        "atomic_risk_fallback_calls": counts["risk_above_threshold"],
        "late_exact_calls": summary.get("late_exact_calls", 0),
        "nan_adapter_fallback_calls": counts["non_finite_adapter_output"],
        "nan_risk_fallback_calls": counts["non_finite_uncertainty"],
        "invalid_risk_fallback_calls": counts["missing_or_invalid_calibration"],
        "coverage": summary.get("adapter_acceptance_overall", 0.0),
        "second_forward_avoidance": summary.get("second_forward_avoidance", 0.0),
        "forward_reduction": summary["forward_reduction"],
        "peak_gpu_memory_bytes": summary["peak_allocated_bytes"],
        "formula": summary["formula"],
        "num_atoms": summary["num_atoms"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "adapter_checkpoint_sha256": summary.get("adapter_checkpoint_sha256") or "",
        "output_path": str(structure_path.resolve()),
        "output_sha256": sha256(structure_path),
        "run_summary_path": str(summary_path.resolve()),
        "run_summary_sha256": sha256(summary_path),
        "failure_summary_path": "",
    }


def validate_row(row: dict[str, Any]) -> list[str]:
    errors = []
    label = str(row["method"])
    if row.get("success") is not True:
        return ["missing or failed generation"]
    if label == "C0":
        expectations = {
            "logical_score_calls": 2000,
            "second_forward_calls": 1000,
            "adapter_calls": 0,
            "fallback_calls": 0,
        }
    else:
        expectations = {
            "late_exact_calls": 300,
            "adapter_calls": 700,
        }
        if int(row["logical_score_calls"]) + int(row["adapter_accept_calls"]) != 2000:
            errors.append("V2 exact and accepted-adapter calls do not sum to 2000")
        if int(row["atomic_risk_fallback_calls"]) + int(row["late_exact_calls"]) + int(
            row["nan_adapter_fallback_calls"]
        ) + int(row["nan_risk_fallback_calls"]) + int(row["invalid_risk_fallback_calls"]) != int(
            row["fallback_calls"]
        ):
            errors.append("V2 fallback reason counts do not sum to fallback_calls")
    for name, expected in expectations.items():
        if int(row[name]) != expected:
            errors.append(f"{name}={row[name]} expected {expected}")
    return errors


def main() -> None:
    args = parse_args()
    preflight = validate_frozen_protocol()
    if not preflight["passed"]:
        raise RuntimeError(preflight["failures"])
    if args.mode == "main":
        root = EXPERIMENT_ROOT
        seeds = FORMAL_SEEDS
        csv_path = root / "generation_per_seed.csv"
        audit_path = root / "formal256_generation_audit.json"
        structures_dir = root / "structures"
    else:
        root = EXPERIMENT_ROOT / "single_h20"
        seeds = SINGLE_H20_SEEDS
        csv_path = EXPERIMENT_ROOT / "single_h20_benchmark.csv"
        audit_path = root / "generation_audit.json"
        structures_dir = root / "structures"
    rows = [row_for(root, method, seed) for seed in seeds for method in METHODS]
    for path in (csv_path, audit_path):
        if path.exists():
            raise FileExistsError(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = []
    for row in rows:
        for error in validate_row(row):
            errors.append({"method": row["method"], "seed": row["seed"], "error": error})
    methods = {
        method: {
            "expected": len(seeds),
            "success": sum(row.get("success") is True for row in rows if row["method"] == method),
            "seed_set_exact": sorted(row["seed"] for row in rows if row["method"] == method) == list(seeds),
        }
        for method in METHODS
    }
    hashes_unique_within_method = {
        method: len({row.get("output_sha256") for row in rows if row["method"] == method}) == len(seeds)
        for method in METHODS
    }
    audit = {
        "schema_version": 1,
        "mode": args.mode,
        "methods": methods,
        "errors": errors,
        "only_allowed_methods": sorted({row["method"] for row in rows}) == sorted(METHODS),
        "structure_hashes_unique_within_method": hashes_unique_within_method,
        "fixed_gpu_shards": (
            {str(gpu): [values[0], values[-1]] for gpu, values in fixed_shards().items()}
            if args.mode == "main"
            else {"0": [seeds[0], seeds[-1]]}
        ),
        "passed": not errors
        and all(item["success"] == item["expected"] and item["seed_set_exact"] for item in methods.values()),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError(f"generation audit failed: {errors[:5]}")

    structures_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        atoms = []
        for row in (item for item in rows if item["method"] == method):
            generated = read(row["output_path"], index=":")
            if len(generated) != 1:
                raise ValueError(f"expected one structure for {method} seed {row['seed']}")
            generated[0].info["sample_seed"] = int(row["seed"])
            generated[0].info["sample_index_within_seed"] = 0
            generated[0].info["benchmark_method"] = method
            atoms.extend(generated)
        write(structures_dir / f"{method}_generated.extxyz", atoms)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
