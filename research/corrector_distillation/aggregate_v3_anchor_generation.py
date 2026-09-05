"""Aggregate V3 generation outputs and materialize per-method structures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ase.io import read, write

from research.corrector_distillation.formal256_protocol import sha256
from research.corrector_distillation.v3_anchor_protocol import (
    METHOD_TO_K,
    methods_for,
    output_root_for,
    seeds_for,
    validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("smoke", "stage-b", "single-h20-b", "stage-c", "single-h20-c"),
        required=True,
    )
    return parser.parse_args()


def reason_counts(path: Path) -> Counter[str]:
    if not path.is_file():
        return Counter()
    with path.open(newline="", encoding="utf-8") as stream:
        return Counter(
            row["fallback_reason"]
            for row in csv.DictReader(stream)
            if row["fallback_reason"]
        )


def row_for(root: Path, method: str, seed: int) -> dict[str, Any]:
    run_dir = root / "generation" / method / str(seed)
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        return {"method": method, "seed": seed, "success": False}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    structure_path = run_dir / "generated_crystals.extxyz"
    counts = reason_counts(run_dir / "residual_trace.csv")
    return {
        "method": method,
        "seed": seed,
        "success": summary.get("success") is True and structure_path.is_file(),
        "generation_time_seconds": summary["elapsed_seconds"],
        "samples_per_hour": summary["samples_per_hour"],
        "logical_score_calls": summary["mattergen_score_calls"],
        "second_forward_calls": int(summary["mattergen_score_calls"]) - 1000,
        "adapter_calls": summary.get("adapter_calls", 0),
        "adapter_accept_calls": summary.get("adapter_accept_calls", 0),
        "fallback_calls": summary.get("fallback_calls", 0),
        "atomic_risk_fallback_calls": summary.get("field_risk_fallback_calls", 0),
        "periodic_exact_calls": summary.get("periodic_exact_calls", 0),
        "late_exact_calls": summary.get("late_exact_calls", 0),
        "nonfinite_adapter_fallback_calls": counts["non_finite_adapter_output"],
        "nonfinite_risk_fallback_calls": counts["non_finite_uncertainty"],
        "invalid_risk_fallback_calls": counts["missing_or_invalid_calibration"],
        "periodic_trace_calls": counts["periodic_exact_anchor"],
        "coverage": summary.get("adapter_acceptance_overall", 0.0),
        "forward_reduction": summary["forward_reduction"],
        "anchor_k": summary.get("periodic_exact_anchor_k") or 0,
        "adapter_streak_mean": summary.get("adapter_streak_mean", 0.0),
        "adapter_streak_p95": summary.get("adapter_streak_p95", 0),
        "adapter_streak_max": summary.get("adapter_streak_max", 0),
        "peak_gpu_memory_bytes": summary["peak_allocated_bytes"],
        "formula": summary["formula"],
        "num_atoms": summary["num_atoms"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "adapter_checkpoint_sha256": summary.get("adapter_checkpoint_sha256") or "",
        "output_path": str(structure_path.resolve()),
        "output_sha256": sha256(structure_path),
        "run_summary_path": str(summary_path.resolve()),
    }


def validate_row(row: dict[str, Any]) -> list[str]:
    if row.get("success") is not True:
        return ["missing or failed generation"]
    method = str(row["method"])
    errors: list[str] = []
    if method == "C0":
        if int(row["logical_score_calls"]) != 2000:
            errors.append("C0 logical_score_calls must equal 2000")
        if int(row["adapter_calls"]) or int(row["fallback_calls"]):
            errors.append("C0 unexpectedly used adapter/fallback")
        return errors
    if int(row["adapter_calls"]) != 700 or int(row["late_exact_calls"]) != 300:
        errors.append("adapter/late-exact opportunity count mismatch")
    if int(row["logical_score_calls"]) + int(row["adapter_accept_calls"]) != 2000:
        errors.append("exact plus accepted-adapter calls do not sum to 2000")
    if int(row["periodic_exact_calls"]) != int(row["periodic_trace_calls"]):
        errors.append("periodic metric and trace count differ")
    expected_k = METHOD_TO_K.get(method, 0)
    if int(row["anchor_k"]) != expected_k:
        errors.append(f"anchor_k={row['anchor_k']} expected {expected_k}")
    if expected_k and int(row["adapter_streak_max"]) > expected_k:
        errors.append("adapter streak exceeded configured K")
    if not expected_k and int(row["periodic_exact_calls"]):
        errors.append("V2 periodic anchor was not disabled")
    return errors


def main() -> None:
    args = parse_args()
    preflight = validate_protocol()
    if not preflight["passed"]:
        raise RuntimeError(preflight["failures"])
    root = output_root_for(args.stage)
    methods = methods_for(args.stage)
    seeds = seeds_for(args.stage)
    rows = [row_for(root, method, seed) for seed in seeds for method in methods]
    errors = [
        {"method": row["method"], "seed": row["seed"], "error": error}
        for row in rows
        for error in validate_row(row)
    ]
    output_csv = root / "generation_per_seed.csv"
    if output_csv.exists():
        raise FileExistsError(output_csv)
    with output_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema_version": 1,
        "stage": args.stage,
        "expected_per_method": len(seeds),
        "methods": {
            method: {
                "success": sum(
                    row.get("success") is True
                    for row in rows
                    if row["method"] == method
                ),
                "seed_set_exact": sorted(
                    int(row["seed"]) for row in rows if row["method"] == method
                )
                == list(seeds),
            }
            for method in methods
        },
        "errors": errors,
        "passed": not errors,
    }
    audit_path = root / "generation_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["passed"]:
        raise RuntimeError(f"V3 generation audit failed: {errors[:5]}")
    structures_dir = root / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
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
