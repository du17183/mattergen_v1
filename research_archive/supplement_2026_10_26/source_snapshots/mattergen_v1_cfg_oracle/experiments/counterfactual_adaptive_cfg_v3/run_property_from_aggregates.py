"""Resume the frozen CHGNet evaluation from validated aggregate structures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ase.io import read

import prepare_oracle_outputs as frozen_property


ORACLE = frozen_property.ORACLE


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def main() -> None:
    short_atoms = read(ORACLE / "short_horizon_x0_all.extxyz", index=":")
    full_atoms = read(ORACLE / "full_horizon_final_all.extxyz", index=":")
    base_atoms = read(ORACLE / "base_cfg2_final_all.extxyz", index=":")
    observed = {
        "short_structures": len(short_atoms),
        "full_structures": len(full_atoms),
        "base_structures": len(base_atoms),
        "online_features": csv_rows(ORACLE / "online_features_raw.csv"),
        "branch_manifest": csv_rows(ORACLE / "branch_manifest.csv"),
    }
    expected = {
        "short_structures": 768,
        "full_structures": 192,
        "base_structures": 32,
        "online_features": 256,
        "branch_manifest": 768,
    }
    if observed != expected:
        raise RuntimeError(f"aggregate count mismatch: observed={observed}, expected={expected}")
    if (ORACLE / "property_metrics.csv").exists() or (ORACLE / "evaluation_structures").exists():
        raise FileExistsError("refusing to overwrite an existing property evaluation")
    (ORACLE / "aggregate_resume_validation.json").write_text(
        json.dumps({"observed": observed, "expected": expected, "valid": True}, indent=2) + "\n"
    )
    frozen_property.evaluate_properties(short_atoms, full_atoms, base_atoms)


if __name__ == "__main__":
    main()
