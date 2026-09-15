"""Register disjoint dataset and future evaluation seeds before any outcomes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/risk_calibrated_adaptive_cfg"
SEARCH_ROOT = Path("/mnt/datasets-livsyn/dxl")
RNG_SEED = 20260915
COUNTS = {"dataset": 96, "p0": 32, "formal256": 256}


def values_from_json(value: Any, key: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.update(values_from_json(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.update(values_from_json(child, key))
    elif "seed" in key.lower() and isinstance(value, (int, float)):
        if float(value).is_integer():
            found.add(int(value))
    elif "seed" in key.lower() and isinstance(value, str):
        found.update(map(int, re.findall(r"(?<![\w.])-?\d+(?![\w.])", value)))
    return found


def historical_seeds() -> tuple[set[int], list[dict[str, Any]]]:
    found: set[int] = set()
    audit: list[dict[str, Any]] = []
    current = ROOT.resolve()
    roots = sorted(
        path for path in SEARCH_ROOT.glob("mattergen_v1*")
        if path.is_dir() and path.resolve() != PROJECT.resolve()
    )
    roots.append(Path("/mnt/datasets-livsyn/dxl/mattergen_v1"))
    experiment_roots = [root / "experiments" for root in roots if (root / "experiments").exists()]
    listed = subprocess.run(
        ["rg", "--files", *map(str, experiment_roots)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    exact_names = {
        "generation_manifest.csv", "branch_manifest.csv", "controller_trace.csv",
        "oracle_dataset.csv",
    }
    candidates = {
        Path(raw) for raw in listed
        if "seed" in Path(raw).name.lower()
        or Path(raw).name in exact_names
    }
    for path in sorted(candidates):
            if not path.is_file() or current in path.resolve().parents:
                continue
            values: set[int] = set()
            try:
                if path.suffix == ".json" and "seed" in path.name.lower():
                    values = values_from_json(json.loads(path.read_text(errors="replace")))
                elif path.suffix == ".json" and path.stat().st_size <= 20_000_000:
                    payload = json.loads(path.read_text(errors="replace"))
                    values = values_from_json(payload)
                elif path.suffix == ".csv" and path.stat().st_size <= 100_000_000:
                    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
                        reader = csv.DictReader(stream)
                        seed_columns = [name for name in (reader.fieldnames or []) if "seed" in name.lower()]
                        for row in reader:
                            for name in seed_columns:
                                raw = row.get(name, "")
                                try:
                                    values.add(int(float(raw)))
                                except (TypeError, ValueError):
                                    pass
                elif path.suffix in (".yaml", ".yml") and path.stat().st_size <= 5_000_000:
                    for line in path.read_text(errors="replace").splitlines():
                        if "seed" in line.lower():
                            values.update(map(int, re.findall(r"(?<![\w.])-?\d+(?![\w.])", line)))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if values:
                found.update(values)
                audit.append({"path": str(path), "seed_count": len(values)})
    return found, audit


def main() -> None:
    dataset_path = ROOT / "dataset/seeds.json"
    reservation_path = ROOT / "future_seed_reservations.json"
    audit_path = ROOT / "fresh_seed_audit.json"
    for path in (dataset_path, reservation_path, audit_path):
        if path.exists():
            raise FileExistsError(f"refusing to replace registered seeds: {path}")
    historical, sources = historical_seeds()
    rng = np.random.default_rng(RNG_SEED)
    needed = sum(COUNTS.values())
    selected: list[int] = []
    while len(selected) < needed:
        candidate = int(rng.integers(100_000_000, 2_000_000_000))
        if candidate not in historical and candidate not in selected:
            selected.append(candidate)
    dataset = selected[: COUNTS["dataset"]]
    p0_start = COUNTS["dataset"]
    formal_start = p0_start + COUNTS["p0"]
    p0 = selected[p0_start:formal_start]
    formal = selected[formal_start:]
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "full_outcome_counterfactual_dataset",
                "registration_rng_seed": RNG_SEED,
                "seeds": dataset,
                "count": len(dataset),
                "historical_overlap": len(set(dataset) & historical),
            },
            indent=2,
        )
        + "\n"
    )
    reservation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frozen_before_dataset_outcomes": True,
                "p0_seeds": p0,
                "formal256_seeds": formal,
                "all_reserved_unique": len(set(selected)) == needed,
                "historical_overlap": len(set(selected) & historical),
                "use_rule": "P0 only after offline GO; Formal256 only after P0 GO",
            },
            indent=2,
        )
        + "\n"
    )
    audit_path.write_text(
        json.dumps(
            {
                "historical_unique_seed_count": len(historical),
                "historical_files_with_seed_values": len(sources),
                "historical_sources": sources,
                "new_counts": COUNTS,
                "new_unique_seed_count": len(set(selected)),
                "new_vs_historical_overlap": sorted(set(selected) & historical),
                "cross_cohort_overlap": {
                    "dataset_p0": sorted(set(dataset) & set(p0)),
                    "dataset_formal256": sorted(set(dataset) & set(formal)),
                    "p0_formal256": sorted(set(p0) & set(formal)),
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"historical": len(historical), "registered": COUNTS, "overlap": 0}))


if __name__ == "__main__":
    main()
