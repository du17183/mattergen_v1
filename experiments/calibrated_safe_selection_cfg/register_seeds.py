"""Register all fresh cohorts and trajectory splits before any new outcome exists."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/calibrated_safe_selection_cfg"
SEARCH_ROOT = Path("/mnt/datasets-livsyn/dxl")
REGISTRATION_RNG_SEED = 2026091502
COUNTS = {"dataset": 128, "p0": 32, "formal256": 256}
SPLIT_COUNTS = {"train": 64, "calibration": 26, "validation": 19, "test": 19}


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


def candidate_files() -> list[Path]:
    roots = sorted(path for path in SEARCH_ROOT.glob("mattergen_v1*") if path.is_dir())
    scan_roots: list[Path] = []
    for repository in roots:
        for name in ("experiments", "diagnostics", "research"):
            path = repository / name
            if path.is_dir():
                scan_roots.append(path)
    if not scan_roots:
        raise RuntimeError("no historical experiment roots found")
    listed = subprocess.run(
        ["rg", "--files", *map(str, scan_roots)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    exact_names = {
        "generation_manifest.csv",
        "branch_manifest.csv",
        "controller_trace.csv",
        "oracle_dataset.csv",
        "split_manifest.json",
        "future_seed_reservations.json",
    }
    return sorted(
        {
            Path(raw)
            for raw in listed
            if "seed" in Path(raw).name.lower() or Path(raw).name in exact_names
        }
    )


def historical_seeds() -> tuple[set[int], list[dict[str, Any]]]:
    found: set[int] = set()
    sources: list[dict[str, Any]] = []
    for path in candidate_files():
        if ROOT.resolve() in path.resolve().parents or not path.is_file():
            continue
        values: set[int] = set()
        try:
            suffix = path.suffix.lower()
            size = path.stat().st_size
            if suffix == ".json" and size <= 25_000_000:
                values = values_from_json(json.loads(path.read_text(errors="replace")))
            elif suffix == ".csv" and size <= 150_000_000:
                with path.open(newline="", encoding="utf-8", errors="replace") as stream:
                    reader = csv.DictReader(stream)
                    columns = [name for name in (reader.fieldnames or []) if "seed" in name.lower()]
                    for row in reader:
                        for name in columns:
                            try:
                                values.add(int(float(row.get(name, ""))))
                            except (TypeError, ValueError):
                                pass
            elif suffix in (".yaml", ".yml") and size <= 5_000_000:
                for line in path.read_text(errors="replace").splitlines():
                    if "seed" in line.lower():
                        values.update(
                            map(int, re.findall(r"(?<![\w.])-?\d+(?![\w.])", line))
                        )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if values:
            found.update(values)
            sources.append({"path": str(path), "seed_count": len(values)})
    return found, sources


def write_json_once(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace frozen registration: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    destinations = (
        ROOT / "dataset/seeds.json",
        ROOT / "dataset/seed_overlap_audit.json",
        ROOT / "future_seed_reservations.json",
        ROOT / "splits/split_manifest.json",
    )
    if any(path.exists() for path in destinations):
        raise FileExistsError("one or more frozen seed outputs already exist")
    historical, sources = historical_seeds()
    rng = np.random.default_rng(REGISTRATION_RNG_SEED)
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
    split: dict[str, list[int]] = {}
    cursor = 0
    for name, count in SPLIT_COUNTS.items():
        split[name] = dataset[cursor : cursor + count]
        cursor += count
    if cursor != len(dataset):
        raise RuntimeError("split counts do not consume the registered dataset")
    write_json_once(
        ROOT / "dataset/seeds.json",
        {
            "schema_version": 1,
            "purpose": "independent_step250_full_outcome_dataset",
            "registration_rng_seed": REGISTRATION_RNG_SEED,
            "seeds": dataset,
            "count": len(dataset),
            "historical_overlap": len(set(dataset) & historical),
        },
    )
    write_json_once(
        ROOT / "future_seed_reservations.json",
        {
            "schema_version": 1,
            "frozen_before_dataset_outcomes": True,
            "p0_seeds": p0,
            "formal256_seeds": formal,
            "all_reserved_unique": len(set(selected)) == needed,
            "historical_overlap": len(set(selected) & historical),
            "use_rule": "P0 only after offline GO; Formal256 only after P0 GO",
        },
    )
    write_json_once(
        ROOT / "splits/split_manifest.json",
        {
            "schema_version": 1,
            "frozen_before_generation": True,
            "split_unit": "trajectory_seed",
            "assignment_rule": "contiguous frozen registration order",
            "counts": SPLIT_COUNTS,
            **split,
            "disjoint": len(set().union(*map(set, split.values()))) == len(dataset),
            "all_registered_dataset_seeds_assigned": set().union(*map(set, split.values()))
            == set(dataset),
        },
    )
    write_json_once(
        ROOT / "dataset/seed_overlap_audit.json",
        {
            "scan_roots": ["experiments", "diagnostics", "research", "all mattergen_v1* worktrees"],
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
    )
    print(
        json.dumps(
            {
                "historical_unique": len(historical),
                "historical_sources": len(sources),
                "registered": COUNTS,
                "splits": SPLIT_COUNTS,
                "overlap": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
