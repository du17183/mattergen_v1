"""Reserve mutually disjoint calibration, P0, and Formal cohorts up front."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/stage_calibrated_bounded_cfg"
SEARCH_ROOT = Path("/mnt/datasets-livsyn/dxl")
REGISTRATION_RNG_SEED = 2026091503
COUNTS = {"calibration": 32, "p0": 32, "formal256": 256}


def values_from_json(value: Any, key: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.update(values_from_json(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.update(values_from_json(child, key))
    elif "seed" in key.lower() and isinstance(value, (int, float)) and float(value).is_integer():
        found.add(int(value))
    elif "seed" in key.lower() and isinstance(value, str):
        found.update(map(int, re.findall(r"(?<![\w.])-?\d+(?![\w.])", value)))
    return found


def candidate_files() -> list[Path]:
    scan_roots: list[Path] = []
    for repository in sorted(path for path in SEARCH_ROOT.glob("mattergen_v1*") if path.is_dir()):
        for name in ("experiments", "diagnostics", "research"):
            path = repository / name
            if path.is_dir():
                scan_roots.append(path)
    listed = subprocess.run(
        ["rg", "--files", *map(str, scan_roots)], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    exact = {"generation_manifest.csv", "branch_manifest.csv", "controller_trace.csv", "oracle_dataset.csv", "future_seed_reservations.json"}
    return sorted({Path(raw) for raw in listed if "seed" in Path(raw).name.lower() or Path(raw).name in exact})


def historical_seeds() -> tuple[set[int], list[dict[str, Any]]]:
    found: set[int] = set()
    sources: list[dict[str, Any]] = []
    for path in candidate_files():
        if ROOT.resolve() in path.resolve().parents or not path.is_file():
            continue
        values: set[int] = set()
        try:
            suffix, size = path.suffix.lower(), path.stat().st_size
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
                        values.update(map(int, re.findall(r"(?<![\w.])-?\d+(?![\w.])", line)))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if values:
            found.update(values)
            sources.append({"path": str(path), "seed_count": len(values)})
    return found, sources


def write_json_once(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    destinations = (ROOT / "calibration/seeds.json", ROOT / "future_seed_reservations.json", ROOT / "seed_overlap_audit.json")
    if any(path.exists() for path in destinations):
        raise FileExistsError("seed registration already exists")
    historical, sources = historical_seeds()
    rng = np.random.default_rng(REGISTRATION_RNG_SEED)
    selected: list[int] = []
    while len(selected) < sum(COUNTS.values()):
        candidate = int(rng.integers(100_000_000, 2_000_000_000))
        if candidate not in historical and candidate not in selected:
            selected.append(candidate)
    calibration = selected[: COUNTS["calibration"]]
    p0 = selected[COUNTS["calibration"] : COUNTS["calibration"] + COUNTS["p0"]]
    formal = selected[-COUNTS["formal256"] :]
    write_json_once(ROOT / "calibration/seeds.json", {"schema_version": 1, "purpose": "schedule_calibration", "registration_rng_seed": REGISTRATION_RNG_SEED, "count": len(calibration), "seeds": calibration, "historical_overlap": len(set(calibration) & historical)})
    write_json_once(ROOT / "future_seed_reservations.json", {"schema_version": 1, "frozen_before_calibration_outcomes": True, "p0_seeds": p0, "formal256_seeds": formal, "all_reserved_unique": len(set(selected)) == len(selected), "historical_overlap": len(set(selected) & historical), "use_rule": "P0 only after calibration GO; Formal256 only after P0 GO"})
    write_json_once(ROOT / "seed_overlap_audit.json", {"historical_unique_seed_count": len(historical), "historical_files_with_seed_values": len(sources), "historical_sources": sources, "new_counts": COUNTS, "new_unique_seed_count": len(set(selected)), "new_vs_historical_overlap": sorted(set(selected) & historical), "cross_cohort_overlap": {"calibration_p0": sorted(set(calibration) & set(p0)), "calibration_formal256": sorted(set(calibration) & set(formal)), "p0_formal256": sorted(set(p0) & set(formal))}})
    print(json.dumps({"registered": COUNTS, "historical_unique": len(historical), "overlap": 0}))


if __name__ == "__main__":
    main()
