"""Register the fresh Phase-B cohort and immutable train/validation/test split."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/reference_preserved_budgeted_cfg"
FEATURE_ROOT = ROOT / "feature_study"
SEARCH_ROOT = Path("/mnt/datasets-livsyn/dxl")
PROTOCOL = yaml.safe_load((ROOT / "feature_study_protocol.yaml").read_text())
TOTAL = int(PROTOCOL["cohort"]["total"])
REGISTRATION_SALT = str(PROTOCOL["cohort"]["registration_salt"])
SPLIT_SALT = str(PROTOCOL["cohort"]["split_salt"])
SPLIT_COUNTS = {key: int(value) for key, value in PROTOCOL["cohort"]["split_counts"].items()}


def seed_values(value: Any, key: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.update(seed_values(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            found.update(seed_values(child, key))
    elif "seed" in key.lower() and isinstance(value, (int, float)) and float(value).is_integer():
        found.add(int(value))
    elif "seed" in key.lower() and isinstance(value, str):
        found.update(map(int, re.findall(r"(?<![\w.])-?\d+(?![\w.])", value)))
    return found


def candidate_files() -> list[Path]:
    roots: list[Path] = []
    for repository in sorted(path for path in SEARCH_ROOT.glob("mattergen_v1*") if path.is_dir()):
        for name in ("experiments", "diagnostics", "research"):
            candidate = repository / name
            if candidate.is_dir():
                roots.append(candidate)
    command = [
        "rg", "--files", "--hidden", "--no-ignore", "-g", "!**/.git/**",
        "-g", "*.json", "-g", "*.yaml", "-g", "*.yml", "-g", "*.csv",
        *map(str, roots),
    ]
    return sorted({Path(raw) for raw in subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()})


def historical_seeds() -> tuple[set[int], list[dict[str, Any]]]:
    historical: set[int] = set()
    sources: list[dict[str, Any]] = []
    destinations = {
        (FEATURE_ROOT / "feature_study_seed_manifest.json").resolve(),
        (FEATURE_ROOT / "feature_study_seed_audit.json").resolve(),
    }
    for path in candidate_files():
        if not path.is_file() or path.resolve() in destinations:
            continue
        values: set[int] = set()
        try:
            size = path.stat().st_size
            suffix = path.suffix.lower()
            if suffix == ".json" and size <= 50_000_000:
                values = seed_values(json.loads(path.read_text(errors="replace")))
            elif suffix in (".yaml", ".yml") and size <= 10_000_000:
                values = seed_values(yaml.safe_load(path.read_text(errors="replace")))
            elif suffix == ".csv" and size <= 250_000_000:
                with path.open(newline="", encoding="utf-8", errors="replace") as stream:
                    reader = csv.DictReader(stream)
                    columns = [name for name in (reader.fieldnames or []) if "seed" in name.lower()]
                    for row in reader:
                        for name in columns:
                            try:
                                values.add(int(float(row.get(name, ""))))
                            except (TypeError, ValueError):
                                pass
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
            continue
        values = {value for value in values if 0 <= value <= 2**63 - 1}
        if values:
            historical.update(values)
            sources.append({"path": str(path), "seed_count": len(values), "minimum": min(values), "maximum": max(values)})
    return historical, sources


def derived_seed(ordinal: int) -> int:
    digest = hashlib.sha256(f"{REGISTRATION_SALT}:{ordinal}".encode()).digest()
    return 100_000_000 + int.from_bytes(digest[:8], "big") % 1_900_000_000


def atomic_write_once(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    manifest_path = FEATURE_ROOT / "feature_study_seed_manifest.json"
    audit_path = FEATURE_ROOT / "feature_study_seed_audit.json"
    if manifest_path.exists() or audit_path.exists():
        raise FileExistsError("Phase-B seed registration is immutable and already exists")
    if sum(SPLIT_COUNTS.values()) != TOTAL:
        raise RuntimeError("split counts do not sum to cohort total")
    historical, sources = historical_seeds()
    selected: list[int] = []
    ordinal = 0
    while len(selected) < TOTAL:
        candidate = derived_seed(ordinal)
        ordinal += 1
        if candidate not in historical and candidate not in selected:
            selected.append(candidate)
    ordered = sorted(selected, key=lambda seed: hashlib.sha256(f"{SPLIT_SALT}:{seed}".encode()).hexdigest())
    train_end = SPLIT_COUNTS["train"]
    validation_end = train_end + SPLIT_COUNTS["validation"]
    splits = {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }
    assignment = {str(seed): split for split, seeds in splits.items() for seed in seeds}
    manifest = {
        "schema_version": 1,
        "preregistered_before_generation": True,
        "registration_method": "SHA256-derived candidates rejected on any historical/reserved overlap",
        "registration_salt": REGISTRATION_SALT,
        "split_method": "ascending SHA256(split_salt:seed)",
        "split_salt": SPLIT_SALT,
        "count": TOTAL,
        "split_counts": SPLIT_COUNTS,
        "seeds": selected,
        "splits": splits,
        "split_assignment": assignment,
        "all_unique": len(set(selected)) == TOTAL,
        "historical_overlap": sorted(set(selected) & historical),
    }
    audit = {
        "scan_root": str(SEARCH_ROOT),
        "scan_scope": "JSON/YAML/CSV under experiments, diagnostics, research in every mattergen_v1* worktree",
        "historical_unique_seed_count": len(historical),
        "historical_files_with_seed_values": len(sources),
        "historical_sources": sources,
        "candidate_ordinals_examined": ordinal,
        "new_unique_seed_count": len(set(selected)),
        "new_vs_historical_overlap": sorted(set(selected) & historical),
        "cross_split_overlap": {
            "train_validation": sorted(set(splits["train"]) & set(splits["validation"])),
            "train_test": sorted(set(splits["train"]) & set(splits["test"])),
            "validation_test": sorted(set(splits["validation"]) & set(splits["test"])),
        },
    }
    if manifest["historical_overlap"] or any(audit["cross_split_overlap"].values()):
        raise RuntimeError("fresh-seed audit failed")
    atomic_write_once(manifest_path, manifest)
    atomic_write_once(audit_path, audit)
    print(json.dumps({"registered": TOTAL, "splits": SPLIT_COUNTS, "historical_unique": len(historical), "overlap": 0}))


if __name__ == "__main__":
    main()
