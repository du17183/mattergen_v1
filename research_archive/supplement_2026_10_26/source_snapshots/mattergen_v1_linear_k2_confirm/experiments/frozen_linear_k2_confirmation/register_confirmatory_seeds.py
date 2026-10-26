"""Register all 256 fresh confirmatory seeds before generation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml


WORKTREE = Path(__file__).resolve().parents[2]
ROOT = WORKTREE / "experiments/frozen_linear_k2_confirmation"
PROTOCOL_ROOT = ROOT / "protocol"
SEARCH_ROOT = Path("/mnt/datasets-livsyn/dxl")
TOTAL = 256
REGISTRATION_SALT = "frozen_linear_k2_confirm_seed_registry_v1"
RANDOM_SALT = "frozen_linear_k2_confirm_random_v1"
POLICIES = ("GPulse", "APulse", "PPulse", "CPulse")


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
    roots = []
    for repository in sorted(path for path in SEARCH_ROOT.glob("mattergen_v1*") if path.is_dir()):
        for name in ("experiments", "diagnostics", "research"):
            path = repository / name
            if path.is_dir():
                roots.append(path)
    command = ["rg", "--files", "--hidden", "--no-ignore", "-g", "!**/.git/**", "-g", "*.json", "-g", "*.yaml", "-g", "*.yml", "-g", "*.csv", *map(str, roots)]
    return sorted({Path(raw) for raw in subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()})


def historical_seeds() -> tuple[set[int], list[dict[str, Any]]]:
    historical: set[int] = set()
    sources: list[dict[str, Any]] = []
    destinations = {(PROTOCOL_ROOT / "seed_manifest_256.json").resolve(), (PROTOCOL_ROOT / "seed_overlap_audit.json").resolve()}
    for path in candidate_files():
        if not path.is_file() or path.resolve() in destinations:
            continue
        values: set[int] = set()
        try:
            suffix, size = path.suffix.lower(), path.stat().st_size
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


def random_k2(seed: int) -> list[str]:
    return sorted(POLICIES, key=lambda policy: hashlib.sha256(f"{RANDOM_SALT}:{seed}:{policy}".encode()).hexdigest())[:2]


def write_once(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    audit = json.loads((PROTOCOL_ROOT / "reconstruction_audit.json").read_text())
    manifest = json.loads((PROTOCOL_ROOT / "frozen_method_manifest.json").read_text())
    if not audit.get("success") or audit.get("artifact_status") != "RECONSTRUCTED_ARTIFACT":
        raise RuntimeError("successful reconstructed artifact is required before seed registration")
    artifact = PROTOCOL_ROOT / "frozen_linear_k2_pipeline.joblib"
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["allocator"]["artifact_sha256"]:
        raise RuntimeError("artifact hash mismatch before seed registration")
    historical, sources = historical_seeds()
    selected: list[int] = []
    ordinal = 0
    while len(selected) < TOTAL:
        candidate = derived_seed(ordinal)
        ordinal += 1
        if candidate not in historical and candidate not in selected:
            selected.append(candidate)
    c1, c2 = selected[:128], selected[128:]
    assignments = {str(seed): random_k2(seed) for seed in selected}
    seed_manifest = {
        "schema_version": 1,
        "registered_before_any_confirmatory_generation": True,
        "count": TOTAL,
        "registration_salt": REGISTRATION_SALT,
        "random_k2_salt": RANDOM_SALT,
        "seeds": selected,
        "c1_128": c1,
        "c2_128": c2,
        "random_k2_assignments": assignments,
        "all_unique": len(set(selected)) == TOTAL,
        "historical_overlap": sorted(set(selected) & historical),
        "c1_c2_overlap": sorted(set(c1) & set(c2)),
    }
    overlap_audit = {
        "scan_root": str(SEARCH_ROOT),
        "scan_scope": "all JSON/YAML/CSV under experiments, diagnostics, research in every mattergen_v1* worktree",
        "historical_unique_seed_count": len(historical),
        "historical_files_with_seed_values": len(sources),
        "historical_sources": sources,
        "candidate_ordinals_examined": ordinal,
        "new_unique_seed_count": len(set(selected)),
        "new_vs_historical_overlap": sorted(set(selected) & historical),
        "c1_c2_overlap": sorted(set(c1) & set(c2)),
    }
    if seed_manifest["historical_overlap"] or seed_manifest["c1_c2_overlap"]:
        raise RuntimeError("confirmatory seed overlap audit failed")
    write_once(PROTOCOL_ROOT / "seed_manifest_256.json", seed_manifest)
    write_once(PROTOCOL_ROOT / "seed_overlap_audit.json", overlap_audit)
    print(json.dumps({"registered": TOTAL, "c1": len(c1), "c2": len(c2), "historical_unique": len(historical), "overlap": 0}))


if __name__ == "__main__":
    main()
