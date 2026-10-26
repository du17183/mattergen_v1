"""Audit preregistered probe/search seeds against historical text artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/self_correcting_search_p0"
SEARCH_ROOTS = (
    Path("/mnt/datasets-livsyn/dxl/mattergen_v1/experiments"),
    Path("/mnt/datasets-livsyn/dxl/mattergen_v1_adaptive_neighbor/experiments"),
    Path("/mnt/datasets-livsyn/dxl/mattergen_v1_self_condition/experiments"),
)
SELECTED = set(range(98000, 98192)) | set(range(98200, 98208))
SELECTED_PATTERN = re.compile(r"(?<![0-9.])(98(?:0\d{2}|1(?:[0-8]\d|9[01])|20[0-7]))(?![0-9.])")
SEED_ASSIGNMENT = re.compile(
    r"(?i)(?:\bseeds?\b\s*[:=]|--seed(?:=|\s)|\brange\s*\()\s*"
    r"(98(?:0\d{2}|1(?:[0-8]\d|9[01])|20[0-7]))"
)


def main() -> None:
    matches = []
    scanned = 0
    for search_root in SEARCH_ROOTS:
        files = subprocess.run(
            ["rg", "--files", str(search_root), "-g", "*.py", "-g", "*.json",
             "-g", "*.yaml", "-g", "*.yml", "-g", "*.csv", "-g", "*.md",
             "-g", "*.txt", "-g", "*.sh"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        scanned += len(files)
        for file_text in files:
            path = Path(file_text)
            if any(part.isdigit() and int(part) in SELECTED for part in path.parts):
                matches.append({"seed": next(int(part) for part in path.parts
                                               if part.isdigit() and int(part) in SELECTED),
                                "path": str(path), "line": 0})
            if path.suffix.lower() == ".csv":
                try:
                    with path.open(newline="", encoding="utf-8", errors="ignore") as stream:
                        reader = csv.DictReader(stream)
                        seed_columns = [name for name in (reader.fieldnames or [])
                                        if name and (name.lower() == "seed" or name.lower().endswith("_seed"))]
                        for line_number, row in enumerate(reader, start=2):
                            for column in seed_columns:
                                try:
                                    value = int(float(row[column]))
                                except (TypeError, ValueError):
                                    continue
                                if value in SELECTED:
                                    matches.append({"seed": value, "path": str(path),
                                                    "line": line_number})
                except OSError:
                    continue
            else:
                try:
                    text = path.read_text(errors="ignore")
                except OSError:
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    for found in SEED_ASSIGNMENT.finditer(line):
                        matches.append({"seed": int(found.group(1)), "path": str(path),
                                        "line": line_number})
    result = {
        "selected_probe_range": [98000, 98191],
        "selected_search_range": [98200, 98207],
        "explicitly_forbidden_range": [94000, 96031],
        "scanned_roots": [str(path) for path in SEARCH_ROOTS],
        "text_files_scanned": scanned,
        "selected_seed_matches_in_historical_roots": matches,
        "historical_overlap": bool(matches),
        "numeric_rule": "seed-key/assignment, --seed, range start, CSV seed column, or numeric path component; hashes and decimals excluded",
    }
    (ROOT / "seed_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if matches:
        raise RuntimeError(f"selected seed overlap: {matches[:10]}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
