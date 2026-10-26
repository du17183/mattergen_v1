"""Audit all preregistered diagnostic/A/B seeds across every MatterGen worktree."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/shared_field_dynamics"
MAIN = Path("/mnt/datasets-livsyn/dxl/mattergen_v1")
DIAGNOSTIC = tuple(range(99000, 99064))
ASYNC_FREEZE = tuple(range(99100, 99132))
FIELD_SCHEDULE = tuple(range(99200, 99232))
SELECTED = set(DIAGNOSTIC + ASYNC_FREEZE + FIELD_SCHEDULE)
NUMBER_PATTERN = "(?:" + "|".join(map(str, sorted(SELECTED))) + ")"
SEED_ASSIGNMENT = re.compile(
    rf"(?i)(?:\bseeds?\b\s*[:=]|--seed(?:=|\s)|\brange\s*\()\s*({NUMBER_PATTERN})(?![0-9.])"
)
TEXT_GLOBS = ("*.py", "*.json", "*.yaml", "*.yml", "*.csv", "*.md", "*.txt", "*.sh")


def worktrees() -> tuple[Path, ...]:
    output = subprocess.run(
        ["git", "-C", str(MAIN), "worktree", "list", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout
    return tuple(
        Path(line.split(" ", 1)[1])
        for line in output.splitlines() if line.startswith("worktree ")
    )


def excluded(path: Path) -> bool:
    resolved = str(path)
    return any(token in resolved for token in (
        "/experiments/shared_field_dynamics/",
        "/experiments/field_async_freeze_p0/",
        "/experiments/field_time_schedule_p0/",
    ))


def main() -> None:
    roots = []
    for tree in worktrees():
        for name in ("experiments", "diagnostics", "research", "thesis"):
            candidate = tree / name
            if candidate.is_dir():
                roots.append(candidate)
    matches = []
    scanned = 0
    for search_root in roots:
        command = ["rg", "--files", str(search_root)]
        for glob in TEXT_GLOBS:
            command.extend(("-g", glob))
        files = subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()
        for file_text in files:
            path = Path(file_text)
            if excluded(path):
                continue
            scanned += 1
            for part in path.parts:
                if part.isdigit() and int(part) in SELECTED:
                    matches.append({"seed": int(part), "path": str(path), "line": 0,
                                    "context": "numeric_path_component"})
            if path.suffix.lower() == ".csv":
                try:
                    with path.open(newline="", encoding="utf-8", errors="ignore") as stream:
                        reader = csv.DictReader(stream)
                        columns = [name for name in (reader.fieldnames or [])
                                   if name and (name.lower() == "seed" or name.lower().endswith("_seed"))]
                        for line_number, row in enumerate(reader, start=2):
                            for column in columns:
                                try:
                                    value = int(float(row[column]))
                                except (TypeError, ValueError):
                                    continue
                                if value in SELECTED:
                                    matches.append({"seed": value, "path": str(path),
                                                    "line": line_number, "context": column})
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
                                        "line": line_number, "context": "seed_assignment"})
    result = {
        "diagnostic_seeds": list(DIAGNOSTIC),
        "async_freeze_p0_seeds": list(ASYNC_FREEZE),
        "field_schedule_p0_seeds": list(FIELD_SCHEDULE),
        "explicitly_forbidden_range": [94000, 96031],
        "worktrees_scanned": [str(path) for path in worktrees()],
        "search_roots": [str(path) for path in roots],
        "text_files_scanned": scanned,
        "matches": matches,
        "historical_overlap": bool(matches),
        "sets_pairwise_disjoint": not (
            set(DIAGNOSTIC) & set(ASYNC_FREEZE)
            or set(DIAGNOSTIC) & set(FIELD_SCHEDULE)
            or set(ASYNC_FREEZE) & set(FIELD_SCHEDULE)
        ),
        "numeric_rule": "seed assignment/--seed/range start, seed CSV column, or numeric path component",
    }
    output = ROOT / "seeds.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if matches or not result["sets_pairwise_disjoint"]:
        raise RuntimeError(f"seed audit failed: {matches[:10]}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
