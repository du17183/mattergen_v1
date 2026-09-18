#!/usr/bin/env python3
"""Build the archive inventory, index and per-experiment landing pages."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path

from catalog import EXPERIMENTS


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "research_archive"
CSV_FIELDS = [
    "experiment_id",
    "experiment_name",
    "source_worktree",
    "source_branch",
    "source_commit",
    "source_path",
    "category",
    "status",
    "scientific_question",
    "main_result",
    "used_in_thesis",
    "covered_by_final_release",
    "archive_target",
    "size",
    "notes",
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def size_text(row: dict[str, str]) -> str:
    if row["source_worktree"] in {"remote-only"} or row["source_path"].startswith("git:"):
        return "git-object"
    path = Path(row["source_worktree"]) / row["source_path"]
    if not path.exists():
        return "missing-at-inventory"
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())
    return str(total)


def stable_commit(row: dict[str, str]) -> str:
    value = row["source_commit"]
    try:
        return git("rev-parse", value)
    except subprocess.CalledProcessError:
        return value


def link_for(row: dict[str, str]) -> str:
    target = Path(row["archive_target"])
    return target.relative_to("research_archive").as_posix() + "/README.md"


def write_csv(rows: list[dict[str, str]]) -> None:
    out = ARCHIVE / "experiment_inventory.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def write_index(rows: list[dict[str, str]]) -> None:
    lines = [
        "# Experiment Index",
        "",
        "This index is ordered by research lineage. Statuses use the archive's frozen vocabulary; historical GO/FAIL wording remains in the source reports.",
        "",
        "| ID | Experiment | Question | Status | Key finding | Thesis relevance | Archive level |",
        "| -- | ---------- | -------- | ------ | ----------- | ---------------- | ------------- |",
    ]
    for row in rows:
        question = row["scientific_question"].replace("|", "/")
        finding = row["main_result"].replace("|", "/")
        lines.append(
            f'| {row["experiment_id"]} | [{row["experiment_name"]}]({link_for(row)}) | '
            f'{question} | {row["status"]} | {finding} | {row["used_in_thesis"]} | {row["category"]} |'
        )
    (ARCHIVE / "EXPERIMENT_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_inventory_md(rows: list[dict[str, str]]) -> None:
    categories = Counter(row["category"] for row in rows)
    statuses = Counter(row["status"] for row in rows)
    lines = [
        "# Exploration Archive Inventory",
        "",
        f"Inventory date: {date.today().isoformat()}",
        "",
        f"Experiments found: **{len(rows)}**.",
        "",
        "## Category counts",
        "",
    ]
    for category in ("FULL_ARCHIVE", "SUMMARY_ONLY", "EXTERNAL_DATA_ONLY", "IGNORE"):
        lines.append(f"- `{category}`: {categories[category]}")
    lines += ["", "## Status counts", ""]
    for status, count in sorted(statuses.items()):
        lines.append(f"- `{status}`: {count}")
    lines += [
        "",
        "## Manual review required",
        "",
    ]
    manual = [row for row in rows if "MANUAL_REVIEW_REQUIRED" in row["notes"]]
    for row in manual:
        lines.append(f'- `{row["experiment_id"]}` {row["experiment_name"]}: {row["main_result"]}')
    lines += [
        "",
        "## Full table",
        "",
        "The machine-readable source of truth is [experiment_inventory.csv](experiment_inventory.csv).",
        "",
        "| ID | Name | Category | Status | Source | Covered by final release |",
        "| -- | ---- | -------- | ------ | ------ | ------------------------ |",
    ]
    for row in rows:
        source = f'{row["source_branch"]} @ {row["source_commit"][:12]}'
        lines.append(
            f'| {row["experiment_id"]} | [{row["experiment_name"]}]({link_for(row)}) | '
            f'{row["category"]} | {row["status"]} | {source} | {row["covered_by_final_release"]} |'
        )
    (ARCHIVE / "exploration_archive_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_locator(row: dict[str, str]) -> str:
    if row["source_worktree"] == "remote-only":
        return f'Git object path `{row["source_branch"]}:{row["source_path"].removeprefix("git:")}`'
    return f'Original server path `{Path(row["source_worktree"]) / row["source_path"]}`'


def write_experiment_page(row: dict[str, str]) -> None:
    target = ROOT / row["archive_target"]
    if row["category"] == "IGNORE":
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.mkdir(parents=True, exist_ok=True)
    historical = (
        "See the immutable source report for the exact historical GO/FAIL wording. "
        "This archive does not overwrite or retroactively relabel that source."
    )
    body = f'''# {row["experiment_name"]}

## Research Question

{row["scientific_question"]}

## Motivation

{row["hypothesis"]}

## Change from Previous Stage

{row["change"]}

## Implementation

Archive category: `{row["category"]}`. Reproducibility level: `{row["reproducibility_level"]}`.

{source_locator(row)}. Portable reproduction must use repository-relative paths or an explicit user-supplied data root; the original server path is provenance only.

## Dataset / Seeds

{row["evidence"]} Seed manifests are copied when small; otherwise their immutable source path and external-data hash are recorded.

## Main Results

{row["main_result"]}

## Decision

Archive status: **{row["status"]}**.

Historical conclusion at the time: {historical}

Final thesis interpretation after later confirmation: {row["result"] if "result" in row else row["main_result"]}

## Why This Route Was Stopped

{row["why_stopped"]}

## What It Motivated Next

{row["next_direction"]}

## Provenance

- Source worktree: `{row["source_worktree"]}`
- Source branch: `{row["source_branch"]}`
- Source commit: `{row["source_commit"]}`
- Original path: `{row["source_path"]}`
- Archive date: `{date.today().isoformat()}`
- Covered by final release: `{row["covered_by_final_release"]}`
- Used in thesis: `{row["used_in_thesis"]}`
- Notes: `{row["notes"] or "none"}`
'''
    (target / "README.md").write_text(body, encoding="utf-8")
    (target / "SOURCE.md").write_text(
        f'''# Source locator

- Branch: `{row["source_branch"]}`
- Commit: `{row["source_commit"]}`
- Worktree at inventory time: `{row["source_worktree"]}`
- Original path: `{row["source_path"]}`
- Category: `{row["category"]}`
- Reproducibility: `{row["reproducibility_level"]}`

Absolute paths above are provenance, not portable reproduction defaults.
''',
        encoding="utf-8",
    )
    if row["category"] == "FULL_ARCHIVE":
        for subdir, purpose in {
            "code": "Selected implementation sources or a stable pointer to already-tracked code.",
            "config": "Frozen configuration and protocol artifacts.",
            "results": "Small decisions, metrics, bootstrap summaries and reports.",
            "figures": "Small key figures; large plots remain externally manifested.",
            "seeds": "Seed manifests or an immutable pointer to them.",
        }.items():
            folder = target / subdir
            folder.mkdir(exist_ok=True)
            marker = folder / "README.md"
            if not marker.exists():
                marker.write_text(
                    f"# {subdir.title()}\n\n{purpose}\n\nSee `../SOURCE.md` for the immutable source locator.\n",
                    encoding="utf-8",
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate catalog without writing")
    args = parser.parse_args()
    rows = []
    for source in EXPERIMENTS:
        row = dict(source)
        row["source_commit"] = stable_commit(row)
        row["size"] = size_text(row)
        rows.append(row)
    if args.check:
        print(f"catalog_ok experiments={len(rows)}")
        return 0
    ARCHIVE.mkdir(exist_ok=True)
    for row in rows:
        write_experiment_page(row)
    write_csv(rows)
    write_index(rows)
    write_inventory_md(rows)
    # User requested the long descriptive name as well as the canonical file.
    (ARCHIVE / "exploration_archive_inventory.csv").write_bytes((ARCHIVE / "experiment_inventory.csv").read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
