#!/usr/bin/env python3
"""Validate archive schema, structure, claims and new-file size policy."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "research_archive"
BASE = "5b572c61c70c4287147616beae7f72d28955cf98"
ALLOWED_CATEGORIES = {"FULL_ARCHIVE", "SUMMARY_ONLY", "EXTERNAL_DATA_ONLY", "IGNORE"}
ALLOWED_STATUSES = {"SUPPORTED", "MIXED", "NOT_SUPPORTED", "FAIL", "INCONCLUSIVE", "ENGINEERING_ONLY", "ABANDONED_BEFORE_EVALUATION"}
REQUIRED_COLUMNS = {
    "experiment_id", "experiment_name", "source_worktree", "source_branch", "source_commit",
    "source_path", "category", "status", "scientific_question", "main_result", "used_in_thesis",
    "covered_by_final_release", "archive_target", "size", "notes",
}


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(ROOT), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []
    required_files = [
        "README.md", "ARCHIVE_POLICY.md", "EXPERIMENT_INDEX.md", "experiment_inventory.csv",
        "exploration_archive_inventory.csv", "exploration_archive_inventory.md", "external_data_manifest.csv",
        "selected_artifact_manifest.csv", "source_branch_inventory.csv", "worktree_inventory.csv",
        "research_evolution.png", "research_evolution.pdf", "research_evolution.svg", "archive_checksums.json",
    ]
    for name in required_files:
        if not (ARCHIVE / name).is_file():
            fail(errors, f"missing required archive file: {name}")

    rows: list[dict[str, str]] = []
    inventory = ARCHIVE / "experiment_inventory.csv"
    if inventory.is_file():
        with inventory.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
                fail(errors, "experiment inventory is missing required columns")
            rows = list(reader)
    ids = [row.get("experiment_id", "") for row in rows]
    if len(rows) != 81:
        fail(errors, f"expected 81 experiment records, found {len(rows)}")
    if len(set(ids)) != len(ids):
        fail(errors, "duplicate experiment IDs")
    for row in rows:
        if row.get("category") not in ALLOWED_CATEGORIES:
            fail(errors, f"invalid category for {row.get('experiment_id')}")
        if row.get("status") not in ALLOWED_STATUSES:
            fail(errors, f"invalid status for {row.get('experiment_id')}")
        target_value = row.get("archive_target", "")
        target = ROOT / target_value
        try:
            target.resolve().relative_to(ARCHIVE.resolve())
        except ValueError:
            fail(errors, f"archive target escapes research_archive: {target_value}")
            continue
        for name in ("README.md", "SOURCE.md"):
            if not (target / name).is_file():
                fail(errors, f"{row.get('experiment_id')} missing {name}")
        if row.get("category") == "FULL_ARCHIVE":
            for subdir in ("code", "config", "results", "figures", "seeds"):
                if not (target / subdir).is_dir():
                    fail(errors, f"{row.get('experiment_id')} missing {subdir}/")

    index_text = (ARCHIVE / "EXPERIMENT_INDEX.md").read_text(encoding="utf-8") if (ARCHIVE / "EXPERIMENT_INDEX.md").is_file() else ""
    for experiment_id in ids:
        if experiment_id and experiment_id not in index_text:
            fail(errors, f"experiment missing from index: {experiment_id}")

    category_counts = Counter(row["category"] for row in rows)
    expected_counts = {"FULL_ARCHIVE": 21, "SUMMARY_ONLY": 56, "EXTERNAL_DATA_ONLY": 3, "IGNORE": 1}
    if dict(category_counts) != expected_counts:
        fail(errors, f"category counts changed: {dict(category_counts)}")

    external = ARCHIVE / "external_data_manifest.csv"
    external_rows = []
    if external.is_file():
        with external.open(encoding="utf-8", newline="") as handle:
            external_rows = list(csv.DictReader(handle))
    if len(external_rows) != 37:
        fail(errors, f"expected 37 external manifest rows, found {len(external_rows)}")
    for row in external_rows:
        digest = row.get("sha256", "")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            fail(errors, f"invalid external digest for {row.get('experiment')} / {row.get('artifact')}")

    readme = (ARCHIVE / "README.md").read_text(encoding="utf-8") if (ARCHIVE / "README.md").is_file() else ""
    for claim in ("Fixed-K2 reference-preserved budgeted branching", "Linear-K2", "NOT_SUPPORTED", "RC-NFGD", "DFT verified: **false**"):
        if claim not in readme:
            fail(errors, f"final-claim lock missing from README: {claim}")

    branch = git("branch", "--show-current").stdout.strip()
    if branch != "archive/thesis-exploration-2026":
        fail(errors, f"unexpected branch: {branch}")
    ancestry = git("merge-base", "--is-ancestor", BASE, "HEAD")
    if ancestry.returncode != 0:
        fail(errors, "frozen final-release base is not an ancestor")
    final_status = git("-C", "/mnt/datasets-livsyn/dxl/mattergen_v1_thesis_final", "status", "--porcelain")
    if final_status.returncode == 0 and final_status.stdout.strip():
        fail(errors, "final release worktree is modified")

    reviewed = []
    for path in ARCHIVE.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        size = path.stat().st_size
        if size > 100 * 1024 * 1024:
            fail(errors, f"ordinary Git file exceeds 100 MB: {path.relative_to(ROOT)}")
        elif size > 50 * 1024 * 1024:
            fail(errors, f"ordinary Git file exceeds 50 MB: {path.relative_to(ROOT)}")
        elif size > 10 * 1024 * 1024:
            reviewed.append(path.relative_to(ROOT).as_posix())

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        print("ARCHIVE_VALIDATION=FAIL")
        return 1
    print(
        "ARCHIVE_VALIDATION=PASS "
        f"experiments={len(rows)} full={category_counts['FULL_ARCHIVE']} "
        f"summary={category_counts['SUMMARY_ONLY']} external_only={category_counts['EXTERNAL_DATA_ONLY']} "
        f"ignored={category_counts['IGNORE']} external_artifacts={len(external_rows)}"
    )
    print(f"LARGE_FILE_AUDIT=PASS review_over_10mb={len(reviewed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
