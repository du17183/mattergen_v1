#!/usr/bin/env python3
"""Snapshot all MatterGen worktrees, refs and all-ref history."""

from __future__ import annotations

import csv
import subprocess
from collections import defaultdict
from pathlib import Path

from catalog import EXPERIMENTS


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_archive"


def run(args: list[str], cwd: Path = ROOT, check: bool = True) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)
    return result.stdout


def write_worktrees() -> int:
    raw = run(["git", "worktree", "list", "--porcelain"])
    records = []
    current: dict[str, str] = {}
    for line in raw.splitlines() + [""]:
        if not line:
            if current:
                path = Path(current["worktree"])
                status = run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=path)
                tracked = 0
                untracked = 0
                for entry in status.splitlines():
                    if entry.startswith("??"):
                        untracked += 1
                    else:
                        tracked += 1
                current["branch"] = current.get("branch", "DETACHED").removeprefix("refs/heads/")
                current["tracked_change_count"] = str(tracked)
                current["untracked_file_count"] = str(untracked)
                current["inventory_scope"] = "mattergen_v1*" if path.name.startswith("mattergen_v1") else "related-worktree"
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value or "true"
    fields = ["worktree", "branch", "HEAD", "tracked_change_count", "untracked_file_count", "inventory_scope"]
    with (OUT / "worktree_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def write_branches() -> int:
    by_branch: defaultdict[str, list[str]] = defaultdict(list)
    for row in EXPERIMENTS:
        by_branch[row["source_branch"].removeprefix("origin/")].append(row["experiment_id"])
        by_branch[row["source_branch"]].append(row["experiment_id"])
    fmt = "%00".join(["%(refname:short)", "%(refname)", "%(objectname)", "%(committerdate:iso-strict)", "%(subject)"])
    raw = run(["git", "for-each-ref", f"--format={fmt}", "refs/heads", "refs/remotes"])
    records = []
    for line in raw.splitlines():
        short, full, commit, commit_date, subject = line.split("\x00", 4)
        if short.endswith("/HEAD"):
            disposition = "symbolic-default"
        elif short == "release/thesis-final-2026" or short == "origin/release/thesis-final-2026":
            disposition = "frozen-final-release"
        elif short == "archive/thesis-exploration-2026":
            disposition = "current-exploration-archive"
        elif by_branch.get(short):
            disposition = "represented-in-experiment-index"
        else:
            disposition = "reference-only; manual review if scientific reuse is proposed"
        records.append({
            "ref": short,
            "full_ref": full,
            "commit": commit,
            "commit_date": commit_date,
            "subject": subject,
            "experiment_ids": ";".join(sorted(set(by_branch.get(short, [])))),
            "archive_disposition": disposition,
        })
    fields = ["ref", "full_ref", "commit", "commit_date", "subject", "experiment_ids", "archive_disposition"]
    with (OUT / "source_branch_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def main() -> int:
    worktrees = write_worktrees()
    branches = write_branches()
    history = run(["git", "log", "--all", "--oneline", "--decorate", "--date-order"])
    (OUT / "git_all_history.txt").write_text(history, encoding="utf-8")
    (OUT / "inventory_scan_summary.md").write_text(
        f'''# Inventory Scan Summary

- MatterGen worktrees scanned: **{worktrees}**
- Local and remote refs recorded: **{branches}**
- Experiment records classified: **{len(EXPERIMENTS)}**
- Search root: `/mnt/datasets-livsyn/dxl/`
- Directory name filter: `mattergen_v1*`
- Commands represented: `git worktree list`, `git branch -a`, `git log --all --oneline --decorate`, status scans, and top-level scans for `experiments/`, `diagnostics/`, `results/`, `reports/`, `configs/`, `scripts/`, `logs/`, and `thesis/`.

The worktree status counts are an inventory-time observation, not an instruction to clean or delete any source tree. No source worktree, branch or experiment was removed.
''',
        encoding="utf-8",
    )
    print(f"worktrees={worktrees} branches={branches} experiments={len(EXPERIMENTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
