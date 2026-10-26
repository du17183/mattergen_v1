#!/usr/bin/env python3
"""Preserve safe directory aliases without duplicating their data blobs."""
import collections
import hashlib
import json
import os
from pathlib import Path

from build_snapshot import CACHE_DIRS

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = [json.loads(line) for line in (HERE / "files.jsonl").read_text().splitlines() if line]
    excluded = [json.loads(line) for line in (HERE / "excluded.jsonl").read_text().splitlines() if line]
    trees = {r["name"]:r for r in json.loads((HERE / "worktrees.json").read_text())}
    known = {(r["worktree"],r["path"]) for r in rows}
    retained, links = [], 0
    for row in excluded:
        tree = trees[row["worktree"]]
        root = Path(tree["original_root"])
        source = root / row["path"]
        if (row.get("kind") == "directory" and not any(part in CACHE_DIRS for part in Path(row["path"]).parts)
                and source.is_symlink() and source.is_dir() and source.resolve().is_relative_to(root)):
            if (row["worktree"], row["path"]) not in known:
                rows.append({"worktree": row["worktree"], "path": row["path"], "kind": "directory_symlink",
                             "base_commit": tree["commit"], "target_path": source.resolve().relative_to(root).as_posix(),
                             "source_symlink_target": os.readlink(source), "source": str(source)})
                links += 1
        else:
            retained.append(row)
    manifest = HERE / "files.jsonl"
    manifest.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows))
    (HERE / "excluded.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in retained))
    summary_path = HERE / "snapshot_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["directory_symlinks"] = sum(r["kind"] == "directory_symlink" for r in rows)
    summary["overlay_files"] = len(rows)
    summary["excluded_by_reason"] = dict(collections.Counter(r["reason"] for r in retained))
    summary["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    print(f"DIRECTORY_LINK_METADATA=PASS added={links}")


if __name__ == "__main__":
    main()
