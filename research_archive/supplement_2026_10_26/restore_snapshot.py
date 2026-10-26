#!/usr/bin/env python3
"""Verify Release bundles or restore one recorded research worktree overlay.

Clone and check out the worktree's recorded commit first. Existing different
files are not overwritten unless --overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--worktree")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and (not args.worktree or not args.destination):
        parser.error("restoration requires --worktree and --destination; use --verify-only for read-only verification")
    rows = [json.loads(line) for line in (HERE / "files.jsonl").read_text().splitlines() if line]
    assets = json.loads((HERE / "assets.json").read_text())["assets"]
    targets = collections.defaultdict(list)
    expected = collections.defaultdict(set)
    for row in rows:
        if row["kind"] != "file":
            continue
        expected[row["asset"]].add(row["sha256"])
        if not args.verify_only and row["worktree"] == args.worktree:
            rel = Path(row["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError("Unsafe destination path in manifest")
            destination = args.destination.resolve() / rel
            if destination.exists():
                if destination.is_file() and sha256(destination) == row["sha256"]:
                    continue
                if not args.overwrite:
                    raise RuntimeError(f"Destination differs: {destination}; use a clean checkout or explicitly --overwrite")
            targets[row["sha256"]].append((row, destination))
    if not args.verify_only:
        trees = {x["name"]:x for x in json.loads((HERE / "worktrees.json").read_text())}
        if args.worktree not in trees:
            raise RuntimeError("Unknown worktree")
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.destination).decode().strip()
        if actual != trees[args.worktree]["commit"]:
            raise RuntimeError("Check out the recorded source commit before restoring its overlay")
        if any(r["worktree"] == args.worktree and r["kind"] == "deleted" for r in rows):
            raise RuntimeError("Recorded deletions require manual review; this helper never deletes source files")
    restored = 0
    for asset in assets:
        if not args.verify_only and not any(row["asset"] == asset["name"] and row["sha256"] in targets for row in rows if row["kind"] == "file"):
            continue
        path = args.assets_dir / asset["name"]
        if sha256(path) != asset["sha256"]:
            raise RuntimeError(f"Asset SHA mismatch: {path}")
        proc = subprocess.Popen(["zstd", "-d", "-q", "-c", str(path)], stdout=subprocess.PIPE)
        seen = set()
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    if not member.isfile() or not re.fullmatch(r"blobs/[0-9a-f]{64}", member.name):
                        raise RuntimeError("Unexpected tar member")
                    digest = member.name.split("/",1)[1]
                    h = hashlib.sha256()
                    stream = tar.extractfile(member)
                    selected = targets.get(digest, [])
                    with tempfile.SpooledTemporaryFile(max_size=4*1024*1024) as cache:
                        for block in iter(lambda: stream.read(1024*1024), b""):
                            h.update(block)
                            if selected:
                                cache.write(block)
                        if h.hexdigest() != digest:
                            raise RuntimeError(f"Internal blob SHA mismatch: {member.name}")
                        for row, destination in selected:
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            if not destination.resolve().is_relative_to(args.destination.resolve()):
                                raise RuntimeError("Destination symlink escapes restore root")
                            cache.seek(0)
                            with destination.open("wb") as output:
                                for block in iter(lambda: cache.read(1024*1024), b""):
                                    output.write(block)
                            destination.chmod(row["mode"] & 0o777)
                            restored += 1
                    seen.add(digest)
            if proc.wait() != 0:
                raise RuntimeError("zstd decompression failed")
        except BaseException:
            proc.kill(); proc.wait(); raise
        if seen != expected[asset["name"]]:
            raise RuntimeError(f"Bundle membership mismatch: {asset['name']}")
        print(f"VERIFIED {asset['name']}: {len(seen)} blobs", flush=True)
    restored_links = 0
    if not args.verify_only:
        for row in rows:
            if row["kind"] != "directory_symlink" or row["worktree"] != args.worktree:
                continue
            root = args.destination.resolve()
            link = root / row["path"]
            target = root / row["target_path"]
            if not link.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
                raise RuntimeError("Directory alias escapes restore root")
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            if link.exists() or link.is_symlink():
                raise RuntimeError("Existing directory alias conflicts; review it manually")
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
            restored_links += 1
    print(f"SNAPSHOT_VERIFICATION=PASS restored_files={restored} restored_directory_links={restored_links}", flush=True)


if __name__ == "__main__":
    main()
