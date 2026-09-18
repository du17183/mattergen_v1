#!/usr/bin/env python3
"""Verify archived-file hashes and optionally re-read all external sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_external_manifest import hash_path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "research_archive"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_selected() -> tuple[int, list[str]]:
    failures = []
    count = 0
    with (ARCHIVE / "selected_artifact_manifest.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            count += 1
            path = ROOT / row["archive_path"]
            if not path.is_file():
                failures.append(f"selected artifact missing: {row['archive_path']}")
            elif path.stat().st_size != int(row["archive_size_bytes"]):
                failures.append(f"selected artifact size mismatch: {row['archive_path']}")
            elif sha256(path) != row["archive_sha256"]:
                failures.append(f"selected artifact hash mismatch: {row['archive_path']}")
    return count, failures


def verify_archive() -> tuple[int, list[str]]:
    failures = []
    payload = json.loads((ARCHIVE / "archive_checksums.json").read_text(encoding="utf-8"))
    records = payload["files"]
    for row in records:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append(f"archive file missing: {row['path']}")
        elif path.stat().st_size != int(row["size_bytes"]):
            failures.append(f"archive file size mismatch: {row['path']}")
        elif sha256(path) != row["sha256"]:
            failures.append(f"archive file hash mismatch: {row['path']}")
    return len(records), failures


def verify_external() -> tuple[int, list[str]]:
    failures = []
    count = 0
    with (ARCHIVE / "external_data_manifest.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            count += 1
            path = Path(row["original_path"])
            if not path.exists():
                failures.append(f"external artifact missing: {row['experiment']} / {row['artifact']}")
                continue
            size, file_count, digest, scheme = hash_path(path)
            if size != int(row["size_bytes"]):
                failures.append(f"external size mismatch: {row['experiment']} / {row['artifact']}")
            if file_count != int(row["file_count"]):
                failures.append(f"external file-count mismatch: {row['experiment']} / {row['artifact']}")
            if digest != row["sha256"] or scheme != row["hash_scheme"]:
                failures.append(f"external hash mismatch: {row['experiment']} / {row['artifact']}")
            print(f"verified external {count}: {row['experiment']}")
    return count, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", action="store_true", help="re-read and hash every external source tree")
    args = parser.parse_args()
    selected_count, failures = verify_selected()
    archive_count, archive_failures = verify_archive()
    failures.extend(archive_failures)
    external_count = 0
    if args.external:
        external_count, external_failures = verify_external()
        failures.extend(external_failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(
        f"HASH_VALIDATION=PASS selected={selected_count} archive={archive_count} "
        f"external={external_count if args.external else 'manifest-only'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
