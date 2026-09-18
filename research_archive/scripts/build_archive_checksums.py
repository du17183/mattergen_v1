#!/usr/bin/env python3
"""Create SHA-256 checksums for every file in research_archive/."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "research_archive"
OUT = ARCHIVE / "archive_checksums.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    records = []
    for path in sorted(ARCHIVE.rglob("*")):
        if not path.is_file() or path == OUT or "__pycache__" in path.parts:
            continue
        records.append({
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    payload = {
        "schema_version": 1,
        "algorithm": "sha256",
        "scope": "research_archive files excluding archive_checksums.json and __pycache__",
        "file_count": len(records),
        "files": records,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"archive_checksums={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
