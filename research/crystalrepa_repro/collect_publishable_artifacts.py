"""Collect all small, non-weight CrystalREPA outputs for the Git branch."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from research.crystalrepa_repro.common import (
    PROJECT,
    REPORTS,
    RESULTS,
    atomic_json,
    now,
)


DESTINATION = PROJECT / "research/crystalrepa_repro/artifacts"
ALLOWED_SUFFIXES = {
    ".csv",
    ".extxyz",
    ".json",
    ".jsonl",
    ".md",
    ".sha256",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".pt", ".pth", ".bin", ".npy", ".npz"}
FORBIDDEN_PARTS = {
    "checkpoints",
    "cache",
    "datasets",
    "environment",
    "inference",
    "logs",
    "__pycache__",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def allowed(path: Path) -> bool:
    lowered = {part.lower() for part in path.parts}
    if lowered & FORBIDDEN_PARTS:
        return False
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False
    if path.name.endswith(".lock") or path.name.startswith("."):
        return False
    return path.suffix.lower() in ALLOWED_SUFFIXES


def copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and allowed(path.relative_to(source)):
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if target.suffix.lower() in ALLOWED_SUFFIXES:
                payload = target.read_bytes()
                normalized = payload.replace(b"\r\n", b"\n")
                if normalized != payload:
                    target.write_bytes(normalized)


def main() -> None:
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    DESTINATION.mkdir(parents=True)
    copy_tree(REPORTS, DESTINATION / "reports")
    copy_tree(RESULTS / "progress", DESTINATION / "results/progress")
    copy_tree(RESULTS / "generation", DESTINATION / "results/generation")
    copy_tree(
        RESULTS / "determinism_repeats",
        DESTINATION / "results/determinism_repeats",
    )
    copy_tree(RESULTS / "relaxed", DESTINATION / "results/relaxed")
    training = RESULTS / "training/r1"
    for name in (
        "training_config.json",
        "training_summary_1000.json",
        "training_summary_10000.json",
        "runtime_telemetry.json",
        "checkpoint_progress.json",
    ):
        source = training / name
        if source.is_file() and allowed(Path(name)):
            target = DESTINATION / "results/training" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    copy_tree(training / "csv", DESTINATION / "results/training/csv")
    records = []
    for path in sorted(DESTINATION.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "path": str(path.relative_to(DESTINATION)),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    manifest = {
        "created_at": now(),
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "exclusions": [
            "model/training/inference weights",
            "Teacher and dataset caches",
            "Conda environment",
            "large task logs",
            "NumPy cache arrays",
        ],
        "files": records,
    }
    atomic_json(DESTINATION / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {key: manifest[key] for key in ("file_count", "total_bytes")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
