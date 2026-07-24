#!/usr/bin/env python3
"""Atomic progress updates for budget-aware corrector experiments."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path("/data/dxl")
PROGRESS = ROOT / "results/budget_aware_gating/progress"
MASTER = PROGRESS / "master_progress.json"
MASTER_CSV = PROGRESS / "master_progress.csv"
EVENTS = PROGRESS / "events.jsonl"
LOCK = PROGRESS / "master_progress.lock"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(progress: dict) -> None:
    fields = ["stage", "status", "started_at", "finished_at", "message"]
    fd, name = tempfile.mkstemp(prefix=f".{MASTER_CSV.name}.tmp.", dir=PROGRESS)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in progress["stages"]:
                writer.writerow({key: row.get(key) for key in fields})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, MASTER_CSV)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def update(stage: str | None, status: str | None, message: str, fields: dict) -> dict:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        progress = json.loads(MASTER.read_text())
        stamp = now()
        if stage is not None:
            row = next(item for item in progress["stages"] if item["stage"] == stage)
            if status == "running" and row.get("started_at") is None:
                row["started_at"] = stamp
            row["status"] = status
            row["message"] = message
            if status in {"success", "failed", "interrupted", "incomplete", "not_applicable"}:
                row["finished_at"] = stamp
            progress["current_stage"] = stage
        progress.update(fields)
        progress["updated_at"] = stamp
        atomic_json(MASTER, progress)
        write_csv(progress)
        event = {"at": stamp, "stage": stage, "status": status, "message": message, **fields}
        with EVENTS.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage")
    parser.add_argument("--status")
    parser.add_argument("--message", default="")
    parser.add_argument("--fields-json", default="{}")
    args = parser.parse_args()
    fields = json.loads(args.fields_json)
    print(json.dumps(update(args.stage, args.status, args.message, fields), ensure_ascii=False))


if __name__ == "__main__":
    main()
