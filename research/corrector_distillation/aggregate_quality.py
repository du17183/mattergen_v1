"""Flatten per-method official quality summaries into one auditable CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in sorted(args.quality_root.expanduser().resolve().glob("*/quality_summary.json")):
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        official = summary.pop("official_metrics")
        rows.append(
            {
                **summary,
                **official,
                "source_summary": str(summary_path.resolve()),
            }
        )
    if not rows:
        raise FileNotFoundError(args.quality_root)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output_path), "methods": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
