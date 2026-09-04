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
    parser.add_argument(
        "--ablation",
        type=Path,
        help="Optional existing speed ablation CSV to enrich in place with quality columns.",
    )
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
    ablation_path = None
    if args.ablation is not None:
        ablation_path = args.ablation.expanduser().resolve()
        with ablation_path.open(newline="", encoding="utf-8") as stream:
            ablation_rows = list(csv.DictReader(stream))
        quality_by_method = {row["method"]: row for row in rows}
        quality_fields = [key for key in rows[0] if key not in ("method", "n")]
        for ablation_row in ablation_rows:
            quality_row = quality_by_method.get(ablation_row["method"])
            if quality_row is None:
                raise ValueError(f"missing quality row for {ablation_row['method']}")
            if int(ablation_row["n"]) != int(quality_row["n"]):
                raise ValueError(f"sample count mismatch for {ablation_row['method']}")
            ablation_row.update({key: quality_row[key] for key in quality_fields})
        fieldnames = list(ablation_rows[0])
        with ablation_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ablation_rows)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "ablation": None if ablation_path is None else str(ablation_path),
                "methods": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
