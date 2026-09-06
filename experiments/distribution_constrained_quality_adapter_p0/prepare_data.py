"""Build a deterministic clean-x0 subset from the official Alex-MP train split."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
from pathlib import Path
from zipfile import ZipFile

from mattergen.common.data.dataset import CrystalDataset


def eligible(row: dict[str, str]) -> bool:
    try:
        mag = float(row["dft_mag_density"])
        num_sites = int(float(row["num_sites"]))
    except (KeyError, TypeError, ValueError):
        return False
    return bool(row.get("cif")) and math.isfinite(mag) and 0 < num_sites <= 20


def reservoir_sample(
    zip_path: Path, member: str, *, count: int, seed: int
) -> tuple[list[dict[str, str]], list[str], int]:
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    eligible_count = 0
    with ZipFile(zip_path) as archive, archive.open(member, "r") as binary:
        reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8", newline=""))
        if reader.fieldnames is None:
            raise RuntimeError("Alex-MP CSV has no header")
        fieldnames = list(reader.fieldnames)
        for source_row, row in enumerate(reader):
            if not eligible(row):
                continue
            row["source_row"] = str(source_row)
            eligible_count += 1
            if len(selected) < count:
                selected.append(row)
            else:
                replace = rng.randrange(eligible_count)
                if replace < count:
                    selected[replace] = row
    if len(selected) != count:
        raise RuntimeError(f"requested {count} samples but found only {len(selected)}")
    return selected, fieldnames + ["source_row"], eligible_count


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--train-count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if not 0 < args.train_count < args.count:
        raise ValueError("train-count must lie strictly between zero and count")

    rows, fieldnames, eligible_count = reservoir_sample(
        args.zip_path, "alex_mp_20/train.csv", count=args.count, seed=args.seed
    )
    split_rng = random.Random(args.seed + 1)
    split_rng.shuffle(rows)
    train_rows = rows[: args.train_count]
    val_rows = rows[args.train_count :]
    csv_root = args.output_root / "csv"
    cache_root = args.output_root / "cache"
    write_csv(csv_root / "train.csv", train_rows, fieldnames)
    write_csv(csv_root / "val.csv", val_rows, fieldnames)

    for split in ("train", "val"):
        CrystalDataset.from_csv(
            csv_path=str(csv_root / f"{split}.csv"),
            cache_path=str(cache_root / split),
        )

    summary = {
        "source": "official MatterGen Alex-MP training split",
        "source_zip": str(args.zip_path.resolve()),
        "source_member": "alex_mp_20/train.csv",
        "selection": "reservoir sample with finite dft_mag_density and 1-20 sites",
        "selection_seed": args.seed,
        "eligible_source_rows": eligible_count,
        "selected": args.count,
        "train": len(train_rows),
        "validation": len(val_rows),
        "uses_historical_generated_or_formal_data": False,
        "train_material_ids": [row["material_id"] for row in train_rows],
        "validation_material_ids": [row["material_id"] for row in val_rows],
    }
    (args.output_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("material_ids")}, indent=2))


if __name__ == "__main__":
    main()
