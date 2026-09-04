"""Build and audit the cross-seed teacher data manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def expand_range(bounds: list[int]) -> set[int]:
    return set(range(int(bounds[0]), int(bounds[1]) + 1))


def main() -> None:
    args = parse_args()
    with args.experiment_config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    historical = set()
    for bounds in config["excluded_historical_seed_ranges"]:
        historical |= expand_range(bounds)

    teacher_root = args.teacher_root.expanduser().resolve()
    split_seeds: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
    runs = []
    for manifest_path in sorted(teacher_root.glob("*/*/manifest.json")):
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        split = str(manifest["split"])
        seed = int(manifest["seed"])
        if split not in split_seeds:
            raise ValueError(f"unexpected split {split!r} in {manifest_path}")
        if seed in split_seeds[split]:
            raise ValueError(f"duplicate seed {seed} in split {split}")
        if seed in historical:
            raise ValueError(f"teacher seed {seed} overlaps historical evaluation data")
        split_seeds[split].add(seed)
        shard_records = 0
        shard_bytes = 0
        for shard in manifest["shards"]:
            shard_path = manifest_path.parent / shard["path"]
            actual = sha256(shard_path)
            if actual != shard["sha256"]:
                raise ValueError(f"checksum mismatch: {shard_path}")
            shard_records += int(shard["records"])
            shard_bytes += int(shard["bytes"])
        if shard_records != int(manifest["records"]):
            raise ValueError(f"record count mismatch: {manifest_path}")
        runs.append(
            {
                "split": split,
                "seed": seed,
                "manifest": str(manifest_path.relative_to(teacher_root.parent)),
                "manifest_sha256": sha256(manifest_path),
                "records": shard_records,
                "bytes": shard_bytes,
                "storage_dtype": manifest["storage_dtype"],
                "completed": bool(manifest["completed"]),
            }
        )
    if not runs:
        raise FileNotFoundError(f"no teacher manifests below {teacher_root}")
    split_names = list(split_seeds)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_seeds[left] & split_seeds[right]
            if overlap:
                raise ValueError(f"teacher split leakage {left}/{right}: {sorted(overlap)}")
    output = {
        "schema_version": 1,
        "teacher_schema_version": 1,
        "runs": runs,
        "split_seeds": {key: sorted(values) for key, values in split_seeds.items()},
        "total_records": sum(run["records"] for run in runs),
        "total_bytes": sum(run["bytes"] for run in runs),
        "historical_seed_overlap": [],
        "cross_split_overlap": [],
        "source_experiment_config": str(args.experiment_config.resolve()),
        "source_experiment_config_sha256": sha256(args.experiment_config.resolve()),
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

