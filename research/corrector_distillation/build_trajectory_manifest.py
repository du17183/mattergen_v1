"""Build a checksummed cross-seed manifest for teacher or DAgger trajectories."""

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
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--trajectory-kind", choices=("exact_teacher", "dagger"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def expand(bounds: list[int]) -> set[int]:
    return set(range(int(bounds[0]), int(bounds[1]) + 1))


def main() -> None:
    args = parse_args()
    config_path = args.experiment_config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    expected = {
        split: expand(bounds)
        for split, bounds in config["trajectory_seed_splits"].items()
    }
    excluded = set()
    for bounds in config["excluded_seed_ranges"]:
        excluded |= expand(bounds)
    root = args.trajectory_root.expanduser().resolve()
    observed: dict[str, set[int]] = {}
    runs = []
    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("trajectory_kind", "exact_teacher") != args.trajectory_kind:
            raise ValueError(f"unexpected trajectory kind in {manifest_path}")
        if not bool(manifest.get("completed")):
            raise ValueError(f"incomplete trajectory: {manifest_path}")
        split = str(manifest["split"])
        seed = int(manifest["seed"])
        if split not in expected:
            raise ValueError(f"unexpected split {split!r}")
        if seed not in expected[split]:
            raise ValueError(f"seed {seed} is outside configured {split} range")
        if seed in excluded:
            raise ValueError(f"seed {seed} overlaps frozen/historical ranges")
        if seed in observed.setdefault(split, set()):
            raise ValueError(f"duplicate seed {seed} in {split}")
        observed[split].add(seed)
        records = 0
        byte_count = 0
        for shard in manifest["shards"]:
            shard_path = manifest_path.parent / shard["path"]
            if sha256(shard_path) != shard["sha256"]:
                raise ValueError(f"checksum mismatch: {shard_path}")
            records += int(shard["records"])
            byte_count += int(shard["bytes"])
        if records != int(manifest["records"]):
            raise ValueError(f"record count mismatch: {manifest_path}")
        runs.append(
            {
                "split": split,
                "seed": seed,
                "records": records,
                "bytes": byte_count,
                "manifest": str(manifest_path.relative_to(root.parent)),
                "manifest_sha256": sha256(manifest_path),
                "adapter_checkpoint_sha256": manifest.get("provenance", {}).get(
                    "adapter_checkpoint_sha256"
                ),
                "teacher_target": manifest.get("provenance", {}).get("teacher_target"),
                "decision_codes": manifest.get("decision_codes"),
            }
        )
    if not runs:
        raise FileNotFoundError(f"no trajectory manifests below {root}")
    split_names = sorted(observed)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = observed[left] & observed[right]
            if overlap:
                raise ValueError(f"cross-split leakage {left}/{right}: {sorted(overlap)}")
    output = {
        "schema_version": 1,
        "trajectory_schema_version": 1,
        "trajectory_kind": args.trajectory_kind,
        "runs": runs,
        "split_seeds": {key: sorted(value) for key, value in observed.items()},
        "expected_split_seeds": {key: sorted(value) for key, value in expected.items()},
        "missing_split_seeds": {
            key: sorted(value - observed.get(key, set())) for key, value in expected.items()
        },
        "total_records": sum(item["records"] for item in runs),
        "total_bytes": sum(item["bytes"] for item in runs),
        "excluded_seed_overlap": [],
        "cross_split_overlap": [],
        "source_experiment_config": str(config_path),
        "source_experiment_config_sha256": sha256(config_path),
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
