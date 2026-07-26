"""Memory-efficient cache merger used to resume after the initial compressed-array audit."""

from __future__ import annotations

import json

import numpy as np

from research.fn_pra import merge_teacher_cache as base
from research.fn_pra.phase1_common import CACHE


def merge_split(split: str) -> dict:
    staging = CACHE / "staging" / split
    paths = sorted(staging.glob("shard_*.npz"))
    metric_paths = sorted(staging.glob("shard_*.json"))
    if len(paths) != 8 or len(metric_paths) != 8:
        raise RuntimeError(f"{split}: expected 8 cache shards, found {len(paths)}/{len(metric_paths)}")
    records = []
    for path in paths:
        with np.load(path) as data:
            dataset_indices = data["dataset_indices"]
            offsets = data["offsets"]
            structure_ids = data["structure_ids"]
            num_atoms = data["num_atoms"]
            dft_mag_density = data["dft_mag_density"]
            atomic_numbers_hash = data["atomic_numbers_hash"]
            cell_hash = data["cell_hash"]
            positions_hash = data["positions_hash"]
            combined_structure_hash = data["combined_structure_hash"]
            features = data["features"]
            atomic_numbers = data["atomic_numbers"]
            for local, dataset_index in enumerate(dataset_indices):
                start = int(offsets[local])
                stop = int(offsets[local + 1])
                records.append(
                    {
                        "dataset_index": int(dataset_index),
                        "structure_id": str(structure_ids[local]),
                        "num_atoms": int(num_atoms[local]),
                        "dft_mag_density": float(dft_mag_density[local]),
                        "atomic_numbers_hash": str(atomic_numbers_hash[local]),
                        "cell_hash": str(cell_hash[local]),
                        "positions_hash": str(positions_hash[local]),
                        "combined_structure_hash": str(combined_structure_hash[local]),
                        "features": features[start:stop].copy(),
                        "atomic_numbers": atomic_numbers[start:stop].copy(),
                    }
                )
    records.sort(key=lambda item: item["dataset_index"])
    dataset_indices = np.asarray([item["dataset_index"] for item in records], dtype=np.int64)
    if len(dataset_indices) != len(np.unique(dataset_indices)):
        raise RuntimeError(f"{split}: duplicate dataset indices")
    counts = np.asarray([item["num_atoms"] for item in records], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    features = np.concatenate([item["features"] for item in records], axis=0)
    atomic_numbers = np.concatenate([item["atomic_numbers"] for item in records], axis=0)
    if features.shape != (int(counts.sum()), 64) or not np.isfinite(features).all():
        raise RuntimeError(f"{split}: invalid merged representation array {features.shape}")
    output = CACHE / split
    base.atomic_npy(output / "features.f16.npy", features.astype(np.float16, copy=False))
    base.atomic_npy(output / "atomic_numbers.npy", atomic_numbers.astype(np.int16, copy=False))
    base.atomic_npy(output / "offsets.npy", offsets)
    base.atomic_npy(output / "num_atoms.npy", counts)
    base.atomic_npy(output / "dataset_indices.npy", dataset_indices)
    base.atomic_npy(output / "structure_ids.npy", np.asarray([item["structure_id"] for item in records]))
    base.atomic_npy(
        output / "dft_mag_density.npy",
        np.asarray([item["dft_mag_density"] for item in records], dtype=np.float32),
    )
    for key in ("atomic_numbers_hash", "cell_hash", "positions_hash", "combined_structure_hash"):
        base.atomic_npy(output / f"{key}.npy", np.asarray([item[key] for item in records]))
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    return {
        "split": split,
        "structures": len(records),
        "atoms": int(counts.sum()),
        "dimension": 64,
        "dtype": "float16",
        "bytes": int(features.nbytes),
        "peak_vram_bytes_max": int(max(item["peak_vram_bytes"] for item in metrics)),
        "elapsed_seconds_max_rank": float(max(item["elapsed_seconds"] for item in metrics)),
        "structures_per_second_aggregate": float(
            sum(item["structures_per_second"] for item in metrics)
        ),
    }


def main() -> None:
    base.merge_split = merge_split
    base.main()


if __name__ == "__main__":
    main()
