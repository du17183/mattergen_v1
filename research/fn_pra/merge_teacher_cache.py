from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from research.fn_pra.phase1_common import (
    CACHE,
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
    set_stage,
    sha256_file,
)


CHECKPOINT = Path(
    "/data/dxl/envs/fn_pra_teacher/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/"
    "chgnet_0.3.0_e29f68s314m37.pth.tar"
)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(tmp, path)


def merge_split(split: str) -> dict:
    staging = CACHE / "staging" / split
    paths = sorted(staging.glob("shard_*.npz"))
    metric_paths = sorted(staging.glob("shard_*.json"))
    if len(paths) != 8 or len(metric_paths) != 8:
        raise RuntimeError(f"{split}: expected 8 cache shards, found {len(paths)}/{len(metric_paths)}")
    records = []
    for path in paths:
        data = np.load(path)
        for local, dataset_index in enumerate(data["dataset_indices"]):
            start = int(data["offsets"][local])
            stop = int(data["offsets"][local + 1])
            records.append(
                {
                    "dataset_index": int(dataset_index),
                    "structure_id": str(data["structure_ids"][local]),
                    "num_atoms": int(data["num_atoms"][local]),
                    "dft_mag_density": float(data["dft_mag_density"][local]),
                    "atomic_numbers_hash": str(data["atomic_numbers_hash"][local]),
                    "cell_hash": str(data["cell_hash"][local]),
                    "positions_hash": str(data["positions_hash"][local]),
                    "combined_structure_hash": str(data["combined_structure_hash"][local]),
                    "features": data["features"][start:stop].copy(),
                    "atomic_numbers": data["atomic_numbers"][start:stop].copy(),
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
    if features.shape != (int(counts.sum()), 64):
        raise RuntimeError(f"{split}: merged representation shape mismatch {features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError(f"{split}: merged cache contains non-finite values")
    output = CACHE / split
    atomic_npy(output / "features.f16.npy", features.astype(np.float16, copy=False))
    atomic_npy(output / "atomic_numbers.npy", atomic_numbers.astype(np.int16, copy=False))
    atomic_npy(output / "offsets.npy", offsets)
    atomic_npy(output / "num_atoms.npy", counts)
    atomic_npy(output / "dataset_indices.npy", dataset_indices)
    atomic_npy(output / "structure_ids.npy", np.asarray([item["structure_id"] for item in records]))
    atomic_npy(
        output / "dft_mag_density.npy",
        np.asarray([item["dft_mag_density"] for item in records], dtype=np.float32),
    )
    for key in ("atomic_numbers_hash", "cell_hash", "positions_hash", "combined_structure_hash"):
        atomic_npy(output / f"{key}.npy", np.asarray([item[key] for item in records]))
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


def probe_reference() -> dict[int, tuple[np.ndarray, np.ndarray, str, str]]:
    reference = {}
    for path in sorted((RESULTS / "teacher_probe/chgnet").glob("shard_*.npz")):
        data = np.load(path)
        for local, dataset_index in enumerate(data["dataset_indices"]):
            start = int(data["offsets"][local])
            stop = int(data["offsets"][local + 1])
            reference[int(dataset_index)] = (
                data["features"][start:stop].copy(),
                data["atomic_numbers"][start:stop].copy(),
                str(data["structure_ids"][local]),
                str(data["structure_hashes"][local]),
            )
    return reference


def validate_online_cache() -> dict:
    reference = probe_reference()
    selected = sorted(reference)[:64]
    output = CACHE / "train"
    cache_indices = np.load(output / "dataset_indices.npy")
    lookup = {int(index): offset for offset, index in enumerate(cache_indices)}
    offsets = np.load(output / "offsets.npy")
    features = np.load(output / "features.f16.npy", mmap_mode="r")
    atomic_numbers = np.load(output / "atomic_numbers.npy", mmap_mode="r")
    structure_ids = np.load(output / "structure_ids.npy")
    structure_hashes = np.load(output / "combined_structure_hash.npy")
    errors = []
    compared_atoms = 0
    for index in selected:
        if index not in lookup:
            raise RuntimeError(f"Online/cache validation index missing: {index}")
        local = lookup[index]
        start, stop = int(offsets[local]), int(offsets[local + 1])
        online, online_z, structure_id, structure_hash = reference[index]
        cached = np.asarray(features[start:stop], dtype=np.float32)
        cached_z = np.asarray(atomic_numbers[start:stop])
        if str(structure_ids[local]) != structure_id or str(structure_hashes[local]) != structure_hash:
            raise RuntimeError(f"Online/cache identity mismatch at dataset index {index}")
        if not np.array_equal(cached_z, online_z):
            raise RuntimeError(f"Online/cache atom order mismatch at dataset index {index}")
        if cached.shape != online.shape:
            raise RuntimeError(f"Online/cache representation shape mismatch at dataset index {index}")
        errors.append(np.abs(cached - online).reshape(-1))
        compared_atoms += len(cached_z)
    error = np.concatenate(errors)
    result = {
        "schema_version": 1,
        "created_at": now(),
        "structures": len(selected),
        "atoms": compared_atoms,
        "identity_match": True,
        "atom_order_match": True,
        "shape_match": True,
        "online_dtype": "float32",
        "cache_dtype": "float16",
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "tolerance": 0.02,
        "passed": bool(float(error.max()) <= 0.02),
        "note": "Difference includes independent GPU inference and float16 cache quantization.",
    }
    if not result["passed"]:
        raise RuntimeError(f"Online/cache error exceeded tolerance: {result}")
    atomic_json(REPORTS / "online_teacher_validation.json", result)
    atomic_text(
        REPORTS / "online_teacher_validation.md",
        "# Online Teacher / Cache Validation\n\n"
        f"- Structures: {result['structures']}\n"
        f"- Atoms: {result['atoms']}\n"
        "- Identity and atom order: passed\n"
        f"- Max absolute error: {result['max_abs_error']:.8f}\n"
        f"- Mean absolute error: {result['mean_abs_error']:.8f}\n"
        f"- Tolerance: {result['tolerance']}\n"
        "- Cache precision: float16; online reference: float32.\n",
    )
    return result


def write_hash_manifest() -> None:
    files = [
        path
        for path in sorted(CACHE.rglob("*"))
        if path.is_file()
        and "staging" not in path.parts
        and path.name != "cache_sha256_manifest.csv"
    ]
    target = CACHE / "cache_sha256_manifest.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for path in files:
            writer.writerow([str(path.relative_to(CACHE)), path.stat().st_size, sha256_file(path)])


def main() -> None:
    set_stage("teacher_cache", "running", "Merging CHGNet train/val shards and validating integrity.")
    splits = [merge_split(split) for split in ("train", "val")]
    validation = validate_online_cache()
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "dataset": "mp_20",
        "validity_filter": "finite dft_mag_density",
        "teacher_name": "CHGNet 0.3.0",
        "teacher_package_version": "0.4.2",
        "teacher_checkpoint": str(CHECKPOINT),
        "teacher_checkpoint_sha256": sha256_file(CHECKPOINT),
        "teacher_layer": "atom_fea before final CHGNet convolution",
        "representation_dimension": 64,
        "representation_dtype": "float16",
        "cache_bytes_per_atom": 128,
        "key_fields": [
            "split",
            "structure_id",
            "combined_structure_hash",
            "num_atoms",
            "atomic_numbers_hash",
            "atom_index",
            "teacher_checkpoint_sha256",
            "teacher_layer",
        ],
        "splits": splits,
    }
    integrity = {
        "schema_version": 1,
        "created_at": now(),
        "passed": True,
        "structure_id_match": True,
        "structure_hash_match": True,
        "atom_order_match": True,
        "row_count_match": True,
        "finite": True,
        "online_cache_validation": validation,
        "splits": splits,
    }
    atomic_json(CACHE / "cache_manifest.json", manifest)
    atomic_json(CACHE / "cache_integrity_report.json", integrity)
    write_hash_manifest()
    set_stage(
        "online_teacher_validation",
        "success",
        f"64 structures/{validation['atoms']} atoms matched cache; max_abs={validation['max_abs_error']:.6g}.",
        validation,
    )
    set_stage(
        "teacher_cache",
        "success",
        f"CHGNet cache complete: train={splits[0]['structures']}, val={splits[1]['structures']} structures.",
        {"train": splits[0], "val": splits[1]},
    )


if __name__ == "__main__":
    main()
