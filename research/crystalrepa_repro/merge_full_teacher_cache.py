from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from research.crystalrepa_repro.common import atomic_json, now, sha256_file

DATASET = Path("/data/dxl/datasets/cache/mp_20")
OLD_CACHE = Path("/data/dxl/data/fn_pra_teacher_cache")
NEW_CACHE = Path("/data/dxl/data/crystalrepa_teacher_cache")


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def merge_split(split: str) -> dict:
    source = DATASET / split
    output = NEW_CACHE / split
    output.mkdir(parents=True, exist_ok=True)
    num_atoms = np.asarray(np.load(source / "num_atoms.npy", mmap_mode="r"), dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(num_atoms, dtype=np.int64)))
    z_all = np.load(source / "atomic_numbers.npy", mmap_mode="r")
    pos_all = np.load(source / "pos.npy", mmap_mode="r")
    cells = np.load(source / "cell.npy", mmap_mode="r")
    structure_ids = np.load(source / "structure_id.npy", allow_pickle=True)
    old_indices = np.load(OLD_CACHE / split / "dataset_indices.npy")
    old_offsets = np.load(OLD_CACHE / split / "offsets.npy")
    old_features = np.load(OLD_CACHE / split / "features.f16.npy", mmap_mode="r")
    missing_records: dict[int, np.ndarray] = {}
    shard_paths = sorted((NEW_CACHE / "staging" / split).glob("missing_shard_*.npz"))
    if len(shard_paths) != 8:
        raise RuntimeError(f"{split}: expected 8 successful missing shards, found {len(shard_paths)}")
    for shard_path in shard_paths:
        with np.load(shard_path) as shard:
            for local, index in enumerate(shard["dataset_indices"]):
                start, stop = int(shard["offsets"][local]), int(shard["offsets"][local + 1])
                missing_records[int(index)] = shard["features"][start:stop].copy()
    expected_missing = set(range(len(num_atoms))) - set(int(value) for value in old_indices)
    if set(missing_records) != expected_missing:
        raise RuntimeError(f"{split}: missing supplement index mismatch")
    temporary_features = output / f".features.f16.npy.{os.getpid()}.tmp"
    full_features = np.lib.format.open_memmap(temporary_features, mode="w+", dtype=np.float16, shape=(int(offsets[-1]), 64))
    old_lookup = {int(index): local for local, index in enumerate(old_indices)}
    for index in range(len(num_atoms)):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if index in old_lookup:
            local = old_lookup[index]
            old_start, old_stop = int(old_offsets[local]), int(old_offsets[local + 1])
            full_features[start:stop] = old_features[old_start:old_stop]
        else:
            full_features[start:stop] = missing_records[index]
    full_features.flush()
    del full_features
    os.replace(temporary_features, output / "features.f16.npy")
    atomic_npy(output / "atomic_numbers.npy", np.asarray(z_all, dtype=np.int16))
    atomic_npy(output / "offsets.npy", offsets)
    atomic_npy(output / "num_atoms.npy", num_atoms)
    atomic_npy(output / "dataset_indices.npy", np.arange(len(num_atoms), dtype=np.int64))
    atomic_npy(output / "structure_ids.npy", np.asarray(structure_ids, dtype=str))
    hashes = {key: [] for key in ("atomic_numbers_hash", "cell_hash", "positions_hash", "combined_structure_hash")}
    for index in range(len(num_atoms)):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        z = np.asarray(z_all[start:stop], dtype=np.int64)
        pos = np.asarray(pos_all[start:stop], dtype=np.float64)
        cell = np.asarray(cells[index], dtype=np.float64)
        z_hash, pos_hash, cell_hash = array_hash(z), array_hash(pos), array_hash(cell)
        hashes["atomic_numbers_hash"].append(z_hash)
        hashes["positions_hash"].append(pos_hash)
        hashes["cell_hash"].append(cell_hash)
        hashes["combined_structure_hash"].append(hashlib.sha256(f"{structure_ids[index]}|{z_hash}|{cell_hash}|{pos_hash}".encode()).hexdigest())
    for name, values in hashes.items():
        atomic_npy(output / f"{name}.npy", np.asarray(values))
    return {
        "split": split, "structures": len(num_atoms), "atoms": int(offsets[-1]),
        "reused_structures": len(old_indices), "supplemented_structures": len(missing_records),
        "features_sha256": sha256_file(output / "features.f16.npy"), "passed": True,
    }


def main() -> None:
    records = [merge_split(split) for split in ("train", "val")]
    manifest = {
        "schema_version": 1, "created_at": now(), "dataset": "mp_20 unconditional full train/val",
        "teacher_name": "CHGNet 0.3.0", "teacher_checkpoint_sha256": "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1",
        "teacher_layer": "atom_fea before final CHGNet convolution", "representation_dimension": 64,
        "representation_dtype": "float16", "source_cache": str(OLD_CACHE),
        "source_cache_modified": False, "splits": records,
    }
    atomic_json(NEW_CACHE / "cache_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
