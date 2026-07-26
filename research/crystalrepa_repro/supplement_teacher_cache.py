from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from research.crystalrepa_repro.common import atomic_json, now

DATASET = Path("/data/dxl/datasets/cache/mp_20")
OLD_CACHE = Path("/data/dxl/data/fn_pra_teacher_cache")
NEW_CACHE = Path("/data/dxl/data/crystalrepa_teacher_cache")


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def run_split(split: str, shard: int, num_shards: int, batch_size: int) -> dict:
    output = NEW_CACHE / "staging" / split / f"missing_shard_{shard:02d}.npz"
    metrics_path = output.with_suffix(".json")
    if output.exists() and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("success"):
            return metrics
    source = DATASET / split
    num_atoms = np.load(source / "num_atoms.npy", mmap_mode="r")
    offsets = np.concatenate(([0], np.cumsum(num_atoms[:-1], dtype=np.int64)))
    z_all = np.load(source / "atomic_numbers.npy", mmap_mode="r")
    pos_all = np.load(source / "pos.npy", mmap_mode="r")
    cells = np.load(source / "cell.npy", mmap_mode="r")
    structure_ids = np.load(source / "structure_id.npy", allow_pickle=True)
    old_indices = np.load(OLD_CACHE / split / "dataset_indices.npy")
    missing = np.setdiff1d(np.arange(len(num_atoms), dtype=np.int64), old_indices, assume_unique=True)
    selected = missing[shard::num_shards]
    from chgnet.model.model import CHGNet
    model = CHGNet.load(model_name="0.3.0", use_device="cuda", verbose=False)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    feature_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    started = time.perf_counter()
    chunk_size = 256
    for chunk_start in range(0, len(selected), chunk_size):
        indices = selected[chunk_start:chunk_start + chunk_size]
        structures, z_rows = [], []
        for index in indices:
            start = int(offsets[index])
            stop = start + int(num_atoms[index])
            z = np.asarray(z_all[start:stop], dtype=np.int64)
            pos = np.asarray(pos_all[start:stop], dtype=np.float64)
            cell = np.asarray(cells[index], dtype=np.float64)
            structures.append(Structure(Lattice(cell), z, pos, coords_are_cartesian=False, to_unit_cell=True))
            z_rows.append(z)
        predictions = model.predict_structure(structures, task="e", return_atom_feas=True, batch_size=batch_size)
        for index, z, prediction in zip(indices, z_rows, predictions, strict=True):
            features = np.asarray(prediction["atom_fea"], dtype=np.float32)
            if features.shape != (len(z), 64) or not np.isfinite(features).all():
                raise RuntimeError(f"Invalid CHGNet features for {split}[{index}]: {features.shape}")
            feature_parts.append(features.astype(np.float16))
            z_parts.append(z.astype(np.int16))
    torch.cuda.synchronize()
    counts = np.asarray(num_atoms[selected], dtype=np.int32)
    features = np.concatenate(feature_parts) if feature_parts else np.empty((0, 64), dtype=np.float16)
    atomic_numbers = np.concatenate(z_parts) if z_parts else np.empty((0,), dtype=np.int16)
    atomic_npz(
        output, features=features, atomic_numbers=atomic_numbers,
        offsets=np.concatenate(([0], np.cumsum(counts, dtype=np.int64))),
        num_atoms=counts, dataset_indices=selected,
        structure_ids=np.asarray(structure_ids[selected], dtype=str),
    )
    elapsed = time.perf_counter() - started
    metrics = {
        "success": True, "created_at": now(), "split": split, "shard": shard,
        "num_shards": num_shards, "structures": len(selected), "atoms": len(atomic_numbers),
        "elapsed_seconds": elapsed, "structures_per_second": len(selected) / elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "output": str(output),
    }
    atomic_json(metrics_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("Invalid shard")
    print(json.dumps([run_split(split, args.shard, args.num_shards, args.batch_size) for split in ("train", "val")], indent=2))


if __name__ == "__main__":
    main()
