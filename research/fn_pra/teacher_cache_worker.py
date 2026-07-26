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

from research.fn_pra.phase1_common import CACHE, atomic_json, now


DATASET_ROOT = Path("/data/dxl/datasets/cache/mp_20")
CHECKPOINT = Path(
    "/data/dxl/envs/fn_pra_teacher/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/"
    "chgnet_0.3.0_e29f68s314m37.pth.tar"
)


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_property(path: Path, name: str) -> np.ndarray:
    payload = json.loads((path / f"{name}.json").read_text(encoding="utf-8"))
    return np.asarray(payload["values"], dtype=np.float64)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("Invalid shard")

    split_dir = DATASET_ROOT / args.split
    num_atoms = np.load(split_dir / "num_atoms.npy", mmap_mode="r")
    offsets = np.concatenate(([0], np.cumsum(num_atoms[:-1], dtype=np.int64)))
    z_all = np.load(split_dir / "atomic_numbers.npy", mmap_mode="r")
    pos_all = np.load(split_dir / "pos.npy", mmap_mode="r")
    cells = np.load(split_dir / "cell.npy", mmap_mode="r")
    structure_ids = np.load(split_dir / "structure_id.npy", allow_pickle=True)
    dft_mag = load_property(split_dir, "dft_mag_density")
    valid = np.flatnonzero(np.isfinite(dft_mag))
    selected = valid[args.shard :: args.num_shards]

    from chgnet.model.model import CHGNet

    model = CHGNet.load(model_name="0.3.0", use_device="cuda", verbose=False)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    feature_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    atom_hashes: list[str] = []
    cell_hashes: list[str] = []
    position_hashes: list[str] = []
    structure_hashes: list[str] = []
    start_time = time.perf_counter()

    for chunk_start in range(0, len(selected), args.chunk_size):
        chunk_indices = selected[chunk_start : chunk_start + args.chunk_size]
        structures = []
        chunk_z = []
        for index in chunk_indices:
            atom_start = int(offsets[index])
            atom_stop = atom_start + int(num_atoms[index])
            z = np.asarray(z_all[atom_start:atom_stop], dtype=np.int64)
            pos = np.asarray(pos_all[atom_start:atom_stop], dtype=np.float64)
            cell = np.asarray(cells[index], dtype=np.float64)
            structures.append(
                Structure(
                    Lattice(cell),
                    z,
                    pos,
                    coords_are_cartesian=False,
                    to_unit_cell=True,
                )
            )
            chunk_z.append(z)
            z_hash = array_hash(z)
            cell_hash = array_hash(cell)
            pos_hash = array_hash(pos)
            atom_hashes.append(z_hash)
            cell_hashes.append(cell_hash)
            position_hashes.append(pos_hash)
            structure_hashes.append(
                hashlib.sha256(
                    f"{structure_ids[index]}|{z_hash}|{cell_hash}|{pos_hash}".encode()
                ).hexdigest()
            )
        outputs = model.predict_structure(
            structures,
            task="e",
            return_atom_feas=True,
            batch_size=args.batch_size,
        )
        for index, z, output in zip(chunk_indices, chunk_z, outputs, strict=True):
            features = np.asarray(output["atom_fea"], dtype=np.float32)
            if features.shape != (len(z), 64):
                raise RuntimeError(
                    f"Unexpected representation shape at {args.split}[{index}]: {features.shape}"
                )
            if not np.isfinite(features).all():
                raise RuntimeError(f"Non-finite representation at {args.split}[{index}]")
            feature_parts.append(features.astype(np.float16))
            z_parts.append(z.astype(np.int16))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    counts = np.asarray(num_atoms[selected], dtype=np.int32)
    feature_array = np.concatenate(feature_parts, axis=0)
    z_array = np.concatenate(z_parts, axis=0)
    shard_offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    out_dir = CACHE / "staging" / args.split
    output_path = out_dir / f"shard_{args.shard:02d}.npz"
    atomic_npz(
        output_path,
        features=feature_array,
        atomic_numbers=z_array,
        offsets=shard_offsets,
        num_atoms=counts,
        dataset_indices=selected.astype(np.int64),
        structure_ids=np.asarray(structure_ids[selected], dtype=str),
        dft_mag_density=np.asarray(dft_mag[selected], dtype=np.float32),
        atomic_numbers_hash=np.asarray(atom_hashes),
        cell_hash=np.asarray(cell_hashes),
        positions_hash=np.asarray(position_hashes),
        combined_structure_hash=np.asarray(structure_hashes),
    )
    metrics = {
        "schema_version": 1,
        "created_at": now(),
        "split": args.split,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "structures": int(len(selected)),
        "atoms": int(len(z_array)),
        "dimension": 64,
        "dtype": "float16",
        "elapsed_seconds": elapsed,
        "structures_per_second": len(selected) / elapsed,
        "atoms_per_second": len(z_array) / elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(CHECKPOINT),
        "representation_layer": "atom_fea before final CHGNet convolution",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "output": str(output_path),
    }
    atomic_json(out_dir / f"shard_{args.shard:02d}.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
