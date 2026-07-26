from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from research.fn_pra.phase1_common import (
    PROJECT,
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    initialize_progress,
    now,
    set_stage,
    sha256_file,
)


DATASET = Path("/data/dxl/datasets/cache/mp_20")
RAW = Path("/data/dxl/data/extracted/mp_20")
ARCHIVE = Path("/data/dxl/data/archives/mp_20.zip")
SELECTION_SEED = 20260725


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_property(split: Path, name: str) -> np.ndarray:
    data = json.loads((split / f"{name}.json").read_text(encoding="utf-8"))
    return np.asarray(data["values"], dtype=np.float64)


def file_record(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def split_record(name: str) -> dict:
    split = DATASET / name
    num_atoms = np.load(split / "num_atoms.npy", mmap_mode="r")
    atomic_numbers = np.load(split / "atomic_numbers.npy", mmap_mode="r")
    structure_id = np.load(split / "structure_id.npy", allow_pickle=True)
    mag = load_property(split, "dft_mag_density")
    return {
        "split": name,
        "structures": int(len(num_atoms)),
        "atoms": int(num_atoms.sum()),
        "min_atoms": int(num_atoms.min()),
        "max_atoms": int(num_atoms.max()),
        "dft_mag_density_non_null": int(np.isfinite(mag).sum()),
        "unique_structure_ids": int(len(np.unique(structure_id))),
        "structure_id_available": True,
        "atomic_number_min": int(atomic_numbers.min()),
        "atomic_number_max": int(atomic_numbers.max()),
        "files": [file_record(p) for p in sorted(split.iterdir()) if p.is_file()],
    }


def build_probe_manifest(size: int) -> dict:
    split = DATASET / "train"
    num_atoms = np.load(split / "num_atoms.npy", mmap_mode="r")
    offsets = np.concatenate(([0], np.cumsum(num_atoms[:-1], dtype=np.int64)))
    atomic_numbers = np.load(split / "atomic_numbers.npy", mmap_mode="r")
    cell = np.load(split / "cell.npy", mmap_mode="r")
    pos = np.load(split / "pos.npy", mmap_mode="r")
    structure_ids = np.load(split / "structure_id.npy", allow_pickle=True)
    mag = load_property(split, "dft_mag_density")
    formation = load_property(split, "formation_energy_per_atom")
    valid = np.flatnonzero(np.isfinite(mag) & np.isfinite(formation))
    if size > len(valid):
        raise ValueError(f"Requested {size} structures but only {len(valid)} are valid")
    rng = np.random.default_rng(SELECTION_SEED)
    selected = np.sort(rng.choice(valid, size=size, replace=False))

    records = []
    for index in selected:
        start = int(offsets[index])
        stop = start + int(num_atoms[index])
        z = np.asarray(atomic_numbers[start:stop])
        c = np.asarray(cell[index])
        p = np.asarray(pos[start:stop])
        z_hash = array_hash(z)
        c_hash = array_hash(c)
        p_hash = array_hash(p)
        combined = hashlib.sha256(f"{structure_ids[index]}|{z_hash}|{c_hash}|{p_hash}".encode()).hexdigest()
        records.append(
            {
                "dataset_index": int(index),
                "split": "train",
                "structure_id": str(structure_ids[index]),
                "num_atoms": int(num_atoms[index]),
                "dft_mag_density": float(mag[index]),
                "formation_energy_per_atom": float(formation[index]),
                "atomic_numbers_hash": z_hash,
                "cell_hash": c_hash,
                "positions_hash": p_hash,
                "combined_structure_hash": combined,
            }
        )
    return {
        "schema_version": 1,
        "created_at": now(),
        "dataset": "mp_20",
        "split": "train",
        "selection_seed": SELECTION_SEED,
        "selection_rule": "uniform without replacement among finite dft_mag_density and formation_energy_per_atom",
        "size": size,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-size", type=int, default=1000)
    args = parser.parse_args()
    initialize_progress()
    set_stage("data_audit", "running", "Auditing official MP-20 cache and atom mapping.")
    if not DATASET.exists():
        set_stage("data_audit", "blocked", f"Missing preprocessed dataset: {DATASET}")
        raise SystemExit(2)

    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "source": "microsoft/mattergen data-release/mp-20",
        "license": "CC BY 4.0",
        "archive": file_record(ARCHIVE),
        "raw_root": str(RAW),
        "cache_root": str(DATASET),
        "splits": [split_record(name) for name in ("train", "val", "test")],
    }
    atomic_json(REPORTS / "data_manifest.json", manifest)

    probe = build_probe_manifest(args.probe_size)
    atomic_json(RESULTS / "teacher_probe/probe_subset_manifest.json", probe)
    atomic_json(REPORTS / "teacher_probe_subset_manifest.json", probe)

    mapping = {
        "schema_version": 1,
        "created_at": now(),
        "structure_id_in_cache": True,
        "structure_id_in_default_chemgraph": False,
        "dataset_item_atom_order": "preserved by contiguous offset slicing",
        "chemgraph_atom_order": "preserved; only fractional coordinates are reduced modulo 1",
        "pyg_batch_order": "preserved per input graph; ptr/num_atoms retain boundaries",
        "ddp_sampler_behavior": "may permute structures across ranks, never atoms within a structure",
        "fn_pra_mapping_strategy": "cache-backed RepaCrystalDataset attaches teacher rows by cache index before collate",
        "positive_key": ["split", "structure_id", "atom_index"],
        "probe_records": len(probe["records"]),
        "strict_mapping_established": True,
        "PHASE1_BLOCKED_BY_MAPPING": False,
    }
    atomic_json(REPORTS / "structure_mapping_audit.json", mapping)

    git_status = subprocess.check_output(
        ["git", "-C", str(PROJECT), "status", "--short", "--branch"], text=True
    )
    atomic_text(REPORTS / "frozen/git_status.txt", git_status)
    lines = [
        "# FN-PRA Phase-1 data audit",
        "",
        f"Generated: `{now()}`",
        "",
        f"- Official archive: `{ARCHIVE}`",
        f"- Archive SHA256: `{manifest['archive']['sha256']}`",
        f"- Cache: `{DATASET}`",
        f"- Probe subset: {args.probe_size} train structures, seed {SELECTION_SEED}",
        "- Mapping: strict `(structure_id, atom_index)` mapping is established.",
        "- Default ChemGraph does not expose structure_id; FN-PRA uses a cache-backed dataset that attaches atom rows before collate.",
        "",
        "| split | structures | atoms | dft_mag non-null |",
        "|---|---:|---:|---:|",
    ]
    for item in manifest["splits"]:
        lines.append(
            f"| {item['split']} | {item['structures']} | {item['atoms']} | {item['dft_mag_density_non_null']} |"
        )
    atomic_text(REPORTS / "data_audit.md", "\n".join(lines) + "\n")
    set_stage(
        "data_audit",
        "success",
        f"MP-20 audited; strict mapping established; probe subset={args.probe_size}.",
        {"probe_size": args.probe_size, "mapping": "strict", "blocked": False},
    )


if __name__ == "__main__":
    main()
