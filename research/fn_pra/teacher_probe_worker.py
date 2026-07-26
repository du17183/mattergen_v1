from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from research.fn_pra.phase1_common import RESULTS, atomic_json, now


DATASET = Path("/data/dxl/datasets/cache/mp_20/train")
MANIFEST = RESULTS / "teacher_probe/probe_subset_manifest.json"
MATTERSIM_CHECKPOINT = Path("/data/dxl/mattersim_weights/mattersim-v1.0.0-5M.pth")
CHGNET_CHECKPOINT = Path(
    "/data/dxl/envs/fn_pra_teacher/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/"
    "chgnet_0.3.0_e29f68s314m37.pth.tar"
)


def load_records(shard: int, num_shards: int):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = [r for i, r in enumerate(manifest["records"]) if i % num_shards == shard]
    num_atoms = np.load(DATASET / "num_atoms.npy", mmap_mode="r")
    offsets = np.concatenate(([0], np.cumsum(num_atoms[:-1], dtype=np.int64)))
    z_all = np.load(DATASET / "atomic_numbers.npy", mmap_mode="r")
    pos_all = np.load(DATASET / "pos.npy", mmap_mode="r")
    cells = np.load(DATASET / "cell.npy", mmap_mode="r")
    structures = []
    atomic_numbers = []
    coordination = []
    for record in records:
        idx = record["dataset_index"]
        start = int(offsets[idx])
        stop = start + int(num_atoms[idx])
        z = np.asarray(z_all[start:stop], dtype=np.int64)
        pos = np.asarray(pos_all[start:stop], dtype=np.float64)
        structure = Structure(
            Lattice(np.asarray(cells[idx], dtype=np.float64)),
            z,
            pos,
            coords_are_cartesian=False,
            to_unit_cell=True,
        )
        if len(structure) != record["num_atoms"]:
            raise RuntimeError(f"Atom count mismatch for {record['structure_id']}")
        structures.append(structure)
        atomic_numbers.append(z)
        coordination.append(
            np.asarray([len(neighbors) for neighbors in structure.get_all_neighbors(3.0)], dtype=np.int16)
        )
    return records, structures, atomic_numbers, coordination


def chgnet_features(structures: list[Structure], batch_size: int):
    from chgnet.model.model import CHGNet

    model = CHGNet.load(model_name="0.3.0", use_device="cuda", verbose=False)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = model.predict_structure(
        structures,
        task="e",
        return_atom_feas=True,
        batch_size=batch_size,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    features = [np.asarray(item["atom_fea"], dtype=np.float32) for item in outputs]
    repeat = model.predict_structure(
        structures[: min(batch_size, len(structures))],
        task="e",
        return_atom_feas=True,
        batch_size=batch_size,
    )
    repeat_features = [np.asarray(item["atom_fea"], dtype=np.float32) for item in repeat]
    max_abs = max(
        float(np.max(np.abs(a - b)))
        for a, b in zip(features[: len(repeat_features)], repeat_features, strict=True)
    )
    peak = int(torch.cuda.max_memory_allocated())
    return features, elapsed, peak, max_abs


def mattersim_features(structures: list[Structure], batch_size: int):
    from pymatgen.io.ase import AseAtomsAdaptor
    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield.potential import Potential, batch_to_dict

    atoms = [AseAtomsAdaptor.get_atoms(structure) for structure in structures]
    loader = build_dataloader(
        atoms=atoms,
        batch_size=batch_size,
        model_type="m3gnet",
        only_inference=True,
        shuffle=False,
        num_workers=0,
    )
    potential = Potential.from_checkpoint(
        load_path=str(MATTERSIM_CHECKPOINT),
        device="cuda",
        load_training_state=False,
    )
    potential.model.eval()
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output[0].detach())

    handle = potential.model.graph_conv[-1].register_forward_hook(hook)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output_features = []
    first_input = None
    first_counts = None
    with torch.no_grad():
        for graph_batch in loader:
            graph_batch = graph_batch.to("cuda")
            model_input = batch_to_dict(graph_batch)
            if first_input is None:
                first_input = {k: v.clone() if torch.is_tensor(v) else v for k, v in model_input.items()}
                first_counts = graph_batch.num_atoms.detach().cpu().tolist()
            captured.clear()
            potential.model(model_input)
            node = captured[-1].detach().cpu().numpy().astype(np.float32)
            counts = graph_batch.num_atoms.detach().cpu().tolist()
            cursor = 0
            for count in counts:
                output_features.append(node[cursor : cursor + int(count)])
                cursor += int(count)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    captured.clear()
    with torch.no_grad():
        potential.model(first_input)
    repeat_node = captured[-1].detach().cpu().numpy().astype(np.float32)
    first_total = int(sum(first_counts))
    original_first = np.concatenate(output_features[: len(first_counts)], axis=0)
    max_abs = float(np.max(np.abs(original_first - repeat_node[:first_total])))
    peak = int(torch.cuda.max_memory_allocated())
    handle.remove()
    return output_features, elapsed, peak, max_abs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["chgnet", "mattersim"], required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("Invalid shard")
    records, structures, atomic_numbers, coordination = load_records(args.shard, args.num_shards)
    if args.candidate == "chgnet":
        features, elapsed, peak, repeat_max_abs = chgnet_features(structures, args.batch_size)
        checkpoint = CHGNET_CHECKPOINT
        layer = "atom_fea before final CHGNet convolution"
        version = "CHGNet 0.4.2 / checkpoint 0.3.0"
    else:
        features, elapsed, peak, repeat_max_abs = mattersim_features(structures, args.batch_size)
        checkpoint = MATTERSIM_CHECKPOINT
        layer = "M3GNet graph_conv[-1] output atom_attr"
        version = "MatterSim 1.1.2 / 5M"

    if len(features) != len(records):
        raise RuntimeError("Teacher output structure count mismatch")
    for record, z, rep in zip(records, atomic_numbers, features, strict=True):
        if rep.shape[0] != record["num_atoms"] or rep.shape[0] != len(z):
            raise RuntimeError(f"Teacher row mismatch for {record['structure_id']}: {rep.shape}")
        if not np.isfinite(rep).all():
            raise RuntimeError(f"Non-finite teacher representation for {record['structure_id']}")

    counts = np.asarray([len(x) for x in atomic_numbers], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    feature_array = np.concatenate(features, axis=0)
    z_array = np.concatenate(atomic_numbers, axis=0)
    coordination_array = np.concatenate(coordination, axis=0)
    out_dir = RESULTS / f"teacher_probe/{args.candidate}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shard_{args.shard:02d}.npz"
    np.savez_compressed(
        out,
        features=feature_array,
        atomic_numbers=z_array,
        coordination=coordination_array,
        offsets=offsets,
        dataset_indices=np.asarray([r["dataset_index"] for r in records], dtype=np.int64),
        structure_ids=np.asarray([r["structure_id"] for r in records]),
        structure_hashes=np.asarray([r["combined_structure_hash"] for r in records]),
        formation_energy=np.asarray([r["formation_energy_per_atom"] for r in records], dtype=np.float64),
        dft_mag_density=np.asarray([r["dft_mag_density"] for r in records], dtype=np.float64),
    )
    atoms = int(len(z_array))
    metrics = {
        "schema_version": 1,
        "created_at": now(),
        "candidate": args.candidate,
        "version": version,
        "checkpoint": str(checkpoint),
        "representation_layer": layer,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "structures": len(records),
        "atoms": atoms,
        "dimension": int(feature_array.shape[1]),
        "elapsed_seconds": elapsed,
        "structures_per_second": len(records) / elapsed,
        "atoms_per_second": atoms / elapsed,
        "peak_vram_bytes": peak,
        "repeat_max_abs": repeat_max_abs,
        "nan_count": int(np.isnan(feature_array).sum()),
        "inf_count": int(np.isinf(feature_array).sum()),
        "feature_norm_mean": float(np.linalg.norm(feature_array, axis=1).mean()),
        "feature_std_mean": float(feature_array.std(axis=0).mean()),
        "output": str(out),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    atomic_json(out_dir / f"shard_{args.shard:02d}.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
