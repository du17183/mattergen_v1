"""Label final clean-x0 structures with the frozen calibrated CHGNet surrogate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import ase.io
import ase.constraints
from ase.filters import ExpCellFilter
import numpy as np
import torch
import torch_geometric  # load the PyTorch-2.9-compatible build before adding legacy CHGNet
import torch_scatter

ase.constraints.ExpCellFilter = ExpCellFilter
sys.path.append("/mnt/datasets-livsyn/dxl/mattergen_v1/.venv/lib/python3.10/site-packages")
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/self_correcting_search_p0"
CHGNET_PATH = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar")
SEEDS = tuple(range(98000, 98192))
EXPECTED_SPLITS = {"train": 128, "validation": 32, "test": 32}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for_seed(seed: int) -> str:
    if seed <= 98127:
        return "train"
    if seed <= 98159:
        return "validation"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    structures = []
    metadata = []
    for seed in SEEDS:
        split = split_for_seed(seed)
        run_dir = ROOT / "probe_dataset" / split / str(seed)
        summary_path = run_dir / "run_summary.json"
        hidden_path = run_dir / "hidden_states.npz"
        structure_path = run_dir / "generated_crystals.extxyz"
        if not all(path.is_file() for path in (summary_path, hidden_path, structure_path)):
            raise FileNotFoundError(f"incomplete probe trajectory: {run_dir}")
        summary = json.loads(summary_path.read_text())
        atoms = ase.io.read(structure_path, index=":")
        if len(atoms) != 1:
            raise RuntimeError(f"expected one clean-x0 structure: {structure_path}")
        structure = AseAtomsAdaptor.get_structure(atoms[0])
        structures.append(structure)
        metadata.append((seed, split, run_dir, summary, structure))

    model = CHGNet.from_file(CHGNET_PATH).to(args.device)
    model.eval()
    predictions = model.predict_structure(
        structures, task="efsm", batch_size=args.batch_size
    )
    if isinstance(predictions, dict):
        predictions = [predictions]
    if len(predictions) != len(SEEDS):
        raise RuntimeError("CHGNet prediction count mismatch")

    rows = []
    for (seed, split, run_dir, summary, structure), prediction in zip(metadata, predictions):
        moment = np.asarray(prediction["m"], dtype=float)
        value = float(np.abs(moment).sum() / structure.volume)
        rows.append({
            "trajectory_id": f"probe_{seed}",
            "seed": seed,
            "split": split,
            "formula": structure.composition.reduced_formula,
            "num_atoms": len(structure),
            "volume_angstrom3": float(structure.volume),
            "final_chgnet_mag_density": value,
            "target_mag_density": 0.2,
            "target_absolute_error": abs(value - 0.2),
            "target_hit_tau_0p01": abs(value - 0.2) <= 0.01,
            "hidden_states_path": str((run_dir / "hidden_states.npz").relative_to(PROJECT_ROOT)),
            "clean_x0_structure_path": str((run_dir / "generated_crystals.extxyz").relative_to(PROJECT_ROOT)),
            "hidden_states_sha256": sha256(run_dir / "hidden_states.npz"),
            "clean_x0_structure_sha256": sha256(run_dir / "generated_crystals.extxyz"),
            "requested_timesteps": ";".join(map(str, summary["requested_timesteps"])),
            "actual_timesteps": ";".join(f"{value:.9f}" for value in summary["actual_timesteps"]),
            "activation_phase": summary["activation_phase"],
            "activation_branch": summary["activation_branch"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "chgnet_checkpoint": str(CHGNET_PATH),
            "chgnet_checkpoint_sha256": sha256(CHGNET_PATH),
            "surrogate_property_eval": True,
            "surrogate_calibration_mae": 0.001449,
            "surrogate_calibration_pearson": 0.9499,
            "dft_verified": False,
        })
    counts = {split: sum(row["split"] == split for row in rows) for split in EXPECTED_SPLITS}
    if counts != EXPECTED_SPLITS:
        raise RuntimeError(f"split counts changed: {counts}")
    output = ROOT / "probe_dataset_manifest.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"manifest": str(output), "counts": counts,
                      "label_min": min(row["final_chgnet_mag_density"] for row in rows),
                      "label_max": max(row["final_chgnet_mag_density"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
