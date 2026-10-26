"""Frozen CHGNet magnetic-density surrogate and validity for both P0 arms."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import ase.constraints
import ase.io
from ase.filters import ExpCellFilter
import numpy as np
import torch
import torch_geometric  # load ABI-compatible extensions before legacy CHGNet
import torch_scatter

ase.constraints.ExpCellFilter = ExpCellFilter
sys.path.append("/mnt/datasets-livsyn/dxl/mattergen_v1/.venv/lib/python3.10/site-packages")
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor


ROOT = Path(__file__).resolve().parent
CHGNET_PATH = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/"
    "0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar"
)
METHODS = ("B0", "F0")
SEEDS = tuple(range(710032, 710048))
TARGET = 0.2
TAU = 0.01


def validity(structure) -> tuple[bool, float]:
    if not np.isfinite(structure.volume) or structure.volume < 0.1 or len(structure) < 1:
        return False, float("nan")
    if len(structure) == 1:
        return True, float("inf")
    distances = np.asarray(structure.distance_matrix, dtype=float)
    np.fill_diagonal(distances, np.inf)
    minimum = float(distances.min())
    return bool(np.isfinite(minimum) and minimum >= 0.5), minimum


def main() -> None:
    by_method = {}
    for method in METHODS:
        atoms = ase.io.read(ROOT / "p0_structures" / f"{method}_generated.extxyz", index=":")
        seeds = tuple(int(item.info["sample_seed"]) for item in atoms)
        if len(atoms) != len(SEEDS) or seeds != SEEDS:
            raise RuntimeError(f"{method}: structure/seed mismatch")
        by_method[method] = atoms
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CHGNet.from_file(CHGNET_PATH).to(device)
    model.eval()
    rows = []
    for method in METHODS:
        atoms = by_method[method]
        structures = [AseAtomsAdaptor.get_structure(item) for item in atoms]
        predictions = model.predict_structure(structures, task="efsm", batch_size=16)
        if isinstance(predictions, dict):
            predictions = [predictions]
        for seed, structure, prediction in zip(SEEDS, structures, predictions):
            value = float(np.abs(np.asarray(prediction["m"], dtype=float)).sum() / structure.volume)
            error = abs(value - TARGET)
            valid, minimum = validity(structure)
            rows.append({
                "method": method, "seed": seed,
                "formula": structure.composition.reduced_formula,
                "num_atoms": len(structure),
                "chgnet_mag_density": value, "target_mag_density": TARGET,
                "mag_absolute_error": error, "mag_hit": error <= TAU, "tau": TAU,
                "structure_valid": valid,
                "minimum_periodic_distance_angstrom": minimum,
                "volume_angstrom3": float(structure.volume),
                "surrogate_property_eval": True,
                "surrogate_calibration_mae": 0.001449,
                "surrogate_calibration_pearson": 0.9499,
                "dft_verified": False,
            })
    path = ROOT / "p0_property_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "n": len(selected),
            "mag_mae": float(np.mean([row["mag_absolute_error"] for row in selected])),
            "mag_hit_count": int(sum(row["mag_hit"] for row in selected)),
            "mag_hit_rate": float(np.mean([row["mag_hit"] for row in selected])),
            "valid_count": int(sum(row["structure_valid"] for row in selected)),
            "valid_fraction": float(np.mean([row["structure_valid"] for row in selected])),
        }
    payload = {
        "schema_version": 1, "device": device, "target": TARGET, "tau": TAU,
        "chgnet_version": "0.3.0", "surrogate_property_eval": True,
        "surrogate_calibration_mae": 0.001449,
        "surrogate_calibration_pearson": 0.9499, "dft_verified": False,
        "methods": summary,
    }
    (ROOT / "p0_property_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
