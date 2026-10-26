"""Frozen CHGNet magnetic-density surrogate and basic validity evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import ase.io
import ase.constraints
from ase.filters import ExpCellFilter
import numpy as np
import torch
import torch_geometric  # load ABI-compatible extensions before legacy CHGNet path
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
TARGET = 0.2
TAU = 0.01


def _validity(structure) -> tuple[bool, float]:
    if not np.isfinite(structure.volume) or structure.volume < 0.1 or len(structure) < 1:
        return False, float("nan")
    if len(structure) == 1:
        return True, float("inf")
    distances = np.asarray(structure.distance_matrix, dtype=float)
    np.fill_diagonal(distances, np.inf)
    minimum = float(distances.min())
    return bool(np.isfinite(minimum) and minimum >= 0.5), minimum


def main() -> None:
    atoms = ase.io.read(ROOT / "generated_diagnostic.extxyz", index=":")
    if len(atoms) != 64:
        raise RuntimeError(f"expected 64 structures, got {len(atoms)}")
    structures = [AseAtomsAdaptor.get_structure(item) for item in atoms]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CHGNet.from_file(CHGNET_PATH).to(device)
    model.eval()
    predictions = model.predict_structure(structures, task="efsm", batch_size=16)
    if isinstance(predictions, dict):
        predictions = [predictions]
    rows = []
    for item, structure, prediction in zip(atoms, structures, predictions):
        seed = int(item.info["sample_seed"])
        value = float(np.abs(np.asarray(prediction["m"], dtype=float)).sum() / structure.volume)
        error = abs(value - TARGET)
        forces = np.asarray(prediction["f"], dtype=float).reshape(len(structure), 3)
        force_norms = np.linalg.norm(forces, axis=1)
        valid, minimum = _validity(structure)
        rows.append({
            "seed": seed,
            "subset": "calibration" if seed <= 99031 else "validation",
            "formula": structure.composition.reduced_formula,
            "num_atoms": len(structure),
            "chgnet_mag_density": value,
            "target_mag_density": TARGET,
            "mag_absolute_error": error,
            "mag_hit": error <= TAU,
            "tau": TAU,
            "structure_valid": valid,
            "minimum_periodic_distance_angstrom": minimum,
            "volume_angstrom3": float(structure.volume),
            "chgnet_force_rms_ev_ang": float(np.sqrt(np.mean(forces**2))),
            "chgnet_force_max_ev_ang": float(force_norms.max()),
            "surrogate_property_eval": True,
            "surrogate_calibration_mae": 0.001449,
            "surrogate_calibration_pearson": 0.9499,
            "dft_verified": False,
        })
    path = ROOT / "diagnostic_property_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n": len(rows),
        "device": device,
        "mag_mae": float(np.mean([row["mag_absolute_error"] for row in rows])),
        "mag_hit_count": int(sum(row["mag_hit"] for row in rows)),
        "mag_hit_rate": float(np.mean([row["mag_hit"] for row in rows])),
        "valid_count": int(sum(row["structure_valid"] for row in rows)),
        "valid_fraction": float(np.mean([row["structure_valid"] for row in rows])),
        "surrogate_property_eval": True,
        "surrogate_calibration_mae": 0.001449,
        "surrogate_calibration_pearson": 0.9499,
        "dft_verified": False,
    }
    (ROOT / "diagnostic_property_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
