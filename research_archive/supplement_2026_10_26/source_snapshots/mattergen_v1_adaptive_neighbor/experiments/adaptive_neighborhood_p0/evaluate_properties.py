"""Evaluate A0/B0 magnetic density with the frozen calibrated CHGNet surrogate."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import ase.io
import numpy as np
import torch
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/adaptive_neighborhood_p0"
CHGNET_PATH = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar")
METHODS = ("B0", "A0")
SEEDS = tuple(range(96000, 96032))
TARGET = 0.2
TAU = 0.01


def main() -> None:
    structures = []
    keys = []
    for method in METHODS:
        atoms = ase.io.read(ROOT / "structures" / f"{method}_generated.extxyz", index=":")
        if len(atoms) != len(SEEDS):
            raise RuntimeError(f"{method}: expected 32 structures")
        for seed, item in zip(SEEDS, atoms):
            structures.append(AseAtomsAdaptor.get_structure(item))
            keys.append((method, seed))
    model = CHGNet.from_file(CHGNET_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
    predictions = model.predict_structure(structures, task="efsm", batch_size=16)
    if isinstance(predictions, dict):
        predictions = [predictions]
    rows = []
    for (method, seed), structure, prediction in zip(keys, structures, predictions):
        value = float(np.abs(np.asarray(prediction["m"], dtype=float)).sum() / structure.volume)
        error = abs(value - TARGET)
        rows.append({
            "method": method,
            "seed": seed,
            "formula": structure.composition.reduced_formula,
            "chgnet_mag_density": value,
            "target_mag_density": TARGET,
            "mag_absolute_error": error,
            "mag_hit": error <= TAU,
            "tau": TAU,
            "surrogate_property_eval": True,
            "surrogate_heldout_mae": 0.001449,
            "surrogate_pearson": 0.9499,
            "dft_verified": False,
        })
    with (ROOT / "heldout_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"surrogate_property_eval": True, "dft_verified": False, "methods": {}}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary["methods"][method] = {
            "n": len(selected),
            "mean_mag_density": float(np.mean([row["chgnet_mag_density"] for row in selected])),
            "mag_mae": float(np.mean([row["mag_absolute_error"] for row in selected])),
            "mag_hits": int(sum(row["mag_hit"] for row in selected)),
            "mag_hit_rate": float(np.mean([row["mag_hit"] for row in selected])),
        }
    base = {row["seed"]: row for row in rows if row["method"] == "B0"}
    selected = [row for row in rows if row["method"] == "A0"]
    summary["methods"]["A0"]["paired_not_worse_count"] = int(sum(
        row["mag_absolute_error"] <= base[row["seed"]]["mag_absolute_error"]
        for row in selected
    ))
    (ROOT / "property_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
