"""Aggregate fresh P0 policies and evaluate the frozen property surrogate."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import ase.constraints
from ase.filters import ExpCellFilter
from ase.io import read, write
import numpy as np
import torch
import torch_geometric  # noqa: F401
import torch_scatter  # noqa: F401

ase.constraints.ExpCellFilter = ExpCellFilter
sys.path.append("/mnt/datasets-livsyn/dxl/mattergen_v1/.venv/lib/python3.10/site-packages")
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/risk_calibrated_adaptive_cfg"
P0 = ROOT / "p0"
CHGNET_PATH = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar")
TARGET = 0.1
POLICIES = ("C0", "T0", "A0")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def validity(atoms) -> tuple[bool, float, str]:
    cell = np.asarray(atoms.cell.array, dtype=float)
    if len(atoms) < 1 or not np.isfinite(cell).all() or np.linalg.det(cell) <= 0.1:
        return False, float("nan"), "invalid_cell"
    if len(atoms) == 1: return True, float("inf"), ""
    distances = atoms.get_all_distances(mic=True); np.fill_diagonal(distances, np.inf)
    minimum = float(np.min(distances))
    return (minimum >= 0.5, minimum, "" if minimum >= 0.5 else "minimum_distance_below_0.5A")


def main() -> None:
    seeds = json.loads((P0 / "seeds.json").read_text())["seeds"]
    atoms_list, traces, manifest = [], [], []
    for seed in seeds:
        source = P0 / "generation" / str(seed)
        if not json.loads((source / "run_summary.json").read_text())["success"]:
            raise RuntimeError(f"P0 generation failed: {seed}")
        items = read(source / "policy_final.extxyz", index=":")
        if len(items) != 3: raise RuntimeError(f"P0 policy count mismatch: {seed}")
        atoms_list.extend(items)
        with (source / "controller_trace.csv").open(newline="", encoding="utf-8") as stream:
            traces.extend(csv.DictReader(stream))
        manifest.extend(
            {"seed": int(seed), "policy": atoms.info["policy"], "selected_action": atoms.info["selected_action"],
             "selected_cfg": atoms.info["selected_cfg"], "sampling_step": atoms.info["sampling_step"],
             "state_sha256": atoms.info["state_sha256"]}
            for atoms in items
        )
    if len(atoms_list) != len(seeds) * 3 or len(traces) != len(seeds):
        raise RuntimeError("fresh-stage aggregate incomplete")
    write(P0 / "policy_final_all.extxyz", atoms_list)
    write_csv(P0 / "controller_trace.csv", traces); write_csv(P0 / "generation_manifest.csv", manifest)
    model = CHGNet.from_file(CHGNET_PATH).to("cuda" if torch.cuda.is_available() else "cpu"); model.eval()
    rows, valid_indices, structures = [], [], []
    for index, atoms in enumerate(atoms_list):
        valid, minimum, reason = validity(atoms)
        row = {"seed": int(atoms.info["seed"]), "policy": str(atoms.info["policy"]),
               "selected_action": str(atoms.info["selected_action"]), "selected_cfg": float(atoms.info["selected_cfg"]),
               "formula": atoms.get_chemical_formula(), "num_atoms": len(atoms), "geometric_valid": valid,
               "minimum_periodic_distance_angstrom": minimum, "validity_reason": reason,
               "chgnet_mag_density": np.nan, "property_absolute_error": 1.0, "structure_valid": False}
        rows.append(row)
        if valid:
            try: structure = AseAtomsAdaptor.get_structure(atoms)
            except BaseException as error: row["validity_reason"] = f"pymatgen:{type(error).__name__}"
            else: valid_indices.append(index); structures.append(structure)
    for start in range(0, len(structures), 32):
        batch, indices = structures[start:start + 32], valid_indices[start:start + 32]
        predictions = model.predict_structure(batch, task="efsm", batch_size=32)
        if isinstance(predictions, dict): predictions = [predictions]
        for index, structure, prediction in zip(indices, batch, predictions):
            value = float(np.abs(np.asarray(prediction["m"])).sum() / structure.volume)
            rows[index].update(chgnet_mag_density=value, property_absolute_error=abs(value - TARGET), structure_valid=True, validity_reason="")
    write_csv(P0 / "property_metrics.csv", rows)
    evaluation = P0 / "evaluation_structures"; evaluation.mkdir(exist_ok=False)
    for policy in POLICIES:
        selected = []
        for atoms, row in zip(atoms_list, rows):
            if atoms.info["policy"] == policy and row["structure_valid"]:
                atoms.info["branch_id"] = f"{row['seed']}:{policy}"; atoms.info["sample_seed"] = row["seed"]; selected.append(atoms)
        if selected: write(evaluation / f"{policy}.extxyz", selected)


if __name__ == "__main__": main()
