"""Aggregate generated schedules and evaluate the frozen conditional property."""

from __future__ import annotations

import argparse
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

from experiments.stage_calibrated_bounded_cfg.run_generation import phase_specs


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/stage_calibrated_bounded_cfg"
CHGNET_PATH = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar")
TARGET = 0.1
INVALID_OBJECTIVE = 1.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def geometric_validity(atoms) -> tuple[bool, float, str]:
    cell = np.asarray(atoms.cell.array, dtype=float)
    if len(atoms) < 1 or not np.isfinite(cell).all():
        return False, float("nan"), "empty_or_nonfinite_cell"
    determinant = float(np.linalg.det(cell))
    if not np.isfinite(determinant) or determinant <= 0.1:
        return False, float("nan"), "nonpositive_or_small_cell"
    if len(atoms) == 1:
        return True, float("inf"), ""
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    np.fill_diagonal(distances, np.inf)
    minimum = float(distances.min())
    if not np.isfinite(minimum) or minimum < 0.5:
        return False, minimum, "minimum_distance_below_0.5A"
    return True, minimum, ""


def main(phase: str) -> None:
    phase_root = ROOT / phase
    seeds = list(map(int, json.loads((phase_root / "seeds.json").read_text())["seeds"]))
    expected_ids = {spec["schedule_id"] for spec in phase_specs(phase)}
    atoms_list, manifests, summaries = [], [], []
    for seed in seeds:
        source = phase_root / "generation" / str(seed)
        summary = json.loads((source / "run_summary.json").read_text())
        if not summary["success"]:
            raise RuntimeError(f"failed generation: {seed}")
        summaries.append(summary)
        atoms_list.extend(read(source / "schedule_final.extxyz", index=":"))
        with (source / "schedule_manifest.csv").open(newline="", encoding="utf-8") as stream:
            manifests.extend(csv.DictReader(stream))
    observed = {(int(atoms.info["seed"]), str(atoms.info["schedule_id"])) for atoms in atoms_list}
    expected = {(seed, identifier) for seed in seeds for identifier in expected_ids}
    if observed != expected or len(atoms_list) != len(expected):
        raise RuntimeError("aggregate seed/schedule matrix is incomplete or duplicated")
    write(phase_root / "schedule_final_all.extxyz", atoms_list)
    write_csv(phase_root / "generation_manifest.csv", manifests)
    (phase_root / "generation_summary.json").write_text(json.dumps({"complete": True, "phase": phase, "seeds": len(seeds), "schedules": len(expected_ids), "structures": len(atoms_list), "total_elapsed_gpu_hours": sum(row["elapsed_seconds"] for row in summaries) / 3600, "max_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in summaries), "score_calls": sum(row["mattergen_score_calls"] for row in summaries)}, indent=2) + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CHGNet.from_file(CHGNET_PATH).to(device)
    model.eval()
    records: list[dict] = []
    valid_indices, valid_structures = [], []
    for index, atoms in enumerate(atoms_list):
        valid, minimum, reason = geometric_validity(atoms)
        record = {
            "branch_id": f"{int(atoms.info['seed'])}:{atoms.info['schedule_id']}",
            "seed": int(atoms.info["seed"]),
            "schedule_id": str(atoms.info["schedule_id"]),
            "schedule_kind": str(atoms.info["schedule_kind"]),
            "active_cfg": float(atoms.info["active_cfg"]),
            "delta_g": float(atoms.info["delta_g"]),
            "start": int(atoms.info["start"]),
            "duration": int(atoms.info["duration"]),
            "formula": atoms.get_chemical_formula(),
            "num_atoms": len(atoms),
            "geometric_valid": valid,
            "minimum_periodic_distance_angstrom": minimum,
            "validity_reason": reason,
            "chgnet_success": False,
            "chgnet_mag_density": float("nan"),
            "target_mag_density": TARGET,
            "property_absolute_error": INVALID_OBJECTIVE,
            "structure_valid": False,
            "surrogate_property_eval": True,
            "dft_verified": False,
        }
        records.append(record)
        if valid:
            try:
                structure = AseAtomsAdaptor.get_structure(atoms)
            except BaseException as error:
                record["validity_reason"] = f"pymatgen_conversion:{type(error).__name__}"
            else:
                valid_indices.append(index)
                valid_structures.append(structure)
    for start in range(0, len(valid_structures), 32):
        structures = valid_structures[start : start + 32]
        indices = valid_indices[start : start + 32]
        try:
            predictions = model.predict_structure(structures, task="efsm", batch_size=32)
            if isinstance(predictions, dict):
                predictions = [predictions]
        except BaseException:
            predictions = []
            for structure in structures:
                try:
                    predictions.append(model.predict_structure(structure, task="efsm"))
                except BaseException as error:
                    predictions.append(error)
        for record_index, structure, prediction in zip(indices, structures, predictions):
            record = records[record_index]
            if isinstance(prediction, BaseException):
                record["validity_reason"] = f"chgnet:{type(prediction).__name__}"
                continue
            value = float(np.abs(np.asarray(prediction["m"], dtype=float)).sum() / structure.volume)
            if not np.isfinite(value):
                record["validity_reason"] = "chgnet_nonfinite"
                continue
            record.update(chgnet_success=True, chgnet_mag_density=value, property_absolute_error=abs(value - TARGET), structure_valid=True, validity_reason="")
    write_csv(phase_root / "property_metrics.csv", records)
    evaluation = phase_root / "evaluation_structures"
    evaluation.mkdir(exist_ok=False)
    lookup = {row["branch_id"]: row for row in records}
    exclusions: list[dict] = []
    for identifier in sorted(expected_ids):
        selected = []
        for atoms in atoms_list:
            if str(atoms.info["schedule_id"]) != identifier:
                continue
            branch = f"{int(atoms.info['seed'])}:{identifier}"
            if lookup[branch]["structure_valid"]:
                atoms.info.update(sample_seed=int(atoms.info["seed"]), branch_id=branch)
                selected.append(atoms)
            else:
                exclusions.append({"branch_id": branch, "schedule_id": identifier, "reason": lookup[branch]["validity_reason"]})
        if selected:
            write(evaluation / f"{identifier}.extxyz", selected)
    if exclusions:
        write_csv(phase_root / "quality_exclusions.csv", exclusions)
    (phase_root / "property_summary.json").write_text(json.dumps({"total": len(records), "valid": sum(row["structure_valid"] for row in records), "invalid": sum(not row["structure_valid"] for row in records), "device": device, "chgnet_version": "0.3.0", "target": TARGET, "invalid_objective": INVALID_OBJECTIVE, "surrogate_property_eval": True, "dft_verified": False}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "p0", "formal256"), required=True)
    arguments = parser.parse_args()
    main(arguments.phase)
