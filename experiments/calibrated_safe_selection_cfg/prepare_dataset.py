"""Aggregate full-outcome branches and evaluate the frozen property objective."""

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
ROOT = PROJECT / "experiments/calibrated_safe_selection_cfg"
DATASET = ROOT / "dataset"
CHGNET_PATH = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/"
    "0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar"
)
TARGET = 0.1
INVALID_OBJECTIVE = 1.0
ACTIONS = ("DOWN", "KEEP", "UP")


def write_csv_once(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def branch_id(atoms) -> str:
    return f"{int(atoms.info['seed'])}:250:{str(atoms.info['candidate_action'])}"


def aggregate() -> list:
    seeds = list(map(int, json.loads((DATASET / "seeds.json").read_text())["seeds"]))
    branches: list = []
    bases: list = []
    features: list[dict] = []
    manifests: list[dict] = []
    summaries: list[dict] = []
    for seed in seeds:
        source = DATASET / "generation" / str(seed)
        summary = json.loads((source / "run_summary.json").read_text())
        if not summary["success"]:
            raise RuntimeError(f"failed generation: {seed}")
        summaries.append(summary)
        branches.extend(read(source / "full_horizon_final.extxyz", index=":"))
        bases.extend(read(source / "base_cfg2_final.extxyz", index=":"))
        with (source / "online_features.csv").open(newline="", encoding="utf-8") as stream:
            features.extend(csv.DictReader(stream))
        with (source / "branch_manifest.csv").open(newline="", encoding="utf-8") as stream:
            manifests.extend(csv.DictReader(stream))
    observed = (len(branches), len(bases), len(features), len(manifests))
    expected = (128 * 3, 128, 128, 128 * 3)
    if observed != expected:
        raise RuntimeError(f"aggregate count mismatch: observed={observed}, expected={expected}")
    write(DATASET / "full_horizon_final_all.extxyz", branches)
    write(DATASET / "base_cfg2_final_all.extxyz", bases)
    write_csv_once(DATASET / "online_features_raw.csv", features)
    write_csv_once(DATASET / "generation_manifest.csv", manifests)
    (DATASET / "generation_summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "trajectories": 128,
                "decision_states": 128,
                "full_branches": 384,
                "base_final_structures": 128,
                "total_elapsed_gpu_hours": sum(row["elapsed_seconds"] for row in summaries) / 3600,
                "mean_elapsed_seconds_per_trajectory": float(
                    np.mean([row["elapsed_seconds"] for row in summaries])
                ),
                "maximum_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in summaries),
                "all_keep_branches_exactly_paired": all(
                    row["pairing_check"]["atomic_numbers_equal"]
                    and row["pairing_check"]["max_fractional_position_abs_error"] == 0
                    and row["pairing_check"]["max_cell_abs_error"] == 0
                    for row in summaries
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return branches


def evaluate_properties(branches: list) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CHGNet.from_file(CHGNET_PATH).to(device)
    model.eval()
    records: list[dict] = []
    valid_indices: list[int] = []
    valid_structures: list = []
    for index, atoms in enumerate(branches):
        valid, minimum, reason = geometric_validity(atoms)
        record = {
            "branch_id": branch_id(atoms),
            "seed": int(atoms.info["seed"]),
            "decision_ordinal": int(atoms.info["decision_ordinal"]),
            "sampling_step": int(atoms.info["sampling_step"]),
            "decision_t_norm": float(atoms.info["decision_t_norm"]),
            "candidate_action": str(atoms.info["candidate_action"]),
            "candidate_cfg": float(atoms.info["candidate_cfg"]),
            "state_sha256": str(atoms.info["state_sha256"]),
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
            record.update(
                chgnet_success=True,
                chgnet_mag_density=value,
                property_absolute_error=abs(value - TARGET),
                structure_valid=True,
                validity_reason="",
            )
    write_csv_once(DATASET / "property_metrics.csv", records)
    evaluation = DATASET / "evaluation_structures"
    evaluation.mkdir(exist_ok=False)
    lookup = {row["branch_id"]: row for row in records}
    exclusions: list[dict] = []
    for action in ACTIONS:
        selected = []
        for atoms in branches:
            if str(atoms.info["candidate_action"]) != action:
                continue
            identifier = branch_id(atoms)
            if lookup[identifier]["structure_valid"]:
                atoms.info.update(sample_seed=int(atoms.info["seed"]), branch_id=identifier)
                selected.append(atoms)
            else:
                exclusions.append({"branch_id": identifier, "candidate_action": action,
                                   "reason": lookup[identifier]["validity_reason"]})
        if selected:
            write(evaluation / f"STEP250_{action}.extxyz", selected)
    if exclusions:
        write_csv_once(DATASET / "quality_exclusions.csv", exclusions)
    (DATASET / "property_summary.json").write_text(
        json.dumps(
            {
                "total": len(records),
                "valid": sum(row["structure_valid"] for row in records),
                "invalid": sum(not row["structure_valid"] for row in records),
                "device": device,
                "chgnet_version": "0.3.0",
                "target": TARGET,
                "invalid_objective": INVALID_OBJECTIVE,
                "surrogate_property_eval": True,
                "dft_verified": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    evaluate_properties(aggregate())
