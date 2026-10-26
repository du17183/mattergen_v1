"""Aggregate Oracle branches, validate them, and evaluate magnetic target error."""

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
import torch_geometric  # noqa: F401 -- initializes ABI-compatible extensions first
import torch_scatter  # noqa: F401

ase.constraints.ExpCellFilter = ExpCellFilter
sys.path.append("/mnt/datasets-livsyn/dxl/mattergen_v1/.venv/lib/python3.10/site-packages")
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/counterfactual_adaptive_cfg_v3"
ORACLE = ROOT / "oracle"
CHGNET_PATH = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/"
    "0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar"
)
TARGET = 0.1
INVALID_PENALTY = 10.0
KINDS = ("SHORT", "FULL")
ACTIONS = ("DOWN", "KEEP", "UP")


def write_csv(path: Path, rows: list[dict]) -> None:
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


def branch_id(kind: str, atoms) -> str:
    if kind == "BASE":
        return f"{int(atoms.info['seed'])}:BASE"
    return (
        f"{int(atoms.info['seed'])}:{int(atoms.info['sampling_step'])}:"
        f"{str(atoms.info['candidate_action'])}:{kind}"
    )


def load_all() -> tuple[list, list, list]:
    seeds = json.loads((ORACLE / "oracle_seeds.json").read_text())["seeds"]
    short_atoms: list = []
    full_atoms: list = []
    base_atoms: list = []
    feature_rows: list[dict] = []
    branch_rows: list[dict] = []
    for seed in seeds:
        source = ORACLE / "generation" / str(seed)
        summary = json.loads((source / "run_summary.json").read_text())
        if not summary["success"]:
            raise RuntimeError(f"failed Oracle generation: {seed}")
        short_atoms.extend(read(source / "short_horizon_x0.extxyz", index=":"))
        full_atoms.extend(read(source / "full_horizon_final.extxyz", index=":"))
        base_atoms.extend(read(source / "base_cfg2_final.extxyz", index=":"))
        with (source / "online_features.csv").open(newline="", encoding="utf-8") as stream:
            feature_rows.extend(csv.DictReader(stream))
        with (source / "branch_manifest.csv").open(newline="", encoding="utf-8") as stream:
            branch_rows.extend(csv.DictReader(stream))
    expected = (32 * 24, 32 * 6, 32, 32 * 8, 32 * 24)
    observed = (
        len(short_atoms), len(full_atoms), len(base_atoms), len(feature_rows), len(branch_rows)
    )
    if observed != expected:
        raise RuntimeError(f"Oracle aggregate count mismatch: observed={observed}, expected={expected}")
    write(ORACLE / "short_horizon_x0_all.extxyz", short_atoms)
    write(ORACLE / "full_horizon_final_all.extxyz", full_atoms)
    write(ORACLE / "base_cfg2_final_all.extxyz", base_atoms)
    write_csv(ORACLE / "online_features_raw.csv", feature_rows)
    write_csv(ORACLE / "branch_manifest.csv", branch_rows)
    return short_atoms, full_atoms, base_atoms


def evaluate_properties(short_atoms: list, full_atoms: list, base_atoms: list) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CHGNet.from_file(CHGNET_PATH).to(device)
    model.eval()
    combined = [("SHORT", item) for item in short_atoms]
    combined += [("FULL", item) for item in full_atoms]
    combined += [("BASE", item) for item in base_atoms]
    records: list[dict] = []
    valid_indices: list[int] = []
    valid_structures: list = []
    for index, (kind, atoms) in enumerate(combined):
        valid, minimum, reason = geometric_validity(atoms)
        action = "KEEP" if kind == "BASE" else str(atoms.info["candidate_action"])
        record = {
            "branch_id": branch_id(kind, atoms),
            "outcome_kind": kind,
            "seed": int(atoms.info["seed"]),
            "decision_ordinal": "" if kind == "BASE" else int(atoms.info["decision_ordinal"]),
            "sampling_step": "" if kind == "BASE" else int(atoms.info["sampling_step"]),
            "decision_t_norm": "" if kind == "BASE" else float(atoms.info["decision_t_norm"]),
            "candidate_action": action,
            "candidate_cfg": 2.0 if kind == "BASE" else float(atoms.info["candidate_cfg"]),
            "formula": atoms.get_chemical_formula(),
            "num_atoms": len(atoms),
            "geometric_valid": valid,
            "minimum_periodic_distance_angstrom": minimum,
            "validity_reason": reason,
            "chgnet_success": False,
            "chgnet_mag_density": float("nan"),
            "target_mag_density": TARGET,
            "property_absolute_error": float("nan"),
            "normalized_property_error": float("nan"),
            "oracle_objective": INVALID_PENALTY,
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
    for start in range(0, len(valid_structures), 16):
        batch_structures = valid_structures[start : start + 16]
        batch_indices = valid_indices[start : start + 16]
        try:
            predictions = model.predict_structure(batch_structures, task="efsm", batch_size=16)
            if isinstance(predictions, dict):
                predictions = [predictions]
        except BaseException:
            predictions = []
            for structure in batch_structures:
                try:
                    predictions.append(model.predict_structure(structure, task="efsm"))
                except BaseException as error:
                    predictions.append(error)
        for record_index, structure, prediction in zip(batch_indices, batch_structures, predictions):
            record = records[record_index]
            if isinstance(prediction, BaseException):
                record["validity_reason"] = f"chgnet:{type(prediction).__name__}"
                continue
            value = float(np.abs(np.asarray(prediction["m"], dtype=float)).sum() / structure.volume)
            if not np.isfinite(value):
                record["validity_reason"] = "chgnet_nonfinite"
                continue
            error = abs(value - TARGET)
            normalized = error / TARGET
            record.update(
                chgnet_success=True,
                chgnet_mag_density=value,
                property_absolute_error=error,
                normalized_property_error=normalized,
                oracle_objective=normalized,
                structure_valid=True,
                validity_reason="",
            )
    write_csv(ORACLE / "property_metrics.csv", records)
    evaluation_dir = ORACLE / "evaluation_structures"
    evaluation_dir.mkdir(exist_ok=False)
    excluded_rows: list[dict] = []
    lookup = {record["branch_id"]: record for record in records}
    for kind, atoms_list in (("SHORT", short_atoms), ("FULL", full_atoms)):
        for action in ACTIONS:
            selected = []
            for atoms in atoms_list:
                if str(atoms.info["candidate_action"]) != action:
                    continue
                identifier = branch_id(kind, atoms)
                if bool(lookup[identifier]["structure_valid"]):
                    atoms.info["sample_seed"] = int(atoms.info["seed"])
                    atoms.info["branch_id"] = identifier
                    selected.append(atoms)
                else:
                    excluded_rows.append(
                        {
                            "branch_id": identifier,
                            "outcome_kind": kind,
                            "candidate_action": action,
                            "reason": lookup[identifier]["validity_reason"],
                        }
                    )
            if selected:
                write(evaluation_dir / f"{kind}_{action}.extxyz", selected)
    if excluded_rows:
        write_csv(ORACLE / "quality_exclusions.csv", excluded_rows)
    summary = {
        "total": len(records),
        "short": len(short_atoms),
        "full": len(full_atoms),
        "base": len(base_atoms),
        "valid": sum(bool(row["structure_valid"]) for row in records),
        "invalid": sum(not bool(row["structure_valid"]) for row in records),
        "device": device,
        "chgnet_version": "0.3.0",
        "target": TARGET,
        "invalid_penalty": INVALID_PENALTY,
        "surrogate_property_eval": True,
        "dft_verified": False,
    }
    (ORACLE / "property_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    evaluate_properties(*load_all())
