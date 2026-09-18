"""Aggregate a confirmatory cohort and evaluate CHGNet property error."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
ROOT = PROJECT / "experiments/frozen_linear_k2_confirmation"
PROTOCOL_ROOT = ROOT / "protocol"
CHGNET_PATH = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar")
CHGNET_SHA256 = "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"
TARGET = 0.1
INVALID_OBJECTIVE = 1.0
GROUPS = ("C0", "Fixed_K2_rank1", "Fixed_K2_rank2", "Random_K2_rank1", "Random_K2_rank2", "Linear_K2_rank1", "Linear_K2_rank2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_seeds(phase: str) -> list[int]:
    manifest = json.loads((PROTOCOL_ROOT / "seed_manifest_256.json").read_text())
    return list(map(int, manifest[phase]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: raise ValueError(f"empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def geometric_validity(atoms) -> tuple[bool, float, str]:
    cell = np.asarray(atoms.cell.array, dtype=float)
    if len(atoms) < 1 or not np.isfinite(cell).all(): return False, float("nan"), "empty_or_nonfinite_cell"
    determinant = float(np.linalg.det(cell))
    if not np.isfinite(determinant) or determinant <= 0.1: return False, float("nan"), "nonpositive_or_small_cell"
    if len(atoms) == 1: return True, float("inf"), ""
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float); np.fill_diagonal(distances, np.inf); minimum = float(distances.min())
    if not np.isfinite(minimum) or minimum < 0.5: return False, minimum, "minimum_distance_below_0.5A"
    return True, minimum, ""


def quality_group(method: str, rank: int) -> str:
    return "C0" if method == "C0" else f"{method}_rank{rank}"


def main(phase: str) -> None:
    if sha256(CHGNET_PATH) != CHGNET_SHA256:
        raise RuntimeError("CHGNet checkpoint hash mismatch")
    phase_root = ROOT / phase; seeds = phase_seeds(phase)
    atoms_list, manifests, summaries, feature_rows, prediction_rows = [], [], [], [], []
    for seed in seeds:
        source = phase_root / "generation" / str(seed); summary = json.loads((source / "run_summary.json").read_text())
        if not summary["success"] or not summary["reference_reproduction"]["success"]: raise RuntimeError(f"failed generation/reference: {seed}")
        summaries.append(summary); atoms_list.extend(read(source / "deployment_final.extxyz", index=":"))
        with (source / "deployment_manifest.csv").open(newline="", encoding="utf-8") as stream: manifests.extend(csv.DictReader(stream))
        with (source / "allocator_predictions.csv").open(newline="", encoding="utf-8") as stream: prediction_rows.extend(csv.DictReader(stream))
        features = json.loads((source / "branch_features.json").read_text()); feature_rows.append(features)
    expected = len(seeds) * 7
    if len(atoms_list) != expected or len({str(atom.info["branch_id"]) for atom in atoms_list}) != expected: raise RuntimeError("deployment matrix incomplete")
    write(phase_root / "deployment_final_all.extxyz", atoms_list); write_csv(phase_root / "generation_manifest.csv", manifests); write_csv(phase_root / "allocator_predictions.csv", prediction_rows); write_csv(phase_root / "prebranch_features.csv", feature_rows)
    (phase_root / "generation_summary.json").write_text(json.dumps({
        "complete": True, "phase": phase, "seeds": len(seeds), "structures": len(atoms_list),
        "total_elapsed_gpu_hours": sum(row["elapsed_seconds"] for row in summaries) / 3600.0,
        "acquisition_score_calls": sum(row["mattergen_score_calls_acquisition"] for row in summaries),
        "per_method_score_calls_per_seed": 4400, "per_method_compute_multiplier": 2.2,
        "reference_reproduction_passed": all(row["reference_reproduction"]["success"] for row in summaries),
        "max_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in summaries),
    }, indent=2) + "\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"; model = CHGNet.from_file(CHGNET_PATH).to(device); model.eval()
    records, valid_indices, valid_structures = [], [], []
    for index, atoms in enumerate(atoms_list):
        valid, minimum, reason = geometric_validity(atoms); method = str(atoms.info["method"]); rank = int(atoms.info["allocation_rank"]); group = quality_group(method, rank)
        record = {
            "branch_id": str(atoms.info["branch_id"]), "seed": int(atoms.info["seed"]), "method": method,
            "allocation_rank": rank, "policy_id": str(atoms.info["policy_id"]), "quality_group": group,
            "formula": atoms.get_chemical_formula(), "num_atoms": len(atoms), "geometric_valid": valid,
            "minimum_periodic_distance_angstrom": minimum, "validity_reason": reason, "chgnet_success": False,
            "chgnet_mag_density": float("nan"), "target_mag_density": TARGET, "property_absolute_error": INVALID_OBJECTIVE,
            "structure_valid": False, "surrogate_property_eval": True, "dft_verified": False,
        }
        records.append(record)
        if valid:
            try: structure = AseAtomsAdaptor.get_structure(atoms)
            except BaseException as error: record["validity_reason"] = f"pymatgen_conversion:{type(error).__name__}"
            else: valid_indices.append(index); valid_structures.append(structure)
    for start in range(0, len(valid_structures), 32):
        structures, indices = valid_structures[start:start + 32], valid_indices[start:start + 32]
        try:
            predictions = model.predict_structure(structures, task="efsm", batch_size=32)
            if isinstance(predictions, dict): predictions = [predictions]
        except BaseException:
            predictions = []
            for structure in structures:
                try: predictions.append(model.predict_structure(structure, task="efsm"))
                except BaseException as error: predictions.append(error)
        for record_index, structure, prediction in zip(indices, structures, predictions):
            record = records[record_index]
            if isinstance(prediction, BaseException): record["validity_reason"] = f"chgnet:{type(prediction).__name__}"; continue
            value = float(np.abs(np.asarray(prediction["m"], dtype=float)).sum() / structure.volume)
            if not np.isfinite(value): record["validity_reason"] = "chgnet_nonfinite"; continue
            record.update(chgnet_success=True, chgnet_mag_density=value, property_absolute_error=abs(value - TARGET), structure_valid=True, validity_reason="")
    write_csv(phase_root / "property_metrics_raw.csv", records)
    evaluation = phase_root / "evaluation_structures"; evaluation.mkdir(exist_ok=False); lookup = {row["branch_id"]: row for row in records}; exclusions = []
    for group in GROUPS:
        selected = []
        for atoms in atoms_list:
            method, rank = str(atoms.info["method"]), int(atoms.info["allocation_rank"])
            if quality_group(method, rank) != group: continue
            branch = str(atoms.info["branch_id"])
            if lookup[branch]["structure_valid"]: atoms.info.update(sample_seed=int(atoms.info["seed"]), branch_id=branch); selected.append(atoms)
            else: exclusions.append({"branch_id": branch, "quality_group": group, "reason": lookup[branch]["validity_reason"]})
        if selected: write(evaluation / f"{group}.extxyz", selected)
    if exclusions: write_csv(phase_root / "quality_exclusions.csv", exclusions)
    (phase_root / "property_summary.json").write_text(json.dumps({"total": len(records), "valid": sum(row["structure_valid"] for row in records), "invalid": sum(not row["structure_valid"] for row in records), "device": device, "chgnet_version": "0.3.0", "surrogate_only": True, "dft_verified": False}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("c1_128", "c2_128"), required=True); args = parser.parse_args(); main(args.phase)
