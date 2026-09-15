"""Aggregate and evaluate primary-property outputs for P0 or Formal256."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from ase.io import read, write
import numpy as np
import torch
import torch_geometric  # noqa: F401
import torch_scatter  # noqa: F401

sys.path.append("/mnt/datasets-livsyn/dxl/mattergen_v1/.venv/lib/python3.10/site-packages")
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor

from prepare_dataset import CHGNET_PATH, INVALID_OBJECTIVE, TARGET, geometric_validity


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/calibrated_safe_selection_cfg"
POLICIES = ("C0", "T0", "S0")


def write_csv_once(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(cohort: str) -> None:
    cohort_root = ROOT / cohort
    seeds = list(map(int, json.loads((cohort_root / "seeds.json").read_text())["seeds"]))
    all_atoms = []
    traces = []
    summaries = []
    for seed in seeds:
        source = cohort_root / "generation" / str(seed)
        summary = json.loads((source / "run_summary.json").read_text())
        if not summary["success"]:
            raise RuntimeError(f"failed generation: {seed}")
        summaries.append(summary)
        atoms = read(source / "policy_final.extxyz", index=":")
        if len(atoms) != 3 or {item.info["policy"] for item in atoms} != set(POLICIES):
            raise RuntimeError(f"policy branch mismatch: {seed}")
        all_atoms.extend(atoms)
        with (source / "controller_trace.csv").open(newline="", encoding="utf-8") as stream:
            traces.extend(csv.DictReader(stream))
    if len(all_atoms) != len(seeds) * 3 or len(traces) != len(seeds):
        raise RuntimeError("fresh aggregation count mismatch")
    write(cohort_root / "policy_final_all.extxyz", all_atoms)
    write_csv_once(cohort_root / "controller_trace.csv", traces)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CHGNet.from_file(CHGNET_PATH).to(device)
    model.eval()
    records = []
    valid_indices = []
    valid_structures = []
    for index, atoms in enumerate(all_atoms):
        valid, minimum, reason = geometric_validity(atoms)
        record = {
            "branch_id": f"{int(atoms.info['seed'])}:{atoms.info['policy']}",
            "seed": int(atoms.info["seed"]), "policy": str(atoms.info["policy"]),
            "selected_action": str(atoms.info["selected_action"]),
            "selected_cfg": float(atoms.info["selected_cfg"]),
            "formula": atoms.get_chemical_formula(), "num_atoms": len(atoms),
            "geometric_valid": valid, "minimum_periodic_distance_angstrom": minimum,
            "validity_reason": reason, "chgnet_success": False,
            "chgnet_mag_density": float("nan"), "target_mag_density": TARGET,
            "property_absolute_error": INVALID_OBJECTIVE, "structure_valid": False,
            "surrogate_property_eval": True, "dft_verified": False,
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
                chgnet_success=True, chgnet_mag_density=value,
                property_absolute_error=abs(value - TARGET), structure_valid=True,
                validity_reason="",
            )
    write_csv_once(cohort_root / "property_metrics.csv", records)
    evaluation = cohort_root / "evaluation_structures"
    evaluation.mkdir(exist_ok=False)
    lookup = {row["branch_id"]: row for row in records}
    for policy in POLICIES:
        selected = []
        for atoms in all_atoms:
            if atoms.info["policy"] != policy:
                continue
            identifier = f"{int(atoms.info['seed'])}:{policy}"
            if lookup[identifier]["structure_valid"]:
                atoms.info.update(sample_seed=int(atoms.info["seed"]), branch_id=identifier)
                selected.append(atoms)
        if selected:
            write(evaluation / f"{policy}.extxyz", selected)
    (cohort_root / "generation_summary.json").write_text(
        json.dumps(
            {
                "cohort": cohort, "trajectories": len(seeds), "policy_branches": len(all_atoms),
                "valid_property_branches": sum(row["structure_valid"] for row in records),
                "total_elapsed_gpu_hours": sum(row["elapsed_seconds"] for row in summaries) / 3600,
                "device": device,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("p0", "formal256"), required=True)
    args = parser.parse_args()
    main(args.cohort)
