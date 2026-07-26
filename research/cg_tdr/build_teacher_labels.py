#!/usr/bin/env python3
"""Construct frozen CHGNet 0.3.0 labels for CG-TDR feature records.

MatterSim is deliberately absent from this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import torch
from ase.geometry import find_mic
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor


@dataclass(frozen=True)
class TeacherConfig:
    position_eta: float = 0.01
    position_radius_angstrom: float = 0.02
    cell_eta_per_gpa: float = 0.00025
    cell_strain_radius: float = 0.003
    minimum_distance_angstrom: float = 0.5
    maximum_cell_condition: float = 100.0
    force_loss_weight: float = 0.01
    stress_loss_weight: float = 1.0e-5
    energy_tolerance: float = 1.0e-4
    force_relative_tolerance: float = 0.05
    stress_relative_tolerance: float = 0.05


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def as_predictions(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else [value]


def predict(model: CHGNet, atoms: list[Any]) -> list[dict[str, Any]]:
    structures = [AseAtomsAdaptor.get_structure(item) for item in atoms]
    return as_predictions(
        model.predict_structure(
            structures,
            task="efs",
            batch_size=max(1, len(structures)),
        )
    )


def metrics(prediction: dict[str, Any], config: TeacherConfig) -> dict[str, float]:
    energy = float(np.asarray(prediction["e"]).reshape(-1)[0])
    force = np.asarray(prediction["f"], dtype=float)
    stress = np.asarray(prediction["s"], dtype=float)
    force_rms = float(np.sqrt(np.mean(force**2)))
    max_force = float(np.linalg.norm(force, axis=-1).max())
    stress_rms = float(np.sqrt(np.mean(stress**2)))
    objective = (
        energy
        + config.force_loss_weight * float(np.mean(force**2))
        + config.stress_loss_weight * float(np.mean(stress**2))
    )
    return {
        "energy_per_atom_ev": energy,
        "force_rms_ev_ang": force_rms,
        "max_force_ev_ang": max_force,
        "stress_rms_gpa": stress_rms,
        "objective": objective,
    }


def minimum_distance(atoms: Any) -> float:
    if len(atoms) < 2:
        return math.inf
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def safe(atoms: Any, config: TeacherConfig) -> bool:
    cell = np.asarray(atoms.cell.array, dtype=float)
    volume = float(atoms.get_volume())
    condition = float(np.linalg.cond(cell))
    return (
        np.isfinite(atoms.positions).all()
        and np.isfinite(cell).all()
        and math.isfinite(volume)
        and volume > 0.1
        and math.isfinite(condition)
        and condition <= config.maximum_cell_condition
        and minimum_distance(atoms) >= config.minimum_distance_angstrom
    )


def proposal(
    atoms: Any,
    prediction: dict[str, Any],
    config: TeacherConfig,
    *,
    position_scale: float,
    use_cell: bool,
) -> Any:
    candidate = atoms.copy()
    forces = np.asarray(prediction["f"], dtype=float)
    displacement = config.position_eta * position_scale * forces
    norms = np.linalg.norm(displacement, axis=-1)
    cap = config.position_radius_angstrom * position_scale
    displacement *= np.minimum(1.0, cap / np.maximum(norms, 1.0e-12))[:, None]
    if use_cell:
        stress = np.asarray(prediction["s"], dtype=float)
        stress = 0.5 * (stress + stress.T)
        strain = -config.cell_eta_per_gpa * stress
        strain_norm = float(np.linalg.norm(strain))
        if strain_norm > config.cell_strain_radius:
            strain *= config.cell_strain_radius / strain_norm
        candidate.set_cell(
            candidate.cell.array @ (np.eye(3) + strain).T,
            scale_atoms=True,
        )
    candidate.positions[:] = candidate.positions + displacement
    candidate.wrap()
    return candidate


def candidate_is_improved(
    baseline: dict[str, float],
    candidate: dict[str, float],
    config: TeacherConfig,
) -> bool:
    objective_improved = candidate["objective"] < baseline["objective"] - 1.0e-8
    one_physical_metric_improved = any(
        candidate[key] < baseline[key] - 1.0e-8
        for key in ("energy_per_atom_ev", "max_force_ev_ang", "stress_rms_gpa")
    )
    energy_safe = (
        candidate["energy_per_atom_ev"]
        <= baseline["energy_per_atom_ev"] + config.energy_tolerance
    )
    force_safe = candidate["max_force_ev_ang"] <= baseline["max_force_ev_ang"] * (
        1.0 + config.force_relative_tolerance
    ) + 1.0e-8
    stress_safe = candidate["stress_rms_gpa"] <= baseline["stress_rms_gpa"] * (
        1.0 + config.stress_relative_tolerance
    ) + 1.0e-8
    return objective_improved and one_physical_metric_improved and energy_safe and force_safe and stress_safe


def split_for_seed(seed: int) -> str:
    if 30000 <= seed <= 30383:
        return "train"
    if 30384 <= seed <= 30447:
        return "validation"
    if 30448 <= seed <= 30511:
        return "test"
    raise ValueError(f"Seed outside frozen split: {seed}")


def build_one(
    *,
    seed: int,
    feature_path: Path,
    structure_path: Path,
    output_path: Path,
    model: CHGNet,
    config: TeacherConfig,
) -> dict[str, Any]:
    if output_path.exists():
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        if int(payload["seed"]) == seed and payload.get("label_complete"):
            return dict(payload["manifest_row"])
        raise RuntimeError(f"Refusing to overwrite invalid label: {output_path}")
    feature = torch.load(feature_path, map_location="cpu", weights_only=False)
    atoms = ase.io.read(structure_path)
    if not safe(atoms, config):
        raise ValueError(f"Unsafe A0 structure: seed={seed}")
    baseline_prediction = predict(model, [atoms])[0]
    baseline_metrics = metrics(baseline_prediction, config)
    variants = [
        ("position_small", proposal(atoms, baseline_prediction, config, position_scale=1.0, use_cell=False)),
        ("position_half", proposal(atoms, baseline_prediction, config, position_scale=0.5, use_cell=False)),
        ("position_weak_cell", proposal(atoms, baseline_prediction, config, position_scale=1.0, use_cell=True)),
    ]
    safe_variants = [(name, candidate) for name, candidate in variants if safe(candidate, config)]
    predictions = predict(model, [candidate for _, candidate in safe_variants])
    evaluated: list[tuple[str, Any, dict[str, float]]] = []
    for (name, candidate), prediction in zip(safe_variants, predictions, strict=True):
        candidate_metrics = metrics(prediction, config)
        if candidate_is_improved(baseline_metrics, candidate_metrics, config):
            evaluated.append((name, candidate, candidate_metrics))

    if evaluated:
        selected_name, selected, selected_metrics = min(
            evaluated, key=lambda item: item[2]["objective"]
        )
        confidence_label = 1.0
        displacement, _ = find_mic(
            np.asarray(selected.positions) - np.asarray(atoms.positions),
            cell=np.asarray(atoms.cell.array),
            pbc=True,
        )
        cell_delta = np.asarray(selected.cell.array) @ np.linalg.inv(
            np.asarray(atoms.cell.array)
        ) - np.eye(3)
        teacher_strain = 0.5 * (cell_delta + cell_delta.T)
    else:
        selected_name = "identity"
        selected_metrics = baseline_metrics
        confidence_label = 0.0
        displacement = np.zeros_like(np.asarray(atoms.positions), dtype=float)
        teacher_strain = np.zeros((3, 3), dtype=float)

    improvement = baseline_metrics["objective"] - selected_metrics["objective"]
    row = {
        "seed": seed,
        "split": split_for_seed(seed),
        "num_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "selected_candidate": selected_name,
        "confidence_label": confidence_label,
        "objective_improvement": improvement,
        "baseline_energy_per_atom_ev": baseline_metrics["energy_per_atom_ev"],
        "teacher_energy_per_atom_ev": selected_metrics["energy_per_atom_ev"],
        "baseline_max_force_ev_ang": baseline_metrics["max_force_ev_ang"],
        "teacher_max_force_ev_ang": selected_metrics["max_force_ev_ang"],
        "baseline_stress_rms_gpa": baseline_metrics["stress_rms_gpa"],
        "teacher_stress_rms_gpa": selected_metrics["stress_rms_gpa"],
        "teacher_position_rms_angstrom": float(np.sqrt(np.mean(displacement**2))),
        "teacher_strain_frobenius": float(np.linalg.norm(teacher_strain)),
    }
    payload = {
        **feature,
        "label_complete": True,
        "split": row["split"],
        "teacher_position_residual_cart": torch.as_tensor(displacement, dtype=torch.float32),
        "teacher_strain": torch.as_tensor(teacher_strain[None], dtype=torch.float32),
        "confidence_label": torch.tensor([[confidence_label]], dtype=torch.float32),
        "cell_confidence_label": torch.tensor([[float(selected_name == "position_weak_cell")]], dtype=torch.float32),
        "chgnet_initial": {
            "energy_per_atom_ev": baseline_metrics["energy_per_atom_ev"],
            "forces_ev_ang": torch.as_tensor(baseline_prediction["f"], dtype=torch.float32),
            "stress_gpa": torch.as_tensor(baseline_prediction["s"], dtype=torch.float32),
        },
        "teacher_metrics": selected_metrics,
        "teacher_config": asdict(config),
        "manifest_row": row,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=30000)
    parser.add_argument("--seed-end", type=int, default=30511)
    parser.add_argument("--feature-root", default="/data/dxl/data/cg_tdr_teacher/features")
    parser.add_argument(
        "--generation-root", default="/data/dxl/results/cg_tdr/phase0/teacher_generation"
    )
    parser.add_argument("--label-root", default="/data/dxl/data/cg_tdr_teacher/labels")
    parser.add_argument(
        "--report-root", default="/data/dxl/reports/cg_tdr/phase0/teacher_data"
    )
    args = parser.parse_args()
    feature_root = Path(args.feature_root)
    generation_root = Path(args.generation_root)
    label_root = Path(args.label_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    model = CHGNet.load(model_name="0.3.0", verbose=False, use_device="cuda")
    model.eval()
    config = TeacherConfig()
    rows = []
    for seed in range(args.seed_start, args.seed_end + 1):
        feature_path = feature_root / f"seed_{seed}.pt"
        structure_path = generation_root / f"seed_{seed}" / "generated_crystals.extxyz"
        if not feature_path.exists() or not structure_path.exists():
            raise FileNotFoundError(f"Missing feature or structure for seed {seed}")
        rows.append(
            build_one(
                seed=seed,
                feature_path=feature_path,
                structure_path=structure_path,
                output_path=label_root / f"seed_{seed}.pt",
                model=model,
                config=config,
            )
        )
    manifest_path = report_root / "teacher_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    positive = sum(row["confidence_label"] > 0 for row in rows)
    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    }
    summary = {
        "status": "success",
        "teacher": "CHGNet model 0.3.0",
        "structures": len(rows),
        "split_counts": split_counts,
        "positive_confidence_count": positive,
        "positive_confidence_rate": positive / len(rows),
        "atomic_numbers_modified": False,
        "MatterSim_used": False,
        "config": asdict(config),
        "manifest": str(manifest_path),
    }
    atomic_json(report_root / "teacher_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
