#!/usr/bin/env python3
"""Build leakage-free continuous utility targets for the single Gate V2 repair."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr


SOURCE = Path("/data/dxl/data/cg_tdr_teacher/labels")
OUTPUT = Path("/data/dxl/data/cg_tdr_teacher/labels_v2")
RESULT = Path("/data/dxl/results/cg_tdr/phase0/gate_v2_labels.json")
REPORT = Path("/data/dxl/reports/cg_tdr/phase0/gate_v2_labels.md")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def split(seed: int) -> str:
    if seed <= 30383:
        return "train"
    if seed <= 30447:
        return "validation"
    return "test"


def percentile_scale(values: list[float]) -> float:
    positive = np.asarray([max(value, 0.0) for value in values], dtype=float)
    return max(float(np.quantile(positive, 0.9)), 1.0e-12)


def soft_target(utility: float, quantiles: tuple[float, float, float]) -> float:
    q40, q70, q90 = quantiles
    if utility <= q40:
        return 0.0
    if utility <= q70:
        return 0.5 * (utility - q40) / max(q70 - q40, 1.0e-12)
    if utility <= q90:
        return 0.5 + 0.5 * (utility - q70) / max(q90 - q70, 1.0e-12)
    return 1.0


def main() -> int:
    samples = [
        torch.load(SOURCE / f"seed_{seed}.pt", map_location="cpu", weights_only=False)
        for seed in range(30000, 30512)
    ]
    if any(int(sample["seed"]) != seed for seed, sample in zip(range(30000, 30512), samples)):
        raise ValueError("Frozen V1 label seed mapping mismatch")
    rows: list[dict[str, Any]] = []
    for sample in samples:
        manifest = sample["manifest_row"]
        baseline_energy = float(manifest["baseline_energy_per_atom_ev"])
        teacher_energy = float(manifest["teacher_energy_per_atom_ev"])
        baseline_force = float(manifest["baseline_max_force_ev_ang"])
        teacher_force = float(manifest["teacher_max_force_ev_ang"])
        baseline_stress = float(manifest["baseline_stress_rms_gpa"])
        teacher_stress = float(manifest["teacher_stress_rms_gpa"])
        position = sample["teacher_position_residual_cart"].float()
        strain = sample["teacher_strain"].float()
        rows.append(
            {
                "seed": int(sample["seed"]),
                "split": split(int(sample["seed"])),
                "energy_improvement": baseline_energy - teacher_energy,
                "force_relative_improvement": (
                    baseline_force - teacher_force
                )
                / max(abs(baseline_force), 1.0e-6),
                "stress_relative_improvement": (
                    baseline_stress - teacher_stress
                )
                / max(abs(baseline_stress), 1.0e-6),
                "position_rms": float(position.square().mean().sqrt()),
                "position_max": float(torch.linalg.vector_norm(position, dim=-1).max()),
                "strain_norm": float(torch.linalg.matrix_norm(strain, ord="fro").max()),
                "selected_candidate": str(manifest["selected_candidate"]),
            }
        )
    train = [row for row in rows if row["split"] == "train"]
    scales = {
        "energy": percentile_scale([row["energy_improvement"] for row in train]),
        "force": percentile_scale(
            [row["force_relative_improvement"] for row in train]
        ),
        "stress": percentile_scale(
            [row["stress_relative_improvement"] for row in train]
        ),
    }
    for row in rows:
        components = {
            "energy": row["energy_improvement"] / scales["energy"],
            "force": row["force_relative_improvement"] / scales["force"],
            "stress": row["stress_relative_improvement"] / scales["stress"],
        }
        row["utility_energy"] = float(np.clip(components["energy"], -1.0, 2.0))
        row["utility_force"] = float(np.clip(components["force"], -1.0, 2.0))
        row["utility_stress"] = float(np.clip(components["stress"], -1.0, 2.0))
        row["utility"] = (
            0.45 * row["utility_energy"]
            + 0.45 * row["utility_force"]
            + 0.10 * row["utility_stress"]
        )
        row["material_worsening"] = bool(
            row["energy_improvement"] < -1.0e-4
            or row["force_relative_improvement"] < -0.01
            or row["stress_relative_improvement"] < -0.01
        )
        row["position_eligible"] = bool(
            row["position_rms"] > 1.0e-6
            and row["position_max"] < 0.0198
            and not row["material_worsening"]
        )
        row["cell_eligible"] = bool(
            row["strain_norm"] > 1.0e-6
            and row["strain_norm"] < 0.00297
            and not row["material_worsening"]
        )
    eligible_train_utilities = np.asarray(
        [row["utility"] for row in train if row["position_eligible"]], dtype=float
    )
    if len(eligible_train_utilities) < 100:
        raise RuntimeError("Too few eligible train structures for Gate V2 quantiles")
    quantiles = tuple(
        float(value)
        for value in np.quantile(eligible_train_utilities, [0.4, 0.7, 0.9])
    )
    for sample, row in zip(samples, rows, strict=True):
        base_target = soft_target(row["utility"], quantiles)
        row["position_gate_target"] = base_target if row["position_eligible"] else 0.0
        row["cell_gate_target"] = base_target if row["cell_eligible"] else 0.0
        rebuilt = dict(sample)
        rebuilt.update(
            {
                "schema_version": 2,
                "confidence_label_v1": sample["confidence_label"].clone(),
                "cell_confidence_label_v1": sample["cell_confidence_label"].clone(),
                "confidence_label": torch.tensor(
                    [[row["position_gate_target"]]], dtype=torch.float32
                ),
                "cell_confidence_label": torch.tensor(
                    [[row["cell_gate_target"]]], dtype=torch.float32
                ),
                "gate_v2_utility": torch.tensor(
                    [[row["utility"]]], dtype=torch.float32
                ),
                "gate_v2_components": {
                    "energy": row["utility_energy"],
                    "force": row["utility_force"],
                    "stress": row["utility_stress"],
                },
                "gate_v2_train_quantiles": {
                    "q40": quantiles[0],
                    "q70": quantiles[1],
                    "q90": quantiles[2],
                },
                "MatterSim_used_for_gate_v2": False,
            }
        )
        atomic_torch(OUTPUT / f"seed_{row['seed']}.pt", rebuilt)

    split_summary: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        current = [row for row in rows if row["split"] == name]
        position_targets = np.asarray(
            [row["position_gate_target"] for row in current]
        )
        cell_targets = np.asarray([row["cell_gate_target"] for row in current])
        utilities = np.asarray([row["utility"] for row in current])
        split_summary[name] = {
            "n": len(current),
            "position_target_mean": float(position_targets.mean()),
            "position_high_confidence_rate_ge_0_5": float(
                np.mean(position_targets >= 0.5)
            ),
            "position_zero_low_rate_le_0_1": float(
                np.mean(position_targets <= 0.1)
            ),
            "cell_target_mean": float(cell_targets.mean()),
            "cell_high_confidence_rate_ge_0_5": float(np.mean(cell_targets >= 0.5)),
            "cell_zero_low_rate_le_0_1": float(np.mean(cell_targets <= 0.1)),
            "position_target_utility_spearman": float(
                spearmanr(position_targets, utilities).statistic
            ),
            "cell_target_utility_spearman": float(
                spearmanr(cell_targets, utilities).statistic
            ),
        }
    summary = {
        "status": "success",
        "source_labels": str(SOURCE),
        "output_labels": str(OUTPUT),
        "structures": len(rows),
        "training_seed_range": [30000, 30383],
        "weights": {"energy": 0.45, "force": 0.45, "stress": 0.10},
        "normalization": "positive train-split 90th percentile per component",
        "component_scales": scales,
        "train_utility_quantiles": {
            "q40": quantiles[0],
            "q70": quantiles[1],
            "q90": quantiles[2],
        },
        "splits": split_summary,
        "MatterSim_used": False,
        "teacher_structures_regenerated": False,
    }
    atomic_json(RESULT, summary)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(
            [
                "# CG-TDR Gate V2 utility labels",
                "",
                "- Reused all 512 frozen A0/CHGNet records; no structure or Teacher regeneration.",
                "- Utility weights: energy 0.45, maximum force 0.45, stress 0.10.",
                "- Component normalization and all q40/q70/q90 cutoffs use the train split only.",
                "- Targets are forced to zero for identity/noise residuals, material worsening, or trust-radius impacts.",
                "- MatterSim is not imported or used.",
                "",
                "```json",
                json.dumps(summary["splits"], indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
