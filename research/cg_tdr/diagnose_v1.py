#!/usr/bin/env python3
"""Evaluate the frozen V1 checkpoint against zero and Teacher residual baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.cg_tdr.model import CGTDRConfig, CGTDRRefiner


DEFAULT_LABEL_ROOT = Path("/data/dxl/data/cg_tdr_teacher/labels")
DEFAULT_CHECKPOINT = Path(
    "/data/dxl/results/cg_tdr/phase0/training/checkpoints/best.pt"
)
DEFAULT_RESULT_ROOT = Path("/data/dxl/results/cg_tdr/phase0")
DEFAULT_REPORT_ROOT = Path("/data/dxl/reports/cg_tdr/phase0")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def finite_correlation(
    left: list[float], right: list[float], kind: str
) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.ptp(x) <= 1.0e-15 or np.ptp(y) <= 1.0e-15:
        return None
    value = pearsonr(x, y).statistic if kind == "pearson" else spearmanr(x, y).statistic
    return float(value) if math.isfinite(float(value)) else None


def cosine_and_ratio(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[float, float, float, float]:
    prediction = prediction.detach().double().flatten()
    target = target.detach().double().flatten()
    prediction_norm = float(torch.linalg.vector_norm(prediction))
    target_norm = float(torch.linalg.vector_norm(target))
    dot = float(torch.dot(prediction, target))
    cosine = dot / max(prediction_norm * target_norm, 1.0e-18)
    ratio = prediction_norm / max(target_norm, 1.0e-18)
    return cosine, ratio, dot, prediction_norm


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "fraction_lt_0_1": float(np.mean(array < 0.1)),
        "fraction_lt_0_5": float(np.mean(array < 0.5)),
        "fraction_gt_0_9": float(np.mean(array > 0.9)),
    }


def direction_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    rows = [row for row in rows if row[f"{prefix}_teacher_rms"] > 1.0e-12]
    cosine = np.asarray([row[f"{prefix}_cosine"] for row in rows], dtype=float)
    ratio = np.asarray([row[f"{prefix}_magnitude_ratio"] for row in rows], dtype=float)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "positive_cosine_rate": float(np.mean(cosine > 0.0)),
        "cosine_gt_0_5_rate": float(np.mean(cosine > 0.5)),
        "dot_positive_rate": float(
            np.mean([row[f"{prefix}_dot_product"] > 0.0 for row in rows])
        ),
        "magnitude_ratio_mean": float(ratio.mean()),
        "magnitude_ratio_median": float(np.median(ratio)),
        "over_prediction_rate": float(np.mean(ratio > 1.0)),
        "under_prediction_rate": float(np.mean(ratio < 1.0)),
        "predicted_rms_mean": float(
            np.mean([row[f"{prefix}_predicted_rms"] for row in rows])
        ),
        "teacher_rms_mean": float(
            np.mean([row[f"{prefix}_teacher_rms"] for row in rows])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 100:
        raise ValueError(
            f"V1 diagnostics require step 100, got checkpoint step={payload.get('step')}"
        )
    if int(payload.get("training_seed", -1)) != 3100:
        raise ValueError("Unexpected V1 training seed")
    model = CGTDRRefiner(CGTDRConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    rows: list[dict[str, Any]] = []
    zero_position_losses: list[float] = []
    model_position_losses: list[float] = []
    zero_cell_losses: list[float] = []
    model_cell_losses: list[float] = []
    with torch.inference_mode():
        for seed in range(30448, 30512):
            sample_path = args.label_root / f"seed_{seed}.pt"
            sample = torch.load(sample_path, map_location="cpu", weights_only=False)
            if sample.get("split") != "test" or int(sample["seed"]) != seed:
                raise ValueError(f"Frozen test split mismatch: {sample_path}")
            output = model(
                node_features=sample["node_features"],
                frac_pos=sample["pos"],
                cell=sample["cell"],
                batch_idx=sample["batch_index"],
                convergence=sample["convergence"],
                enable_cell=True,
            )
            teacher_position = sample["teacher_position_residual_cart"]
            teacher_cell = sample["teacher_strain"]
            zero_position_losses.append(
                float(
                    F.smooth_l1_loss(
                        torch.zeros_like(teacher_position),
                        teacher_position,
                        beta=0.002,
                    )
                )
            )
            model_position_losses.append(
                float(
                    F.smooth_l1_loss(
                        output.position_residual_cart,
                        teacher_position,
                        beta=0.002,
                    )
                )
            )
            zero_cell_losses.append(
                float(
                    F.smooth_l1_loss(
                        torch.zeros_like(teacher_cell), teacher_cell, beta=0.0005
                    )
                )
            )
            model_cell_losses.append(
                float(
                    F.smooth_l1_loss(output.strain, teacher_cell, beta=0.0005)
                )
            )
            position_cosine, position_ratio, position_dot, position_norm = (
                cosine_and_ratio(output.position_residual_cart, teacher_position)
            )
            cell_cosine, cell_ratio, cell_dot, cell_norm = cosine_and_ratio(
                output.strain, teacher_cell
            )
            manifest = sample["manifest_row"]
            rows.append(
                {
                    "seed": seed,
                    "num_atoms": int(sample["num_atoms"].sum()),
                    "selected_candidate": manifest["selected_candidate"],
                    "objective_improvement": float(manifest["objective_improvement"]),
                    "position_predicted_rms": float(
                        output.position_residual_cart.square().mean().sqrt()
                    ),
                    "position_teacher_rms": float(
                        teacher_position.square().mean().sqrt()
                    ),
                    "position_magnitude_ratio": position_ratio,
                    "position_cosine": position_cosine,
                    "position_dot_product": position_dot,
                    "position_predicted_norm": position_norm,
                    "position_gate": float(output.position_gate.mean()),
                    "position_loss_zero": zero_position_losses[-1],
                    "position_loss_model": model_position_losses[-1],
                    "cell_predicted_rms": float(output.strain.square().mean().sqrt()),
                    "cell_teacher_rms": float(teacher_cell.square().mean().sqrt()),
                    "cell_magnitude_ratio": cell_ratio,
                    "cell_cosine": cell_cosine,
                    "cell_dot_product": cell_dot,
                    "cell_predicted_norm": cell_norm,
                    "cell_gate": float(output.cell_gate.mean()),
                    "cell_loss_zero": zero_cell_losses[-1],
                    "cell_loss_model": model_cell_losses[-1],
                    "position_clipped": bool(output.position_clipped.any()),
                    "cell_fallback": bool(output.cell_fallback.any()),
                }
            )

    zero_position = float(np.mean(zero_position_losses))
    model_position = float(np.mean(model_position_losses))
    zero_cell = float(np.mean(zero_cell_losses))
    model_cell = float(np.mean(model_cell_losses))
    position_gates = [row["position_gate"] for row in rows]
    cell_gates = [row["cell_gate"] for row in rows]
    utilities = [row["objective_improvement"] for row in rows]
    position_magnitudes = [row["position_teacher_rms"] for row in rows]
    cell_magnitudes = [row["cell_teacher_rms"] for row in rows]
    position_gate_distribution = distribution(position_gates)
    cell_gate_distribution = distribution(cell_gates)
    correlations = {
        "position_gate_utility_pearson": finite_correlation(
            position_gates, utilities, "pearson"
        ),
        "position_gate_utility_spearman": finite_correlation(
            position_gates, utilities, "spearman"
        ),
        "position_gate_magnitude_pearson": finite_correlation(
            position_gates, position_magnitudes, "pearson"
        ),
        "position_gate_magnitude_spearman": finite_correlation(
            position_gates, position_magnitudes, "spearman"
        ),
        "cell_gate_utility_pearson": finite_correlation(
            cell_gates, utilities, "pearson"
        ),
        "cell_gate_utility_spearman": finite_correlation(
            cell_gates, utilities, "spearman"
        ),
        "cell_gate_magnitude_pearson": finite_correlation(
            cell_gates, cell_magnitudes, "pearson"
        ),
        "cell_gate_magnitude_spearman": finite_correlation(
            cell_gates, cell_magnitudes, "spearman"
        ),
    }
    positive_correlations = [
        correlations["position_gate_utility_spearman"],
        correlations["position_gate_magnitude_spearman"],
        correlations["cell_gate_utility_spearman"],
        correlations["cell_gate_magnitude_spearman"],
    ]
    gate_selectivity_valid = bool(
        position_gate_distribution["std"] >= 0.05
        and cell_gate_distribution["std"] >= 0.05
        and position_gate_distribution["fraction_gt_0_9"] <= 0.80
        and cell_gate_distribution["fraction_gt_0_9"] <= 0.80
        and all(value is not None and value > 0.0 for value in positive_correlations)
    )
    summary = {
        "status": "success",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "training_seed": int(payload["training_seed"]),
        "test_seed_start": 30448,
        "test_seed_end": 30511,
        "test_structures": len(rows),
        "identity_baseline": {
            "zero_position_loss": zero_position,
            "model_position_loss": model_position,
            "teacher_position_loss": 0.0,
            "position_loss_improvement": 1.0
            - model_position / max(zero_position, 1.0e-18),
            "zero_cell_loss": zero_cell,
            "model_cell_loss": model_cell,
            "teacher_cell_loss": 0.0,
            "cell_loss_improvement": 1.0 - model_cell / max(zero_cell, 1.0e-18),
            "zero_weighted_residual_loss": 20.0 * (zero_position + zero_cell),
            "model_weighted_residual_loss": 20.0 * (model_position + model_cell),
            "teacher_weighted_residual_loss": 0.0,
        },
        "position": direction_summary(rows, "position"),
        "cell": direction_summary(rows, "cell"),
        "position_gate": position_gate_distribution,
        "cell_gate": cell_gate_distribution,
        "gate_correlations": correlations,
        "gate_selectivity_criteria": {
            "each_gate_std_at_least": 0.05,
            "each_fraction_gt_0_9_at_most": 0.80,
            "all_utility_and_magnitude_spearman_positive": True,
        },
        "GATE_SELECTIVITY_VALID": gate_selectivity_valid,
    }
    result_csv = args.result_root / "residual_learning_per_structure.csv"
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    with result_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(args.result_root / "residual_learning_summary.json", summary)

    identity = summary["identity_baseline"]
    identity_report = f"""# CG-TDR V1 zero-output identity baseline

- Frozen split: seeds 30448--30511 ({len(rows)} structures)
- Checkpoint: `{args.checkpoint}` (strictly verified step 100)
- Loss definition: the same Smooth-L1 residual losses used in V1 training

| Field | Zero output | V1 model | Teacher oracle | V1 improvement vs zero |
|---|---:|---:|---:|---:|
| Position | {zero_position:.10g} | {model_position:.10g} | 0 | {identity['position_loss_improvement']:.2%} |
| Cell | {zero_cell:.10g} | {model_cell:.10g} | 0 | {identity['cell_loss_improvement']:.2%} |

The Teacher oracle row is the target residual evaluated against itself. This report does not use MatterSim and does not modify labels or checkpoints.
"""
    identity_path = args.report_root / "identity_baseline.md"
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(identity_report, encoding="utf-8")

    position = summary["position"]
    cell = summary["cell"]
    diagnostic_report = f"""# CG-TDR V1 residual-learning diagnostics

Checkpoint `best.pt` was strictly loaded at step 100. Diagnostics use the frozen 64-structure test split only.

## Residual direction and magnitude

| Field | Cosine mean | Cosine median | Positive cosine | Cosine > 0.5 | Magnitude ratio median |
|---|---:|---:|---:|---:|---:|
| Position | {position['cosine_mean']:.6f} | {position['cosine_median']:.6f} | {position['positive_cosine_rate']:.2%} | {position['cosine_gt_0_5_rate']:.2%} | {position['magnitude_ratio_median']:.6f} |
| Cell | {cell['cosine_mean']:.6f} | {cell['cosine_median']:.6f} | {cell['positive_cosine_rate']:.2%} | {cell['cosine_gt_0_5_rate']:.2%} | {cell['magnitude_ratio_median']:.6f} |

## Gate selectivity

| Gate | Mean | Median | Std | <0.1 | <0.5 | >0.9 |
|---|---:|---:|---:|---:|---:|---:|
| Position | {position_gate_distribution['mean']:.6f} | {position_gate_distribution['median']:.6f} | {position_gate_distribution['std']:.6f} | {position_gate_distribution['fraction_lt_0_1']:.2%} | {position_gate_distribution['fraction_lt_0_5']:.2%} | {position_gate_distribution['fraction_gt_0_9']:.2%} |
| Cell | {cell_gate_distribution['mean']:.6f} | {cell_gate_distribution['median']:.6f} | {cell_gate_distribution['std']:.6f} | {cell_gate_distribution['fraction_lt_0_1']:.2%} | {cell_gate_distribution['fraction_lt_0_5']:.2%} | {cell_gate_distribution['fraction_gt_0_9']:.2%} |

`GATE_SELECTIVITY_VALID={gate_selectivity_valid}`. The frozen diagnostic criterion requires both gate standard deviations >= 0.05, no more than 80% of either gate above 0.9, and positive Spearman correlation with both Teacher utility and residual magnitude.

Per-structure values are stored in `{result_csv}`.
"""
    diagnostic_path = args.report_root / "residual_learning_diagnostics.md"
    diagnostic_path.write_text(diagnostic_report, encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
