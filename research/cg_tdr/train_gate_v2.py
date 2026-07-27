#!/usr/bin/env python3
"""Train the single allowed utility-calibrated CG-TDR Gate V2 repair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.cg_tdr.model import CGTDRConfig, CGTDRRefiner


LABEL_ROOT = Path("/data/dxl/data/cg_tdr_teacher/labels_v2")
OUTPUT_ROOT = Path("/data/dxl/results/cg_tdr/phase0/training_v2")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_split(start: int, end: int) -> list[dict[str, Any]]:
    result = []
    for seed in range(start, end + 1):
        path = LABEL_ROOT / f"seed_{seed}.pt"
        sample = torch.load(path, map_location="cpu", weights_only=False)
        if int(sample["seed"]) != seed or int(sample["schema_version"]) != 2:
            raise ValueError(f"Invalid Gate V2 label: {path}")
        if sample.get("MatterSim_used_for_gate_v2") is not False:
            raise ValueError(f"MatterSim contamination flag is not false: {path}")
        result.append(sample)
    return result


def tensors(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "node_features",
        "pos",
        "cell",
        "batch_index",
        "convergence",
        "teacher_position_residual_cart",
        "teacher_strain",
        "confidence_label",
        "cell_confidence_label",
        "gate_v2_utility",
    )
    return {name: sample[name].to(device) for name in names}


def cosine_loss(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    prediction = prediction.flatten()
    target = target.flatten()
    if float(weight.max()) <= 0.1 or float(torch.linalg.vector_norm(target)) <= 1.0e-12:
        return prediction.new_zeros(())
    return weight.mean() * (
        1.0
        - F.cosine_similarity(prediction[None], target[None], dim=-1).mean()
    )


def losses(
    model: CGTDRRefiner,
    sample: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = tensors(sample, device)
    output = model(
        node_features=batch["node_features"],
        frac_pos=batch["pos"],
        cell=batch["cell"],
        batch_idx=batch["batch_index"],
        convergence=batch["convergence"],
        enable_cell=True,
    )
    position_target = (
        batch["teacher_position_residual_cart"]
        * batch["confidence_label"][batch["batch_index"]]
    )
    cell_target = batch["teacher_strain"] * batch["cell_confidence_label"][:, None]
    position = F.smooth_l1_loss(
        output.position_residual_cart, position_target, beta=0.002
    )
    cell = F.smooth_l1_loss(output.strain, cell_target, beta=0.0005)
    position_calibration = F.binary_cross_entropy(
        output.position_gate, batch["confidence_label"]
    )
    cell_calibration = F.binary_cross_entropy(
        output.cell_gate, batch["cell_confidence_label"]
    )
    identity = (
        (1.0 - batch["confidence_label"].mean())
        * output.position_residual_cart.square().mean()
        + (1.0 - batch["cell_confidence_label"].mean())
        * output.strain.square().mean()
    )
    direction = cosine_loss(
        output.position_residual_cart,
        batch["teacher_position_residual_cart"],
        batch["confidence_label"],
    ) + cosine_loss(
        output.strain,
        batch["teacher_strain"],
        batch["cell_confidence_label"],
    )
    total = (
        100.0 * position
        + 100.0 * cell
        + 0.5 * position_calibration
        + 0.3 * cell_calibration
        + 10.0 * identity
        + 0.05 * direction
    )
    values = {
        "total": float(total.detach()),
        "position": float(position.detach()),
        "cell": float(cell.detach()),
        "position_calibration": float(position_calibration.detach()),
        "cell_calibration": float(cell_calibration.detach()),
        "identity": float(identity.detach()),
        "direction": float(direction.detach()),
        "position_gate": float(output.position_gate.detach().mean()),
        "cell_gate": float(output.cell_gate.detach().mean()),
        "position_residual_rms": float(
            output.position_residual_cart.detach().square().mean().sqrt()
        ),
        "cell_residual_rms": float(output.strain.detach().square().mean().sqrt()),
    }
    return total, values


@torch.inference_mode()
def evaluate(
    model: CGTDRRefiner,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    rows = []
    for sample in samples:
        batch = tensors(sample, device)
        output = model(
            node_features=batch["node_features"],
            frac_pos=batch["pos"],
            cell=batch["cell"],
            batch_idx=batch["batch_index"],
            convergence=batch["convergence"],
            enable_cell=True,
        )
        _, values = losses(model, sample, device)
        target_position = batch["teacher_position_residual_cart"]
        target_cell = batch["teacher_strain"]
        position_norm = float(torch.linalg.vector_norm(target_position))
        cell_norm = float(torch.linalg.vector_norm(target_cell))
        values.update(
            {
                "utility": float(batch["gate_v2_utility"].mean()),
                "position_target": float(batch["confidence_label"].mean()),
                "cell_target": float(batch["cell_confidence_label"].mean()),
                "position_cosine": (
                    float(
                        F.cosine_similarity(
                            output.position_residual_cart.flatten()[None],
                            target_position.flatten()[None],
                        ).mean()
                    )
                    if position_norm > 1.0e-12
                    else 0.0
                ),
                "cell_cosine": (
                    float(
                        F.cosine_similarity(
                            output.strain.flatten()[None],
                            target_cell.flatten()[None],
                        ).mean()
                    )
                    if cell_norm > 1.0e-12
                    else 0.0
                ),
            }
        )
        rows.append(values)
    model.train()
    position_gates = np.asarray([row["position_gate"] for row in rows])
    cell_gates = np.asarray([row["cell_gate"] for row in rows])
    utilities = np.asarray([row["utility"] for row in rows])
    low_position = [row for row in rows if row["position_target"] <= 0.1]
    low_cell = [row for row in rows if row["cell_target"] <= 0.1]
    high_position = [row for row in rows if row["position_target"] > 0.1]
    high_cell = [row for row in rows if row["cell_target"] > 0.1]
    return {
        "total": float(np.mean([row["total"] for row in rows])),
        "position": float(np.mean([row["position"] for row in rows])),
        "cell": float(np.mean([row["cell"] for row in rows])),
        "position_gate_mean": float(position_gates.mean()),
        "position_gate_std": float(position_gates.std(ddof=0)),
        "position_gate_gt_0_9_rate": float(np.mean(position_gates > 0.9)),
        "cell_gate_mean": float(cell_gates.mean()),
        "cell_gate_std": float(cell_gates.std(ddof=0)),
        "cell_gate_gt_0_9_rate": float(np.mean(cell_gates > 0.9)),
        "position_gate_utility_spearman": float(
            spearmanr(position_gates, utilities).statistic
        ),
        "cell_gate_utility_spearman": float(
            spearmanr(cell_gates, utilities).statistic
        ),
        "position_identity_output_rate": (
            float(
                np.mean(
                    [row["position_residual_rms"] <= 5.0e-5 for row in low_position]
                )
            )
            if low_position
            else 1.0
        ),
        "cell_identity_output_rate": (
            float(
                np.mean([row["cell_residual_rms"] <= 5.0e-5 for row in low_cell])
            )
            if low_cell
            else 1.0
        ),
        "position_cosine_mean_nonlow": (
            float(np.mean([row["position_cosine"] for row in high_position]))
            if high_position
            else 0.0
        ),
        "cell_cosine_mean_nonlow": (
            float(np.mean([row["cell_cosine"] for row in high_cell]))
            if high_cell
            else 0.0
        ),
        "structures": len(rows),
    }


def save_checkpoint(
    path: Path,
    model: CGTDRRefiner,
    optimizer: torch.optim.Optimizer,
    step: int,
    validation: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 2,
        "gate_version": "utility_calibrated_v2",
        "config": model.config.as_dict(),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "validation": validation,
        "training_seed": 3101,
        "initialization": "identity_zero_initialization",
        "MatterGen_frozen": True,
        "CHGNet_in_checkpoint": False,
        "MatterSim_used_for_training": False,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--training-seed", type=int, default=3101)
    args = parser.parse_args()
    if args.training_seed != 3101 or not 1000 <= args.max_steps <= 2000:
        raise ValueError("Gate V2 requires seed 3101 and 1000--2000 steps")
    seed_everything(args.training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split(30000, 30383)
    validation = load_split(30384, 30447)
    test = load_split(30448, 30511)
    convergence = torch.cat([sample["convergence"].float() for sample in train])
    convergence_mean = convergence.mean(dim=0)
    convergence_std = convergence.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    model = CGTDRRefiner(CGTDRConfig()).to(device)
    model.set_convergence_normalization(convergence_mean, convergence_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-6)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoints = OUTPUT_ROOT / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    random_generator = random.Random(args.training_seed)
    curve: list[dict[str, Any]] = []
    best = math.inf
    best_step = 0
    stale = 0
    for step in range(1, args.max_steps + 1):
        sample = train[random_generator.randrange(len(train))]
        optimizer.zero_grad(set_to_none=True)
        loss, training = losses(model, sample, device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite Gate V2 loss at step {step}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step % 100 == 0:
            validation_metrics = evaluate(model, validation, device)
            row = {
                "step": step,
                "train_total": training["total"],
                "validation_total": validation_metrics["total"],
                "position_gate_mean": validation_metrics["position_gate_mean"],
                "position_gate_std": validation_metrics["position_gate_std"],
                "cell_gate_mean": validation_metrics["cell_gate_mean"],
                "cell_gate_std": validation_metrics["cell_gate_std"],
                "position_utility_spearman": validation_metrics[
                    "position_gate_utility_spearman"
                ],
                "cell_utility_spearman": validation_metrics[
                    "cell_gate_utility_spearman"
                ],
                "gradient_norm": gradient_norm,
            }
            curve.append(row)
            if validation_metrics["total"] < best - 1.0e-6:
                best = validation_metrics["total"]
                best_step = step
                stale = 0
                save_checkpoint(
                    checkpoints / "best.pt",
                    model,
                    optimizer,
                    step,
                    validation_metrics,
                )
            else:
                stale += 1
            save_checkpoint(
                checkpoints / "last.pt", model, optimizer, step, validation_metrics
            )
            if step >= 1000 and stale >= 5:
                break
    with (OUTPUT_ROOT / "training_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    best_payload = torch.load(
        checkpoints / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_payload["state_dict"], strict=True)
    test_metrics = evaluate(model, test, device)
    gate_valid = bool(
        test_metrics["position_gate_std"] > 0.008384357197599718
        and test_metrics["cell_gate_std"] > 0.01319300510654945
        and test_metrics["position_gate_gt_0_9_rate"] < 1.0
        and test_metrics["cell_gate_gt_0_9_rate"] < 1.0
        and test_metrics["position_gate_utility_spearman"] > 0.0
        and test_metrics["cell_gate_utility_spearman"] > 0.0
        and test_metrics["position_cosine_mean_nonlow"] > -0.042970844702618936
        and test_metrics["cell_cosine_mean_nonlow"] > 0.0
    )
    summary = {
        "status": "success",
        "training_seed": args.training_seed,
        "steps_completed": int(curve[-1]["step"]),
        "best_step": best_step,
        "best_validation_loss": best,
        "initialization": (
            "Identity/zero initialization was selected because V1 residual loss "
            "was worse than zero and its position cosine was negative."
        ),
        "parameter_count": model.parameter_count,
        "config": asdict(model.config),
        "train_structures": len(train),
        "validation_structures": len(validation),
        "test_structures": len(test),
        "test_metrics": test_metrics,
        "best_checkpoint": str(checkpoints / "best.pt"),
        "last_checkpoint": str(checkpoints / "last.pt"),
        "CG_TDR_GATE_V2_VALID": gate_valid,
        "MatterGen_frozen": True,
        "CHGNet_frozen_and_offline_only": True,
        "MatterSim_used_for_training": False,
    }
    atomic_json(OUTPUT_ROOT / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
