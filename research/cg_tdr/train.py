#!/usr/bin/env python3
"""Train only the lightweight CG-TDR module on frozen terminal features."""

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

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.cg_tdr.model import CGTDRConfig, CGTDRRefiner


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_split(label_root: Path, start: int, end: int) -> list[dict[str, Any]]:
    samples = []
    for seed in range(start, end + 1):
        path = label_root / f"seed_{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        sample = torch.load(path, map_location="cpu", weights_only=False)
        if not sample.get("label_complete"):
            raise ValueError(f"Incomplete label: {path}")
        if int(sample["seed"]) != seed:
            raise ValueError(f"Seed mismatch: {path}")
        samples.append(sample)
    return samples


def to_device(sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
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
    )
    return {name: sample[name].to(device) for name in names}


def losses(
    model: CGTDRRefiner,
    sample: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = to_device(sample, device)
    output = model(
        node_features=batch["node_features"],
        frac_pos=batch["pos"],
        cell=batch["cell"],
        batch_idx=batch["batch_index"],
        convergence=batch["convergence"],
        enable_cell=True,
    )
    position = F.smooth_l1_loss(
        output.position_residual_cart,
        batch["teacher_position_residual_cart"],
        beta=0.002,
    )
    cell = F.smooth_l1_loss(output.strain, batch["teacher_strain"], beta=0.0005)
    position_confidence = F.binary_cross_entropy(
        output.position_gate, batch["confidence_label"]
    )
    cell_confidence = F.binary_cross_entropy(
        output.cell_gate, batch["cell_confidence_label"]
    )
    negative = 1.0 - batch["confidence_label"].mean()
    identity = negative * output.position_residual_cart.square().mean()
    if output.edge_index.shape[1]:
        source, target = output.edge_index
        smoothness = (
            output.position_residual_cart[source]
            - output.position_residual_cart[target]
        ).square().mean()
    else:
        smoothness = output.position_residual_cart.new_zeros(())
    total = (
        20.0 * position
        + 20.0 * cell
        + 0.25 * position_confidence
        + 0.15 * cell_confidence
        + 5.0 * identity
        + 0.01 * smoothness
    )
    values = {
        "total": float(total.detach().item()),
        "position": float(position.detach().item()),
        "cell": float(cell.detach().item()),
        "position_confidence": float(position_confidence.detach().item()),
        "cell_confidence": float(cell_confidence.detach().item()),
        "identity": float(identity.detach().item()),
        "smoothness": float(smoothness.detach().item()),
        "position_gate": float(output.position_gate.detach().mean().item()),
        "cell_gate": float(output.cell_gate.detach().mean().item()),
        "position_clip_rate": float(output.position_clipped.float().mean().item()),
        "cell_fallback_rate": float(output.cell_fallback.float().mean().item()),
    }
    return total, values


@torch.no_grad()
def evaluate(
    model: CGTDRRefiner,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rows = [losses(model, sample, device)[1] for sample in samples]
    result = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
    model.train()
    return result


def save_checkpoint(
    path: Path,
    model: CGTDRRefiner,
    optimizer: torch.optim.Optimizer,
    step: int,
    validation: dict[str, float],
) -> None:
    payload = {
        "schema_version": 1,
        "config": model.config.as_dict(),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "validation": validation,
        "training_seed": 3100,
        "MatterGen_frozen": True,
        "CHGNet_in_checkpoint": False,
        "MatterSim_used_for_training": False,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_overfit(
    *,
    sample: dict[str, Any],
    config: CGTDRConfig,
    convergence_mean: torch.Tensor,
    convergence_std: torch.Tensor,
    device: torch.device,
) -> dict[str, float | bool]:
    model = CGTDRRefiner(config).to(device)
    model.set_convergence_normalization(convergence_mean, convergence_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-6)
    initial = losses(model, sample, device)[1]["total"]
    final = initial
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = losses(model, sample, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.detach().item())
    return {
        "steps": 100,
        "initial_loss": initial,
        "final_loss": final,
        "loss_ratio": final / max(initial, 1.0e-12),
        "passed": math.isfinite(final) and final < initial * 0.5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", default="/data/dxl/data/cg_tdr_teacher/labels")
    parser.add_argument("--output-root", default="/data/dxl/results/cg_tdr/phase0/training")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--training-seed", type=int, default=3100)
    args = parser.parse_args()
    if args.training_seed != 3100:
        raise ValueError("The frozen training seed is 3100")
    seed_everything(args.training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_root = Path(args.label_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    train_samples = load_split(label_root, 30000, 30383)
    validation_samples = load_split(label_root, 30384, 30447)
    test_samples = load_split(label_root, 30448, 30511)
    convergence = torch.cat([sample["convergence"].float() for sample in train_samples])
    convergence_mean = convergence.mean(dim=0)
    convergence_std = convergence.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    config = CGTDRConfig()
    positive_samples = [
        sample for sample in train_samples if float(sample["confidence_label"].item()) > 0
    ]
    if not positive_samples:
        atomic_json(
            output_root / "training_summary.json",
            {"status": "no_go", "reason": "no_positive_teacher_labels"},
        )
        return 3
    overfit = run_overfit(
        sample=positive_samples[0],
        config=config,
        convergence_mean=convergence_mean,
        convergence_std=convergence_std,
        device=device,
    )
    atomic_json(output_root / "overfit_100.json", overfit)
    if not overfit["passed"]:
        atomic_json(
            output_root / "training_summary.json",
            {"status": "no_go", "reason": "100_step_overfit_failed", "overfit": overfit},
        )
        return 4

    seed_everything(args.training_seed)
    model = CGTDRRefiner(config).to(device)
    model.set_convergence_normalization(convergence_mean, convergence_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-6)
    random_generator = random.Random(args.training_seed)
    curve: list[dict[str, float | int]] = []
    best_validation = math.inf
    best_step = 0
    stale_evaluations = 0
    stopped_reason = "max_steps"
    for step in range(1, args.max_steps + 1):
        sample = train_samples[random_generator.randrange(len(train_samples))]
        optimizer.zero_grad(set_to_none=True)
        loss, train_values = losses(model, sample, device)
        if not torch.isfinite(loss):
            stopped_reason = "non_finite_loss"
            break
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        if step % 100 == 0 or step in (1, 1000):
            validation = evaluate(model, validation_samples, device)
            row = {
                "step": step,
                "train_total": train_values["total"],
                "validation_total": validation["total"],
                "validation_position_gate": validation["position_gate"],
                "validation_cell_gate": validation["cell_gate"],
                "validation_position_clip_rate": validation["position_clip_rate"],
                "validation_cell_fallback_rate": validation["cell_fallback_rate"],
                "gradient_norm": gradient_norm,
            }
            curve.append(row)
            if validation["total"] < best_validation - 1.0e-6:
                best_validation = validation["total"]
                best_step = step
                stale_evaluations = 0
                save_checkpoint(
                    checkpoints / "best.pt", model, optimizer, step, validation
                )
            else:
                stale_evaluations += 1
            save_checkpoint(checkpoints / "last.pt", model, optimizer, step, validation)
            if step >= 1000:
                gates_invalid = (
                    validation["position_gate"] > 0.995
                    or validation["cell_gate"] > 0.995
                    or not math.isfinite(validation["total"])
                )
                if gates_invalid:
                    stopped_reason = "invalid_gate_or_validation"
                    break
                if stale_evaluations >= 12:
                    stopped_reason = "validation_early_stopping"
                    break

    with (output_root / "training_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    best_payload = torch.load(checkpoints / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["state_dict"], strict=True)
    test_metrics = evaluate(model, test_samples, device)
    summary = {
        "status": "success",
        "training_seed": args.training_seed,
        "steps_completed": int(curve[-1]["step"]),
        "best_step": best_step,
        "best_validation_loss": best_validation,
        "stopped_reason": stopped_reason,
        "parameter_count": model.parameter_count,
        "mattergen_parameter_count": 57936253,
        "parameter_percent_of_mattergen": model.parameter_count / 57936253 * 100.0,
        "config": asdict(config),
        "overfit_100": overfit,
        "train_structures": len(train_samples),
        "validation_structures": len(validation_samples),
        "test_structures": len(test_samples),
        "test_metrics": test_metrics,
        "best_checkpoint": str(checkpoints / "best.pt"),
        "last_checkpoint": str(checkpoints / "last.pt"),
        "MatterGen_frozen": True,
        "CHGNet_frozen_and_offline_only": True,
        "MatterSim_used_for_training": False,
    }
    atomic_json(output_root / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
