"""Diagnose residual learnability for Reuse, linear, and lightweight Adapter."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from mattergen.diffusion.sampling.residual_adapter import (
    MASKED_LOGIT_ABS_THRESHOLD,
    minimum_image_displacement,
)
from mattergen.diffusion.sampling.residual_distillation import load_adapter_checkpoint
from research.corrector_distillation.train_adapter import batches, manifests, to_device


FIELDS = ("pos", "cell", "atomic_numbers")
MODELS = ("zero_residual_reuse", "linear_residual", "lightweight_adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--split-name", default="validation")
    parser.add_argument("--batch-records", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class Accumulator:
    def __init__(self) -> None:
        self.squared_error = 0.0
        self.count = 0
        self.dot = 0.0
        self.predicted_square = 0.0
        self.target_square = 0.0

    def add(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        self.squared_error += float((predicted - target).square().sum().item())
        self.count += target.numel()
        self.dot += float((predicted * target).sum().item())
        self.predicted_square += float(predicted.square().sum().item())
        self.target_square += float(target.square().sum().item())

    def row(self) -> dict[str, float | int]:
        mse = self.squared_error / max(self.count, 1)
        denominator = math.sqrt(self.predicted_square * self.target_square)
        return {
            "count": self.count,
            "residual_mse": mse,
            "residual_rmse": math.sqrt(mse),
            "residual_cosine": self.dot / denominator if denominator else 0.0,
            "score_after_mse": mse,
        }


def stage_name(progress: float) -> str:
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "middle"
    return "late"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loaded = load_adapter_checkpoint(args.checkpoint, device)
    model = loaded.model.eval()
    linear = loaded.metadata.get("linear_baseline_coefficients")
    if not linear:
        raise ValueError("checkpoint does not contain the fitted linear baseline")
    accumulators: dict[tuple[str, str, int, str], Accumulator] = defaultdict(Accumulator)
    with torch.no_grad():
        for teacher_batch in batches(
            manifests(args.teacher_root), batch_records=args.batch_records
        ):
            teacher_batch = to_device(teacher_batch, device)
            adapter_prediction = model(
                x_before=teacher_batch["x_before"],
                x_after=teacher_batch["x_after"],
                score_before=teacher_batch["score_before"],
                t=teacher_batch["t"],
                progress=teacher_batch["progress"],
            )
            bases = {
                "pos": (
                    minimum_image_displacement(
                        teacher_batch["x_after"]["pos"], teacher_batch["x_before"]["pos"]
                    ),
                    teacher_batch["score_before"]["pos"],
                ),
                "cell": (
                    teacher_batch["x_after"]["cell"] - teacher_batch["x_before"]["cell"],
                    teacher_batch["score_before"]["cell"],
                ),
                "atomic_numbers": (
                    teacher_batch["score_before"]["atomic_numbers"],
                    torch.ones_like(teacher_batch["score_before"]["atomic_numbers"]),
                ),
            }
            predictions = {
                "zero_residual_reuse": {
                    field: torch.zeros_like(teacher_batch["score_residual"][field])
                    for field in FIELDS
                },
                "linear_residual": {
                    field: float(linear[field][0]) * bases[field][0]
                    + float(linear[field][1]) * bases[field][1]
                    for field in FIELDS
                },
                "lightweight_adapter": adapter_prediction.residuals,
            }
            batch_size = teacher_batch["x_after"].get_batch_size()
            for sample_index in range(batch_size):
                progress = float(teacher_batch["progress"][sample_index].item())
                decile = min(int(progress * 10), 9)
                groups = (("overall", -1), (stage_name(progress), -1), ("decile", decile))
                for field in FIELDS:
                    index = teacher_batch["x_after"].get_batch_idx(field)
                    if index is None:
                        selected = torch.zeros(batch_size, dtype=torch.bool, device=device)
                        selected[sample_index] = True
                    else:
                        selected = index == sample_index
                    target = teacher_batch["score_residual"][field][selected].float()
                    valid = torch.ones_like(target, dtype=torch.bool)
                    if field == "atomic_numbers":
                        before = teacher_batch["score_before"][field][selected]
                        valid = before.abs() < MASKED_LOGIT_ABS_THRESHOLD
                    target = target[valid]
                    for model_name in MODELS:
                        predicted = predictions[model_name][field][selected].float()[valid]
                        for group_name, group_index in groups:
                            accumulators[(model_name, group_name, group_index, field)].add(
                                predicted, target
                            )
    rows: list[dict[str, Any]] = []
    for (model_name, group_name, group_index, field), accumulator in sorted(
        accumulators.items()
    ):
        rows.append(
            {
                "split": args.split_name,
                "model": model_name,
                "stage": group_name,
                "timestep_bin": group_index,
                "field": field,
                **accumulator.row(),
            }
        )
    output_path = args.output_csv.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    with output_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    overall = [row for row in rows if row["stage"] == "overall"]
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

