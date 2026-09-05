"""Compare multiple residual Adapter checkpoints by timestep stage and field."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from mattergen.diffusion.sampling.residual_adapter import MASKED_LOGIT_ABS_THRESHOLD
from mattergen.diffusion.sampling.residual_distillation import load_adapter_checkpoint
from research.corrector_distillation.train_adapter import batches, manifests, to_device


FIELDS = ("pos", "cell", "atomic_numbers")


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=named_path, action="append", required=True)
    parser.add_argument("--adapter", type=named_path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
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

    def result(self) -> dict[str, float | int]:
        mse = self.squared_error / max(self.count, 1)
        denominator = math.sqrt(self.predicted_square * self.target_square)
        return {
            "count": self.count,
            "residual_mse": mse,
            "residual_rmse": math.sqrt(mse),
            "residual_cosine": self.dot / denominator if denominator else 0.0,
        }


def stage(progress: float) -> str:
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "middle"
    return "late"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    summary = {}
    for dataset_name, dataset_root in args.dataset:
        paths = manifests(dataset_root)
        for adapter_name, checkpoint in args.adapter:
            loaded = load_adapter_checkpoint(checkpoint.expanduser().resolve(), device)
            model = loaded.model.eval()
            accumulators: dict[tuple[str, int, str], Accumulator] = defaultdict(Accumulator)
            with torch.no_grad():
                for batch in batches(paths, batch_records=args.batch_records):
                    batch = to_device(batch, device)
                    prediction = model(
                        x_before=batch["x_before"],
                        x_after=batch["x_after"],
                        score_before=batch["score_before"],
                        t=batch["t"],
                        progress=batch["progress"],
                    )
                    batch_size = batch["x_after"].get_batch_size()
                    for sample_index in range(batch_size):
                        progress = float(batch["progress"][sample_index].item())
                        groups = (
                            ("overall", -1),
                            (stage(progress), -1),
                            ("decile", min(int(progress * 10), 9)),
                        )
                        for field in FIELDS:
                            index = batch["x_after"].get_batch_idx(field)
                            if index is None:
                                selected = torch.zeros(
                                    batch_size, dtype=torch.bool, device=device
                                )
                                selected[sample_index] = True
                            else:
                                selected = index == sample_index
                            target = batch["score_residual"][field][selected].float()
                            predicted = prediction.residuals[field][selected].float()
                            if field == "atomic_numbers":
                                valid = (
                                    batch["score_before"][field][selected].abs()
                                    < MASKED_LOGIT_ABS_THRESHOLD
                                )
                                target = target[valid]
                                predicted = predicted[valid]
                            for group_name, timestep_bin in groups:
                                accumulators[(group_name, timestep_bin, field)].add(
                                    predicted, target
                                )
            adapter_rows = []
            for (group_name, timestep_bin, field), accumulator in sorted(
                accumulators.items()
            ):
                adapter_rows.append(
                    {
                        "dataset": dataset_name,
                        "adapter": adapter_name,
                        "stage": group_name,
                        "timestep_bin": timestep_bin,
                        "field": field,
                        **accumulator.result(),
                    }
                )
            rows.extend(adapter_rows)
            summary[f"{dataset_name}/{adapter_name}"] = [
                row for row in adapter_rows if row["stage"] in ("overall", "early", "middle", "late")
            ]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
