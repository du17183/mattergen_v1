"""Train the lightweight residual adapter on sharded teacher trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
import torch.nn.functional as F

from mattergen.diffusion.data.batched_data import SimpleBatchedData
from mattergen.diffusion.sampling.residual_adapter import (
    CONTEXT_FEATURE_NAMES,
    MASKED_LOGIT_ABS_THRESHOLD,
    RISK_FEATURE_NAMES,
    FieldwiseResidualAdapter,
    ResidualAdapterArchitecture,
    _per_sample_rms,
    minimum_image_displacement,
)
from mattergen.diffusion.sampling.residual_distillation import (
    iter_teacher_records,
    save_adapter_checkpoint,
)


FIELDS = ("pos", "cell", "atomic_numbers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-records", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--atomic-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def manifests(root: Path) -> list[Path]:
    paths = sorted(root.expanduser().resolve().glob("*/manifest.json"))
    if not paths:
        raise FileNotFoundError(f"no teacher manifests below {root}")
    return paths


def manifest_seeds(paths: Iterable[Path]) -> list[int]:
    seeds = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            seeds.append(int(json.load(stream)["seed"]))
    return seeds


def _combine(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    structure_offset = 0
    atom_indices = []
    x_fields: dict[str, list[torch.Tensor]] = {
        "pos": [],
        "cell": [],
        "atomic_numbers": [],
        "num_atoms": [],
    }
    score_values: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {field: [] for field in FIELDS}
        for name in ("score_before", "score_after", "score_residual")
    }
    timesteps = []
    progresses = []
    sampling_steps = []
    seeds = []
    for record in records:
        batch_size = record["x_after"].get_batch_size()
        index = record["x_after"].get_batch_idx("pos")
        if index is None:
            index = torch.arange(batch_size)
        atom_indices.append(index.long() + structure_offset)
        for field in x_fields:
            x_fields[field].append(record["x_after"][field])
        for name in score_values:
            for field in FIELDS:
                score_values[name][field].append(record[name][field])
        timesteps.append(record["t"].reshape(-1))
        progresses.append(torch.full((batch_size,), float(record["progress"])))
        sampling_steps.extend([int(record["sampling_step"])] * batch_size)
        seeds.extend([int(record["seed"])] * batch_size)
        structure_offset += batch_size
    atom_index = torch.cat(atom_indices)
    x_batch_index = {
        "pos": atom_index.clone(),
        "atomic_numbers": atom_index.clone(),
        "cell": None,
        "num_atoms": None,
    }
    # x_before shares atom types and sizes with x_after, but position/cell must
    # come from the actual pre-Corrector state.
    x_before = SimpleBatchedData(
        data={
            "pos": torch.cat([record["x_before"]["pos"] for record in records]),
            "cell": torch.cat([record["x_before"]["cell"] for record in records]),
            "atomic_numbers": torch.cat(
                [record["x_before"]["atomic_numbers"] for record in records]
            ),
            "num_atoms": torch.cat([record["x_before"]["num_atoms"] for record in records]),
        },
        batch_idx=dict(x_batch_index),
    )
    x_after = SimpleBatchedData(
        data={field: torch.cat(values) for field, values in x_fields.items()},
        batch_idx=dict(x_batch_index),
    )
    scores = {}
    for name, field_values in score_values.items():
        scores[name] = SimpleBatchedData(
            data={field: torch.cat(values) for field, values in field_values.items()},
            batch_idx={
                "pos": atom_index.clone(),
                "atomic_numbers": atom_index.clone(),
                "cell": None,
            },
        )
    return {
        "x_before": x_before,
        "x_after": x_after,
        **scores,
        "t": torch.cat(timesteps),
        "progress": torch.cat(progresses),
        "sampling_step": torch.tensor(sampling_steps),
        "seed": torch.tensor(seeds),
    }


def batches(
    paths: list[Path], *, batch_records: int, shuffle_seed: int | None = None
) -> Iterator[dict[str, Any]]:
    ordered = list(paths)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(ordered)
    buffer = []
    for record in iter_teacher_records(ordered, verify_sha256=shuffle_seed is None):
        buffer.append(record)
        if len(buffer) == batch_records:
            yield _combine(buffer)
            buffer = []
    if buffer:
        yield _combine(buffer)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for name in ("x_before", "x_after", "score_before", "score_after", "score_residual"):
        batch[name].to(device)
    batch["t"] = batch["t"].to(device)
    batch["progress"] = batch["progress"].to(device)
    return batch


def field_loss(
    predicted: torch.Tensor, target: torch.Tensor, field: str, score_before: torch.Tensor
) -> torch.Tensor:
    if field == "atomic_numbers":
        active = score_before.abs() < MASKED_LOGIT_ABS_THRESHOLD
        if not bool(active.any().item()):
            return predicted.sum() * 0.0
        return F.mse_loss(predicted[active], target[active])
    return F.mse_loss(predicted, target)


@torch.no_grad()
def fit_input_statistics_and_linear_baseline(
    model: FieldwiseResidualAdapter,
    paths: list[Path],
    *,
    batch_records: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], dict[str, list[float]]]:
    feature_sum = torch.zeros(len(CONTEXT_FEATURE_NAMES), device=device)
    feature_square_sum = torch.zeros_like(feature_sum)
    feature_count = 0
    target_square = {field: 0.0 for field in FIELDS}
    target_count = {field: 0 for field in FIELDS}
    normal_xx = {field: torch.zeros(2, 2, dtype=torch.float64) for field in FIELDS}
    normal_xy = {field: torch.zeros(2, dtype=torch.float64) for field in FIELDS}
    for teacher_batch in batches(paths, batch_records=batch_records):
        teacher_batch = to_device(teacher_batch, device)
        prediction = model(
            x_before=teacher_batch["x_before"],
            x_after=teacher_batch["x_after"],
            score_before=teacher_batch["score_before"],
            t=teacher_batch["t"],
            progress=teacher_batch["progress"],
        )
        features = prediction.context_features
        feature_sum += features.sum(dim=0)
        feature_square_sum += features.square().sum(dim=0)
        feature_count += features.shape[0]

        pos_delta = minimum_image_displacement(
            teacher_batch["x_after"]["pos"], teacher_batch["x_before"]["pos"]
        )
        cell_delta = teacher_batch["x_after"]["cell"] - teacher_batch["x_before"]["cell"]
        bases = {
            "pos": (pos_delta, teacher_batch["score_before"]["pos"]),
            "cell": (cell_delta, teacher_batch["score_before"]["cell"]),
            "atomic_numbers": (
                teacher_batch["score_before"]["atomic_numbers"],
                torch.ones_like(teacher_batch["score_before"]["atomic_numbers"]),
            ),
        }
        for field in FIELDS:
            target = teacher_batch["score_residual"][field].float()
            valid = torch.ones_like(target, dtype=torch.bool)
            if field == "atomic_numbers":
                valid = (
                    teacher_batch["score_before"][field].abs()
                    < MASKED_LOGIT_ABS_THRESHOLD
                )
            selected_target = target[valid].double()
            target_square[field] += float(selected_target.square().sum().item())
            target_count[field] += selected_target.numel()
            design = torch.stack(
                (bases[field][0][valid].double(), bases[field][1][valid].double()), dim=1
            )
            normal_xx[field] += (design.T @ design).cpu()
            normal_xy[field] += (design.T @ selected_target).cpu()
    mean = feature_sum / feature_count
    variance = feature_square_sum / feature_count - mean.square()
    std = variance.clamp_min(1.0e-8).sqrt()
    scales = {
        field: max(math.sqrt(target_square[field] / max(target_count[field], 1)), 1.0e-6)
        for field in FIELDS
    }
    coefficients = {}
    for field in FIELDS:
        ridge = torch.eye(2, dtype=torch.float64) * 1.0e-8
        coefficients[field] = torch.linalg.solve(
            normal_xx[field] + ridge, normal_xy[field]
        ).tolist()
    return mean, std, scales, coefficients


@torch.no_grad()
def evaluate(
    model: FieldwiseResidualAdapter,
    paths: list[Path],
    *,
    batch_records: int,
    device: torch.device,
) -> dict[str, float]:
    sums = {f"{field}_squared_error": 0.0 for field in FIELDS}
    counts = {field: 0 for field in FIELDS}
    dots = {field: 0.0 for field in FIELDS}
    pred_squares = {field: 0.0 for field in FIELDS}
    target_squares = {field: 0.0 for field in FIELDS}
    model.eval()
    for teacher_batch in batches(paths, batch_records=batch_records):
        teacher_batch = to_device(teacher_batch, device)
        prediction = model(
            x_before=teacher_batch["x_before"],
            x_after=teacher_batch["x_after"],
            score_before=teacher_batch["score_before"],
            t=teacher_batch["t"],
            progress=teacher_batch["progress"],
        )
        for field in FIELDS:
            pred = prediction.residuals[field].float()
            target = teacher_batch["score_residual"][field].float()
            valid = torch.ones_like(target, dtype=torch.bool)
            if field == "atomic_numbers":
                valid = teacher_batch["score_before"][field].abs() < MASKED_LOGIT_ABS_THRESHOLD
            pred = pred[valid]
            target = target[valid]
            sums[f"{field}_squared_error"] += float((pred - target).square().sum().item())
            counts[field] += target.numel()
            dots[field] += float((pred * target).sum().item())
            pred_squares[field] += float(pred.square().sum().item())
            target_squares[field] += float(target.square().sum().item())
    result = {}
    for field in FIELDS:
        result[f"{field}_mse"] = sums[f"{field}_squared_error"] / max(counts[field], 1)
        denominator = math.sqrt(pred_squares[field] * target_squares[field])
        result[f"{field}_cosine"] = dots[field] / denominator if denominator else 0.0
    return result


@torch.no_grad()
def calibrate_uncertainty(
    model: FieldwiseResidualAdapter,
    paths: list[Path],
    *,
    batch_records: int,
    device: torch.device,
    target_scales: Mapping[str, float],
) -> dict[str, Any]:
    all_features = []
    all_targets = []
    all_progress = []
    model.eval()
    for teacher_batch in batches(paths, batch_records=batch_records):
        teacher_batch = to_device(teacher_batch, device)
        prediction = model(
            x_before=teacher_batch["x_before"],
            x_after=teacher_batch["x_after"],
            score_before=teacher_batch["score_before"],
            t=teacher_batch["t"],
            progress=teacher_batch["progress"],
        )
        batch_size = teacher_batch["x_after"].get_batch_size()
        errors = []
        for field in FIELDS:
            index = teacher_batch["x_after"].get_batch_idx(field)
            if index is None:
                index = torch.arange(batch_size, device=device)
            valid = None
            if field == "atomic_numbers":
                valid = teacher_batch["score_before"][field].abs() < MASKED_LOGIT_ABS_THRESHOLD
            error = prediction.residuals[field] - teacher_batch["score_residual"][field]
            errors.append(
                _per_sample_rms(error, index, batch_size, valid=valid)
                / float(target_scales[field])
            )
        normalized_error = torch.stack(errors, dim=1).mean(dim=1)
        all_targets.append(torch.log1p(normalized_error).cpu())
        all_features.append(prediction.risk_features.cpu())
        all_progress.append(teacher_batch["progress"].cpu())
    features = torch.cat(all_features).double()
    targets = torch.cat(all_targets).double()
    progress = torch.cat(all_progress).double()
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1.0e-6)
    normalized = (features - mean) / std
    bias = targets.mean()
    ridge = torch.eye(normalized.shape[1], dtype=torch.float64) * 1.0e-2
    weights = torch.linalg.solve(
        normalized.T @ normalized + ridge,
        normalized.T @ (targets - bias),
    )
    predicted_risk = normalized @ weights + bias
    coverages = (0.25, 0.5, 0.75, 0.9)
    thresholds = {
        str(coverage): float(torch.quantile(predicted_risk, coverage).item())
        for coverage in coverages
    }
    bin_edges = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    bin_errors = []
    for index in range(10):
        if index == 9:
            selected = (progress >= bin_edges[index]) & (progress <= bin_edges[index + 1])
        else:
            selected = (progress >= bin_edges[index]) & (progress < bin_edges[index + 1])
        bin_errors.append(float(targets[selected].mean().item()) if selected.any() else math.nan)
    correlation = torch.corrcoef(torch.stack((predicted_risk, targets)))[0, 1]
    return {
        "risk_feature_names": list(RISK_FEATURE_NAMES),
        "risk_feature_mean": mean.tolist(),
        "risk_feature_std": std.tolist(),
        "risk_linear_weights": weights.tolist(),
        "risk_linear_bias": float(bias.item()),
        "risk_target": "log1p(mean field-normalized residual RMSE)",
        "risk_validation_correlation": float(correlation.item()),
        "coverage_thresholds": thresholds,
        "timestep_bin_edges": bin_edges.tolist(),
        "timestep_bin_mean_log_error": bin_errors,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_paths = manifests(args.train_root)
    validation_paths = manifests(args.validation_root)
    train_seeds = manifest_seeds(train_paths)
    validation_seeds = manifest_seeds(validation_paths)
    if set(train_seeds) & set(validation_seeds):
        raise ValueError("teacher train/validation seed leakage")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    model = FieldwiseResidualAdapter(
        ResidualAdapterArchitecture(
            hidden_dim=args.hidden_dim,
            context_dim=args.context_dim,
            atomic_rank=args.atomic_rank,
        )
    ).to(device)
    mean, std, target_scales, linear_coefficients = fit_input_statistics_and_linear_baseline(
        model,
        train_paths,
        batch_records=args.batch_records,
        device=device,
    )
    model.set_feature_statistics(mean, std)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rows = []
    best_epoch = 0
    best_validation_objective = math.inf
    best_state_dict: dict[str, torch.Tensor] | None = None
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        batches_seen = 0
        for teacher_batch in batches(
            train_paths,
            batch_records=args.batch_records,
            shuffle_seed=args.seed + epoch,
        ):
            teacher_batch = to_device(teacher_batch, device)
            prediction = model(
                x_before=teacher_batch["x_before"],
                x_after=teacher_batch["x_after"],
                score_before=teacher_batch["score_before"],
                t=teacher_batch["t"],
                progress=teacher_batch["progress"],
            )
            losses = {
                field: field_loss(
                    prediction.residuals[field],
                    teacher_batch["score_residual"][field],
                    field,
                    teacher_batch["score_before"][field],
                )
                / (target_scales[field] ** 2)
                for field in FIELDS
            }
            loss = sum(losses.values()) / len(FIELDS)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().item())
            batches_seen += 1
        validation = evaluate(
            model,
            validation_paths,
            batch_records=args.batch_records,
            device=device,
        )
        validation_objective = sum(
            validation[f"{field}_mse"] / (target_scales[field] ** 2)
            for field in FIELDS
        ) / len(FIELDS)
        row = {
            "epoch": epoch + 1,
            "train_normalized_loss": loss_sum / max(batches_seen, 1),
            "validation_normalized_mse": validation_objective,
            **validation,
        }
        rows.append(row)
        if validation_objective < best_validation_objective:
            best_epoch = epoch + 1
            best_validation_objective = validation_objective
            best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(json.dumps(rows[-1], sort_keys=True))

    if best_state_dict is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state_dict)
    selected_validation = evaluate(
        model,
        validation_paths,
        batch_records=args.batch_records,
        device=device,
    )
    calibration = calibrate_uncertainty(
        model,
        validation_paths,
        batch_records=args.batch_records,
        device=device,
        target_scales=target_scales,
    )
    training_hyperparameters = {
        "epochs": args.epochs,
        "batch_records": args.batch_records,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "context_dim": args.context_dim,
        "atomic_rank": args.atomic_rank,
        "seed": args.seed,
        "device": args.device,
    }
    checkpoint_path = save_adapter_checkpoint(
        output_dir / "residual_adapter.pt",
        model=model,
        calibration=calibration,
        metadata={
            "training_seed": args.seed,
            "train_teacher_seeds": train_seeds,
            "validation_teacher_seeds": validation_seeds,
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "selected_validation_normalized_mse": best_validation_objective,
            "parameter_count": model.parameter_count,
            "target_residual_rms": target_scales,
            "linear_baseline_coefficients": linear_coefficients,
            "training_hyperparameters": training_hyperparameters,
        },
    )
    with (output_dir / "training_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(checkpoint_path),
        "parameter_count": model.parameter_count,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "target_residual_rms": target_scales,
        "linear_baseline_coefficients": linear_coefficients,
        "calibration": calibration,
        "last_epoch_validation": rows[-1],
        "selected_epoch": best_epoch,
        "selected_validation_normalized_mse": best_validation_objective,
        "selected_validation": selected_validation,
        "training_hyperparameters": training_hyperparameters,
    }
    with (output_dir / "training_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
