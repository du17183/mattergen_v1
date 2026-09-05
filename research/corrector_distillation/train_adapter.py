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
    parser.add_argument("--extra-train-root", type=Path, action="append", default=[])
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
    parser.add_argument("--early-weight", type=float, default=1.0)
    parser.add_argument("--middle-weight", type=float, default=1.0)
    parser.add_argument("--late-weight", type=float, default=1.0)
    parser.add_argument("--early-end", type=float, default=0.2)
    parser.add_argument("--late-start", type=float, default=0.7)
    parser.add_argument(
        "--cache-data-on-device",
        action="store_true",
        help="Load verified combined batches once and retain them on the training device.",
    )
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


def timestep_weights(
    progress: torch.Tensor,
    *,
    early_weight: float,
    middle_weight: float,
    late_weight: float,
    early_end: float,
    late_start: float,
) -> torch.Tensor:
    if min(early_weight, middle_weight, late_weight) <= 0.0:
        raise ValueError("all timestep weights must be positive")
    if not 0.0 <= early_end <= late_start <= 1.0:
        raise ValueError("timestep boundaries must satisfy 0 <= early_end <= late_start <= 1")
    return torch.where(
        progress < early_end,
        torch.full_like(progress, early_weight),
        torch.where(
            progress < late_start,
            torch.full_like(progress, middle_weight),
            torch.full_like(progress, late_weight),
        ),
    )


def field_row_weights(
    sample_weights: torch.Tensor,
    batch_index: torch.Tensor | None,
    row_count: int,
) -> torch.Tensor:
    if batch_index is None:
        if row_count != sample_weights.numel():
            raise ValueError("field rows do not match unbatched sample weights")
        return sample_weights
    return sample_weights[batch_index.long()]


def field_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    field: str,
    score_before: torch.Tensor,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    squared_error = (predicted - target).square()
    if row_weights is None:
        weights = torch.ones_like(squared_error)
    else:
        shape = (row_weights.shape[0],) + (1,) * (squared_error.ndim - 1)
        weights = row_weights.reshape(shape).expand_as(squared_error)
    if field == "atomic_numbers":
        active = score_before.abs() < MASKED_LOGIT_ABS_THRESHOLD
        if not bool(active.any().item()):
            return predicted.sum() * 0.0
        squared_error = squared_error[active]
        weights = weights[active]
    return (squared_error * weights).sum() / weights.sum().clamp_min(1.0e-12)


@torch.no_grad()
def fit_input_statistics_and_linear_baseline(
    model: FieldwiseResidualAdapter,
    paths: list[Path],
    *,
    batch_records: int,
    device: torch.device,
    cached_batches: list[dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], dict[str, list[float]]]:
    feature_sum = torch.zeros(len(CONTEXT_FEATURE_NAMES), device=device)
    feature_square_sum = torch.zeros_like(feature_sum)
    feature_count = 0
    target_square = {field: 0.0 for field in FIELDS}
    target_count = {field: 0 for field in FIELDS}
    normal_xx = {field: torch.zeros(2, 2, dtype=torch.float64) for field in FIELDS}
    normal_xy = {field: torch.zeros(2, dtype=torch.float64) for field in FIELDS}
    source = (
        cached_batches
        if cached_batches is not None
        else batches(paths, batch_records=batch_records)
    )
    for teacher_batch in source:
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
    stage_weights: tuple[float, float, float, float, float] | None = None,
    cached_batches: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    sums = {f"{field}_squared_error": 0.0 for field in FIELDS}
    counts = {field: 0 for field in FIELDS}
    dots = {field: 0.0 for field in FIELDS}
    pred_squares = {field: 0.0 for field in FIELDS}
    target_squares = {field: 0.0 for field in FIELDS}
    model.eval()
    source = (
        cached_batches
        if cached_batches is not None
        else batches(paths, batch_records=batch_records)
    )
    for teacher_batch in source:
        teacher_batch = to_device(teacher_batch, device)
        prediction = model(
            x_before=teacher_batch["x_before"],
            x_after=teacher_batch["x_after"],
            score_before=teacher_batch["score_before"],
            t=teacher_batch["t"],
            progress=teacher_batch["progress"],
        )
        sample_weights = (
            torch.ones_like(teacher_batch["progress"])
            if stage_weights is None
            else timestep_weights(
                teacher_batch["progress"],
                early_weight=stage_weights[0],
                middle_weight=stage_weights[1],
                late_weight=stage_weights[2],
                early_end=stage_weights[3],
                late_start=stage_weights[4],
            )
        )
        for field in FIELDS:
            pred = prediction.residuals[field].float()
            target = teacher_batch["score_residual"][field].float()
            valid = torch.ones_like(target, dtype=torch.bool)
            if field == "atomic_numbers":
                valid = teacher_batch["score_before"][field].abs() < MASKED_LOGIT_ABS_THRESHOLD
            row_weights = field_row_weights(
                sample_weights,
                teacher_batch["score_before"].get_batch_idx(field),
                target.shape[0],
            )
            weight_shape = (row_weights.shape[0],) + (1,) * (target.ndim - 1)
            weights = row_weights.reshape(weight_shape).expand_as(target)[valid]
            pred = pred[valid]
            target = target[valid]
            sums[f"{field}_squared_error"] += float(
                ((pred - target).square() * weights).sum().item()
            )
            counts[field] += float(weights.sum().item())
            dots[field] += float((pred * target * weights).sum().item())
            pred_squares[field] += float((pred.square() * weights).sum().item())
            target_squares[field] += float((target.square() * weights).sum().item())
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
    cached_batches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    all_features = []
    all_targets = []
    all_field_targets: dict[str, list[torch.Tensor]] = {field: [] for field in FIELDS}
    all_progress = []
    model.eval()
    source = (
        cached_batches
        if cached_batches is not None
        else batches(paths, batch_records=batch_records)
    )
    for teacher_batch in source:
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
            normalized_field_error = (
                _per_sample_rms(error, index, batch_size, valid=valid)
                / float(target_scales[field])
            )
            errors.append(normalized_field_error)
            all_field_targets[field].append(torch.log1p(normalized_field_error).cpu())
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
    field_models = {}
    field_predictions = {}
    for field in FIELDS:
        field_target = torch.cat(all_field_targets[field]).double()
        field_bias = field_target.mean()
        field_weights = torch.linalg.solve(
            normalized.T @ normalized + ridge,
            normalized.T @ (field_target - field_bias),
        )
        field_prediction = normalized @ field_weights + field_bias
        field_predictions[field] = field_prediction
        correlations = {}
        stage_masks = {
            "overall": torch.ones_like(progress, dtype=torch.bool),
            "early": progress < (1.0 / 3.0),
            "middle": (progress >= (1.0 / 3.0)) & (progress < (2.0 / 3.0)),
            "late": progress >= (2.0 / 3.0),
        }
        for stage, selected in stage_masks.items():
            selected_prediction = field_prediction[selected]
            selected_target = field_target[selected]
            if selected_target.numel() < 2 or float(selected_target.std().item()) == 0.0:
                correlations[stage] = math.nan
            else:
                correlations[stage] = float(
                    torch.corrcoef(
                        torch.stack((selected_prediction, selected_target))
                    )[0, 1].item()
                )
        field_models[field] = {
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "linear_weights": field_weights.tolist(),
            "linear_bias": float(field_bias.item()),
            "target": f"log1p({field}-normalized residual RMSE)",
            "correlations": correlations,
        }

    field_thresholds = {}
    field_coverage_calibration = {}
    for coverage in coverages:
        low, high = 0.0, 1.0
        for _ in range(40):
            quantile = (low + high) / 2.0
            trial = {
                field: torch.quantile(values, quantile)
                for field, values in field_predictions.items()
            }
            accepted = torch.ones_like(progress, dtype=torch.bool)
            for field in FIELDS:
                accepted &= field_predictions[field] <= trial[field]
            if float(accepted.double().mean().item()) < coverage:
                low = quantile
            else:
                high = quantile
        quantile = high
        thresholds_for_coverage = {
            field: float(torch.quantile(values, quantile).item())
            for field, values in field_predictions.items()
        }
        accepted = torch.ones_like(progress, dtype=torch.bool)
        for field in FIELDS:
            accepted &= field_predictions[field] <= thresholds_for_coverage[field]
        key = str(coverage)
        field_thresholds[key] = thresholds_for_coverage
        field_coverage_calibration[key] = {
            "marginal_quantile": quantile,
            "joint_coverage": float(accepted.double().mean().item()),
        }

    return {
        "risk_feature_names": list(RISK_FEATURE_NAMES),
        "risk_feature_mean": mean.tolist(),
        "risk_feature_std": std.tolist(),
        "risk_linear_weights": weights.tolist(),
        "risk_linear_bias": float(bias.item()),
        "risk_target": "log1p(mean field-normalized residual RMSE)",
        "risk_validation_correlation": float(correlation.item()),
        "field_risk_models": field_models,
        "field_coverage_thresholds": field_thresholds,
        "field_coverage_calibration": field_coverage_calibration,
        "field_risk_stage_definition": {
            "early": "progress < 1/3",
            "middle": "1/3 <= progress < 2/3",
            "late": "progress >= 2/3",
        },
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
    train_roots = [args.train_root, *args.extra_train_root]
    train_paths = [path for root in train_roots for path in manifests(root)]
    validation_paths = manifests(args.validation_root)
    train_seeds = manifest_seeds(train_paths)
    validation_seeds = manifest_seeds(validation_paths)
    if set(train_seeds) & set(validation_seeds):
        raise ValueError("teacher train/validation seed leakage")
    stage_weight_config = (
        args.early_weight,
        args.middle_weight,
        args.late_weight,
        args.early_end,
        args.late_start,
    )
    timestep_weights(
        torch.tensor([0.0, 0.5, 1.0]),
        early_weight=args.early_weight,
        middle_weight=args.middle_weight,
        late_weight=args.late_weight,
        early_end=args.early_end,
        late_start=args.late_start,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    train_cache = None
    validation_cache = None
    if args.cache_data_on_device:
        print(
            json.dumps(
                {
                    "event": "cache_start",
                    "device": str(device),
                    "train_manifests": len(train_paths),
                    "validation_manifests": len(validation_paths),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        train_cache = [
            to_device(batch, device)
            for batch in batches(train_paths, batch_records=args.batch_records)
        ]
        validation_cache = [
            to_device(batch, device)
            for batch in batches(validation_paths, batch_records=args.batch_records)
        ]
        print(
            json.dumps(
                {
                    "event": "cache_complete",
                    "train_batches": len(train_cache),
                    "validation_batches": len(validation_cache),
                },
                sort_keys=True,
            ),
            flush=True,
        )

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
        cached_batches=train_cache,
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
        if train_cache is None:
            epoch_batches = batches(
                train_paths,
                batch_records=args.batch_records,
                shuffle_seed=args.seed + epoch,
            )
        else:
            shuffled_cache = list(train_cache)
            random.Random(args.seed + epoch).shuffle(shuffled_cache)
            epoch_batches = iter(shuffled_cache)
        for teacher_batch in epoch_batches:
            teacher_batch = to_device(teacher_batch, device)
            prediction = model(
                x_before=teacher_batch["x_before"],
                x_after=teacher_batch["x_after"],
                score_before=teacher_batch["score_before"],
                t=teacher_batch["t"],
                progress=teacher_batch["progress"],
            )
            sample_weights = timestep_weights(
                teacher_batch["progress"],
                early_weight=args.early_weight,
                middle_weight=args.middle_weight,
                late_weight=args.late_weight,
                early_end=args.early_end,
                late_start=args.late_start,
            )
            losses = {
                field: field_loss(
                    prediction.residuals[field],
                    teacher_batch["score_residual"][field],
                    field,
                    teacher_batch["score_before"][field],
                    field_row_weights(
                        sample_weights,
                        teacher_batch["score_before"].get_batch_idx(field),
                        teacher_batch["score_residual"][field].shape[0],
                    ),
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
            cached_batches=validation_cache,
        )
        weighted_validation = evaluate(
            model,
            validation_paths,
            batch_records=args.batch_records,
            device=device,
            stage_weights=stage_weight_config,
            cached_batches=validation_cache,
        )
        validation_objective = sum(
            weighted_validation[f"{field}_mse"] / (target_scales[field] ** 2)
            for field in FIELDS
        ) / len(FIELDS)
        row = {
            "epoch": epoch + 1,
            "train_normalized_loss": loss_sum / max(batches_seen, 1),
            "validation_normalized_mse": validation_objective,
            **validation,
            **{
                f"weighted_{key}": value
                for key, value in weighted_validation.items()
            },
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
        cached_batches=validation_cache,
    )
    selected_weighted_validation = evaluate(
        model,
        validation_paths,
        batch_records=args.batch_records,
        device=device,
        stage_weights=stage_weight_config,
        cached_batches=validation_cache,
    )
    calibration = calibrate_uncertainty(
        model,
        validation_paths,
        batch_records=args.batch_records,
        device=device,
        target_scales=target_scales,
        cached_batches=validation_cache,
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
        "early_weight": args.early_weight,
        "middle_weight": args.middle_weight,
        "late_weight": args.late_weight,
        "early_end": args.early_end,
        "late_start": args.late_start,
        "train_roots": [str(root.expanduser().resolve()) for root in train_roots],
        "cache_data_on_device": args.cache_data_on_device,
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
        "selected_weighted_validation": selected_weighted_validation,
        "training_hyperparameters": training_hyperparameters,
    }
    with (output_dir / "training_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
