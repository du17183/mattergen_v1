#!/usr/bin/env python3
"""Train a deterministic deep ensemble for post-generation quality ranking."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn

from research.postgen_fastgate.model import QualityNetwork

CONTINUOUS_TARGETS = (
    "energy_above_hull_per_atom",
    "rmsd_from_relaxation",
)
BINARY_TARGETS = (
    "stable",
    "novel_unique_stable",
    "comp_validity",
    "novel",
    "unique",
)
NON_FEATURE_COLUMNS = {
    "method",
    "seed",
    "split",
    "input_path",
    *CONTINUOUS_TARGETS,
    *BINARY_TARGETS,
    "structure_validity",
    "novel_unique",
    "converged",
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not columns:
        raise ValueError("no numeric feature columns")
    return columns


def transform_continuous(frame: pd.DataFrame) -> np.ndarray:
    ehull = np.log1p(
        np.maximum(frame["energy_above_hull_per_atom"].to_numpy(float), 0.0)
        / 0.05
    )
    rmsd = np.log1p(
        np.maximum(frame["rmsd_from_relaxation"].to_numpy(float), 0.0)
        / 0.01
    )
    return np.stack([ehull, rmsd], axis=1)


def inverse_continuous(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values)
    output[:, 0] = np.maximum(np.expm1(values[:, 0]) * 0.05, 0.0)
    output[:, 1] = np.maximum(np.expm1(values[:, 1]) * 0.01, 0.0)
    return output


def binary_values(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, BINARY_TARGETS].astype(float).to_numpy()


def pairwise_loss(
    output: dict[str, torch.Tensor],
    continuous_target: torch.Tensor,
    binary_target: torch.Tensor,
) -> torch.Tensor:
    count = len(continuous_target)
    if count < 2:
        return torch.zeros((), device=continuous_target.device)
    permutation = torch.roll(torch.arange(count, device=continuous_target.device), 1)
    true_utility = (
        -continuous_target[:, 0]
        - 0.25 * continuous_target[:, 1]
        + 0.75 * binary_target[:, 0]
        + 1.00 * binary_target[:, 1]
        + 0.50 * binary_target[:, 2]
    )
    probabilities = torch.sigmoid(output["binary_logits"])
    predicted_utility = (
        -output["continuous"][:, 0]
        - 0.25 * output["continuous"][:, 1]
        + 0.75 * probabilities[:, 0]
        + 1.00 * probabilities[:, 1]
        + 0.50 * probabilities[:, 2]
    )
    direction = torch.sign(true_utility - true_utility[permutation])
    valid = direction != 0
    if not torch.any(valid):
        return torch.zeros((), device=continuous_target.device)
    margin = predicted_utility - predicted_utility[permutation]
    return torch.nn.functional.softplus(-direction[valid] * margin[valid]).mean()


def prepare(
    frame: pd.DataFrame,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    feature_columns = select_feature_columns(frame)
    train = frame["split"] == "train"
    features = frame.loc[:, feature_columns].to_numpy(float)
    feature_mean = features[train].mean(axis=0)
    feature_std = features[train].std(axis=0)
    feature_std[feature_std < 1.0e-8] = 1.0
    features = (features - feature_mean) / feature_std

    continuous_raw = transform_continuous(frame)
    continuous_mean = continuous_raw[train].mean(axis=0)
    continuous_std = continuous_raw[train].std(axis=0)
    continuous_std[continuous_std < 1.0e-8] = 1.0
    continuous = (continuous_raw - continuous_mean) / continuous_std
    binary = binary_values(frame)
    tensors = {
        "features": torch.tensor(features, dtype=torch.float32),
        "continuous": torch.tensor(continuous, dtype=torch.float32),
        "binary": torch.tensor(binary, dtype=torch.float32),
    }
    metadata = {
        "feature_columns": feature_columns,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "continuous_mean": continuous_mean.tolist(),
        "continuous_std": continuous_std.tolist(),
    }
    return tensors, metadata


def total_loss(
    model: QualityNetwork,
    features: torch.Tensor,
    continuous: torch.Tensor,
    binary: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(features)
    continuous_loss = nn.functional.smooth_l1_loss(
        output["continuous"], continuous
    )
    binary_loss = nn.functional.binary_cross_entropy_with_logits(
        output["binary_logits"],
        binary,
        pos_weight=pos_weight,
    )
    ranking_loss = pairwise_loss(output, continuous, binary)
    loss = continuous_loss + binary_loss + 0.25 * ranking_loss
    return loss, {
        "continuous": float(continuous_loss.detach()),
        "binary": float(binary_loss.detach()),
        "pairwise": float(ranking_loss.detach()),
        "total": float(loss.detach()),
    }


def train_member(
    *,
    frame: pd.DataFrame,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    seed: int,
    output_dir: Path,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    seed_everything(seed)
    train_mask = torch.tensor(
        (frame["split"] == "train").to_numpy(), dtype=torch.bool
    )
    validation_mask = torch.tensor(
        (frame["split"] == "validation").to_numpy(), dtype=torch.bool
    )
    features = tensors["features"].to(device)
    continuous = tensors["continuous"].to(device)
    binary = tensors["binary"].to(device)
    train_binary = binary[train_mask.to(device)]
    positives = train_binary.sum(dim=0)
    negatives = len(train_binary) - positives
    pos_weight = (negatives / torch.clamp(positives, min=1.0)).clamp(0.25, 4.0)

    model = QualityNetwork(
        input_dim=features.shape[1],
        hidden_dim=128,
        dropout=0.10,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-3,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )
    best_loss = math.inf
    best_epoch = -1
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    train_index = train_mask.to(device)
    validation_index = validation_mask.to(device)
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, train_parts = total_loss(
            model,
            features[train_index],
            continuous[train_index],
            binary[train_index],
            pos_weight,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_loss, validation_parts = total_loss(
                model,
                features[validation_index],
                continuous[validation_index],
                binary[validation_index],
                pos_weight,
            )
        value = float(validation_loss)
        if epoch % 10 == 0 or value < best_loss:
            history.append(
                {
                    "epoch": epoch,
                    "train": train_parts,
                    "validation": validation_parts,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
        if value < best_loss - 1.0e-5:
            best_loss = value
            best_epoch = epoch
            stale = 0
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"quality_member_{seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": int(features.shape[1]),
            "hidden_dim": 128,
            "dropout": 0.10,
            "metadata": metadata,
            "seed": seed,
        },
        checkpoint,
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "checkpoint": str(checkpoint),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "history": history,
    }


def predict_ensemble(
    *,
    frame: pd.DataFrame,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    members: list[dict[str, Any]],
    device: torch.device,
) -> pd.DataFrame:
    feature = tensors["features"].to(device)
    continuous_predictions: list[np.ndarray] = []
    probability_predictions: list[np.ndarray] = []
    continuous_mean = np.asarray(metadata["continuous_mean"], dtype=float)
    continuous_std = np.asarray(metadata["continuous_std"], dtype=float)
    for member in members:
        payload = torch.load(
            member["checkpoint"],
            map_location=device,
            weights_only=True,
        )
        model = QualityNetwork(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload["dropout"]),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            output = model(feature)
        standardized = output["continuous"].cpu().numpy()
        transformed = standardized * continuous_std + continuous_mean
        continuous_predictions.append(inverse_continuous(transformed))
        probability_predictions.append(
            torch.sigmoid(output["binary_logits"]).cpu().numpy()
        )
    continuous_stack = np.stack(continuous_predictions)
    probability_stack = np.stack(probability_predictions)
    result = frame.loc[:, ["method", "seed", "split"]].copy()
    for index, target in enumerate(CONTINUOUS_TARGETS):
        result[f"pred_{target}"] = continuous_stack[:, :, index].mean(axis=0)
        result[f"std_{target}"] = continuous_stack[:, :, index].std(axis=0)
    for index, target in enumerate(BINARY_TARGETS):
        result[f"prob_{target}"] = probability_stack[:, :, index].mean(axis=0)
        result[f"std_prob_{target}"] = probability_stack[:, :, index].std(
            axis=0
        )
    return result


def prediction_metrics(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    merged = frame.merge(predictions, on=["method", "seed", "split"])
    report: dict[str, Any] = {}
    for split in ("validation", "test"):
        selected = merged[merged["split"] == split]
        values: dict[str, Any] = {"rows": len(selected)}
        for target in CONTINUOUS_TARGETS:
            observed = selected[target].to_numpy(float)
            predicted = selected[f"pred_{target}"].to_numpy(float)
            values[target] = {
                "mae": float(np.mean(np.abs(predicted - observed))),
                "spearman": float(spearmanr(predicted, observed).statistic),
            }
        for target in BINARY_TARGETS:
            observed = selected[target].astype(int).to_numpy()
            predicted = selected[f"prob_{target}"].to_numpy(float)
            values[target] = {
                "positive_rate": float(observed.mean()),
                "auroc": (
                    float(roc_auc_score(observed, predicted))
                    if len(np.unique(observed)) == 2
                    else None
                ),
            }
        report[split] = values
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "/data/dxl/results/postgen_fastgate/features/"
            "historical_features.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/dxl/results/postgen_fastgate/quality_model"),
    )
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.features)
    tensors, metadata = prepare(frame)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    seeds = [4101 + index for index in range(args.members)]
    members = [
        train_member(
            frame=frame,
            tensors=tensors,
            metadata=metadata,
            seed=seed,
            output_dir=args.output_dir / "checkpoints",
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        for seed in seeds
    ]
    predictions = predict_ensemble(
        frame=frame,
        tensors=tensors,
        metadata=metadata,
        members=members,
        device=device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics = prediction_metrics(frame, predictions)
    summary = {
        "schema_version": 1,
        "device": str(device),
        "feature_count": len(metadata["feature_columns"]),
        "train_rows": int((frame["split"] == "train").sum()),
        "validation_rows": int((frame["split"] == "validation").sum()),
        "test_rows": int((frame["split"] == "test").sum()),
        "members": members,
        "prediction_metrics": metrics,
        "mattergen_backbone_trainable": False,
        "chgnet_trainable": False
    }
    atomic_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
