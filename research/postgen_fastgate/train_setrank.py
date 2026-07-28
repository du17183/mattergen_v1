#!/usr/bin/env python3
"""Train and evaluate the pool-aware novelty-stability SetRank network."""

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
from torch import nn

from research.postgen_fastgate.setrank import SetRankNetwork
from research.postgen_fastgate.train_quality import (
    prepare,
    seed_everything,
)

OUTCOMES = (
    "energy_above_hull_per_atom",
    "rmsd_from_relaxation",
    "stable",
    "novel_unique_stable",
    "comp_validity",
    "structure_validity",
    "novel",
    "unique",
    "converged",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def true_utility(frame: pd.DataFrame) -> np.ndarray:
    ehull = np.log1p(
        np.maximum(frame["energy_above_hull_per_atom"].to_numpy(float), 0.0)
        / 0.05
    )
    rmsd = np.log1p(
        np.maximum(frame["rmsd_from_relaxation"].to_numpy(float), 0.0)
        / 0.01
    )
    return (
        5.0 * frame["novel_unique_stable"].to_numpy(float)
        + 2.0 * frame["stable"].to_numpy(float)
        + 3.0 * frame["novel"].to_numpy(float)
        + 1.0 * frame["unique"].to_numpy(float)
        + 1.0 * frame["comp_validity"].to_numpy(float)
        - 0.5 * ehull
        - 0.1 * rmsd
    )


def make_pools(
    indexes: np.ndarray,
    utility: np.ndarray,
    *,
    count: int,
    pool_size: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    pools = np.empty((count, pool_size), dtype=np.int64)
    targets = np.empty(count, dtype=np.int64)
    for row in range(count):
        selected = rng.choice(indexes, size=pool_size, replace=False)
        pools[row] = selected
        targets[row] = int(np.argmax(utility[selected]))
    return pools, targets


def train_member(
    *,
    features: torch.Tensor,
    utility: np.ndarray,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    output_dir: Path,
    seed: int,
    device: torch.device,
    pool_size: int,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    seed_everything(seed)
    train_pools, train_targets = make_pools(
        train_indexes,
        utility,
        count=12000,
        pool_size=pool_size,
        random_seed=seed,
    )
    validation_pools, validation_targets = make_pools(
        validation_indexes,
        utility,
        count=3000,
        pool_size=pool_size,
        random_seed=seed + 10000,
    )
    model = SetRankNetwork(
        input_dim=features.shape[1],
        hidden_dim=96,
        dropout=0.10,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=1.0e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )
    feature_device = features.to(device)
    train_targets_tensor = torch.tensor(
        train_targets, dtype=torch.long, device=device
    )
    validation_targets_tensor = torch.tensor(
        validation_targets, dtype=torch.long, device=device
    )
    batch_size = 256
    best_loss = math.inf
    best_accuracy = 0.0
    best_epoch = -1
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None
    rng = np.random.default_rng(seed + 20000)
    history: list[dict[str, float | int]] = []
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(train_pools))
        train_loss_sum = 0.0
        train_count = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            pool_index = torch.tensor(
                train_pools[selected], dtype=torch.long, device=device
            )
            target = train_targets_tensor[
                torch.tensor(selected, dtype=torch.long, device=device)
            ]
            optimizer.zero_grad(set_to_none=True)
            score = model(feature_device[pool_index])
            loss = nn.functional.cross_entropy(score, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(selected)
            train_count += len(selected)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_index = torch.tensor(
                validation_pools, dtype=torch.long, device=device
            )
            validation_score = model(feature_device[validation_index])
            validation_loss = nn.functional.cross_entropy(
                validation_score, validation_targets_tensor
            )
            validation_accuracy = float(
                (
                    validation_score.argmax(dim=1)
                    == validation_targets_tensor
                )
                .float()
                .mean()
            )
        value = float(validation_loss)
        if epoch % 5 == 0 or value < best_loss:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_sum / train_count,
                    "validation_loss": value,
                    "validation_accuracy": validation_accuracy,
                }
            )
        if value < best_loss - 1.0e-5:
            best_loss = value
            best_accuracy = validation_accuracy
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
        raise RuntimeError("SetRank produced no checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"setrank_member_{seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": int(features.shape[1]),
            "hidden_dim": 96,
            "dropout": 0.10,
            "seed": seed,
        },
        checkpoint,
    )
    return {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_validation_accuracy": best_accuracy,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "history": history,
    }


def load_ensemble(
    members: list[dict[str, Any]],
    device: torch.device,
) -> list[SetRankNetwork]:
    models = []
    for member in members:
        payload = torch.load(
            member["checkpoint"], map_location=device, weights_only=True
        )
        model = SetRankNetwork(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload["dropout"]),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    return models


def aggregate(selected: pd.DataFrame) -> dict[str, float]:
    return {
        column: float(selected[column].astype(float).mean())
        for column in OUTCOMES
    }


def evaluate_c0_test(
    *,
    frame: pd.DataFrame,
    features: torch.Tensor,
    models: list[SetRankNetwork],
    device: torch.device,
    pool_size: int,
    trials: int,
    random_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    eligible = np.flatnonzero(
        ((frame["method"] == "C0") & (frame["split"] == "test")).to_numpy()
    )
    rng = np.random.default_rng(random_seed)
    feature_device = features.to(device)
    rows = []
    for trial in range(trials):
        permutation = rng.permutation(eligible)
        pools = permutation.reshape(-1, pool_size)
        baseline_indexes = pools[:, 0]
        pool_tensor = torch.tensor(pools, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = np.stack(
                [model(feature_device[pool_tensor]).cpu().numpy() for model in models]
            )
        conservative_score = scores.mean(axis=0) - 0.25 * scores.std(axis=0)
        selected_indexes = pools[
            np.arange(len(pools)), conservative_score.argmax(axis=1)
        ]
        baseline = aggregate(frame.iloc[baseline_indexes])
        selected = aggregate(frame.iloc[selected_indexes])
        rows.append(
            {
                "trial": trial,
                **{f"baseline_{key}": value for key, value in baseline.items()},
                **{f"selected_{key}": value for key, value in selected.items()},
                "ehull_change": (
                    selected["energy_above_hull_per_atom"]
                    - baseline["energy_above_hull_per_atom"]
                ),
                "rmsd_relative_change": (
                    selected["rmsd_from_relaxation"]
                    / baseline["rmsd_from_relaxation"]
                    - 1.0
                ),
                "stable_change": selected["stable"] - baseline["stable"],
                "nus_change": (
                    selected["novel_unique_stable"]
                    - baseline["novel_unique_stable"]
                ),
                "composition_change": (
                    selected["comp_validity"] - baseline["comp_validity"]
                ),
                "structure_change": (
                    selected["structure_validity"]
                    - baseline["structure_validity"]
                ),
                "novel_change": selected["novel"] - baseline["novel"],
                "unique_change": selected["unique"] - baseline["unique"],
            }
        )
    details = pd.DataFrame(rows)
    columns = (
        "ehull_change",
        "rmsd_relative_change",
        "stable_change",
        "nus_change",
        "composition_change",
        "structure_change",
        "novel_change",
        "unique_change",
    )
    summary = {
        column: {
            "mean": float(details[column].mean()),
            "median": float(details[column].median()),
            "q025": float(details[column].quantile(0.025)),
            "q975": float(details[column].quantile(0.975)),
        }
        for column in columns
    }
    safety = (
        summary["structure_change"]["mean"] >= 0.0
        and summary["composition_change"]["mean"] >= -0.03125
        and summary["stable_change"]["mean"] >= -0.03125
        and summary["nus_change"]["mean"] >= -0.03125
        and summary["novel_change"]["mean"] >= -0.02
        and summary["unique_change"]["mean"] >= -0.02
    )
    positive = (
        summary["ehull_change"]["mean"] <= -0.005
        or summary["stable_change"]["mean"] >= 0.03125
        or summary["nus_change"]["mean"] >= 0.03125
        or summary["rmsd_relative_change"]["mean"] <= -0.10
    )
    return {
        "rows": len(eligible),
        "pools_per_trial": len(eligible) // pool_size,
        "trials": trials,
        "summary": summary,
        "safety_gate": bool(safety),
        "positive_gate": bool(positive),
        "Q6_NS_SETRANK_LEARNED_OFFLINE_GO": bool(safety and positive),
    }, details


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
        default=Path("/data/dxl/results/postgen_fastgate/setrank"),
    )
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.features)
    tensors, metadata = prepare(frame)
    features = tensors["features"]
    utility = true_utility(frame)
    train_indexes = np.flatnonzero((frame["split"] == "train").to_numpy())
    validation_indexes = np.flatnonzero(
        (frame["split"] == "validation").to_numpy()
    )
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    members = [
        train_member(
            features=features,
            utility=utility,
            train_indexes=train_indexes,
            validation_indexes=validation_indexes,
            output_dir=args.output_dir / "checkpoints",
            seed=5101 + index,
            device=device,
            pool_size=args.pool_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        for index in range(args.members)
    ]
    models = load_ensemble(members, device)
    evaluation, details = evaluate_c0_test(
        frame=frame,
        features=features,
        models=models,
        device=device,
        pool_size=args.pool_size,
        trials=args.trials,
        random_seed=20260728,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_dir / "heldout_trials.csv", index=False)
    summary = {
        "schema_version": 1,
        "device": str(device),
        "feature_count": len(metadata["feature_columns"]),
        "pool_size": args.pool_size,
        "members": members,
        "evaluation": evaluation,
        "mattergen_backbone_trainable": False,
        "chgnet_trainable": False,
    }
    atomic_json(args.output_dir / "setrank_summary.json", summary)
    print(
        json.dumps(
            {
                "Q6_NS_SETRANK_LEARNED_OFFLINE_GO": evaluation[
                    "Q6_NS_SETRANK_LEARNED_OFFLINE_GO"
                ],
                "summary": evaluation["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
