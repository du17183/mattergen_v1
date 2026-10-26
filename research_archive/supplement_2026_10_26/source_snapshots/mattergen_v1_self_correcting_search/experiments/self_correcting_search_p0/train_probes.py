"""Train and gate the pre-registered Linear/Tiny-MLP hidden-state probes."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/self_correcting_search_p0"
TIMESTEPS = (0.50, 0.25, 0.10, 0.05, 0.02)
TARGET = 0.2
BASE_SEED = 20260911


class LinearProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(513, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TinyMLPProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(513, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(fn(x, y).statistic)


def metrics(y: np.ndarray, pred: np.ndarray, train_mean: float) -> dict[str, float]:
    mae = float(np.mean(np.abs(pred - y)))
    baseline_mae = float(np.mean(np.abs(train_mean - y)))
    residual = float(np.sum((pred - y) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "pearson": safe_corr(pearsonr, pred, y),
        "spearman": safe_corr(spearmanr, pred, y),
        "r2": 1.0 - residual / total if total > 1e-15 else float("nan"),
        "target_distance_spearman": safe_corr(
            spearmanr, np.abs(pred - TARGET), np.abs(y - TARGET)
        ),
        "train_mean_baseline_mae": baseline_mae,
        "mae_improvement_fraction": 1.0 - mae / baseline_mae if baseline_mae > 0 else float("nan"),
    }


def load_data(manifest: pd.DataFrame, block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, splits, ts = [], [], [], []
    for row in manifest.itertuples(index=False):
        archive = np.load(PROJECT_ROOT / row.hidden_states_path)
        hidden = archive["hidden"][:, block, :].astype(np.float32)
        actual_t = archive["actual_t"].astype(np.float32)
        if hidden.shape != (5, 512):
            raise RuntimeError(f"bad hidden shape for seed {row.seed}: {hidden.shape}")
        xs.append(hidden)
        ys.extend([float(row.final_chgnet_mag_density)] * 5)
        splits.extend([row.split] * 5)
        ts.extend(actual_t.tolist())
    return np.concatenate(xs), np.asarray(ts, np.float32), np.asarray(ys, np.float32), np.asarray(splits)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_input(hidden: np.ndarray, t: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = (hidden - mean) / std
    return np.concatenate([standardized, t[:, None]], axis=1).astype(np.float32)


def train_one(
    hidden: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    architecture: str,
    seed: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray, int, float]:
    train = splits == "train"
    validation = splits == "validation"
    mean = hidden[train].mean(axis=0)
    std = hidden[train].std(axis=0)
    std[std < 1e-6] = 1.0
    x = make_input(hidden, t, mean, std)
    set_seed(seed)
    model = LinearProbe() if architecture == "linear" else TinyMLPProbe()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_train = torch.from_numpy(x[train])
    y_train = torch.from_numpy(y[train])
    x_val = torch.from_numpy(x[validation])
    y_val = y[validation]
    best_state = None
    best_mae = math.inf
    best_epoch = -1
    stale = 0
    for epoch in range(2000):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train)
        loss = torch.mean((prediction - y_train) ** 2)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prediction = model(x_val).numpy()
        val_mae = float(np.mean(np.abs(val_prediction - y_val)))
        if val_mae < best_mae - 1e-8:
            best_mae = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 150:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, mean.astype(np.float32), std.astype(np.float32), best_epoch, best_mae


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def passes(values: dict[str, float]) -> bool:
    return (
        values["pearson"] >= 0.50
        and values["spearman"] >= 0.50
        and values["mae_improvement_fraction"] >= 0.25
        and values["target_distance_spearman"] >= 0.50
    )


def main() -> None:
    manifest = pd.read_csv(ROOT / "probe_dataset_manifest.csv")
    expected = {"train": 128, "validation": 32, "test": 32}
    actual = manifest.groupby("split").size().to_dict()
    if actual != expected or manifest["seed"].nunique() != 192:
        raise RuntimeError(f"trajectory split contract changed: {actual}")
    train_mean = float(manifest.loc[manifest.split == "train", "final_chgnet_mag_density"].mean())

    candidates = []
    layer_rows = []
    for block in range(4):
        hidden, t, y, splits = load_data(manifest, block)
        for arch_index, architecture in enumerate(("linear", "tiny_mlp")):
            model, mean, std, best_epoch, best_val_mae = train_one(
                hidden, t, y, splits, architecture, BASE_SEED + block * 10 + arch_index
            )
            x = make_input(hidden, t, mean, std)
            with torch.no_grad():
                prediction = model(torch.from_numpy(x)).numpy()
            train_metrics = metrics(y[splits == "train"], prediction[splits == "train"], train_mean)
            val_metrics = metrics(y[splits == "validation"], prediction[splits == "validation"], train_mean)
            row = {"block": block, "architecture": architecture,
                   "parameter_count": sum(p.numel() for p in model.parameters()),
                   "best_epoch": best_epoch}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"validation_{key}": value for key, value in val_metrics.items()})
            layer_rows.append(row)
            candidates.append({"block": block, "architecture": architecture,
                               "model": model, "mean": mean, "std": std,
                               "validation_mae": best_val_mae,
                               "validation_target_distance_spearman": val_metrics["target_distance_spearman"],
                               "parameter_count": row["parameter_count"]})
    write_csv(ROOT / "probe_layer_metrics.csv", layer_rows)
    candidates.sort(key=lambda item: (
        item["validation_mae"],
        -np.nan_to_num(item["validation_target_distance_spearman"], nan=-1.0),
        item["parameter_count"], item["block"],
    ))
    selected = candidates[0]

    hidden, t, y, splits = load_data(manifest, selected["block"])
    x = make_input(hidden, t, selected["mean"], selected["std"])
    with torch.no_grad():
        prediction = selected["model"](torch.from_numpy(x)).numpy()

    timestep_rows = []
    val_pass_by_t = {}
    for split in ("validation", "test"):
        for requested in TIMESTEPS:
            mask = (splits == split) & np.isclose(t, requested, atol=6e-4)
            value = metrics(y[mask], prediction[mask], train_mean)
            timestep_rows.append({"split": split, "requested_timestep": requested,
                                  "n_trajectories": int(mask.sum()), **value,
                                  "passes_all_thresholds": passes(value)})
            if split == "validation":
                val_pass_by_t[requested] = passes(value)

    reliable = []
    for requested in sorted(TIMESTEPS):
        if val_pass_by_t[requested]:
            reliable.append(requested)
        else:
            break
    reliable_max_t = max(reliable) if reliable else None
    in_window = np.zeros_like(t, dtype=bool)
    if reliable_max_t is not None:
        in_window = t <= reliable_max_t + 6e-4
    if reliable_max_t is None:
        # The test set is still reported once for completeness after validation has
        # frozen an empty search window.  These descriptive numbers cannot rescue
        # the gate because ``reliable`` remains empty.
        test_mask = splits == "test"
        test_scope = "all_preregistered_timesteps_descriptive_no_reliable_window"
    else:
        test_mask = (splits == "test") & in_window
        test_scope = "frozen_validation_selected_reliable_window"
    test_metrics = metrics(y[test_mask], prediction[test_mask], train_mean)
    probe_go = bool(reliable and passes(test_metrics))
    write_csv(ROOT / "probe_timestep_metrics.csv", timestep_rows)

    test_rows = []
    test_indices = np.flatnonzero(splits == "test")
    test_manifest = manifest[manifest.split == "test"].reset_index(drop=True)
    for flat_index in test_indices:
        trajectory_index = flat_index // 5 - 160  # train+validation occupy 160 trajectories
        local_t = flat_index % 5
        row = test_manifest.iloc[trajectory_index]
        test_rows.append({
            "trajectory_id": row.trajectory_id,
            "seed": int(row.seed),
            "requested_timestep": TIMESTEPS[local_t],
            "actual_timestep": float(t[flat_index]),
            "true_final_chgnet_mag_density": float(y[flat_index]),
            "predicted_final_chgnet_mag_density": float(prediction[flat_index]),
            "true_target_distance": float(abs(y[flat_index] - TARGET)),
            "predicted_target_distance": float(abs(prediction[flat_index] - TARGET)),
            "in_frozen_reliable_window": bool(in_window[flat_index]),
        })
    write_csv(ROOT / "probe_test_predictions.csv", test_rows)

    checkpoint = {
        "method": "Hidden-State Probe P0 (no gradients, no steering)",
        "architecture": selected["architecture"],
        "block": selected["block"],
        "input_dim": 513,
        "hidden_dim": 512,
        "state_dict": selected["model"].state_dict(),
        "feature_mean": torch.from_numpy(selected["mean"]),
        "feature_std": torch.from_numpy(selected["std"]),
        "requested_timesteps": TIMESTEPS,
        "reliable_timesteps_from_validation": reliable,
        "reliable_max_t": reliable_max_t,
        "target": TARGET,
        "acceptance_temperature": 0.01,
        "training_seed": BASE_SEED,
        "selection_used_test": False,
        "probe_go": probe_go,
    }
    torch.save(checkpoint, ROOT / "probe_checkpoint.pt")
    decision = {
        "PROBE_P0": "GO" if probe_go else "FAIL",
        "SELF_CORRECTING_SEARCH_P0": "PENDING" if probe_go else "NOT_RUN_PROBE_FAIL",
        "NEXT": "RUN_SINGLE_PROPOSAL_SEARCH" if probe_go else "STOP",
        "best_probe_block": selected["block"],
        "best_probe_architecture": selected["architecture"],
        "selection_basis": "validation_only",
        "reliable_timesteps": reliable,
        "reliable_max_t": reliable_max_t,
        "test_scope": test_scope,
        "test_n_rows": int(test_mask.sum()),
        "test_metrics": test_metrics,
        "probe_gradient_used": False,
        "hidden_state_steering_used": False,
        "surrogate_property_eval": True,
        "dft_verified": False,
    }
    (ROOT / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
