"""Train simple Oracle-action classifiers and apply the predictability gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/counterfactual_adaptive_cfg_v3"
ORACLE = ROOT / "oracle"
FEATURES_ROOT = ROOT / "features"
CONTROLLER = ROOT / "controller"
PLOTS = ROOT / "plots"
SPLIT_SEED = 20260918
ACTIONS = ("DOWN", "KEEP", "UP")
THRESHOLDS = (0.50, 0.60, 0.70, 0.80)


FEATURE_COLUMNS = (
    "t_norm",
    "z_atomic", "z_pos", "z_cell",
    "r_atomic", "r_pos", "r_cell",
    "score_ratio_atomic", "score_ratio_pos", "score_ratio_cell",
    "score_alignment_atomic", "score_alignment_pos", "score_alignment_cell",
    "residual_std", "residual_range", "residual_median",
    "log_residual_std", "log_residual_range", "log_residual_median",
    "previous_log_residual", "ema_log_residual", "ema_slope", "previous_cfg",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_split(seeds: np.ndarray) -> dict[str, list[int]]:
    values = np.asarray(sorted(map(int, seeds)), dtype=int)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(values)
    return {
        "train": sorted(map(int, values[:19])),
        "validation": sorted(map(int, values[19:25])),
        "test": sorted(map(int, values[25:])),
    }


def build_features() -> tuple[pd.DataFrame, dict[str, list[int]], dict]:
    raw = pd.read_csv(ORACLE / "online_features_raw.csv").rename(
        columns={
            "r_atomic_numbers": "r_atomic",
            "score_ratio_atomic_numbers": "score_ratio_atomic",
            "score_alignment_atomic_numbers": "score_alignment_atomic",
        }
    )
    labels = pd.read_csv(ORACLE / "short_oracle_labels.csv")
    data = raw.merge(
        labels[["seed", "sampling_step", "oracle_label", "oracle_relative_gain"]],
        on=["seed", "sampling_step"], how="inner",
    )
    if len(data) != 256 or data["seed"].nunique() != 32:
        raise RuntimeError("feature/label join is incomplete")
    split = make_split(data["seed"].unique())
    membership = {
        seed: name for name, seeds in split.items() for seed in seeds
    }
    data["split"] = data["seed"].map(membership)
    normalization: dict[str, dict[str, dict[str, float]]] = {}
    for step in sorted(data["sampling_step"].unique()):
        normalization[str(int(step))] = {}
        train_mask = (data["sampling_step"] == step) & (data["split"] == "train")
        step_mask = data["sampling_step"] == step
        for source, destination in (("r_atomic", "z_atomic"), ("r_pos", "z_pos"), ("r_cell", "z_cell")):
            log_values = np.log(data.loc[train_mask, source].astype(float).to_numpy() + 1e-12)
            mean = float(log_values.mean())
            std = float(log_values.std(ddof=1))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            data.loc[step_mask, destination] = (
                np.log(data.loc[step_mask, source].astype(float) + 1e-12) - mean
            ) / std
            normalization[str(int(step))][source] = {"log_mean": mean, "log_std": std}
    if data[list(FEATURE_COLUMNS)].isna().any().any():
        raise RuntimeError("non-finite controller feature")
    data.to_csv(FEATURES_ROOT / "feature_dataset.csv", index=False)
    schema = {
        "features": list(FEATURE_COLUMNS),
        "label": "oracle_label",
        "normalization": "log residual z-score using training seeds only, separately per decision timestep",
        "split_unit": "seed trajectory",
        "split_seed": SPLIT_SEED,
        "temporal_ema_beta": 0.9,
        "normalization_parameters": normalization,
    }
    (FEATURES_ROOT / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    (CONTROLLER / "split_manifest.json").write_text(
        json.dumps({"split_seed": SPLIT_SEED, "split_unit": "seed", **split}, indent=2) + "\n"
    )
    return data, split, schema


def probability_matrix(model, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features[list(FEATURE_COLUMNS)])
    output = np.zeros((len(features), len(ACTIONS)), dtype=float)
    for source_index, label in enumerate(model.classes_):
        output[:, ACTIONS.index(str(label))] = raw[:, source_index]
    return output


def predictions_at_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    best = np.argmax(probabilities, axis=1)
    labels = np.asarray([ACTIONS[index] for index in best], dtype=object)
    labels[np.max(probabilities, axis=1) < threshold] = "KEEP"
    return labels


def regret(data: pd.DataFrame, candidate: pd.DataFrame, predictions: np.ndarray) -> dict:
    state = data[["seed", "sampling_step", "oracle_label"]].copy()
    state["prediction"] = predictions
    objectives = candidate.pivot_table(
        index=["seed", "sampling_step"], columns="candidate_action",
        values="oracle_objective_short", aggfunc="first",
    ).reset_index()
    merged = state.merge(objectives, on=["seed", "sampling_step"], how="inner")
    oracle_objective = np.asarray(
        [row[getattr(row, "oracle_label")] for row in merged.itertuples()], dtype=float
    )
    selected_objective = np.asarray(
        [row[getattr(row, "prediction")] for row in merged.itertuples()], dtype=float
    )
    keep_objective = merged["KEEP"].to_numpy(dtype=float)
    fixed_regret = keep_objective - oracle_objective
    selected_regret = selected_objective - oracle_objective
    fixed_mean = float(fixed_regret.mean())
    selected_mean = float(selected_regret.mean())
    harmful = (
        (merged["prediction"].to_numpy() != "KEEP")
        & (selected_objective > keep_objective * 1.05)
    )
    return {
        "n": len(merged),
        "fixed_cfg2_regret_mean": fixed_mean,
        "controller_regret_mean": selected_mean,
        "regret_reduction_fraction": (fixed_mean - selected_mean) / max(abs(fixed_mean), 1e-12),
        "wrong_non_keep_property_harm_rate": float(harmful.mean()),
    }


def evaluate_thresholds(model, data: pd.DataFrame, candidate: pd.DataFrame) -> list[dict]:
    probabilities = probability_matrix(model, data)
    rows = []
    for threshold in THRESHOLDS:
        predictions = predictions_at_threshold(probabilities, threshold)
        result = regret(data, candidate, predictions)
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": float(balanced_accuracy_score(data["oracle_label"], predictions)),
                "macro_f1": float(f1_score(data["oracle_label"], predictions, labels=ACTIONS, average="macro", zero_division=0)),
                **result,
            }
        )
    return rows


def chosen_threshold(rows: list[dict]) -> dict:
    return sorted(rows, key=lambda row: (row["controller_regret_mean"], -row["threshold"]))[0]


def main() -> None:
    headroom = json.loads((ORACLE / "oracle_summary.json").read_text())
    if headroom["ORACLE_HEADROOM"] != "GO":
        raise RuntimeError("predictability may only run after ORACLE_HEADROOM=GO")
    data, split, schema = build_features()
    candidate = pd.read_csv(ORACLE / "oracle_dataset.csv")
    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()
    majority_label = str(train["oracle_label"].value_counts().index[0])
    majority_predictions = np.full(len(test), majority_label, dtype=object)
    majority_result = {
        "label": majority_label,
        "balanced_accuracy": float(balanced_accuracy_score(test["oracle_label"], majority_predictions)),
        "macro_f1": float(f1_score(test["oracle_label"], majority_predictions, labels=ACTIONS, average="macro", zero_division=0)),
        **regret(test, candidate, majority_predictions),
    }

    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs", class_weight="balanced",
                max_iter=2000, random_state=SPLIT_SEED,
            )),
        ]
    )
    logistic.fit(train[list(FEATURE_COLUMNS)], train["oracle_label"])
    logistic_validation = evaluate_thresholds(logistic, validation, candidate)
    logistic_choice = chosen_threshold(logistic_validation)
    logistic_result = {
        "validation_thresholds": logistic_validation,
        "selected_validation": logistic_choice,
    }
    (CONTROLLER / "logistic_results.json").write_text(json.dumps(logistic_result, indent=2) + "\n")

    train_mlp = (
        logistic_choice["regret_reduction_fraction"] < 0.30
        or logistic_choice["balanced_accuracy"] <= 1.0 / 3.0
    )
    candidates = [("logistic", logistic, logistic_choice)]
    mlp_result = {"trained": False, "reason": "logistic validation gate not clearly insufficient"}
    if train_mlp:
        mlp = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", MLPClassifier(
                    hidden_layer_sizes=(16,), alpha=0.001, max_iter=1000,
                    random_state=SPLIT_SEED, early_stopping=False,
                )),
            ]
        )
        mlp.fit(train[list(FEATURE_COLUMNS)], train["oracle_label"])
        hidden_parameters = (len(FEATURE_COLUMNS) + 1) * 16 + (16 + 1) * len(mlp.classes_)
        if hidden_parameters >= 1000:
            raise RuntimeError(f"tiny MLP parameter budget exceeded: {hidden_parameters}")
        mlp_validation = evaluate_thresholds(mlp, validation, candidate)
        mlp_choice = chosen_threshold(mlp_validation)
        mlp_result = {
            "trained": True,
            "parameter_count": hidden_parameters,
            "validation_thresholds": mlp_validation,
            "selected_validation": mlp_choice,
        }
        candidates.append(("tiny_mlp", mlp, mlp_choice))
    (CONTROLLER / "mlp_results.json").write_text(json.dumps(mlp_result, indent=2) + "\n")
    selected_name, selected_model, selected_validation = sorted(
        candidates,
        key=lambda item: (
            item[2]["controller_regret_mean"],
            0 if item[0] == "logistic" else 1,
        ),
    )[0]
    threshold = float(selected_validation["threshold"])
    test_probabilities = probability_matrix(selected_model, test)
    test_predictions = predictions_at_threshold(test_probabilities, threshold)
    test_result = {
        "model": selected_name,
        "confidence_threshold": threshold,
        "balanced_accuracy": float(balanced_accuracy_score(test["oracle_label"], test_predictions)),
        "macro_f1": float(f1_score(test["oracle_label"], test_predictions, labels=ACTIONS, average="macro", zero_division=0)),
        **regret(test, candidate, test_predictions),
    }
    matrix = confusion_matrix(test["oracle_label"], test_predictions, labels=ACTIONS)
    confusion_rows = []
    for actual_index, actual in enumerate(ACTIONS):
        for predicted_index, predicted in enumerate(ACTIONS):
            confusion_rows.append({"actual": actual, "predicted": predicted, "count": int(matrix[actual_index, predicted_index])})
    write_csv(CONTROLLER / "confusion_matrix.csv", confusion_rows)
    regret_rows = [
        {"model": "majority", "split": "test", **majority_result},
        {"model": selected_name, "split": "test", **test_result},
    ]
    write_csv(CONTROLLER / "regret_analysis.csv", regret_rows)

    checks = {
        "counterfactual_regret_reduction_ge_30pct": test_result["regret_reduction_fraction"] >= 0.30,
        "balanced_accuracy_above_majority": test_result["balanced_accuracy"] > majority_result["balanced_accuracy"],
        "wrong_non_keep_property_harm_rate_le_10pct": test_result["wrong_non_keep_property_harm_rate"] <= 0.10,
    }
    decision = "GO" if all(checks.values()) else "FAIL"
    payload = {
        "ORACLE_PREDICTABILITY": decision,
        "next": "IMPLEMENT_FROZEN_V3_AND_FRESH_P0" if decision == "GO" else "STOP",
        "selected_model": selected_name,
        "confidence_threshold": threshold,
        "majority_test": majority_result,
        "controller_test": test_result,
        "gate_checks": checks,
        "trajectory_disjoint_split": split,
        "parameters_retuned_after_test": False,
    }
    (CONTROLLER / "decision_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    joblib.dump(selected_model, CONTROLLER / "selected_model.joblib")
    (CONTROLLER / "selected_model_metadata.json").write_text(
        json.dumps({
            "model": selected_name,
            "confidence_threshold": threshold,
            "features": list(FEATURE_COLUMNS),
            "classes": list(map(str, selected_model.classes_)),
        }, indent=2) + "\n"
    )

    classifier = selected_model.named_steps["classifier"]
    if selected_name == "logistic":
        importance = np.mean(np.abs(classifier.coef_), axis=0)
    else:
        importance = np.mean(np.abs(classifier.coefs_[0]), axis=1)
    order = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.asarray(FEATURE_COLUMNS)[order][:15][::-1], importance[order][:15][::-1])
    ax.set_xlabel("Mean absolute model weight")
    fig.tight_layout()
    fig.savefig(PLOTS / "feature_importance.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), ACTIONS)
    ax.set_yticks(range(3), ACTIONS)
    ax.set(xlabel="Predicted", ylabel="Actual")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "controller_confusion_matrix.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(["CFG2", selected_name], [test_result["fixed_cfg2_regret_mean"], test_result["controller_regret_mean"]])
    ax.set_ylabel("Mean counterfactual regret")
    fig.tight_layout()
    fig.savefig(PLOTS / "controller_regret.png", dpi=180)
    plt.close(fig)

    if decision == "FAIL":
        not_run = {
            "V3_P0": "NOT_RUN",
            "V3_FORMAL256": "NOT_RUN",
            "reason": "ORACLE_PREDICTABILITY=FAIL; frozen stop rule applied",
            "rescue_tuning": False,
        }
        (ROOT / "p0/decision_summary.json").write_text(json.dumps(not_run, indent=2) + "\n")
        final = {
            "ORACLE_HEADROOM": "GO",
            "ORACLE_PREDICTABILITY": "FAIL",
            "V3_P0": "NOT_RUN",
            "V3_FORMAL256": "NOT_RUN",
            "INNOVATION1_FINAL_STATUS": "MIXED",
            "parameters_retuned_after_results": False,
            "all_data_retained": True,
            "surrogate_property_eval": True,
            "dft_verified": False,
        }
        (ROOT / "final_status.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
