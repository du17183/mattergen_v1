"""Fit, calibrate, select once on Validation, and gate the untouched Test split."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from itertools import product
from pathlib import Path

from ase.io import read
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yaml


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/calibrated_safe_selection_cfg"
DATASET = ROOT / "dataset"
SPLITS = ROOT / "splits"
MODELS = ROOT / "models"
CALIBRATION = ROOT / "calibration"
VALIDATION = ROOT / "validation"
TEST = ROOT / "test"
IMPLEMENTATION = ROOT / "implementation"
PLOTS = ROOT / "plots"
ACTIONS = ("DOWN", "KEEP", "UP")
NON_KEEP = ("DOWN", "UP")
TIE_ORDER = {"KEEP": 0, "DOWN": 1, "UP": 2}
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CONFORMAL_ALPHAS = (0.10, 0.15, 0.20)
DELTA_SAFE = (0.0, 0.0025, 0.005, 0.01)
ETA = (0.05, 0.10, 0.15)
MIN_COVERAGE = 0.15
MAX_SELECTIVE_HARM = 0.10
MIN_ORACLE_RECOVERY = 0.20
MIN_PRECISION = 0.85

RAW_FEATURE_COLUMNS = (
    "t_norm",
    "r_atomic", "r_pos", "r_cell",
    "score_ratio_atomic", "score_ratio_pos", "score_ratio_cell",
    "score_alignment_atomic", "score_alignment_pos", "score_alignment_cell",
    "residual_mean", "residual_median", "residual_std", "residual_range",
    "ratio_mean", "ratio_median", "ratio_std", "ratio_range",
    "alignment_mean", "alignment_median", "alignment_std", "alignment_range",
    "log_residual_mean", "log_residual_median", "log_residual_std", "log_residual_range",
    "previous_log_residual", "ema_log_residual", "ema_slope",
    "recent_residual_trend", "previous_cfg",
    "x0_structural_valid", "x0_cell_volume", "x0_density_g_cm3", "x0_num_atoms",
    "x0_num_elements", "x0_mean_atomic_number", "x0_std_atomic_number",
    "x0_max_element_fraction", "x0_element_entropy",
    "x0_min_periodic_distance", "x0_mean_periodic_distance",
)
FEATURE_COLUMNS = (
    "t_norm", "z_atomic", "z_pos", "z_cell", *RAW_FEATURE_COLUMNS[1:]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")


def quality_enriched_dataset() -> pd.DataFrame:
    values = pd.read_csv(DATASET / "property_metrics.csv")
    for column, default in (
        ("e_hull", np.nan), ("stable", False), ("nus", False), ("novel", False),
        ("unique", False), ("rmsd", np.nan), ("pre_relaxation_max_force", np.nan),
        ("relaxation_steps", np.nan), ("quality_success", False),
    ):
        values[column] = default
    for action in ACTIONS:
        group = f"STEP250_{action}"
        structure_path = DATASET / "evaluation_structures" / f"{group}.extxyz"
        quality_root = DATASET / "quality" / group
        if not structure_path.exists():
            continue
        if not (quality_root / "official_detailed.json.gz").exists():
            raise FileNotFoundError(f"quality result missing: {group}")
        atoms = read(structure_path, index=":")
        per_structure = pd.read_csv(quality_root / "per_structure.csv")
        with gzip.open(quality_root / "official_detailed.json.gz", "rt") as stream:
            detailed = json.load(stream)
        if len(per_structure) != len(detailed["energy_above_hull_per_atom"]):
            raise RuntimeError(f"quality length mismatch: {group}")
        for output_index, row in per_structure.reset_index(drop=True).iterrows():
            identifier = str(atoms[int(row["index"])].info["branch_id"])
            match = values.index[values["branch_id"] == identifier]
            if len(match) != 1:
                raise RuntimeError(f"branch lookup failed: {identifier}")
            target = match[0]
            values.loc[target, "e_hull"] = float(detailed["energy_above_hull_per_atom"][output_index])
            values.loc[target, "stable"] = bool(detailed["stable"][output_index])
            values.loc[target, "nus"] = bool(detailed["novel_unique_stable"][output_index])
            values.loc[target, "novel"] = bool(detailed["novel"][output_index])
            values.loc[target, "unique"] = bool(detailed["unique"][output_index])
            values.loc[target, "rmsd"] = float(detailed["rmsd_from_relaxation"][output_index])
            values.loc[target, "pre_relaxation_max_force"] = float(row["pre_relaxation_max_force"])
            values.loc[target, "relaxation_steps"] = int(row["relaxation_steps"])
            values.loc[target, "quality_success"] = True
    for column in ("stable", "nus", "novel", "unique", "quality_success"):
        values[column] = values[column].fillna(False).astype(bool)
    values["evaluation_valid"] = values["structure_valid"].astype(bool) & values["quality_success"]
    values.to_csv(DATASET / "counterfactual_dataset.csv", index=False)
    return values


def build_treatment_dataset(counterfactual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = (
        "property_absolute_error", "e_hull", "stable", "nus", "novel", "unique",
        "evaluation_valid",
    )
    for seed, state in counterfactual.groupby("seed", sort=False):
        if len(state) != 3 or set(state["candidate_action"]) != set(ACTIONS):
            raise RuntimeError(f"candidate mismatch: seed={seed}")
        selected = {action: state[state["candidate_action"] == action].iloc[0] for action in ACTIONS}
        base = selected["KEEP"]
        row: dict = {
            "seed": int(seed), "sampling_step": 250,
            "decision_t_norm": float(base["decision_t_norm"]),
            "state_sha256": str(base["state_sha256"]),
        }
        for action in ACTIONS:
            for metric in metrics:
                row[f"{metric}_{action}"] = selected[action][metric]
        base_error = float(row["property_absolute_error_KEEP"])
        for action in NON_KEEP:
            delta = float(row[f"property_absolute_error_{action}"]) - base_error
            row[f"delta_property_{action}"] = delta
            row[f"harm_property_{action}"] = bool(
                float(row[f"property_absolute_error_{action}"]) > 1.05 * base_error
            )
            action_ehull = float(row[f"e_hull_{action}"])
            base_ehull = float(row["e_hull_KEEP"])
            row[f"delta_ehull_{action}"] = action_ehull - base_ehull
            row[f"harm_ehull_{action}"] = bool(
                np.isfinite(action_ehull) and np.isfinite(base_ehull)
                and action_ehull - base_ehull > 0.02
            )
            row[f"harm_stable_{action}"] = bool(
                row["stable_KEEP"] and not row[f"stable_{action}"]
            )
        ranked = sorted(
            ACTIONS,
            key=lambda action: (float(row[f"property_absolute_error_{action}"]), TIE_ORDER[action]),
        )
        row["oracle_action"] = ranked[0]
        row["oracle_gain"] = base_error - float(row[f"property_absolute_error_{ranked[0]}"])
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 128 or result["seed"].nunique() != 128:
        raise RuntimeError("treatment-effect dataset is incomplete")
    result.to_csv(DATASET / "treatment_effect_dataset.csv", index=False)
    return result


def merge_features(treatment: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    raw = pd.read_csv(DATASET / "online_features_raw.csv").rename(
        columns={
            "r_atomic_numbers": "r_atomic",
            "score_ratio_atomic_numbers": "score_ratio_atomic",
            "score_alignment_atomic_numbers": "score_alignment_atomic",
        }
    )
    data = raw.merge(treatment, on=["seed", "sampling_step"], how="inner", validate="one_to_one")
    split = json.loads((SPLITS / "split_manifest.json").read_text())
    membership = {int(seed): name for name in ("train", "calibration", "validation", "test") for seed in split[name]}
    data["split"] = data["seed"].map(membership)
    if data["split"].isna().any():
        raise RuntimeError("unassigned dataset seed")
    train = data[data["split"] == "train"]
    normalizers: dict[str, dict[str, float]] = {}
    for source, destination in (("r_atomic", "z_atomic"), ("r_pos", "z_pos"), ("r_cell", "z_cell")):
        values = np.log(train[source].astype(float).to_numpy() + 1e-12)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        data[destination] = (np.log(data[source].astype(float) + 1e-12) - mean) / std
        normalizers[source] = {"log_mean": mean, "log_std": std}
    numeric = data[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        bad = numeric.columns[~np.isfinite(numeric.to_numpy()).all(axis=0)].tolist()
        raise RuntimeError(f"non-finite online features: {bad}")
    schema = {
        "features": list(FEATURE_COLUMNS), "online_only": True,
        "residual_normalization": "train-only log z-score",
        "normalization_parameters": normalizers,
        "temporal_ema_beta": 0.9, "recent_window": 5,
        "future_information_prohibited": True,
    }
    json_write(MODELS / "feature_schema.json", schema)
    return data, split, schema


def select_ridge_alpha(x: np.ndarray, y: np.ndarray, action_offset: int) -> tuple[float, list[dict]]:
    folds = KFold(n_splits=5, shuffle=True, random_state=20261001 + action_offset)
    rows = []
    for alpha_value in RIDGE_ALPHAS:
        errors = []
        for train_index, valid_index in folds.split(x):
            model = Pipeline([
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha_value)),
            ])
            model.fit(x[train_index], y[train_index])
            errors.append(mean_squared_error(y[valid_index], model.predict(x[valid_index])))
        rows.append({"alpha": alpha_value, "mean_inner_cv_mse": float(np.mean(errors))})
    chosen = min(rows, key=lambda row: (row["mean_inner_cv_mse"], -row["alpha"]))
    return float(chosen["alpha"]), rows


def positive_probability(model, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(map(int, model.classes_))
    return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(x), dtype=float)


def fit_models(data: pd.DataFrame) -> tuple[dict, dict]:
    train = data[data["split"] == "train"].reset_index(drop=True)
    calibration = data[data["split"] == "calibration"].reset_index(drop=True)
    x_train_raw = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    x_cal_raw = calibration[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_cal = scaler.transform(x_cal_raw)
    models: dict = {"scaler": scaler, "ridge": {}, "harm": {}, "platt": {}}
    fit_report: dict = {"ridge_inner_cv": {}, "harm": {}, "platt": {}}
    conformal_rows: list[dict] = []
    quantiles: dict[str, dict[str, float]] = {}
    for offset, action in enumerate(NON_KEEP):
        target = train[f"delta_property_{action}"].to_numpy(dtype=float)
        alpha_value, cv_rows = select_ridge_alpha(x_train_raw, target, offset)
        ridge_model = Ridge(alpha=alpha_value).fit(x_train, target)
        models["ridge"][action] = ridge_model
        fit_report["ridge_inner_cv"][action] = {"selected_alpha": alpha_value, "candidates": cv_rows}
        labels = train[f"harm_property_{action}"].astype(int).to_numpy()
        if len(np.unique(labels)) < 2:
            harm_model = DummyClassifier(strategy="prior").fit(x_train, labels)
            harm_mode = "one_class_dummy"
        else:
            harm_model = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=5000,
                random_state=20261003 + offset,
            ).fit(x_train, labels)
            harm_mode = "logistic_balanced"
        models["harm"][action] = harm_model
        cal_labels = calibration[f"harm_property_{action}"].astype(int).to_numpy()
        if hasattr(harm_model, "decision_function"):
            raw_score = np.asarray(harm_model.decision_function(x_cal), dtype=float).reshape(-1, 1)
        else:
            raw_probability = np.clip(positive_probability(harm_model, x_cal), 1e-8, 1 - 1e-8)
            raw_score = np.log(raw_probability / (1 - raw_probability)).reshape(-1, 1)
        if len(np.unique(cal_labels)) >= 2:
            platt = LogisticRegression(C=1.0, max_iter=5000, random_state=20261005 + offset)
            platt.fit(raw_score, cal_labels)
            platt_record = {
                "mode": "platt_logistic", "positive_count": int(cal_labels.sum()),
                "n": len(cal_labels), "coefficient": float(platt.coef_[0, 0]),
                "intercept": float(platt.intercept_[0]),
            }
        else:
            probability = float((cal_labels.sum() + 1) / (len(cal_labels) + 2))
            platt = {"mode": "constant", "probability": probability}
            platt_record = {
                "mode": "laplace_smoothed_constant", "positive_count": int(cal_labels.sum()),
                "n": len(cal_labels), "probability": probability,
            }
        models["platt"][action] = platt
        fit_report["harm"][action] = {
            "mode": harm_mode, "positive_count": int(labels.sum()), "n": len(labels)
        }
        fit_report["platt"][action] = platt_record
        predictions = ridge_model.predict(x_cal)
        actual = calibration[f"delta_property_{action}"].to_numpy(dtype=float)
        residuals = np.abs(actual - predictions)
        quantiles[action] = {}
        for alpha_candidate in CONFORMAL_ALPHAS:
            rank = min(len(residuals), int(math.ceil((len(residuals) + 1) * (1 - alpha_candidate))))
            q = float(np.sort(residuals)[rank - 1])
            quantiles[action][str(alpha_candidate)] = q
        conformal_rows.extend(
            {
                "seed": int(seed), "action": action, "true_delta": float(truth),
                "predicted_delta": float(prediction), "absolute_residual": float(residual),
            }
            for seed, truth, prediction, residual in zip(
                calibration["seed"], actual, predictions, residuals
            )
        )
    models["feature_columns"] = list(FEATURE_COLUMNS)
    models["normalization_parameters"] = json.loads((MODELS / "feature_schema.json").read_text())["normalization_parameters"]
    models["conformal_quantiles"] = quantiles
    pd.DataFrame(conformal_rows).to_csv(CALIBRATION / "conformal_residuals.csv", index=False)
    json_write(CALIBRATION / "conformal_quantiles.json", quantiles)
    json_write(CALIBRATION / "harm_probability_calibration.json", fit_report)
    return models, fit_report


def calibrated_harm(models: dict, action: str, x_scaled: np.ndarray) -> np.ndarray:
    harm_model = models["harm"][action]
    if hasattr(harm_model, "decision_function"):
        raw_score = np.asarray(harm_model.decision_function(x_scaled), dtype=float).reshape(-1, 1)
    else:
        raw_probability = np.clip(positive_probability(harm_model, x_scaled), 1e-8, 1 - 1e-8)
        raw_score = np.log(raw_probability / (1 - raw_probability)).reshape(-1, 1)
    platt = models["platt"][action]
    if isinstance(platt, dict):
        return np.full(len(x_scaled), float(platt["probability"]), dtype=float)
    return positive_probability(platt, raw_score)


def predict(models: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    x = models["scaler"].transform(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
    output: dict[str, np.ndarray] = {}
    for action in NON_KEEP:
        output[f"mean_{action}"] = models["ridge"][action].predict(x)
        output[f"harm_{action}"] = calibrated_harm(models, action, x)
    return output


def safe_decisions(prediction: dict, models: dict, alpha_value: float, delta_safe: float, eta: float) -> np.ndarray:
    chosen: list[str] = []
    for index in range(len(prediction["mean_DOWN"])):
        eligible: list[tuple[float, int, str]] = []
        for action in NON_KEEP:
            upper = float(prediction[f"mean_{action}"][index]) + float(
                models["conformal_quantiles"][action][str(alpha_value)]
            )
            if upper < -delta_safe and float(prediction[f"harm_{action}"][index]) < eta:
                eligible.append((upper, TIE_ORDER[action], action))
        chosen.append(min(eligible)[2] if eligible else "KEEP")
    return np.asarray(chosen, dtype=object)


def r0_decisions(prediction: dict) -> np.ndarray:
    result = []
    for index in range(len(prediction["mean_DOWN"])):
        candidates = [(float(prediction[f"mean_{action}"][index]), TIE_ORDER[action], action) for action in NON_KEEP]
        best = min(candidates)
        result.append(best[2] if best[0] < 0 else "KEEP")
    return np.asarray(result, dtype=object)


def policy_metrics(frame: pd.DataFrame, actions: np.ndarray) -> dict:
    delta = np.asarray([
        0.0 if action == "KEEP" else float(row[f"delta_property_{action}"])
        for action, (_, row) in zip(actions, frame.iterrows())
    ])
    harm = np.asarray([
        False if action == "KEEP" else bool(row[f"harm_property_{action}"])
        for action, (_, row) in zip(actions, frame.iterrows())
    ])
    adapted = actions != "KEEP"
    base = frame["property_absolute_error_KEEP"].to_numpy(dtype=float)
    chosen_error = base + delta
    oracle_gain = float(frame["oracle_gain"].mean())
    overall_gain = float(-delta.mean())
    wins = int((delta[adapted] < 0).sum())
    losses = int((delta[adapted] > 0).sum())
    ties = int((delta[adapted] == 0).sum())
    selective_harm = float(harm[adapted].mean()) if adapted.any() else 1.0
    precision = float((delta[adapted] < 0).mean()) if adapted.any() else 0.0
    selected_quality: dict[str, float] = {}
    for metric in ("e_hull", "stable", "nus", "novel", "unique", "evaluation_valid"):
        values = np.asarray([
            row[f"{metric}_{action}"] for action, (_, row) in zip(actions, frame.iterrows())
        ], dtype=float)
        selected_quality[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return {
        "n": len(frame), "adaptive_count": int(adapted.sum()),
        "coverage": float(adapted.mean()), "selective_property_harm_rate": selective_harm,
        "overall_property_harm_rate": float(harm.mean()), "adaptive_precision": precision,
        "adaptive_subset_mean_delta": float(delta[adapted].mean()) if adapted.any() else float("nan"),
        "adaptive_subset_gain": float(-delta[adapted].mean()) if adapted.any() else 0.0,
        "overall_gain": overall_gain, "oracle_gain": oracle_gain,
        "oracle_recovery": overall_gain / max(oracle_gain, 1e-12),
        "property_mae": float(chosen_error.mean()), "c0_property_mae": float(base.mean()),
        "relative_property_mae_improvement": overall_gain / max(float(base.mean()), 1e-12),
        "wins": wins, "ties": ties, "losses": losses,
        "false_positive_adaptations": int((adapted & (delta >= 0)).sum()),
        "false_negative_missed_benefits": int((~adapted & ((frame["delta_property_DOWN"] < 0) | (frame["delta_property_UP"] < 0))).sum()),
        "down_count": int((actions == "DOWN").sum()), "keep_count": int((actions == "KEEP").sum()),
        "up_count": int((actions == "UP").sum()),
        **{f"mean_{key}": value for key, value in selected_quality.items()},
        "actions": actions, "selected_delta": delta, "selected_harm": harm,
    }


def public_metrics(values: dict) -> dict:
    return {key: value for key, value in values.items() if not isinstance(value, np.ndarray)}


def best_t0_action(train: pd.DataFrame) -> str:
    means = {"KEEP": 0.0, **{action: float(train[f"delta_property_{action}"].mean()) for action in NON_KEEP}}
    return min(ACTIONS, key=lambda action: (means[action], TIE_ORDER[action]))


def constant_actions(n: int, action: str) -> np.ndarray:
    return np.full(n, action, dtype=object)


def clopper_pearson(harms: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    lower = 0.0 if harms == 0 else float(beta.ppf(0.025, harms, n - harms + 1))
    upper = 1.0 if harms == n else float(beta.ppf(0.975, harms + 1, n - harms))
    return lower, upper


def validation_search(models: dict, validation: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    prediction = predict(models, validation)
    rows = []
    for alpha_value, delta_safe, eta in product(CONFORMAL_ALPHAS, DELTA_SAFE, ETA):
        actions = safe_decisions(prediction, models, alpha_value, delta_safe, eta)
        metrics = policy_metrics(validation, actions)
        feasible = metrics["coverage"] >= MIN_COVERAGE and metrics["selective_property_harm_rate"] <= MAX_SELECTIVE_HARM
        rows.append({
            "alpha": alpha_value, "delta_safe": delta_safe, "eta": eta,
            "feasible": feasible, **public_metrics(metrics),
        })
    frame = pd.DataFrame(rows)
    feasible_rows = frame[frame["feasible"]].to_dict("records")
    selected = max(
        feasible_rows,
        key=lambda row: (
            row["overall_gain"], row["oracle_recovery"], row["adaptive_precision"],
            -row["selective_property_harm_rate"], -row["coverage"], -row["eta"],
            row["delta_safe"], -row["alpha"],
        ),
        default=None,
    )
    return frame, selected


def freeze_controller(models: dict, selected: dict, t0_action: str) -> dict:
    joblib.dump(models["scaler"], MODELS / "scaler.pkl")
    joblib.dump(models["ridge"]["DOWN"], MODELS / "ridge_low.pkl")
    joblib.dump(models["ridge"]["UP"], MODELS / "ridge_high.pkl")
    joblib.dump(models["harm"], MODELS / "harm_classifier.pkl")
    joblib.dump(models["platt"], MODELS / "harm_probability_calibrators.pkl")
    joblib.dump(models, IMPLEMENTATION / "controller_bundle.pkl")
    frozen = {
        "schema_version": 1, "frozen_before_test": True, "decision_index": 250,
        "actions": {"DOWN": 1.75, "KEEP": 2.0, "UP": 2.25},
        "one_shot": True, "feature_columns": list(FEATURE_COLUMNS),
        "normalization_parameters": models["normalization_parameters"],
        "conformal_alpha": float(selected["alpha"]),
        "delta_safe": float(selected["delta_safe"]), "eta": float(selected["eta"]),
        "conformal_q": {
            action: float(models["conformal_quantiles"][action][str(float(selected["alpha"]))])
            for action in NON_KEEP
        },
        "t0_action": t0_action,
        "benefit_gate": "predicted_delta + q < -delta_safe",
        "harm_gate": "platt_calibrated_probability < eta",
        "both_actions": "minimum_conformal_upper_bound",
    }
    (IMPLEMENTATION / "frozen_safe_controller.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False))
    files = [
        MODELS / "scaler.pkl", MODELS / "ridge_low.pkl", MODELS / "ridge_high.pkl",
        MODELS / "harm_classifier.pkl", MODELS / "harm_probability_calibrators.pkl",
        IMPLEMENTATION / "controller_bundle.pkl",
        IMPLEMENTATION / "frozen_safe_controller.yaml",
    ]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    json_write(IMPLEMENTATION / "model_hashes.json", hashes)
    return frozen


def make_plots(data: pd.DataFrame, models: dict, candidates: pd.DataFrame, selected: dict | None, test_rows: pd.DataFrame | None) -> None:
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    prediction = predict(models, validation)
    actual = np.concatenate([validation[f"delta_property_{action}"].to_numpy(float) for action in NON_KEEP])
    predicted = np.concatenate([prediction[f"mean_{action}"] for action in NON_KEEP])
    fig, ax = plt.subplots(figsize=(5.5, 5.5)); ax.scatter(actual, predicted, alpha=0.7)
    limits = [float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))]
    ax.plot(limits, limits, "k--"); ax.axhline(0, color="grey", linewidth=0.8); ax.axvline(0, color="grey", linewidth=0.8)
    ax.set(xlabel="True paired treatment effect", ylabel="Ridge predicted treatment effect")
    fig.tight_layout(); fig.savefig(PLOTS / "predicted_vs_true_delta.png", dpi=180); plt.close(fig)

    coverage_rows = []
    for action in NON_KEEP:
        truth = validation[f"delta_property_{action}"].to_numpy(float)
        estimate = prediction[f"mean_{action}"]
        for alpha_value in CONFORMAL_ALPHAS:
            q = models["conformal_quantiles"][action][str(alpha_value)]
            coverage_rows.append({"action": action, "alpha": alpha_value, "coverage": float((np.abs(truth - estimate) <= q).mean())})
    cov = pd.DataFrame(coverage_rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for action, group in cov.groupby("action"):
        ax.plot(1 - group["alpha"], group["coverage"], marker="o", label=action)
    ax.plot([0.8, 0.9], [0.8, 0.9], "k--", label="nominal")
    ax.set(xlabel="Nominal marginal coverage", ylabel="Validation empirical coverage"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "conformal_interval_coverage.png", dpi=180); plt.close(fig)

    calibration = data[data["split"] == "calibration"].reset_index(drop=True)
    cal_prediction = predict(models, calibration)
    fig, ax = plt.subplots(figsize=(6, 5))
    for action in NON_KEEP:
        probabilities = cal_prediction[f"harm_{action}"]
        labels = calibration[f"harm_property_{action}"].astype(int).to_numpy()
        bins = pd.qcut(pd.Series(probabilities), q=min(5, len(np.unique(probabilities))), duplicates="drop")
        grouped = pd.DataFrame({"p": probabilities, "y": labels, "bin": bins}).groupby("bin", observed=True)[["p", "y"]].mean()
        ax.plot(grouped["p"], grouped["y"], marker="o", label=action)
    ax.plot([0, 1], [0, 1], "k--"); ax.set(xlabel="Calibrated harm probability", ylabel="Observed harm frequency"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "harm_probability_calibration.png", dpi=180); plt.close(fig)

    for filename, ycolumn, ylabel in (
        ("risk_coverage_curve.png", "selective_property_harm_rate", "Selective property harm"),
        ("coverage_vs_gain.png", "overall_gain", "Actual overall paired gain"),
        ("coverage_vs_oracle_recovery.png", "oracle_recovery", "Oracle recovery"),
        ("adaptive_precision_vs_coverage.png", "adaptive_precision", "Adaptive precision"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4.5)); ax.scatter(candidates["coverage"], candidates[ycolumn], s=20, alpha=0.6)
        ax.axvline(MIN_COVERAGE, color="black", linestyle="--")
        if ycolumn == "selective_property_harm_rate": ax.axhline(MAX_SELECTIVE_HARM, color="red", linestyle="--")
        ax.set(xlabel="Coverage", ylabel=ylabel); fig.tight_layout(); fig.savefig(PLOTS / filename, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if selected is not None:
        actions_s0 = safe_decisions(prediction, models, selected["alpha"], selected["delta_safe"], selected["eta"])
        r0 = policy_metrics(validation, r0_decisions(prediction))
        s0 = policy_metrics(validation, actions_s0)
        ax.bar(["R0", "S0"], [r0["selective_property_harm_rate"], s0["selective_property_harm_rate"]])
        ax.axhline(MAX_SELECTIVE_HARM, color="red", linestyle="--")
    ax.set(ylabel="Selective property harm")
    fig.tight_layout(); fig.savefig(PLOTS / "r0_vs_s0_harm.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if test_rows is not None:
        policy = test_rows[test_rows["record_type"] == "policy"]
        ax.bar(policy["policy"], policy["property_mae"])
    ax.set(ylabel="Test property MAE (lower is better)")
    fig.tight_layout(); fig.savefig(PLOTS / "test_c0_t0_r0_s0.png", dpi=180); plt.close(fig)


def write_final(status: dict, report: str) -> None:
    json_write(ROOT / "final_status.json", status)
    (ROOT / "final_report.md").write_text(report)


def main() -> None:
    counterfactual = quality_enriched_dataset()
    treatment = build_treatment_dataset(counterfactual)
    data, split, _ = merge_features(treatment)
    models, fit_report = fit_models(data)
    train = data[data["split"] == "train"].reset_index(drop=True)
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    candidates, selected = validation_search(models, validation)
    candidates.to_csv(VALIDATION / "operating_point_results.csv", index=False)
    prereg = {
        "alpha": list(CONFORMAL_ALPHAS), "delta_safe": list(DELTA_SAFE), "eta": list(ETA),
        "minimum_coverage": MIN_COVERAGE, "maximum_selective_harm": MAX_SELECTIVE_HARM,
        "selection": "maximum realized overall paired gain among feasible points",
    }
    frozen_prereg = yaml.safe_load((VALIDATION / "preregistered_candidates.yaml").read_text())
    for key, expected in prereg.items():
        if frozen_prereg[key] != expected:
            raise RuntimeError(f"frozen validation candidate mismatch: {key}")
    t0_action = best_t0_action(train)
    if selected is None:
        decision = {
            "FULL_OUTCOME_DATASET": "COMPLETE", "VALIDATION_SAFE_GATE": "FAIL",
            "OFFLINE_SAFE_SELECTION": "NOT_RUN", "SAFE_SELECTION_P0": "NOT_RUN",
            "SAFE_SELECTION_FORMAL256": "NOT_RUN", "SAMPLE_ADAPTIVITY": "NOT_RUN",
            "INNOVATION1_FINAL_STATUS": "MIXED", "next": "STOP",
            "reason": "No preregistered Validation operating point met coverage >=15% and selective property harm <=10%.",
            "rescue_tuning": False,
        }
        json_write(VALIDATION / "selected_operating_point.json", {"status": "FAIL", "selected": None})
        json_write(VALIDATION / "decision_summary.json", decision)
        (TEST / "metrics.csv").write_text("status,reason\nNOT_RUN,VALIDATION_SAFE_GATE_FAIL\n")
        (TEST / "final_report.md").write_text("# Offline Test\n\nNOT RUN: Validation safe gate failed.\n")
        make_plots(data, models, candidates, None, None)
        report = "# Calibrated Safe-Selection Adaptive CFG\n\nValidation gate failed under the frozen candidate grid. Test, P0, and Formal256 were not run; no rescue tuning is permitted.\n\n" + "\n".join(f"- `{key} = {value}`" for key, value in decision.items() if key.isupper()) + "\n"
        write_final(decision, report)
        print(json.dumps(decision, indent=2))
        return

    selected = {key: (bool(value) if isinstance(value, np.bool_) else float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value) for key, value in selected.items()}
    frozen = freeze_controller(models, selected, t0_action)
    json_write(VALIDATION / "selected_operating_point.json", {"status": "GO", "selected": selected, "t0_action": t0_action})
    json_write(VALIDATION / "decision_summary.json", {"VALIDATION_SAFE_GATE": "GO", "selected": selected, "controller_frozen_before_test": True})

    test = data[data["split"] == "test"].reset_index(drop=True)
    prediction = predict(models, test)
    policies = {
        "C0": constant_actions(len(test), "KEEP"),
        "T0": constant_actions(len(test), t0_action),
        "R0": r0_decisions(prediction),
        "S0": safe_decisions(prediction, models, selected["alpha"], selected["delta_safe"], selected["eta"]),
    }
    metrics = {name: policy_metrics(test, actions) for name, actions in policies.items()}
    metric_rows = [{"record_type": "policy", "policy": name, **public_metrics(value)} for name, value in metrics.items()]
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(TEST / "metrics.csv", index=False)
    trace_rows = []
    for index, row in test.iterrows():
        record = {
            "seed": int(row["seed"]),
            "predicted_delta_DOWN": float(prediction["mean_DOWN"][index]),
            "predicted_delta_UP": float(prediction["mean_UP"][index]),
            "calibrated_harm_DOWN": float(prediction["harm_DOWN"][index]),
            "calibrated_harm_UP": float(prediction["harm_UP"][index]),
            "true_delta_DOWN": float(row["delta_property_DOWN"]),
            "true_delta_UP": float(row["delta_property_UP"]),
            "oracle_action": row["oracle_action"],
        }
        for name, actions in policies.items():
            action = actions[index]
            record[f"action_{name}"] = action
            record[f"selected_delta_{name}"] = 0.0 if action == "KEEP" else float(row[f"delta_property_{action}"])
        trace_rows.append(record)
    pd.DataFrame(trace_rows).to_csv(TEST / "treatment_effect_results.csv", index=False)
    pd.DataFrame([
        {"policy": name, "overall_gain": value["overall_gain"], "full_oracle_gain": value["oracle_gain"], "oracle_recovery": value["oracle_recovery"]}
        for name, value in metrics.items()
    ]).to_csv(TEST / "oracle_recovery.csv", index=False)
    validation_curve = candidates.copy(); validation_curve.insert(0, "split", "validation")
    test_point = {"split": "test", "alpha": selected["alpha"], "delta_safe": selected["delta_safe"], "eta": selected["eta"], **public_metrics(metrics["S0"])}
    pd.concat([validation_curve, pd.DataFrame([test_point])], ignore_index=True).to_csv(TEST / "risk_coverage.csv", index=False)
    s0 = metrics["S0"]
    harm_lower, harm_upper = clopper_pearson(int(s0["selected_harm"].sum()), s0["adaptive_count"])
    checks = {
        "coverage_ge_15pct": s0["coverage"] >= MIN_COVERAGE,
        "selective_property_harm_le_10pct": s0["selective_property_harm_rate"] <= MAX_SELECTIVE_HARM,
        "adaptive_subset_mean_delta_lt_zero": s0["adaptive_subset_mean_delta"] < 0,
        "adaptive_wins_gt_losses": s0["wins"] > s0["losses"],
        "oracle_recovery_ge_20pct": s0["oracle_recovery"] >= MIN_ORACLE_RECOVERY,
        "adaptive_precision_ge_85pct": s0["adaptive_precision"] >= MIN_PRECISION,
        "s0_property_mae_better_than_t0": s0["property_mae"] < metrics["T0"]["property_mae"],
    }
    offline = "GO" if all(checks.values()) else "FAIL"
    sample_adaptivity = "SUPPORTED" if checks["s0_property_mae_better_than_t0"] else "NOT_SUPPORTED"
    status = {
        "FULL_OUTCOME_DATASET": "COMPLETE", "VALIDATION_SAFE_GATE": "GO",
        "OFFLINE_SAFE_SELECTION": offline, "SAFE_SELECTION_P0": "NOT_RUN",
        "SAFE_SELECTION_FORMAL256": "NOT_RUN", "SAMPLE_ADAPTIVITY": sample_adaptivity,
        "INNOVATION1_FINAL_STATUS": "MIXED", "next": "RUN_FRESH_P0" if offline == "GO" else "STOP",
        "selected_operating_point": {key: frozen[key] for key in ("conformal_alpha", "delta_safe", "eta", "conformal_q")},
        "t0_action": t0_action, "test_gate_checks": checks,
        "test_s0": public_metrics(s0), "selective_harm_ci95": [harm_lower, harm_upper],
        "rescue_tuning": False,
    }
    json_write(TEST / "decision_summary.json", status)
    report_lines = [
        "# Calibrated Safe-Selection — Untouched Offline Test", "",
        "The controller, Ridge/scaler weights, conformal q, Platt calibrators, and operating point were frozen before Test evaluation.",
        "The primary property is the frozen CHGNet magnetic-density surrogate, not DFT.", "",
        f"- Validation gate: `GO`", f"- Offline safe-selection gate: `{offline}`",
        f"- S0 coverage: `{s0['coverage']:.4f}` ({s0['adaptive_count']}/{s0['n']})",
        f"- S0 selective harm: `{s0['selective_property_harm_rate']:.4f}`; exact 95% CI `{harm_lower:.4f}, {harm_upper:.4f}`",
        f"- S0 adaptive precision: `{s0['adaptive_precision']:.4f}`",
        f"- S0 overall gain: `{s0['overall_gain']:.6f}`; oracle recovery `{s0['oracle_recovery']:.4f}`",
        f"- Property MAE C0/T0/R0/S0: `{metrics['C0']['property_mae']:.6f} / {metrics['T0']['property_mae']:.6f} / {metrics['R0']['property_mae']:.6f} / {s0['property_mae']:.6f}`",
        "", "## Frozen gate checks", "",
    ]
    report_lines.extend(f"- `{key}`: `{value}`" for key, value in checks.items())
    report_lines += ["", "No Test-driven retuning or rescue was performed.\n"]
    report = "\n".join(report_lines)
    (TEST / "final_report.md").write_text(report)
    make_plots(data, models, candidates, selected, metric_frame)
    root_report = "# Calibrated Safe-Selection Adaptive CFG\n\n" + "\n".join(
        f"- `{key} = {value}`" for key, value in status.items() if key.isupper()
    ) + "\n\n" + report
    write_final(status, root_report)
    print(json.dumps(status, indent=2, default=float))


if __name__ == "__main__":
    main()
