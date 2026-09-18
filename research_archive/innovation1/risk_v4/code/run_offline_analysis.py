"""Build treatment effects, fit simple predictors, and apply frozen gates."""

from __future__ import annotations

import gzip
import json
from itertools import product
from pathlib import Path

from ase.io import read
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/risk_calibrated_adaptive_cfg"
DATASET = ROOT / "dataset"
FEATURES = ROOT / "features"
SPLITS = ROOT / "splits"
MODELS = ROOT / "models"
OFFLINE = ROOT / "offline"
PLOTS = ROOT / "plots"
ACTIONS = ("DOWN", "KEEP", "UP")
NON_KEEP = ("DOWN", "UP")
STEPS = (100, 250, 400, 550)
SPLIT_SEED = 20260921
BOOTSTRAP_SEEDS = (20260930, 20260931, 20260932, 20260933, 20260934)
DELTA_MIN = (0.0, 0.0025, 0.005, 0.01)
DELTA_SAFE = (0.0, 0.0025, 0.005)
ETA = (0.10, 0.20, 0.30, 0.40)
CONSERVATIVE_K = 1.0
MIN_COVERAGE = 0.15
MAX_SELECTIVE_HARM = 0.10
MIN_ORACLE_RECOVERY = 0.20
TIE_ORDER = {"KEEP": 0, "DOWN": 1, "UP": 2}

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


def ensure_directories() -> None:
    for path in (FEATURES, SPLITS, MODELS, OFFLINE, PLOTS, ROOT / "p0"):
        path.mkdir(exist_ok=True)


def quality_enriched_dataset() -> pd.DataFrame:
    values = pd.read_csv(DATASET / "property_metrics.csv")
    for column, default in (
        ("e_hull", np.nan), ("stable", False), ("nus", False), ("novel", False),
        ("unique", False), ("rmsd", np.nan), ("pre_relaxation_max_force", np.nan),
        ("relaxation_steps", np.nan), ("quality_success", False),
    ):
        values[column] = default
    for step in STEPS:
        for action in ACTIONS:
            group = f"STEP{step}_{action}"
            structure_path = DATASET / "evaluation_structures" / f"{group}.extxyz"
            quality_root = DATASET / "quality" / group
            if not structure_path.exists():
                continue
            detailed_path = quality_root / "official_detailed.json.gz"
            if not detailed_path.exists():
                raise FileNotFoundError(f"quality result missing: {group}")
            atoms = read(structure_path, index=":")
            per_structure = pd.read_csv(quality_root / "per_structure.csv")
            with gzip.open(detailed_path, "rt") as stream:
                detailed = json.load(stream)
            if len(per_structure) != len(detailed["energy_above_hull_per_atom"]):
                raise RuntimeError(f"quality length mismatch: {group}")
            for output_index, row in per_structure.reset_index(drop=True).iterrows():
                input_index = int(row["index"])
                identifier = str(atoms[input_index].info["branch_id"])
                match = values.index[values["branch_id"] == identifier]
                if len(match) != 1:
                    raise RuntimeError(f"branch lookup failed: {identifier}")
                target = match[0]
                values.loc[target, "e_hull"] = float(
                    detailed["energy_above_hull_per_atom"][output_index]
                )
                values.loc[target, "stable"] = bool(detailed["stable"][output_index])
                values.loc[target, "nus"] = bool(
                    detailed["novel_unique_stable"][output_index]
                )
                values.loc[target, "novel"] = bool(detailed["novel"][output_index])
                values.loc[target, "unique"] = bool(detailed["unique"][output_index])
                values.loc[target, "rmsd"] = float(
                    detailed["rmsd_from_relaxation"][output_index]
                )
                values.loc[target, "pre_relaxation_max_force"] = float(
                    row["pre_relaxation_max_force"]
                )
                values.loc[target, "relaxation_steps"] = int(row["relaxation_steps"])
                values.loc[target, "quality_success"] = True
    for column in ("stable", "nus", "novel", "unique", "quality_success"):
        values[column] = values[column].fillna(False).astype(bool)
    values["evaluation_valid"] = (
        values["structure_valid"].astype(bool) & values["quality_success"].astype(bool)
    )
    values.to_csv(DATASET / "counterfactual_dataset.csv", index=False)
    return values


def build_treatment_dataset(counterfactual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = (
        "property_absolute_error", "e_hull", "stable", "nus", "novel", "unique",
        "evaluation_valid",
    )
    for (seed, step), state in counterfactual.groupby(["seed", "sampling_step"], sort=True):
        if len(state) != 3 or set(state["candidate_action"]) != set(ACTIONS):
            raise RuntimeError(f"candidate mismatch: seed={seed}, step={step}")
        selected = {action: state[state["candidate_action"] == action].iloc[0] for action in ACTIONS}
        base = selected["KEEP"]
        row: dict = {
            "seed": int(seed),
            "sampling_step": int(step),
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
            row[f"harm_property_{action}"] = bool(delta > 0.05 * max(base_error, 1e-12))
            action_ehull = float(row[f"e_hull_{action}"])
            base_ehull = float(row["e_hull_KEEP"])
            row[f"delta_ehull_{action}"] = action_ehull - base_ehull
            row[f"harm_ehull_{action}"] = bool(
                np.isfinite(action_ehull)
                and np.isfinite(base_ehull)
                and action_ehull - base_ehull > 0.02
            )
            row[f"harm_stable_{action}"] = bool(
                row["stable_KEEP"] and not row[f"stable_{action}"]
            )
        ranked = sorted(
            ACTIONS,
            key=lambda action: (
                float(row[f"property_absolute_error_{action}"]), TIE_ORDER[action]
            ),
        )
        row["oracle_action"] = ranked[0]
        row["oracle_gain"] = base_error - float(row[f"property_absolute_error_{ranked[0]}"])
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 96 * 4 or result["seed"].nunique() != 96:
        raise RuntimeError("treatment-effect dataset is incomplete")
    result.to_csv(DATASET / "treatment_effect_dataset.csv", index=False)
    return result


def make_split(seeds: np.ndarray) -> dict[str, list[int]]:
    shuffled = np.asarray(sorted(map(int, seeds)), dtype=int)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(shuffled)
    return {
        "train": sorted(map(int, shuffled[:58])),
        "validation": sorted(map(int, shuffled[58:77])),
        "test": sorted(map(int, shuffled[77:])),
    }


def build_features(treatment: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    raw = pd.read_csv(DATASET / "online_features_raw.csv").rename(
        columns={
            "r_atomic_numbers": "r_atomic",
            "score_ratio_atomic_numbers": "score_ratio_atomic",
            "score_alignment_atomic_numbers": "score_alignment_atomic",
        }
    )
    data = raw.merge(treatment, on=["seed", "sampling_step"], how="inner")
    split = make_split(data["seed"].unique())
    membership = {seed: name for name, seeds in split.items() for seed in seeds}
    data["split"] = data["seed"].map(membership)
    normalization: dict[str, dict] = {}
    for step in STEPS:
        normalization[str(step)] = {}
        train_mask = (data["sampling_step"] == step) & (data["split"] == "train")
        step_mask = data["sampling_step"] == step
        for source, destination in (
            ("r_atomic", "z_atomic"), ("r_pos", "z_pos"), ("r_cell", "z_cell")
        ):
            train_values = np.log(data.loc[train_mask, source].astype(float).to_numpy() + 1e-12)
            mean = float(train_values.mean())
            std = float(train_values.std(ddof=1))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            data.loc[step_mask, destination] = (
                np.log(data.loc[step_mask, source].astype(float) + 1e-12) - mean
            ) / std
            normalization[str(step)][source] = {"log_mean": mean, "log_std": std}
    feature_values = data[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(feature_values.to_numpy()).all():
        bad = feature_values.columns[~np.isfinite(feature_values.to_numpy()).all(axis=0)].tolist()
        raise RuntimeError(f"non-finite online features: {bad}")
    data.to_csv(FEATURES / "feature_dataset.csv", index=False)
    schema = {
        "features": list(FEATURE_COLUMNS),
        "targets": [
            "delta_property_DOWN", "delta_property_UP",
            "harm_property_DOWN", "harm_property_UP",
        ],
        "online_only": True,
        "normalization": "training-only log-residual z-score separately per timestep",
        "normalization_parameters": normalization,
        "temporal_ema_beta": 0.9,
        "recent_window": 5,
        "future_information_prohibited": True,
    }
    (FEATURES / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    manifest = {
        "split_seed": SPLIT_SEED,
        "split_unit": "trajectory_seed",
        "fractions_requested": [0.60, 0.20, 0.20],
        "counts": {name: len(seeds) for name, seeds in split.items()},
        **split,
        "disjoint": len(set(split["train"]) | set(split["validation"]) | set(split["test"])) == 96,
        "all_timesteps_coassigned": True,
    }
    (SPLITS / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return data, split, schema


def make_regressor(family: str, random_state: int):
    if family == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if family == "tree":
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.03, subsample=0.8,
            min_samples_leaf=5, random_state=random_state, loss="huber",
        )
    raise ValueError(family)


def make_classifier(family: str, random_state: int, y: np.ndarray):
    if len(np.unique(y)) < 2:
        return DummyClassifier(strategy="prior")
    if family == "ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=2000,
                    random_state=random_state,
                )),
            ]
        )
    if family == "tree":
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.03, subsample=0.8,
            min_samples_leaf=5, random_state=random_state,
        )
    raise ValueError(family)


def harm_probability(model, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = list(map(int, model.classes_))
    return raw[:, classes.index(1)] if 1 in classes else np.zeros(len(x), dtype=float)


def fit_ensemble(train: pd.DataFrame, family: str) -> dict:
    x = train[list(FEATURE_COLUMNS)]
    rng = np.random.default_rng(20260929)
    members = []
    for member_seed in BOOTSTRAP_SEEDS:
        indices = rng.choice(np.arange(len(train)), size=len(train), replace=True)
        bx = x.iloc[indices]
        member = {"seed": member_seed, "regression": {}, "harm": {}}
        for action in NON_KEEP:
            regression = make_regressor(family, member_seed)
            regression.fit(bx, train.iloc[indices][f"delta_property_{action}"])
            labels = train.iloc[indices][f"harm_property_{action}"].astype(int).to_numpy()
            classifier = make_classifier(family, member_seed, labels)
            classifier.fit(bx, labels)
            member["regression"][action] = regression
            member["harm"][action] = classifier
        members.append(member)
    return {"family": family, "features": list(FEATURE_COLUMNS), "members": members}


def ensemble_predict(ensemble: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    x = frame[list(FEATURE_COLUMNS)]
    output: dict[str, np.ndarray] = {}
    for action in NON_KEEP:
        effect = np.vstack(
            [member["regression"][action].predict(x) for member in ensemble["members"]]
        )
        harm = np.vstack(
            [harm_probability(member["harm"][action], x) for member in ensemble["members"]]
        )
        output[f"mean_{action}"] = effect.mean(axis=0)
        output[f"std_{action}"] = effect.std(axis=0, ddof=1)
        output[f"harm_{action}"] = harm.mean(axis=0)
    return output


def decisions(prediction: dict, delta_min: float, delta_safe: float, eta: float, umax: float) -> np.ndarray:
    chosen: list[str] = []
    for index in range(len(prediction["mean_DOWN"])):
        eligible = []
        for action in NON_KEEP:
            mean = float(prediction[f"mean_{action}"][index])
            std = float(prediction[f"std_{action}"][index])
            harm = float(prediction[f"harm_{action}"][index])
            if (
                mean < -delta_min
                and mean + CONSERVATIVE_K * std < -delta_safe
                and harm < eta
                and std <= umax
            ):
                eligible.append((mean, TIE_ORDER[action], action))
        chosen.append(min(eligible)[2] if eligible else "KEEP")
    return np.asarray(chosen, dtype=object)


def policy_metrics(frame: pd.DataFrame, actions: np.ndarray) -> dict:
    selected_delta = np.asarray(
        [0.0 if action == "KEEP" else float(row[f"delta_property_{action}"])
         for action, (_, row) in zip(actions, frame.iterrows())],
        dtype=float,
    )
    selected_harm = np.asarray(
        [False if action == "KEEP" else bool(row[f"harm_property_{action}"])
         for action, (_, row) in zip(actions, frame.iterrows())],
        dtype=bool,
    )
    adapted = actions != "KEEP"
    coverage = float(adapted.mean())
    selective_harm = float(selected_harm[adapted].mean()) if adapted.any() else 1.0
    subset_gain = float((-selected_delta[adapted]).mean()) if adapted.any() else 0.0
    overall_gain = float(-selected_delta.mean())
    oracle_gain = float(frame["oracle_gain"].astype(float).mean())
    recovery = overall_gain / max(oracle_gain, 1e-12)
    base = frame["property_absolute_error_KEEP"].astype(float).to_numpy()
    return {
        "n": len(frame),
        "coverage": coverage,
        "adaptive_count": int(adapted.sum()),
        "selective_property_harm_rate": selective_harm,
        "property_harm_rate_all": float(selected_harm.mean()),
        "adaptive_action_precision": float((selected_delta[adapted] < 0).mean()) if adapted.any() else 0.0,
        "expected_treatment_gain_adaptive_subset": subset_gain,
        "controller_gain_overall": overall_gain,
        "full_oracle_gain": oracle_gain,
        "oracle_recovery": recovery,
        "c0_property_mae": float(base.mean()),
        "a0_property_mae": float((base + selected_delta).mean()),
        "chosen_actions": actions,
        "selected_delta": selected_delta,
    }


def best_t0_action(train: pd.DataFrame) -> str:
    means = {
        "KEEP": 0.0,
        "DOWN": float(train["delta_property_DOWN"].mean()),
        "UP": float(train["delta_property_UP"].mean()),
    }
    return min(ACTIONS, key=lambda action: (means[action], TIE_ORDER[action]))


def t0_metrics(frame: pd.DataFrame, action: str) -> dict:
    return policy_metrics(frame, np.full(len(frame), action, dtype=object))


def regression_metrics(frame: pd.DataFrame, prediction: dict) -> list[dict]:
    rows = []
    actual_all = []
    predicted_all = []
    for action in NON_KEEP:
        actual = frame[f"delta_property_{action}"].astype(float).to_numpy()
        predicted = prediction[f"mean_{action}"]
        actual_all.extend(actual)
        predicted_all.extend(predicted)
        rows.append(
            {
                "action": action,
                "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
                "mae": float(mean_absolute_error(actual, predicted)),
                "spearman": float(pd.Series(actual).corr(pd.Series(predicted), method="spearman")),
                "sign_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
            }
        )
    actual = np.asarray(actual_all)
    predicted = np.asarray(predicted_all)
    rows.append(
        {
            "action": "ALL",
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
            "mae": float(mean_absolute_error(actual, predicted)),
            "spearman": float(pd.Series(actual).corr(pd.Series(predicted), method="spearman")),
            "sign_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
        }
    )
    return rows


def validation_search(data: pd.DataFrame, split: dict) -> tuple[list[dict], list[dict], dict]:
    comparison: list[dict] = []
    risk_rows: list[dict] = []
    fitted: dict = {}
    train_seeds, validation_seeds = set(split["train"]), set(split["validation"])
    for step in STEPS:
        step_data = data[data["sampling_step"] == step]
        train = step_data[step_data["seed"].isin(train_seeds)].reset_index(drop=True)
        validation = step_data[step_data["seed"].isin(validation_seeds)].reset_index(drop=True)
        t0_action = best_t0_action(train)
        for family in ("ridge", "tree"):
            ensemble = fit_ensemble(train, family)
            prediction = ensemble_predict(ensemble, validation)
            uncertainty_pool = np.concatenate(
                [prediction["std_DOWN"], prediction["std_UP"]]
            )
            umax_values = sorted(set(float(np.quantile(uncertainty_pool, q)) for q in (0.25, 0.50, 0.75, 1.0)))
            candidates: list[dict] = []
            for dmin, dsafe, eta, umax in product(DELTA_MIN, DELTA_SAFE, ETA, umax_values):
                action = decisions(prediction, dmin, dsafe, eta, umax)
                metrics = policy_metrics(validation, action)
                feasible = (
                    metrics["coverage"] >= MIN_COVERAGE
                    and metrics["selective_property_harm_rate"] <= MAX_SELECTIVE_HARM
                )
                row = {
                    "sampling_step": step,
                    "model_family": family,
                    "delta_min": dmin,
                    "delta_safe": dsafe,
                    "eta": eta,
                    "uncertainty_max": umax,
                    "split": "validation",
                    "feasible_selective_gate": feasible,
                    **{k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)},
                }
                candidates.append(row)
                risk_rows.append(row)
            feasible_rows = [row for row in candidates if row["feasible_selective_gate"]]
            selected = max(
                feasible_rows,
                key=lambda row: (
                    row["controller_gain_overall"], row["oracle_recovery"],
                    -row["selective_property_harm_rate"], -row["uncertainty_max"],
                    -row["coverage"],
                ),
                default=None,
            )
            record = {
                "sampling_step": step,
                "model_family": family,
                "validation_oracle_headroom": float(validation["oracle_gain"].mean()),
                "validation_oracle_non_keep_fraction": float((validation["oracle_action"] != "KEEP").mean()),
                "t0_action_from_train": t0_action,
                "selective_gate_feasible": selected is not None,
            }
            if selected is not None:
                record.update({f"selected_{key}": value for key, value in selected.items() if key not in ("sampling_step", "model_family", "split")})
            comparison.append(record)
            fitted[(step, family)] = {"ensemble": ensemble, "prediction": prediction, "selected": selected, "t0_action": t0_action}
    return comparison, risk_rows, fitted


def write_model_results(comparison: list[dict], fitted: dict) -> None:
    for family, filename in (("ridge", "ridge_results.json"), ("tree", "tree_results.json")):
        rows = [row for row in comparison if row["model_family"] == family]
        (MODELS / filename).write_text(
            json.dumps(
                {
                    "model_family": family,
                    "implementation": "Ridge+LogisticRegression" if family == "ridge" else "sklearn GradientBoosting regressor+classifier",
                    "ensemble_members": 5,
                    "validation_by_timestep": rows,
                },
                indent=2,
                default=float,
            )
            + "\n"
        )
    (MODELS / "mlp_results.json").write_text(
        json.dumps(
            {
                "status": "NOT_RUN",
                "reason": "optional M3 not needed for the pre-registered M1/M2 comparison",
                "parameter_limit": 100000,
            },
            indent=2,
        )
        + "\n"
    )
    (MODELS / "ensemble_results.json").write_text(
        json.dumps(
            {
                "ensemble_size": 5,
                "bootstrap_seeds": BOOTSTRAP_SEEDS,
                "uncertainty": "sample standard deviation of five treatment-effect regressors",
                "harm_probability": "mean probability from five harm classifiers",
                "families": ["ridge", "tree"],
                "fitted_timestep_family_pairs": len(fitted),
            },
            indent=2,
        )
        + "\n"
    )


def make_plots(data: pd.DataFrame, comparison: pd.DataFrame, risk: pd.DataFrame, selected: dict | None, test_payload: dict | None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(data["delta_property_DOWN"], bins=30, alpha=0.65, label="DOWN")
    ax.hist(data["delta_property_UP"], bins=30, alpha=0.65, label="UP")
    ax.axvline(0, color="black", linestyle="--")
    ax.set(xlabel="Paired property-error treatment effect (action - CFG2)", ylabel="states")
    ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "treatment_effect_distribution.png", dpi=180); plt.close(fig)

    headroom = data.groupby("sampling_step")["oracle_gain"].mean()
    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.plot(headroom.index, headroom.values, marker="o")
    ax.set(xlabel="Decision sampling index", ylabel="Mean full-oracle property gain")
    fig.tight_layout(); fig.savefig(PLOTS / "timestep_oracle_headroom.png", dpi=180); plt.close(fig)

    feasible = comparison[comparison["selective_gate_feasible"]].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for family, group in feasible.groupby("model_family"):
        ax.plot(group["sampling_step"], group["selected_controller_gain_overall"], marker="o", label=family)
    ax.set(xlabel="Decision sampling index", ylabel="Best feasible validation gain")
    if not feasible.empty: ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "timestep_predictability.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for family, group in risk.groupby("model_family"):
        ax.scatter(group["coverage"], group["selective_property_harm_rate"], s=9, alpha=0.35, label=family)
    ax.axhline(MAX_SELECTIVE_HARM, color="red", linestyle="--"); ax.axvline(MIN_COVERAGE, color="black", linestyle="--")
    ax.set(xlabel="Coverage", ylabel="Selective property harm rate")
    ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "risk_coverage_curve.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.scatter(risk["coverage"], risk["controller_gain_overall"], s=9, alpha=0.35)
    ax.set(xlabel="Coverage", ylabel="Validation controller gain")
    fig.tight_layout(); fig.savefig(PLOTS / "coverage_vs_gain.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.scatter(risk["coverage"], risk["selective_property_harm_rate"], s=9, alpha=0.35)
    ax.axhline(MAX_SELECTIVE_HARM, color="red", linestyle="--"); ax.set(xlabel="Coverage", ylabel="Selective harm")
    fig.tight_layout(); fig.savefig(PLOTS / "coverage_vs_harm.png", dpi=180); plt.close(fig)

    recovery = feasible.pivot_table(index="sampling_step", columns="model_family", values="selected_oracle_recovery") if not feasible.empty else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not recovery.empty: recovery.plot(kind="bar", ax=ax)
    ax.axhline(MIN_ORACLE_RECOVERY, color="black", linestyle="--"); ax.set(ylabel="Validation oracle recovery")
    fig.tight_layout(); fig.savefig(PLOTS / "oracle_recovery.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if test_payload is not None:
        ax.scatter(test_payload["actual"], test_payload["predicted"], alpha=0.65)
        limits = [min(test_payload["actual"] + test_payload["predicted"]), max(test_payload["actual"] + test_payload["predicted"])]
        ax.plot(limits, limits, "k--")
    ax.set(xlabel="True paired treatment effect", ylabel="Predicted treatment effect")
    fig.tight_layout(); fig.savefig(PLOTS / "predicted_vs_true_treatment_effect.png", dpi=180); plt.close(fig)


def main() -> None:
    ensure_directories()
    counterfactual = quality_enriched_dataset()
    treatment = build_treatment_dataset(counterfactual)
    data, split, _ = build_features(treatment)
    comparison, risk_rows, fitted = validation_search(data, split)
    write_model_results(comparison, fitted)
    comparison_frame = pd.DataFrame(comparison)
    risk_frame = pd.DataFrame(risk_rows)
    comparison_frame.to_csv(OFFLINE / "timestep_comparison.csv", index=False)
    risk_frame.to_csv(OFFLINE / "risk_coverage.csv", index=False)
    feasible = [
        (key, value) for key, value in fitted.items() if value["selected"] is not None
    ]
    if not feasible:
        decision = {
            "FULL_OUTCOME_DATASET": "COMPLETE",
            "OFFLINE_SELECTIVE_GATE": "FAIL",
            "OFFLINE_PREDICTABILITY": "NOT_RUN",
            "RISK_CALIBRATED_CFG_P0": "NOT_RUN",
            "RISK_CALIBRATED_CFG_FORMAL256": "NOT_RUN",
            "SAMPLE_ADAPTIVITY": "NOT_RUN",
            "INNOVATION1_FINAL_STATUS": "MIXED",
            "reason": "No validation operating point reached coverage >=15% and selective property harm <=10%.",
            "next": "STOP",
            "rescue_tuning": False,
        }
        (OFFLINE / "decision_summary.json").write_text(json.dumps(decision, indent=2) + "\n")
        (OFFLINE / "offline_test_results.csv").write_text("status,reason\nNOT_RUN,OFFLINE_SELECTIVE_GATE_FAIL\n")
        (OFFLINE / "oracle_recovery.csv").write_text("split,status\nvalidation,NO_FEASIBLE_OPERATING_POINT\n")
        (ROOT / "p0/decision_summary.json").write_text(json.dumps({"status": "NOT_RUN", "reason": "offline selective gate failed"}, indent=2) + "\n")
        make_plots(data, comparison_frame, risk_frame, None, None)
        write_reports(decision, None)
        print(json.dumps(decision, indent=2))
        return
    selected_key, selected_bundle = max(
        feasible,
        key=lambda item: (
            item[1]["selected"]["controller_gain_overall"],
            item[1]["selected"]["oracle_recovery"],
            -item[1]["selected"]["selective_property_harm_rate"],
            1 if item[0][1] == "ridge" else 0,
        ),
    )
    step, family = selected_key
    operating = selected_bundle["selected"]
    test = data[(data["sampling_step"] == step) & (data["seed"].isin(split["test"]))].reset_index(drop=True)
    prediction = ensemble_predict(selected_bundle["ensemble"], test)
    actions = decisions(
        prediction, operating["delta_min"], operating["delta_safe"],
        operating["eta"], operating["uncertainty_max"],
    )
    controller = policy_metrics(test, actions)
    t0 = t0_metrics(test, selected_bundle["t0_action"])
    controller["t0_action"] = selected_bundle["t0_action"]
    controller["t0_property_mae"] = t0["a0_property_mae"]
    controller["a0_gain_vs_t0"] = t0["a0_property_mae"] - controller["a0_property_mae"]
    regression = regression_metrics(test, prediction)
    test_rows = [{"record_type": "regression", "model_family": family, "sampling_step": step, **row} for row in regression]
    test_rows += [
        {
            "record_type": "policy", "model_family": family, "sampling_step": step,
            "action": "A0", **{k: v for k, v in controller.items() if not isinstance(v, np.ndarray)},
        },
        {
            "record_type": "policy", "model_family": "timestep_only", "sampling_step": step,
            "action": selected_bundle["t0_action"], **{k: v for k, v in t0.items() if not isinstance(v, np.ndarray)},
        },
    ]
    pd.DataFrame(test_rows).to_csv(OFFLINE / "offline_test_results.csv", index=False)
    pd.DataFrame(
        [
            {"split": "validation", "policy": "A0", "sampling_step": step, "model_family": family,
             "controller_gain": operating["controller_gain_overall"], "full_oracle_gain": operating["full_oracle_gain"], "oracle_recovery": operating["oracle_recovery"]},
            {"split": "test", "policy": "A0", "sampling_step": step, "model_family": family,
             "controller_gain": controller["controller_gain_overall"], "full_oracle_gain": controller["full_oracle_gain"], "oracle_recovery": controller["oracle_recovery"]},
        ]
    ).to_csv(OFFLINE / "oracle_recovery.csv", index=False)
    checks = {
        "expected_treatment_gain_positive": controller["expected_treatment_gain_adaptive_subset"] > 0 and controller["controller_gain_overall"] > 0,
        "coverage_ge_15pct": controller["coverage"] >= MIN_COVERAGE,
        "selective_property_harm_le_10pct": controller["selective_property_harm_rate"] <= MAX_SELECTIVE_HARM,
        "oracle_recovery_ge_20pct": controller["oracle_recovery"] >= MIN_ORACLE_RECOVERY,
        "a0_offline_better_than_t0": controller["a0_property_mae"] < controller["t0_property_mae"],
    }
    offline_decision = "GO" if all(checks.values()) else "FAIL"
    decision = {
        "FULL_OUTCOME_DATASET": "COMPLETE",
        "OFFLINE_SELECTIVE_GATE": "GO",
        "OFFLINE_PREDICTABILITY": offline_decision,
        "RISK_CALIBRATED_CFG_P0": "NOT_RUN",
        "RISK_CALIBRATED_CFG_FORMAL256": "NOT_RUN",
        "SAMPLE_ADAPTIVITY": "NOT_RUN",
        "INNOVATION1_FINAL_STATUS": "MIXED",
        "selected_timestep": step,
        "selected_model_family": family,
        "selected_operating_point": {
            key: operating[key] for key in ("delta_min", "delta_safe", "eta", "uncertainty_max")
        },
        "t0_action": selected_bundle["t0_action"],
        "test_controller": {k: v for k, v in controller.items() if not isinstance(v, np.ndarray)},
        "test_t0": {k: v for k, v in t0.items() if not isinstance(v, np.ndarray)},
        "gate_checks": checks,
        "next": "RUN_FRESH_P0" if offline_decision == "GO" else "STOP",
        "parameters_retuned_after_test": False,
        "rescue_tuning": False,
    }
    (OFFLINE / "decision_summary.json").write_text(json.dumps(decision, indent=2, default=float) + "\n")
    joblib.dump(selected_bundle["ensemble"], MODELS / "selected_ensemble.joblib")
    (MODELS / "selected_model_metadata.json").write_text(
        json.dumps(
            {
                "sampling_step": step, "model_family": family,
                "features": list(FEATURE_COLUMNS),
                "normalization_parameters": json.loads((FEATURES / "feature_schema.json").read_text())["normalization_parameters"],
                "operating_point": decision["selected_operating_point"],
                "t0_action": selected_bundle["t0_action"],
                "one_shot": True,
            },
            indent=2,
        )
        + "\n"
    )
    actual = np.concatenate([test["delta_property_DOWN"], test["delta_property_UP"]]).astype(float).tolist()
    predicted = np.concatenate([prediction["mean_DOWN"], prediction["mean_UP"]]).astype(float).tolist()
    make_plots(data, comparison_frame, risk_frame, decision, {"actual": actual, "predicted": predicted})
    if offline_decision == "FAIL":
        (ROOT / "p0/decision_summary.json").write_text(json.dumps({"status": "NOT_RUN", "reason": "offline predictability gate failed"}, indent=2) + "\n")
    write_reports(decision, controller)
    print(json.dumps(decision, indent=2, default=float))


def write_reports(decision: dict, controller: dict | None) -> None:
    lines = [
        "# Long-horizon Risk-Calibrated Selective Adaptive CFG — Offline Report",
        "",
        "This report is generated under the protocol frozen before new full-outcome generation.",
        "The primary property metric uses the frozen CHGNet magnetic-density surrogate; it is not DFT verification.",
        "",
        "## Status",
        "",
    ]
    for key in (
        "FULL_OUTCOME_DATASET", "OFFLINE_SELECTIVE_GATE", "OFFLINE_PREDICTABILITY",
        "RISK_CALIBRATED_CFG_P0", "RISK_CALIBRATED_CFG_FORMAL256",
        "SAMPLE_ADAPTIVITY", "INNOVATION1_FINAL_STATUS",
    ):
        lines.append(f"- `{key} = {decision[key]}`")
    lines += ["", "## Frozen decision", "", f"- Next action: `{decision['next']}`"]
    if "selected_timestep" in decision:
        lines += [
            f"- Decision timestep: `{decision['selected_timestep']}`",
            f"- Predictor: `{decision['selected_model_family']}` with five bootstrap members",
            f"- T0 global action: `{decision['t0_action']}` (training split only)",
            f"- Operating point: `{json.dumps(decision['selected_operating_point'], default=float)}`",
            "",
            "## Held-out test gate",
            "",
        ]
        for key, value in decision["gate_checks"].items():
            lines.append(f"- `{key}`: `{value}`")
        if controller is not None:
            lines += [
                "",
                f"Coverage: {controller['coverage']:.3f}; selective property harm: {controller['selective_property_harm_rate']:.3f}; "
                f"overall gain: {controller['controller_gain_overall']:.6f}; oracle recovery: {controller['oracle_recovery']:.3f}.",
                f"C0 MAE: {controller['c0_property_mae']:.6f}; A0 offline MAE: {controller['a0_property_mae']:.6f}; "
                f"T0 offline MAE: {controller['t0_property_mae']:.6f}.",
            ]
    else:
        lines += ["", decision["reason"]]
    lines += [
        "", "No threshold, timestep, model, feature, candidate CFG, harm definition, or seed was changed after test outcomes.",
    ]
    text = "\n".join(lines) + "\n"
    (OFFLINE / "final_report.md").write_text(text)
    (ROOT / "final_report.md").write_text(text)
    (ROOT / "final_status.json").write_text(json.dumps(decision, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
