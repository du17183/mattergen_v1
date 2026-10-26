"""Leakage-controlled allocator training, one-shot test, and Phase-B gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from ase.io import read, write
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/reference_preserved_budgeted_cfg"
FEATURE_ROOT = ROOT / "feature_study"
PROTOCOL = yaml.safe_load((ROOT / "feature_study_protocol.yaml").read_text())
POLICIES = ("GPulse", "APulse", "PPulse", "CPulse")
METHODS = ("C0", "Fixed_K2", "Random_K2", "Residual_K2", "Linear_K2", "GBDT_K2", "Oracle_K2", "Oracle_All")
METHOD_COMPLEXITY = {"Residual_K2": 0, "Linear_K2": 1, "GBDT_K2": 2}
TOL = float(PROTOCOL["labels"]["property_tolerance"])
N_BOOT = 20_000
BOOTSTRAP_SEED = 2026091621
EPS = 1e-12


def json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")


def enrich_policy_quality() -> pd.DataFrame:
    values = pd.read_csv(FEATURE_ROOT / "property_metrics_raw.csv")
    for column, default in (("e_hull", np.nan), ("stable", False), ("nus", False), ("novel", False), ("unique", False), ("quality_success", False)):
        values[column] = default
    for policy in ("C0",) + POLICIES:
        structure_path = FEATURE_ROOT / "evaluation_structures" / f"{policy}.extxyz"
        quality_root = FEATURE_ROOT / "quality_policies" / policy
        if not structure_path.exists():
            continue
        if not (quality_root / "official_detailed.json.gz").exists():
            raise FileNotFoundError(f"missing policy quality: {policy}")
        atoms = read(structure_path, index=":")
        per = pd.read_csv(quality_root / "per_structure.csv")
        with gzip.open(quality_root / "official_detailed.json.gz", "rt") as stream:
            detailed = json.load(stream)
        if len(per) != len(detailed["energy_above_hull_per_atom"]):
            raise RuntimeError(f"quality length mismatch: {policy}")
        for output_index, row in per.reset_index(drop=True).iterrows():
            identifier = str(atoms[int(row["index"])].info["branch_id"])
            match = values.index[values.branch_id == identifier]
            if len(match) != 1:
                raise RuntimeError(f"branch lookup failed: {identifier}")
            target = match[0]
            values.loc[target, "e_hull"] = float(detailed["energy_above_hull_per_atom"][output_index])
            values.loc[target, "stable"] = bool(detailed["stable"][output_index])
            values.loc[target, "nus"] = bool(detailed["novel_unique_stable"][output_index])
            values.loc[target, "novel"] = bool(detailed["novel"][output_index])
            values.loc[target, "unique"] = bool(detailed["unique"][output_index])
            values.loc[target, "quality_success"] = True
    for column in ("structure_valid", "stable", "nus", "novel", "unique", "quality_success"):
        values[column] = values[column].fillna(False).astype(bool)
    values["evaluation_valid"] = values.structure_valid & values.quality_success
    values.to_csv(FEATURE_ROOT / "evaluated_policy_results.csv", index=False)
    return values


def add_labels(values: pd.DataFrame) -> pd.DataFrame:
    base = values[values.policy_id == "C0"].set_index("seed")
    if len(base) != values.seed.nunique():
        raise RuntimeError("C0 rows incomplete")
    rows: list[dict[str, Any]] = []
    for _, row in values.iterrows():
        c0 = base.loc[int(row.seed)]
        is_candidate = str(row.policy_id) != "C0"
        property_gain = float(c0.property_absolute_error - row.property_absolute_error)
        quality_comparable = bool(np.isfinite(row.e_hull) and np.isfinite(c0.e_hull))
        safe = bool(
            is_candidate
            and property_gain > TOL
            and int(bool(row.evaluation_valid)) >= int(bool(c0.evaluation_valid))
            and int(bool(row.stable)) >= int(bool(c0.stable))
            and quality_comparable
            and float(row.e_hull) <= float(c0.e_hull) + 0.01 + EPS
        )
        rows.append({**row.to_dict(), "c0_property_absolute_error": float(c0.property_absolute_error), "property_gain_vs_c0": property_gain, "safe_a": safe, "safe_beneficial": safe})
    labeled = pd.DataFrame(rows)
    candidates = labeled[labeled.policy_id.isin(POLICIES)].copy()
    candidates["policy_rank"] = candidates.groupby("seed").property_absolute_error.rank(method="first")
    best: dict[int, str] = {}
    for seed, frame in candidates.groupby("seed"):
        safe = frame[frame.safe_beneficial].sort_values(["property_absolute_error", "policy_id"])
        best[int(seed)] = "NONE" if safe.empty else str(safe.iloc[0].policy_id)
    labeled["best_safe_policy"] = labeled.seed.map(best)
    rank = candidates.set_index(["seed", "policy_id"]).policy_rank
    labeled["policy_rank"] = [float(rank.get((int(row.seed), str(row.policy_id)), np.nan)) for _, row in labeled.iterrows()]
    labeled.to_csv(FEATURE_ROOT / "labeled_candidate_dataset.csv", index=False)
    return labeled


def feature_columns(features: pd.DataFrame) -> list[str]:
    prohibited = {"seed", "split", "prefix_state_sha256", "branch_point", "trace_rows"}
    columns = [name for name in features.columns if name not in prohibited and pd.api.types.is_numeric_dtype(features[name])]
    if any(any(token in name.lower() for token in ("final", "property_absolute_error", "e_hull", "stable", "novel", "oracle", "label")) for name in columns):
        raise RuntimeError("future-leaking feature name detected")
    if not columns:
        raise RuntimeError("no numeric prebranch features")
    return columns


def candidate_frame(features: pd.DataFrame, labeled: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = feature_columns(features)
    base = features.set_index("seed")
    records = []
    labels = labeled[labeled.policy_id.isin(POLICIES)].set_index(["seed", "policy_id"])
    for seed in map(int, features.seed):
        for policy in POLICIES:
            record = {name: float(base.loc[seed, name]) for name in columns}
            record.update(seed=seed, split=str(base.loc[seed, "split"]), policy_id=policy, label=int(bool(labels.loc[(seed, policy), "safe_beneficial"])))
            records.append(record)
    return pd.DataFrame(records), columns


def outcome_for(seed: int, selected: tuple[str, ...], labeled: pd.DataFrame) -> pd.Series:
    seed_rows = labeled[labeled.seed == seed]
    c0 = seed_rows[seed_rows.policy_id == "C0"].iloc[0]
    eligible = seed_rows[seed_rows.policy_id.isin(selected) & seed_rows.safe_beneficial]
    return c0 if eligible.empty else eligible.sort_values(["property_absolute_error", "policy_id"]).iloc[0]


def metrics_for(name: str, seeds: list[int], selections: Mapping[int, tuple[str, ...]], labeled: pd.DataFrame, oracle_mae: float | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    result_rows = []
    for seed in seeds:
        selected = tuple(selections.get(seed, ()))
        outcome = outcome_for(seed, selected, labeled)
        seed_candidates = labeled[(labeled.seed == seed) & labeled.policy_id.isin(POLICIES)]
        positives = set(map(str, seed_candidates[seed_candidates.safe_beneficial].policy_id))
        selected_positive = positives & set(selected)
        best = str(seed_candidates.best_safe_policy.iloc[0])
        result_rows.append({
            **outcome.to_dict(), "method": name, "allocated_policies": ";".join(selected),
            "terminal_policy": str(outcome.policy_id), "safe_beneficial_total": len(positives),
            "safe_beneficial_selected": len(selected_positive), "candidate_hit": bool(selected_positive),
            "best_safe_available": best != "NONE", "best_safe_hit": best in set(selected),
            "fallback_to_c0": str(outcome.policy_id) == "C0",
        })
    results = pd.DataFrame(result_rows)
    c0_mae = float(labeled[(labeled.seed.isin(seeds)) & (labeled.policy_id == "C0")].property_absolute_error.mean())
    mae = float(results.property_absolute_error.mean())
    if oracle_mae is None:
        recovery = float("nan")
    else:
        recovery = (c0_mae - mae) / max(c0_mae - oracle_mae, EPS)
    available = results[results.best_safe_available]
    total_positive = int(results.safe_beneficial_total.sum())
    metrics = {
        "method": name, "n": len(results), "property_mae": mae,
        "recovered_oracle_headroom": recovery,
        "safe_beneficial_coverage": float(results.safe_beneficial_selected.sum() / max(total_positive, 1)),
        "candidate_hit_rate": float(results.candidate_hit.mean()),
        "top2_safe_oracle_recall": float(available.best_safe_hit.mean()) if len(available) else 1.0,
        "fallback_rate": float(results.fallback_to_c0.mean()),
        "e_hull": float(results.e_hull.mean()), "stable": float(results.stable.mean()),
        "novel": float(results.novel.mean()), "unique": float(results.unique.mean()),
        "nus": float(results.nus.mean()), "validity": float(results.evaluation_valid.mean()),
    }
    return metrics, results


def evaluate_selection(name: str, seeds: list[int], selections: Mapping[int, tuple[str, ...]], labeled: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    oracle = {seed: tuple(POLICIES) for seed in seeds}
    oracle_pre, _ = metrics_for("Oracle_All", seeds, oracle, labeled)
    return metrics_for(name, seeds, selections, labeled, oracle_mae=float(oracle_pre["property_mae"]))


def rank_key(metrics: Mapping[str, Any], tie: str = "") -> tuple[float, float, str]:
    return (-float(metrics["recovered_oracle_headroom"]), float(metrics["property_mae"]), tie)


def fixed_selection(train: list[int], validation: list[int], labeled: pd.DataFrame) -> tuple[tuple[str, str], list[dict[str, Any]]]:
    records = []
    subsets = list(combinations(POLICIES, 2))
    train_ranked = []
    for subset in subsets:
        metrics, _ = evaluate_selection("Fixed_K2", train, {seed: subset for seed in train}, labeled)
        train_ranked.append((rank_key(metrics, ";".join(subset)), subset, metrics))
    shortlist = [item[1] for item in sorted(train_ranked)[:3]]
    for _, subset, metrics in train_ranked:
        records.append({"stage": "train", "policies": ";".join(subset), "shortlisted": subset in shortlist, **metrics})
    validation_ranked = []
    for subset in shortlist:
        metrics, _ = evaluate_selection("Fixed_K2", validation, {seed: subset for seed in validation}, labeled)
        validation_ranked.append((rank_key(metrics, ";".join(subset)), subset, metrics))
        records.append({"stage": "validation", "policies": ";".join(subset), "shortlisted": True, **metrics})
    return sorted(validation_ranked)[0][1], records


def residual_selections(seeds: list[int], features: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    base = features.set_index("seed")
    result = {}
    for seed in seeds:
        row = base.loc[seed]
        atomic = float(row["predictor_atomic_residual_over_ema_last"])
        pos = float(row["predictor_pos_residual_over_ema_last"])
        cell = float(row["predictor_cell_residual_over_ema_last"])
        scores = {"APulse": atomic, "PPulse": pos, "CPulse": cell, "GPulse": float(np.mean([atomic, pos, cell]))}
        result[seed] = tuple(sorted(POLICIES, key=lambda policy: (-scores[policy], policy))[:2])
    return result


def random_selections(seeds: list[int]) -> dict[int, tuple[str, ...]]:
    salt = str(PROTOCOL["allocators"]["m1_random"]["salt"])
    result = {}
    for seed in seeds:
        scored = sorted(POLICIES, key=lambda policy: hashlib.sha256(f"{salt}:{seed}:{policy}".encode()).hexdigest())
        result[seed] = tuple(scored[:2])
    return result


def model_frame(candidates: pd.DataFrame, columns: list[str], interactions: bool) -> tuple[pd.DataFrame, list[str]]:
    frame = candidates[["seed", "split", "policy_id", "label", *columns]].copy()
    if interactions:
        for policy in POLICIES:
            indicator = (frame.policy_id == policy).astype(float)
            for column in columns:
                frame[f"ix_{policy}_{column}"] = indicator * frame[column].astype(float)
    return frame, [name for name in frame.columns if name not in {"seed", "split", "label"}]


def fit_score_model(kind: str, hyperparameter: float | int, candidates: pd.DataFrame, columns: list[str], train: list[int], score_seeds: list[int]) -> dict[int, tuple[str, ...]]:
    interactions = kind == "Linear_K2"
    frame, model_columns = model_frame(candidates, columns, interactions)
    train_frame = frame[frame.seed.isin(train)]
    score_frame = frame[frame.seed.isin(score_seeds)]
    numeric = [name for name in model_columns if name != "policy_id"]
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler() if kind == "Linear_K2" else "passthrough", numeric),
        ("policy", OneHotEncoder(categories=[list(POLICIES)], handle_unknown="ignore", sparse_output=False), ["policy_id"]),
    ])
    if kind == "Linear_K2":
        estimator = LogisticRegression(C=float(hyperparameter), class_weight="balanced", max_iter=5000, random_state=20260916)
    else:
        estimator = GradientBoostingClassifier(max_depth=int(hyperparameter), n_estimators=100, learning_rate=0.05, random_state=20260916)
    pipeline = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
    labels = train_frame.label.to_numpy(int)
    if len(np.unique(labels)) < 2:
        scores = np.full(len(score_frame), float(labels[0]), dtype=float)
    else:
        pipeline.fit(train_frame[model_columns], labels)
        scores = pipeline.predict_proba(score_frame[model_columns])[:, list(pipeline.named_steps["estimator"].classes_).index(1)]
    scored = score_frame[["seed", "policy_id"]].copy()
    scored["score"] = scores
    result = {}
    for seed, group in scored.groupby("seed"):
        ordered = group.sort_values(["score", "policy_id"], ascending=[False, True])
        result[int(seed)] = tuple(map(str, ordered.policy_id.iloc[:2]))
    return result


def train_allocators(candidates: pd.DataFrame, columns: list[str], train: list[int], validation: list[int], features: pd.DataFrame, labeled: pd.DataFrame) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    records = []
    configurations: dict[str, Any] = {"Residual_K2": None}
    residual = residual_selections(validation, features)
    metrics, _ = evaluate_selection("Residual_K2", validation, residual, labeled)
    records.append({"method": "Residual_K2", "hyperparameter": "frozen", **metrics})
    for c_value in PROTOCOL["allocators"]["m3_linear"]["C_grid"]:
        selections = fit_score_model("Linear_K2", float(c_value), candidates, columns, train, validation)
        metrics, _ = evaluate_selection("Linear_K2", validation, selections, labeled)
        records.append({"method": "Linear_K2", "hyperparameter": float(c_value), **metrics})
    for depth in PROTOCOL["allocators"]["m4_gbdt"]["max_depth_grid"]:
        selections = fit_score_model("GBDT_K2", int(depth), candidates, columns, train, validation)
        metrics, _ = evaluate_selection("GBDT_K2", validation, selections, labeled)
        records.append({"method": "GBDT_K2", "hyperparameter": int(depth), **metrics})
    frame = pd.DataFrame(records)
    best_by_method = {}
    for method, group in frame.groupby("method"):
        selected = group.sort_values(["recovered_oracle_headroom", "property_mae", "hyperparameter"], ascending=[False, True, True]).iloc[0]
        best_by_method[method] = selected
        configurations[method] = selected.hyperparameter
    best_performance = sorted(best_by_method.values(), key=lambda row: rank_key(row, str(row.method)))[0]
    recovery_tolerance = float(PROTOCOL["allocators"]["simple_model_preference"]["recovered_headroom_tolerance_points"]) / 100.0
    mae_tolerance = float(PROTOCOL["allocators"]["simple_model_preference"]["property_mae_relative_tolerance"])
    eligible = []
    for method, row in best_by_method.items():
        if float(best_performance.recovered_oracle_headroom) - float(row.recovered_oracle_headroom) <= recovery_tolerance + EPS and float(row.property_mae) <= float(best_performance.property_mae) * (1.0 + mae_tolerance) + EPS:
            eligible.append(method)
    selected_method = min(eligible, key=lambda name: METHOD_COMPLEXITY[name])
    configurations["selected_method"] = selected_method
    configurations["feature_columns"] = columns
    configurations["validation_best_by_method"] = {method: row.to_dict() for method, row in best_by_method.items()}
    return selected_method, configurations, records


def oracle_k2(seeds: list[int], labeled: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    result = {}
    for seed in seeds:
        frame = labeled[(labeled.seed == seed) & labeled.policy_id.isin(POLICIES)].copy()
        frame["oracle_order"] = np.where(frame.safe_beneficial, frame.property_absolute_error, np.inf)
        result[seed] = tuple(map(str, frame.sort_values(["oracle_order", "policy_id"]).policy_id.iloc[:2]))
    return result


def write_mixed_structures(all_results: Mapping[str, pd.DataFrame]) -> None:
    destination = FEATURE_ROOT / "mixed_evaluation_structures"
    destination.mkdir(exist_ok=False)
    atoms = read(FEATURE_ROOT / "policy_final_all.extxyz", index=":")
    lookup = {(int(atom.info["seed"]), str(atom.info["policy_id"])): atom for atom in atoms}
    for method, results in all_results.items():
        selected = []
        for _, row in results.sort_values("seed").iterrows():
            if not bool(row.structure_valid):
                continue
            atom = lookup[(int(row.seed), str(row.terminal_policy))].copy()
            atom.info.update(sample_seed=int(row.seed), branch_id=str(row.branch_id), allocator_method=method)
            selected.append(atom)
        if selected:
            write(destination / f"{method}.extxyz", selected)


def selection_stage() -> None:
    if (FEATURE_ROOT / "allocator_selection.json").exists():
        raise FileExistsError("allocator selection is immutable once written")
    labeled = add_labels(enrich_policy_quality())
    features = pd.read_csv(FEATURE_ROOT / "prebranch_features.csv")
    candidates, columns = candidate_frame(features, labeled)
    candidates.to_csv(FEATURE_ROOT / "allocator_training_rows.csv", index=False)
    seed_manifest = json.loads((FEATURE_ROOT / "feature_study_seed_manifest.json").read_text())
    train, validation, test = [list(map(int, seed_manifest["splits"][name])) for name in ("train", "validation", "test")]
    fixed_subset, fixed_records = fixed_selection(train, validation, labeled)
    pd.DataFrame(fixed_records).to_csv(FEATURE_ROOT / "fixed_k2_selection.csv", index=False)
    selected_method, configuration, model_records = train_allocators(candidates, columns, train, validation, features, labeled)
    pd.DataFrame(model_records).to_csv(FEATURE_ROOT / "allocator_validation_results.csv", index=False)
    test_selections: dict[str, dict[int, tuple[str, ...]]] = {
        "C0": {seed: tuple() for seed in test},
        "Fixed_K2": {seed: fixed_subset for seed in test},
        "Random_K2": random_selections(test),
        "Residual_K2": residual_selections(test, features),
        "Linear_K2": fit_score_model("Linear_K2", configuration["Linear_K2"], candidates, columns, train, test),
        "GBDT_K2": fit_score_model("GBDT_K2", configuration["GBDT_K2"], candidates, columns, train, test),
        "Oracle_K2": oracle_k2(test, labeled),
        "Oracle_All": {seed: tuple(POLICIES) for seed in test},
    }
    oracle_pre, _ = metrics_for("Oracle_All", test, test_selections["Oracle_All"], labeled)
    oracle_mae = float(oracle_pre["property_mae"])
    metric_rows, result_frames, selection_rows = [], {}, []
    for method in METHODS:
        metrics, results = metrics_for(method, test, test_selections[method], labeled, oracle_mae=oracle_mae)
        metric_rows.append(metrics)
        result_frames[method] = results
        for seed, allocated in test_selections[method].items():
            terminal = results[results.seed == seed].iloc[0]
            selection_rows.append({
                "split": "test", "method": method, "seed": seed,
                "allocated_policy_1": allocated[0] if len(allocated) > 0 else "",
                "allocated_policy_2": allocated[1] if len(allocated) > 1 else "",
                "allocated_policy_count": len(allocated), "terminal_policy": str(terminal.terminal_policy),
                "branch_id": str(terminal.branch_id), "best_safe_policy": str(terminal.best_safe_policy),
                "best_safe_hit": bool(terminal.best_safe_hit), "fallback_to_c0": bool(terminal.fallback_to_c0),
            })
    pd.DataFrame(metric_rows).to_csv(FEATURE_ROOT / "test_metrics_provisional.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(FEATURE_ROOT / "allocator_test_selections.csv", index=False)
    pd.concat(result_frames.values(), ignore_index=True).to_csv(FEATURE_ROOT / "allocator_test_outcomes.csv", index=False)
    write_mixed_structures(result_frames)
    selection = {
        "schema_version": 1, "test_not_inspected_before_selection": True,
        "best_fixed_k2": list(fixed_subset), "validation_selected_allocator": selected_method,
        "selected_hyperparameters": {key: value for key, value in configuration.items() if key != "feature_columns"},
        "feature_count": len(columns), "feature_columns": columns,
        "train_seeds": train, "validation_seeds": validation, "test_seeds": test,
    }
    json_write(FEATURE_ROOT / "allocator_selection.json", selection)
    print(json.dumps({"selected_allocator": selected_method, "fixed_k2": fixed_subset, "feature_count": len(columns), "test_n": len(test)}))


def mixed_quality(method: str) -> dict[str, float]:
    structure_path = FEATURE_ROOT / "mixed_evaluation_structures" / f"{method}.extxyz"
    quality_root = FEATURE_ROOT / "quality_mixed" / method
    if not structure_path.exists():
        return {name: 0.0 if name != "e_hull" else float("nan") for name in ("e_hull", "stable", "novel", "unique", "nus", "validity")}
    atoms = read(structure_path, index=":")
    per = pd.read_csv(quality_root / "per_structure.csv")
    with gzip.open(quality_root / "official_detailed.json.gz", "rt") as stream:
        detailed = json.load(stream)
    if len(per) != len(atoms) or len(per) != len(detailed["stable"]):
        raise RuntimeError(f"mixed quality length mismatch: {method}")
    return {
        "e_hull": float(np.mean(detailed["energy_above_hull_per_atom"])),
        "stable": float(np.mean(detailed["stable"])),
        "novel": float(np.mean(detailed["novel"])),
        "unique": float(np.mean(detailed["unique"])),
        "nus": float(np.mean(detailed["novel_unique_stable"])),
        "validity": float(len(atoms) / int(PROTOCOL["cohort"]["split_counts"]["test"])),
    }


def bootstrap(values: np.ndarray, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    return {"n": len(values), "mean": float(values.mean()), "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)), "resamples": N_BOOT}


def finalize_stage() -> None:
    if (FEATURE_ROOT / "phase_b_decision_summary.json").exists():
        raise FileExistsError("Phase-B final decision already exists")
    selection = json.loads((FEATURE_ROOT / "allocator_selection.json").read_text())
    selected_method = str(selection["validation_selected_allocator"])
    metrics = pd.read_csv(FEATURE_ROOT / "test_metrics_provisional.csv").set_index("method")
    for method in METHODS:
        quality = mixed_quality(method)
        for name, value in quality.items():
            metrics.loc[method, name] = value
    metrics.reset_index().to_csv(FEATURE_ROOT / "allocator_test_metrics.csv", index=False)
    adaptive = metrics.loc[selected_method]
    fixed = metrics.loc["Fixed_K2"]
    random = metrics.loc["Random_K2"]
    recovery_advantage = float(adaptive.recovered_oracle_headroom - fixed.recovered_oracle_headroom)
    mae_relative_improvement = float((fixed.property_mae - adaptive.property_mae) / max(float(fixed.property_mae), EPS))
    b1 = bool(recovery_advantage >= 0.15 - EPS or mae_relative_improvement >= 0.05 - EPS)
    b2 = bool(
        float(adaptive.e_hull) <= float(fixed.e_hull) + 0.01 + EPS
        and float(adaptive.stable) >= float(fixed.stable) - EPS
        and float(adaptive.validity) >= float(fixed.validity) - EPS
    )
    b3 = bool(float(adaptive.property_mae) < float(random.property_mae) - EPS)
    b4 = bool(float(adaptive.top2_safe_oracle_recall) >= 0.60 - EPS)
    supported = b1 and b2 and b3 and b4
    outcomes = pd.read_csv(FEATURE_ROOT / "allocator_test_outcomes.csv")
    bootstrap_results = {}
    for offset, comparator in enumerate(("Fixed_K2", "Random_K2")):
        left = outcomes[outcomes.method == selected_method].set_index("seed").property_absolute_error
        right = outcomes[outcomes.method == comparator].set_index("seed").property_absolute_error
        common = sorted(set(left.index) & set(right.index))
        bootstrap_results[f"{selected_method}_property_gain_vs_{comparator}"] = bootstrap((right.loc[common] - left.loc[common]).to_numpy(float), BOOTSTRAP_SEED + offset)
    json_write(FEATURE_ROOT / "allocator_test_bootstrap_20k.json", bootstrap_results)
    decision = {
        "PREBRANCH_FEATURE_DATASET": "COMPLETE",
        "ALLOCATOR_TEST": "COMPLETE",
        "BEST_ALLOCATOR": selected_method.removesuffix("_K2").upper() if supported else "NONE",
        "VALIDATION_SELECTED_ALLOCATOR": selected_method,
        "BEST_FIXED_K2": selection["best_fixed_k2"],
        "BEST_FIXED_K2_TEST_MAE": float(fixed.property_mae),
        "BEST_ADAPTIVE_K2_TEST_MAE": float(adaptive.property_mae),
        "FIXED_K2_RECOVERED_HEADROOM": float(fixed.recovered_oracle_headroom),
        "ADAPTIVE_K2_RECOVERED_HEADROOM": float(adaptive.recovered_oracle_headroom),
        "TOP2_SAFE_ORACLE_RECALL": float(adaptive.top2_safe_oracle_recall),
        "ADAPTIVE_ALLOCATION_VALUE": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "PHASE_B_GATES": {"B1_allocator_advantage": b1, "B2_quality": b2, "B3_random": b3, "B4_recall": b4},
        "RECOVERY_ADVANTAGE_POINTS": 100.0 * recovery_advantage,
        "PROPERTY_MAE_RELATIVE_IMPROVEMENT": mae_relative_improvement,
        "INNOVATION1_FINAL_STATUS": "MIXED",
        "DFT_VERIFIED": False,
        "NEXT": "START_PHASE_C" if supported else "STOP",
        "STOP_REASON": None if supported else "Branch-point state features did not provide enough held-out candidate-allocation value beyond fixed best-of-K under the frozen gates.",
    }
    json_write(FEATURE_ROOT / "phase_b_decision_summary.json", decision)
    pipeline_path = ROOT / "pipeline_status.json"
    pipeline = json.loads(pipeline_path.read_text())
    pipeline.update(decision)
    pipeline["stage"] = "phase_b_complete_supported" if supported else "stopped_after_phase_b_not_supported"
    temporary = pipeline_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(pipeline, indent=2) + "\n")
    temporary.replace(pipeline_path)
    plots(metrics.reset_index(), selected_method)
    report(decision, metrics.reset_index(), bootstrap_results)
    print(json.dumps(decision, indent=2))


def plots(metrics: pd.DataFrame, selected_method: str) -> None:
    destination = FEATURE_ROOT / "plots"
    destination.mkdir(exist_ok=True)
    view = metrics.set_index("method").loc[list(METHODS)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(view)), view.property_mae)
    ax.set_xticks(range(len(view)), view.index, rotation=30, ha="right")
    ax.set_ylabel("Held-out property MAE")
    fig.tight_layout(); fig.savefig(destination / "test_property_mae.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(view)), view.recovered_oracle_headroom)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(view)), view.index, rotation=30, ha="right")
    ax.set_ylabel("Recovered Oracle headroom")
    fig.tight_layout(); fig.savefig(destination / "test_recovered_headroom.png", dpi=180); plt.close(fig)
    selected = view.loc[["Fixed_K2", "Random_K2", selected_method]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, column in zip(axes, ("top2_safe_oracle_recall", "stable", "validity")):
        ax.bar(range(len(selected)), selected[column])
        ax.set_xticks(range(len(selected)), selected.index, rotation=25, ha="right")
        ax.set_title(column)
    fig.tight_layout(); fig.savefig(destination / "allocator_recall_quality.png", dpi=180); plt.close(fig)


def report(decision: Mapping[str, Any], metrics: pd.DataFrame, bootstrap_results: Mapping[str, Any]) -> None:
    selected = str(decision["VALIDATION_SELECTED_ALLOCATOR"])
    rows = metrics.set_index("method")
    lines = [
        "# Phase B final report", "",
        "This is a fresh-seed, surrogate-only study. No DFT verification is claimed.", "",
        "## Decision", "",
        f"ADAPTIVE_ALLOCATION_VALUE = {decision['ADAPTIVE_ALLOCATION_VALUE']}",
        f"Validation-selected allocator = {selected}",
        f"Best fixed K2 = {' + '.join(decision['BEST_FIXED_K2'])}",
        f"Held-out fixed K2 MAE = {decision['BEST_FIXED_K2_TEST_MAE']:.8f}",
        f"Held-out adaptive K2 MAE = {decision['BEST_ADAPTIVE_K2_TEST_MAE']:.8f}",
        f"Fixed recovered headroom = {decision['FIXED_K2_RECOVERED_HEADROOM']:.4f}",
        f"Adaptive recovered headroom = {decision['ADAPTIVE_K2_RECOVERED_HEADROOM']:.4f}",
        f"Adaptive Top-2 safe-oracle recall = {decision['TOP2_SAFE_ORACLE_RECALL']:.4f}", "",
        "## Frozen gates", "",
    ]
    lines.extend(f"{name} = {'PASS' if passed else 'FAIL'}" for name, passed in decision["PHASE_B_GATES"].items())
    lines.extend(["", "## Held-out test metrics", "", "| Method | Property MAE | Recovered headroom | Top2 recall | Stable | NUS | Validity |", "|---|---:|---:|---:|---:|---:|---:|"])
    for method in METHODS:
        row = rows.loc[method]
        lines.append(f"| {method} | {row.property_mae:.8f} | {row.recovered_oracle_headroom:.4f} | {row.top2_safe_oracle_recall:.4f} | {row.stable:.4f} | {row.nus:.4f} | {row.validity:.4f} |")
    lines.extend(["", "Unique and NUS above were recomputed on each method's final mixed held-out batch.", "", "## Bootstrap", "", f"All paired intervals use {N_BOOT:,} deterministic resamples."])
    if decision["ADAPTIVE_ALLOCATION_VALUE"] == "NOT_SUPPORTED":
        lines.extend(["", "Phase C is prohibited by the preregistered stop rule. Innovation 1 remains MIXED."])
    else:
        lines.extend(["", "All Phase-B gates passed; Phase C may start under its separate exact-reproduction gate."])
    (FEATURE_ROOT / "final_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("select", "finalize"), required=True)
    arguments = parser.parse_args()
    selection_stage() if arguments.stage == "select" else finalize_stage()
