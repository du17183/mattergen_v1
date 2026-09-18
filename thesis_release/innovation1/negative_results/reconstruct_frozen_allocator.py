"""Deterministically reconstruct and verify the Phase-B Linear-K2 artifact once.

This is the only user-authorized reconstruction.  It uses the original Phase-B
training rows, train split, source code, feature list, and C=1.0.  No
confirmatory seeds may be registered unless every stored equivalence check
passes and the serialized pipeline is written successfully.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd


WORKTREE = Path(__file__).resolve().parents[2]
ROOT = WORKTREE / "experiments/frozen_linear_k2_confirmation"
PROTOCOL_ROOT = ROOT / "protocol"
PHASE_B_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/reference_preserved_budgeted_cfg")
PHASE_B_DATA = PHASE_B_ROOT / "feature_study"
PHASE_B_SOURCE = PHASE_B_ROOT / "analyze_feature_study.py"
EXPECTED_METHOD = "Linear_K2"
EXPECTED_C = 1.0
FLOAT_TOLERANCE = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def load_original_module():
    specification = importlib.util.spec_from_file_location("frozen_phase_b_analysis", PHASE_B_SOURCE)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load original Phase-B source")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def compare_metrics(actual: dict[str, Any], expected: pd.Series, names: list[str]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name in names:
        observed = float(actual[name])
        reference = float(expected[name])
        checks[name] = {
            "actual": observed,
            "expected": reference,
            "abs_error": abs(observed - reference),
            "exact_within_tolerance": abs(observed - reference) <= FLOAT_TOLERANCE,
        }
    return checks


def fit_exact_pipeline(module: Any, candidates: pd.DataFrame, feature_columns: list[str], train_seeds: list[int]):
    frame, model_columns = module.model_frame(candidates, feature_columns, interactions=True)
    train_frame = frame[frame.seed.isin(train_seeds)]
    numeric = [name for name in model_columns if name != "policy_id"]
    preprocessor = module.ColumnTransformer([
        ("numeric", module.StandardScaler(), numeric),
        ("policy", module.OneHotEncoder(categories=[list(module.POLICIES)], handle_unknown="ignore", sparse_output=False), ["policy_id"]),
    ])
    estimator = module.LogisticRegression(
        C=EXPECTED_C,
        class_weight="balanced",
        max_iter=5000,
        random_state=20260916,
    )
    pipeline = module.Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
    pipeline.fit(train_frame[model_columns], train_frame.label.to_numpy(int))
    return pipeline, frame, model_columns, numeric


def score_with_pipeline(pipeline: Any, frame: pd.DataFrame, model_columns: list[str], seeds: list[int]) -> tuple[dict[int, tuple[str, ...]], pd.DataFrame]:
    selected = frame[frame.seed.isin(seeds)].copy()
    estimator = pipeline.named_steps["estimator"]
    class_index = list(estimator.classes_).index(1)
    selected["score"] = pipeline.predict_proba(selected[model_columns])[:, class_index]
    assignments: dict[int, tuple[str, ...]] = {}
    for seed, group in selected.groupby("seed"):
        ordered = group.sort_values(["score", "policy_id"], ascending=[False, True])
        assignments[int(seed)] = tuple(map(str, ordered.policy_id.iloc[:2]))
    return assignments, selected[["seed", "split", "policy_id", "score", "label"]]


def main() -> None:
    if (PROTOCOL_ROOT / "frozen_linear_k2_pipeline.joblib").exists():
        raise FileExistsError("reconstructed artifact already exists and is immutable")
    PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    module = load_original_module()
    selection = json.loads((PHASE_B_DATA / "allocator_selection.json").read_text())
    if selection["validation_selected_allocator"] != EXPECTED_METHOD:
        raise RuntimeError("Phase-B selected method changed")
    if float(selection["selected_hyperparameters"][EXPECTED_METHOD]) != EXPECTED_C:
        raise RuntimeError("Phase-B C changed")
    feature_columns = list(map(str, selection["feature_columns"]))
    if len(feature_columns) != 179:
        raise RuntimeError(f"Phase-B feature count changed: {len(feature_columns)}")
    candidates = pd.read_csv(PHASE_B_DATA / "allocator_training_rows.csv")
    labeled = pd.read_csv(PHASE_B_DATA / "labeled_candidate_dataset.csv")
    for name in ("safe_beneficial", "evaluation_valid", "stable", "novel", "unique", "nus", "structure_valid"):
        if name in labeled:
            labeled[name] = labeled[name].astype(bool)
    train_seeds = list(map(int, selection["train_seeds"]))
    validation_seeds = list(map(int, selection["validation_seeds"]))
    test_seeds = list(map(int, selection["test_seeds"]))

    original_validation = module.fit_score_model(EXPECTED_METHOD, EXPECTED_C, candidates, feature_columns, train_seeds, validation_seeds)
    original_test = module.fit_score_model(EXPECTED_METHOD, EXPECTED_C, candidates, feature_columns, train_seeds, test_seeds)
    pipeline, frame, model_columns, numeric = fit_exact_pipeline(module, candidates, feature_columns, train_seeds)
    reconstructed_validation, validation_scores = score_with_pipeline(pipeline, frame, model_columns, validation_seeds)
    reconstructed_test, test_scores = score_with_pipeline(pipeline, frame, model_columns, test_seeds)

    expected_test_rows = pd.read_csv(PHASE_B_DATA / "allocator_test_selections.csv")
    expected_test_rows = expected_test_rows[expected_test_rows.method == EXPECTED_METHOD].set_index("seed")
    stored_test = {
        int(seed): (str(row.allocated_policy_1), str(row.allocated_policy_2))
        for seed, row in expected_test_rows.iterrows()
    }
    top2_checks = {
        "original_vs_reconstructed_validation": original_validation == reconstructed_validation,
        "original_vs_reconstructed_test": original_test == reconstructed_test,
        "reconstructed_vs_stored_test": reconstructed_test == stored_test,
        "test_seed_count": len(reconstructed_test),
        "test_exact_matches": sum(reconstructed_test[seed] == stored_test[seed] for seed in test_seeds),
    }

    validation_metrics, validation_outcomes = module.evaluate_selection(EXPECTED_METHOD, validation_seeds, reconstructed_validation, labeled)
    stored_validation = pd.read_csv(PHASE_B_DATA / "allocator_validation_results.csv")
    stored_validation = stored_validation[stored_validation.method == EXPECTED_METHOD].copy()
    stored_validation = stored_validation[stored_validation.hyperparameter.astype(float) == EXPECTED_C].iloc[0]
    validation_metric_checks = compare_metrics(
        validation_metrics,
        stored_validation,
        ["property_mae", "recovered_oracle_headroom", "safe_beneficial_coverage", "candidate_hit_rate", "top2_safe_oracle_recall", "fallback_rate", "e_hull", "stable", "novel", "unique", "nus", "validity"],
    )
    test_metrics, test_outcomes = module.evaluate_selection(EXPECTED_METHOD, test_seeds, reconstructed_test, labeled)
    stored_test_metrics = pd.read_csv(PHASE_B_DATA / "test_metrics_provisional.csv").set_index("method").loc[EXPECTED_METHOD]
    test_metric_checks = compare_metrics(
        test_metrics,
        stored_test_metrics,
        ["property_mae", "recovered_oracle_headroom", "safe_beneficial_coverage", "candidate_hit_rate", "top2_safe_oracle_recall", "fallback_rate"],
    )
    stored_terminal = expected_test_rows.terminal_policy.astype(str).to_dict()
    reconstructed_terminal = test_outcomes.set_index("seed").terminal_policy.astype(str).to_dict()
    terminal_exact = reconstructed_terminal == stored_terminal

    all_pass = bool(
        all(bool(value) for key, value in top2_checks.items() if key.startswith("original_") or key.startswith("reconstructed_"))
        and top2_checks["test_exact_matches"] == 16
        and all(check["exact_within_tolerance"] for check in validation_metric_checks.values())
        and all(check["exact_within_tolerance"] for check in test_metric_checks.values())
        and terminal_exact
    )
    audit = {
        "schema_version": 1,
        "artifact_status": "RECONSTRUCTED_ARTIFACT" if all_pass else "RECONSTRUCTION_FAILED",
        "authorized_once": True,
        "new_confirmatory_seeds_registered_before_reconstruction": False,
        "phase_b_historical_decision_unchanged": True,
        "phase_b_source_sha256": sha256(PHASE_B_SOURCE),
        "phase_b_training_rows_sha256": sha256(PHASE_B_DATA / "allocator_training_rows.csv"),
        "phase_b_selection_sha256": sha256(PHASE_B_DATA / "allocator_selection.json"),
        "feature_count_raw": len(feature_columns),
        "numeric_design_columns_after_interactions": len(numeric),
        "total_design_columns_after_policy_one_hot": len(numeric) + len(module.POLICIES),
        "C": EXPECTED_C,
        "train_seed_count": len(train_seeds),
        "top2_checks": top2_checks,
        "validation_metric_checks": validation_metric_checks,
        "test_metric_checks": test_metric_checks,
        "terminal_policy_exact": terminal_exact,
        "success": all_pass,
    }
    if not all_pass:
        (PROTOCOL_ROOT / "reconstruction_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        raise RuntimeError("RECONSTRUCTION_FAILED: no artifact was serialized")

    artifact_path = PROTOCOL_ROOT / "frozen_linear_k2_pipeline.joblib"
    joblib.dump(pipeline, artifact_path, compress=3)
    scaler = pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
    estimator = pipeline.named_steps["estimator"]
    parameters = {
        "feature_columns": feature_columns,
        "model_columns": model_columns,
        "numeric_columns": numeric,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classes": estimator.classes_.tolist(),
        "coef": estimator.coef_.tolist(),
        "intercept": estimator.intercept_.tolist(),
    }
    (PROTOCOL_ROOT / "frozen_linear_k2_parameters.json").write_text(json.dumps(parameters, separators=(",", ":")) + "\n")
    validation_scores.to_csv(PROTOCOL_ROOT / "reconstruction_validation_scores.csv", index=False)
    test_scores.to_csv(PROTOCOL_ROOT / "reconstruction_test_scores.csv", index=False)
    audit.update(
        artifact_sha256=sha256(artifact_path),
        parameters_sha256=sha256(PROTOCOL_ROOT / "frozen_linear_k2_parameters.json"),
        feature_list_sha256=canonical_hash(feature_columns),
        normalization_sha256=canonical_hash({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}),
        coefficient_sha256=canonical_hash({"coef": estimator.coef_.tolist(), "intercept": estimator.intercept_.tolist()}),
    )
    (PROTOCOL_ROOT / "reconstruction_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
