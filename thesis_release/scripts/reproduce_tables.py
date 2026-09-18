#!/usr/bin/env python3
"""Rebuild every thesis release table from frozen, compact result artifacts.

This script never launches MatterGen, MatterSim, CHGNet, or a GPU job.  All
reported deltas use ``baseline - method`` so positive values denote a reduction
for lower-is-better metrics.  Rate metrics retain the same arithmetic definition
and must be interpreted with their stated direction.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
I1 = ROOT / "innovation1"
I2 = ROOT / "innovation2"
SUMMARY = ROOT / "combined_summary"
N_BOOT = 20_000
BOOT_SEED = 20_260_915
EPS = 1.0e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}: {value}")
    return value


def bootstrap(values: np.ndarray, seed: int = BOOT_SEED) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap input must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "wins": int((values > EPS).sum()),
        "ties": int((np.abs(values) <= EPS).sum()),
        "losses": int((values < -EPS).sum()),
        "resamples": N_BOOT,
        "bootstrap_seed": seed,
        "delta_definition": "baseline_minus_method",
    }


def i1_tables() -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, Any]]:
    metrics = read_csv(I1 / "results/c1_metrics.csv")
    indexed = {row["method"]: row for row in metrics}
    expected = {"C0", "Fixed_K2", "Linear_K2", "Random_K2"}
    if set(indexed) != expected:
        raise ValueError(f"unexpected Innovation 1 methods: {set(indexed)}")

    compute = {"C0": 1.0, "Fixed_K2": 2.2, "Linear_K2": 2.2, "Random_K2": 2.2}
    fields = [
        "Method", "N", "Property MAE", "E-hull", "Stable", "Novel", "Unique",
        "NUS", "Validity", "Compute multiplier", "Fallback rate",
    ]
    rows: list[dict[str, Any]] = []
    numeric: dict[str, dict[str, float]] = {}
    for method in ("C0", "Fixed_K2", "Linear_K2", "Random_K2"):
        source = indexed[method]
        numeric[method] = {
            "property_mae": as_float(source, "property_mae"),
            "e_hull": as_float(source, "e_hull"),
            "stable": as_float(source, "stable"),
            "novel": as_float(source, "novel"),
            "unique": as_float(source, "unique"),
            "nus": as_float(source, "nus"),
            "validity": as_float(source, "validity"),
            "fallback_rate": as_float(source, "fallback_rate"),
        }
        rows.append({
            "Method": method,
            "N": int(float(source["n"])),
            "Property MAE": numeric[method]["property_mae"],
            "E-hull": numeric[method]["e_hull"],
            "Stable": numeric[method]["stable"],
            "Novel": numeric[method]["novel"],
            "Unique": numeric[method]["unique"],
            "NUS": numeric[method]["nus"],
            "Validity": numeric[method]["validity"],
            "Compute multiplier": compute[method],
            "Fallback rate": numeric[method]["fallback_rate"],
        })
    write_csv(I1 / "tables/final_confirmatory_results.csv", rows, fields)

    paired = read_csv(I1 / "results/c1_paired_results.csv")
    fixed_gain = np.asarray([as_float(row, "C0") - as_float(row, "Fixed_K2") for row in paired])
    fixed_boot = bootstrap(fixed_gain)
    fixed_boot["relative_improvement"] = float(
        fixed_gain.mean() / np.mean([as_float(row, "C0") for row in paired])
    )
    write_json(I1 / "results/fixed_vs_c0_bootstrap_20k.json", fixed_boot)
    return rows, numeric, fixed_boot


def find_bootstrap(payload: dict[str, Any], metric: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for value in payload.values():
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, dict))
    for row in entries:
        if row.get("metric") == metric:
            return row
    raise KeyError(metric)


def i2_formal_table() -> list[dict[str, Any]]:
    payload = json.loads((I2 / "results/mattersim_late_force_guidance_formal256/paired_bootstrap.json").read_text())
    specifications = [
        ("MaxF", "maxF_ev_per_a", "eV/angstrom", "lower"),
        ("Mean Force", "atomic_force_mean_ev_per_a", "eV/angstrom", "lower"),
        ("RMSD", "rmsd_a", "angstrom", "lower"),
        ("Property MAE", "mag_absolute_error_a3", "mu_B/angstrom^3", "lower"),
        ("Stable", "stable", "fraction", "higher"),
        ("Validity", "valid", "fraction", "higher"),
    ]
    rows = []
    for label, metric, unit, direction in specifications:
        source = find_bootstrap(payload, metric)
        rows.append({
            "Metric": label,
            "Unit": unit,
            "Preferred direction": direction,
            "Baseline": source["baseline"],
            "Method": "RC-NFGD",
            "N": source["n_pairs"],
            "Baseline value": source["baseline_mean"],
            "Method value": source["method_mean"],
            "Absolute improvement": source["absolute_improvement"],
            "Relative improvement": source["relative_improvement"],
            "CI low": source["relative_ci95"][0],
            "CI high": source["relative_ci95"][1],
            "Wins": source["wins"],
            "Ties": source["ties"],
            "Losses": source["losses"],
        })
    fields = list(rows[0])
    write_csv(I2 / "tables/formal256_main_results.csv", rows, fields)
    return rows


def i2_chgnet_table() -> list[dict[str, Any]]:
    source_rows = read_csv(I2 / "results/chgnet_formal256/independent_per_structure.csv")
    by_metric = [("MaxF", "maxF_ev_a"), ("Mean Force", "mean_force_ev_a")]
    rows: list[dict[str, Any]] = []
    for index, (label, column) in enumerate(by_metric):
        grouped: dict[str, dict[int, float]] = {"C0": {}, "F0": {}}
        for row in source_rows:
            grouped[row["method"]][int(row["seed"])] = as_float(row, column)
        if set(grouped["C0"]) != set(grouped["F0"]) or len(grouped["C0"]) != 256:
            raise ValueError("CHGNet paired seed mismatch")
        seeds = sorted(grouped["C0"])
        baseline = np.asarray([grouped["C0"][seed] for seed in seeds])
        method = np.asarray([grouped["F0"][seed] for seed in seeds])
        result = bootstrap(baseline - method, BOOT_SEED + index)
        rows.append({
            "Metric": label,
            "Evaluator": "CHGNet 0.3.0 independent surrogate",
            "N": len(seeds),
            "Baseline value": float(baseline.mean()),
            "Method value": float(method.mean()),
            "Absolute improvement": result["mean"],
            "Relative improvement": float(result["mean"] / baseline.mean()),
            "Absolute CI low": result["ci95_low"],
            "Absolute CI high": result["ci95_high"],
            "Wins": result["wins"],
            "Ties": result["ties"],
            "Losses": result["losses"],
            "DFT verified": False,
        })
    archived = json.loads((I2 / "results/chgnet_formal256/paired_bootstrap.json").read_text())[0]
    if abs(rows[0]["Absolute improvement"] - archived["absolute_improvement"]) > 1e-12:
        raise ValueError("CHGNet MaxF reconstruction differs from archived result")
    fields = list(rows[0])
    write_csv(I2 / "tables/chgnet_validation.csv", rows, fields)
    return rows


def experiment_status_table() -> None:
    rows = [
        {"Experiment": "Adaptive CFG V1", "Role": "exploratory", "Dataset / N": "Formal256 + fresh8", "Status": "MIXED", "Main finding": "initial directional result did not replicate robustly", "Confirmatory?": "No", "Source": "innovation1/negative_results/historical_status/adaptive_v1_formal.json"},
        {"Experiment": "Robust V2", "Role": "exploratory", "Dataset / N": "P0=32", "Status": "FAIL", "Main finding": "bounded residual rule did not reduce harm", "Confirmatory?": "No", "Source": "innovation1/negative_results/historical_status/robust_v2.json"},
        {"Experiment": "Counterfactual Oracle V3", "Role": "mechanism", "Dataset / N": "32 seeds; 64 full states", "Status": "SUPPORTED_MECHANISM", "Main finding": "27.22% oracle headroom; online short-horizon selection failed", "Confirmatory?": "No", "Source": "innovation1/negative_results/historical_status/oracle_v3.json"},
        {"Experiment": "Risk-Calibrated V4", "Role": "mechanism", "Dataset / N": "96 trajectories", "Status": "FAIL", "Main finding": "selective harm remained above frozen gate", "Confirmatory?": "No", "Source": "innovation1/negative_results/historical_status/risk_v4.json"},
        {"Experiment": "Safe Selection V5", "Role": "mechanism", "Dataset / N": "128 trajectories", "Status": "FAIL", "Main finding": "no validation point met coverage and harm gates", "Confirmatory?": "No", "Source": "innovation1/negative_results/historical_status/safe_v5.json"},
        {"Experiment": "Stage-Calibrated CFG", "Role": "mechanism", "Dataset / N": "calibration32 + P0=32", "Status": "MIXED", "Main finding": "bounded pulse supported; fresh P0 quality gate failed", "Confirmatory?": "No", "Source": "innovation1/negative_results/historical_status/stage_cfg.json"},
        {"Experiment": "Field-Decoupled CFG", "Role": "mechanism", "Dataset / N": "48", "Status": "FAIL", "Main finding": "no field policy passed all frozen gates", "Confirmatory?": "No", "Source": "innovation1/negative_results/historical_status/field_cfg.json"},
        {"Experiment": "Branch-Compatible Oracle", "Role": "mechanism", "Dataset / N": "48 historical; test=12", "Status": "SUPPORTED", "Main finding": "30.41% branchable oracle headroom", "Confirmatory?": "No", "Source": "innovation1/results/branch_oracle_status.json"},
        {"Experiment": "Phase B Linear-K2", "Role": "model selection", "Dataset / N": "64 (32/16/16)", "Status": "NOT_SUPPORTED", "Main finding": "directional only; B1 failed", "Confirmatory?": "Held-out", "Source": "innovation1/negative_results/phase_b_decision_summary.json"},
        {"Experiment": "C1 Fixed-K2", "Role": "confirmatory comparator/final method", "Dataset / N": "128", "Status": "SUPPORTED", "Main finding": "property MAE improved with safety fallback", "Confirmatory?": "Yes", "Source": "innovation1/results/c1_metrics.csv"},
        {"Experiment": "C1 Linear-K2", "Role": "confirmatory learned allocator", "Dataset / N": "128", "Status": "NOT_SUPPORTED", "Main finding": "no advantage over frozen Fixed-K2", "Confirmatory?": "Yes", "Source": "innovation1/results/c1_continuation_decision.json"},
        {"Experiment": "RC-NFGD P0", "Role": "pilot", "Dataset / N": "16", "Status": "PASS", "Main finding": "late force reduction", "Confirmatory?": "No", "Source": "innovation2/results/mattersim_late_force_guidance_p0/decision_summary.json"},
        {"Experiment": "RC-NFGD Formal32", "Role": "formal", "Dataset / N": "32", "Status": "PASS", "Main finding": "effect replicated", "Confirmatory?": "Yes", "Source": "innovation2/results/mattersim_late_force_guidance_formal32/decision_summary.json"},
        {"Experiment": "RC-NFGD Formal256", "Role": "formal", "Dataset / N": "256", "Status": "SUPPORTED", "Main finding": "strong MatterSim force reduction", "Confirmatory?": "Yes", "Source": "innovation2/results/mattersim_late_force_guidance_formal256/decision_summary.json"},
        {"Experiment": "CHGNet evaluation", "Role": "independent surrogate", "Dataset / N": "256", "Status": "POSITIVE", "Main finding": "force direction agrees; not DFT", "Confirmatory?": "Yes", "Source": "innovation2/results/chgnet_formal256/decision_summary.json"},
    ]
    write_csv(SUMMARY / "experiment_status.csv", rows, list(rows[0]))


def compute_summary(i1_numeric: dict[str, dict[str, float]]) -> None:
    formal = read_csv(I2 / "results/mattersim_late_force_guidance_formal256/paired_results.csv")
    runtime = {
        method: float(np.mean([as_float(row, f"{method}_end_to_end_generation_seconds") for row in formal]))
        for method in ("C0", "F0")
    }
    rows = [
        {"Innovation": 1, "Method": "C0", "Role": "deployment", "MatterGen score calls/seed": 2000, "Neural-potential selector calls/seed": 1, "Compute multiplier vs C0": 1.0, "Fallback rate": i1_numeric["C0"]["fallback_rate"], "Note": "exact reference"},
        {"Innovation": 1, "Method": "Fixed-K2", "Role": "deployment", "MatterGen score calls/seed": 4400, "Neural-potential selector calls/seed": 3, "Compute multiplier vs C0": 2.2, "Fallback rate": i1_numeric["Fixed_K2"]["fallback_rate"], "Note": "final frozen method"},
        {"Innovation": 1, "Method": "Linear-K2", "Role": "negative confirmatory", "MatterGen score calls/seed": 4400, "Neural-potential selector calls/seed": 3, "Compute multiplier vs C0": 2.2, "Fallback rate": i1_numeric["Linear_K2"]["fallback_rate"], "Note": "not supported over Fixed-K2"},
        {"Innovation": 1, "Method": "Phase B full candidate acquisition", "Role": "dataset construction", "MatterGen score calls/seed": "", "Neural-potential selector calls/seed": "", "Compute multiplier vs C0": 3.4, "Fallback rate": "", "Note": "not deployment cost"},
        {"Innovation": 2, "Method": "C0", "Role": "Formal256 deployment", "MatterGen score calls/seed": 2000, "Neural-potential selector calls/seed": 0, "Compute multiplier vs C0": 1.0, "Fallback rate": 0.0, "Note": f"mean end-to-end generation {runtime['C0']:.6f} s"},
        {"Innovation": 2, "Method": "RC-NFGD", "Role": "Formal256 deployment", "MatterGen score calls/seed": 2000, "Neural-potential selector calls/seed": 20, "Compute multiplier vs C0": runtime["F0"] / runtime["C0"], "Fallback rate": 0.0, "Note": f"mean end-to-end generation {runtime['F0']:.6f} s"},
    ]
    write_csv(SUMMARY / "compute_summary.csv", rows, list(rows[0]))


def main_results(
    i1_numeric: dict[str, dict[str, float]],
    fixed_boot: dict[str, Any],
    i2_rows: list[dict[str, Any]],
    chgnet_rows: list[dict[str, Any]],
) -> None:
    fields = ["innovation", "method", "cohort", "n", "metric", "preferred_direction", "baseline_value", "method_value", "absolute_delta", "relative_delta", "ci_low", "ci_high", "delta_definition", "status", "source_file"]
    rows: list[dict[str, Any]] = []

    def add(innovation: int, method: str, cohort: str, n: int, metric: str, direction: str, baseline: float, value: float, absolute: float, relative: float, low: Any, high: Any, status: str, source: str) -> None:
        rows.append(dict(zip(fields, [innovation, method, cohort, n, metric, direction, baseline, value, absolute, relative, low, high, "positive_is_favorable", status, source])))

    c0 = i1_numeric["C0"]
    fixed = i1_numeric["Fixed_K2"]
    linear = i1_numeric["Linear_K2"]
    add(1, "Fixed-K2", "C1-128", 128, "Property MAE", "lower", c0["property_mae"], fixed["property_mae"], fixed_boot["mean"], fixed_boot["relative_improvement"], fixed_boot["ci95_low"], fixed_boot["ci95_high"], "SUPPORTED", "innovation1/results/c1_paired_results.csv")
    for metric in ("e_hull", "stable", "nus", "validity"):
        direction = "lower" if metric == "e_hull" else "higher"
        improvement = c0[metric] - fixed[metric] if direction == "lower" else fixed[metric] - c0[metric]
        add(1, "Fixed-K2", "C1-128", 128, metric, direction, c0[metric], fixed[metric], improvement, improvement / max(abs(c0[metric]), EPS), "", "", "SUPPORTED_GUARDRAIL", "innovation1/results/c1_metrics.csv")
    linear_boot = json.loads((I1 / "results/c1_linear_bootstrap_20k.json").read_text())["linear_vs_fixed"]
    add(1, "Linear-K2", "C1-128", 128, "Property MAE vs Fixed-K2", "lower", fixed["property_mae"], linear["property_mae"], linear_boot["mean"], linear_boot["mean"] / fixed["property_mae"], linear_boot["ci95_low"], linear_boot["ci95_high"], "NOT_SUPPORTED", "innovation1/results/c1_linear_bootstrap_20k.json")
    branch = read_csv(I1 / "results/branch_oracle_metrics.csv")
    safe_a = next(
        row for row in branch
        if row.get("selector") == "SAFE-A" and row.get("split") == "offline_test"
    )
    oracle_value = as_float(safe_a, "property_mae")
    baseline_value = oracle_value + as_float(safe_a, "safe_oracle_gain")
    add(1, "Branchable Oracle-All", "historical test", 12, "Property MAE", "lower", baseline_value, oracle_value, baseline_value - oracle_value, as_float(safe_a, "relative_property_improvement_vs_c0"), "", "", "EXPLORATORY_UPPER_BOUND", "innovation1/results/branch_oracle_metrics.csv")

    for source in i2_rows:
        status = "SUPPORTED" if source["Metric"] in {"MaxF", "Mean Force", "RMSD"} else "GUARDRAIL"
        add(2, "RC-NFGD", "Formal256", int(source["N"]), source["Metric"], source["Preferred direction"], float(source["Baseline value"]), float(source["Method value"]), float(source["Absolute improvement"]), float(source["Relative improvement"]), source["CI low"], source["CI high"], status, "innovation2/results/mattersim_late_force_guidance_formal256/paired_bootstrap.json")
    for source in chgnet_rows:
        add(2, "RC-NFGD", "CHGNet Formal256", int(source["N"]), source["Metric"], "lower", float(source["Baseline value"]), float(source["Method value"]), float(source["Absolute improvement"]), float(source["Relative improvement"]), source["Absolute CI low"], source["Absolute CI high"], "SUPPORTED_INDEPENDENT_SURROGATE", "innovation2/results/chgnet_formal256/independent_per_structure.csv")
    write_csv(SUMMARY / "main_results.csv", rows, fields)


def scale_consistency_table() -> None:
    p0 = json.loads((I2 / "results/mattersim_late_force_guidance_p0/decision_summary.json").read_text())
    f32 = json.loads((I2 / "results/mattersim_late_force_guidance_formal32/decision_summary.json").read_text())
    f256 = json.loads((I2 / "results/mattersim_late_force_guidance_formal256/decision_summary.json").read_text())
    rows = [
        {"Cohort": "P0", "N": p0["n_paired"], "Relative MaxF reduction": p0["mean_maxF_improvement_fraction"], "CI low": p0["maxF_improvement_bootstrap_ci95_percent"][0] / 100.0, "CI high": p0["maxF_improvement_bootstrap_ci95_percent"][1] / 100.0, "Status": p0["MATTERSIM_FORCE_GUIDANCE_P0"]},
        {"Cohort": "Formal32", "N": f32["n_paired"], "Relative MaxF reduction": f32["relative_maxF_reduction"], "CI low": f32["relative_maxF_reduction_bootstrap_ci95"][0], "CI high": f32["relative_maxF_reduction_bootstrap_ci95"][1], "Status": f32["MATTERSIM_FORCE_GUIDANCE_FORMAL32"]},
        {"Cohort": "Formal256", "N": f256["n_paired"], "Relative MaxF reduction": f256["primary"]["relative_improvement"], "CI low": f256["primary"]["relative_ci95"][0], "CI high": f256["primary"]["relative_ci95"][1], "Status": f256["status"]},
    ]
    write_csv(I2 / "tables/effect_scale_consistency.csv", rows, list(rows[0]))


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    i1_rows, i1_numeric, fixed_boot = i1_tables()
    del i1_rows
    i2_rows = i2_formal_table()
    chgnet_rows = i2_chgnet_table()
    experiment_status_table()
    compute_summary(i1_numeric)
    main_results(i1_numeric, fixed_boot, i2_rows, chgnet_rows)
    scale_consistency_table()
    print("TABLE_REPRODUCTION=PASS")


if __name__ == "__main__":
    main()
