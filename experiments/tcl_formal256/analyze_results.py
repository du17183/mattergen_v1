"""Formal256 absolute, superiority, non-inferiority, tail, and replication analysis."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/tcl_formal256"
P0_ROOT = PROJECT_ROOT / "experiments/tcl_dml_p0"
P1_ROOT = PROJECT_ROOT / "experiments/tcl_p1"
P2_ROOT = PROJECT_ROOT / "experiments/tcl_p2"
P1_ANALYSIS = P1_ROOT / "analyze_results.py"
METHODS = ("C0", "FT0", "TCL")
SEEDS = tuple(range(83000, 83256))
COMPARISONS = (("FT0", "C0"), ("TCL", "C0"), ("TCL", "FT0"))
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260912
HOLM_ALPHA = 0.05
PRIMARY = (
    "e_hull", "nus", "rmsd", "atomic_force_mean", "structure_max_force",
)
METRICS = {
    "e_hull": ("lower", "eV/atom", 1.0),
    "stable": ("higher", "percentage_points", 100.0),
    "nus": ("higher", "percentage_points", 100.0),
    "novel": ("higher", "percentage_points", 100.0),
    "unique": ("higher", "percentage_points", 100.0),
    "rmsd": ("lower", "angstrom", 1.0),
    "atomic_force_mean": ("lower", "eV/angstrom", 1.0),
    "structure_max_force": ("lower", "eV/angstrom", 1.0),
    "force_gt1": ("lower", "percentage_points", 100.0),
    "force_gt2": ("lower", "percentage_points", 100.0),
    "relaxation_steps": ("lower", "steps", 1.0),
}


def load_frozen_loader():
    spec = importlib.util.spec_from_file_location("frozen_p1_analysis", P1_ANALYSIS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.SEEDS = SEEDS
    return module


def quantile(values: np.ndarray, probability: float, axis=None):
    return np.quantile(np.asarray(values, dtype=float), probability, axis=axis)


def qfloat(values: np.ndarray, probability: float) -> float:
    return float(quantile(values, probability))


def finite_or_raise(method: str, values: dict[str, np.ndarray]) -> None:
    for key, array in values.items():
        if key == "formula":
            continue
        numeric = np.asarray(array, dtype=float)
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"{method}: non-finite values in {key}")


def summary_row(method: str, values: dict[str, np.ndarray]) -> dict[str, object]:
    e_hull = values["e_hull"]
    rmsd = values["rmsd"]
    atomic_force = values["atomic_force_mean"]
    max_force = values["structure_max_force"]
    steps = values["relaxation_steps"]
    row: dict[str, object] = {
        "method": method,
        "n": len(SEEDS),
        "generation_success_count": len(SEEDS),
        "generation_success_rate": 1.0,
        "mattersim_success_count": len(SEEDS),
        "mattersim_success_rate": 1.0,
        "dft_verified": False,
    }
    for name, array, unit in (
        ("e_hull", e_hull, "ev_per_atom"),
        ("rmsd", rmsd, "angstrom"),
    ):
        row.update({
            f"{name}_mean_{unit}": float(array.mean()),
            f"{name}_median_{unit}": float(np.median(array)),
            f"{name}_p90_{unit}": qfloat(array, 0.90),
            f"{name}_p95_{unit}": qfloat(array, 0.95),
            f"{name}_p99_{unit}": qfloat(array, 0.99),
            f"{name}_max_{unit}": float(array.max()),
        })
    for name in ("stable", "nus", "novel", "unique"):
        array = values[name]
        row[f"{name}_count"] = int(array.sum())
        row[f"{name}_rate"] = float(array.mean())
    row.update({
        "atomic_force_mean_ev_per_angstrom": float(atomic_force.mean()),
        "atomic_force_median_ev_per_angstrom": float(np.median(atomic_force)),
        "atomic_force_p95_ev_per_angstrom": qfloat(atomic_force, 0.95),
        "atomic_force_p99_ev_per_angstrom": qfloat(atomic_force, 0.99),
        "atomic_force_max_ev_per_angstrom": float(atomic_force.max()),
        "structure_max_force_mean_ev_per_angstrom": float(max_force.mean()),
        "structure_max_force_median_ev_per_angstrom": float(np.median(max_force)),
        "structure_max_force_p90_ev_per_angstrom": qfloat(max_force, 0.90),
        "structure_max_force_p95_ev_per_angstrom": qfloat(max_force, 0.95),
        "structure_max_force_p99_ev_per_angstrom": qfloat(max_force, 0.99),
        "structure_max_force_max_ev_per_angstrom": float(max_force.max()),
        "structure_max_force_gt1_count": int((max_force > 1.0).sum()),
        "structure_max_force_gt1_rate": float((max_force > 1.0).mean()),
        "structure_max_force_gt2_count": int((max_force > 2.0).sum()),
        "structure_max_force_gt2_rate": float((max_force > 2.0).mean()),
        "relaxation_steps_mean": float(steps.mean()),
        "relaxation_steps_median": float(np.median(steps)),
        "relaxation_steps_p90": qfloat(steps, 0.90),
        "relaxation_steps_p95": qfloat(steps, 0.95),
        "relaxation_steps_p99": qfloat(steps, 0.99),
        "relaxation_steps_max": int(steps.max()),
        "relaxation_steps_gt100_count": int((steps > 100).sum()),
        "relaxation_steps_gt100_rate": float((steps > 100).mean()),
        "relaxation_steps_gt200_count": int((steps > 200).sum()),
        "relaxation_steps_gt200_rate": float((steps > 200).mean()),
        "relaxation_steps_gt300_count": int((steps > 300).sum()),
        "relaxation_steps_gt300_rate": float((steps > 300).mean()),
        "relaxation_steps_gt400_count": int((steps > 400).sum()),
        "relaxation_steps_gt400_rate": float((steps > 400).mean()),
        "generation_seconds_mean": float(values["generation_seconds"].mean()),
        "generation_seconds_total": float(values["generation_seconds"].sum()),
    })
    return row


def null_centered_pvalue(delta: np.ndarray, indices: np.ndarray, direction: str) -> float:
    observed = float(delta.mean())
    centered = delta - observed
    null_means = centered[indices].mean(axis=1)
    if direction == "lower":
        extreme = int((null_means <= observed).sum())
    else:
        extreme = int((null_means >= observed).sum())
    return float((extreme + 1) / (len(null_means) + 1))


def paired_rows(
    data: dict[str, dict[str, np.ndarray]], indices: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for method, baseline in COMPARISONS:
        for metric, (direction, unit, scale) in METRICS.items():
            candidate_raw = data[method][metric]
            baseline_raw = data[baseline][metric]
            raw_delta = candidate_raw - baseline_raw
            delta = raw_delta * scale
            bootstrap = delta[indices].mean(axis=1)
            tolerance = 1e-12
            favorable = raw_delta < -tolerance if direction == "lower" else raw_delta > tolerance
            unfavorable = raw_delta > tolerance if direction == "lower" else raw_delta < -tolerance
            ties = ~(favorable | unfavorable)
            observed = float(delta.mean())
            favorable_direction = observed < 0 if direction == "lower" else observed > 0
            pvalue = null_centered_pvalue(delta, indices, direction)
            rows.append({
                "comparison": f"{method}-{baseline}",
                "metric": metric,
                "preferred_direction": direction,
                "unit": unit,
                "n_pairs": len(SEEDS),
                "candidate_mean": float(candidate_raw.mean() * scale),
                "baseline_mean": float(baseline_raw.mean() * scale),
                "mean_delta_candidate_minus_baseline": observed,
                "median_paired_delta": float(np.median(delta)),
                "bootstrap_ci95_low": qfloat(bootstrap, 0.025),
                "bootstrap_ci95_high": qfloat(bootstrap, 0.975),
                "bootstrap_resamples": N_BOOTSTRAP,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "favorable_pairs": int(favorable.sum()),
                "ties": int(ties.sum()),
                "unfavorable_pairs": int(unfavorable.sum()),
                "raw_one_sided_pvalue": pvalue,
                "raw_superiority_supported": bool(favorable_direction and pvalue < HOLM_ALPHA),
                "holm_adjusted_pvalue": np.nan,
                "holm_superiority_supported": False,
            })
    return rows


def add_holm(paired: pd.DataFrame) -> pd.DataFrame:
    mask = (paired["comparison"] == "TCL-FT0") & paired["metric"].isin(PRIMARY)
    selected = paired.loc[mask]
    if set(selected["metric"]) != set(PRIMARY) or len(selected) != len(PRIMARY):
        raise RuntimeError("primary Holm family mismatch")
    pvalues = selected["raw_one_sided_pvalue"].to_numpy(dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[position])
        adjusted[position] = min(running, 1.0)
    paired.loc[selected.index, "holm_adjusted_pvalue"] = adjusted
    favorable = np.where(
        selected["preferred_direction"].to_numpy() == "lower",
        selected["mean_delta_candidate_minus_baseline"].to_numpy(dtype=float) < 0,
        selected["mean_delta_candidate_minus_baseline"].to_numpy(dtype=float) > 0,
    )
    paired.loc[selected.index, "holm_superiority_supported"] = (
        favorable & (adjusted < HOLM_ALPHA)
    )
    return paired


def quantile_delta_row(
    data: dict[str, dict[str, np.ndarray]], indices: np.ndarray, probability: float,
) -> dict[str, object]:
    candidate = data["TCL"]["structure_max_force"]
    baseline = data["FT0"]["structure_max_force"]
    observed = qfloat(candidate, probability) - qfloat(baseline, probability)
    bootstrap = (
        quantile(candidate[indices], probability, axis=1)
        - quantile(baseline[indices], probability, axis=1)
    )
    low, high = qfloat(bootstrap, 0.025), qfloat(bootstrap, 0.975)
    return {
        "metric": f"structure_max_force_p{int(probability * 100)}",
        "endpoint_class": "secondary",
        "estimand": "distribution_quantile_delta",
        "preferred_direction": "lower",
        "unit": "eV/angstrom",
        "observed_delta": observed,
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "raw_one_sided_pvalue": np.nan,
        "raw_superiority_supported": bool(high < 0),
        "holm_adjusted_pvalue": np.nan,
        "holm_result": "NOT_APPLICABLE_SECONDARY",
        "wins": np.nan,
        "ties": np.nan,
        "losses": np.nan,
    }


def superiority_rows(
    paired: pd.DataFrame, data: dict[str, dict[str, np.ndarray]], indices: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    selected = paired[paired["comparison"] == "TCL-FT0"]
    for _, item in selected.iterrows():
        primary = item["metric"] in PRIMARY
        rows.append({
            "metric": item["metric"],
            "endpoint_class": "primary" if primary else "secondary",
            "estimand": "paired_mean_delta",
            "preferred_direction": item["preferred_direction"],
            "unit": item["unit"],
            "observed_delta": item["mean_delta_candidate_minus_baseline"],
            "bootstrap_ci95_low": item["bootstrap_ci95_low"],
            "bootstrap_ci95_high": item["bootstrap_ci95_high"],
            "raw_one_sided_pvalue": item["raw_one_sided_pvalue"],
            "raw_superiority_supported": item["raw_superiority_supported"],
            "holm_adjusted_pvalue": item["holm_adjusted_pvalue"],
            "holm_result": (
                "SUPPORTED" if item["holm_superiority_supported"] else "NOT_SUPPORTED"
            ) if primary else "NOT_APPLICABLE_SECONDARY",
            "wins": item["favorable_pairs"],
            "ties": item["ties"],
            "losses": item["unfavorable_pairs"],
        })
    rows.extend((
        quantile_delta_row(data, indices, 0.95),
        quantile_delta_row(data, indices, 0.99),
    ))
    return rows


def delta_ni(
    endpoint: str, candidate: np.ndarray, baseline: np.ndarray, indices: np.ndarray,
    scale: float, unit: str, margin: float, lower_bound_rule: bool = False,
) -> dict[str, object]:
    delta = (candidate - baseline) * scale
    bootstrap = delta[indices].mean(axis=1)
    low, high = qfloat(bootstrap, 0.025), qfloat(bootstrap, 0.975)
    passed = low > margin if lower_bound_rule else high < margin
    return {
        "endpoint": endpoint,
        "estimand": "mean_delta_TCL_minus_C0",
        "unit": unit,
        "tcl_observed": float(candidate.mean() * scale),
        "c0_observed": float(baseline.mean() * scale),
        "observed_delta_or_ratio": float(delta.mean()),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "frozen_margin": margin,
        "pass_rule": "lower_ci_greater_than_margin" if lower_bound_rule else "upper_ci_less_than_margin",
        "result": "PASS" if passed else "NOT_ESTABLISHED",
    }


def ratio_ni(
    endpoint: str, candidate: np.ndarray, baseline: np.ndarray, indices: np.ndarray,
    statistic: Callable[[np.ndarray, int | None], np.ndarray], unit: str, margin: float,
) -> dict[str, object]:
    candidate_value = float(statistic(candidate, None))
    baseline_value = float(statistic(baseline, None))
    candidate_boot = statistic(candidate[indices], 1)
    baseline_boot = statistic(baseline[indices], 1)
    if baseline_value <= 0 or np.any(baseline_boot <= 0):
        raise RuntimeError(f"non-positive ratio denominator: {endpoint}")
    bootstrap = candidate_boot / baseline_boot
    ratio = candidate_value / baseline_value
    low, high = qfloat(bootstrap, 0.025), qfloat(bootstrap, 0.975)
    return {
        "endpoint": endpoint,
        "estimand": "ratio_TCL_over_C0",
        "unit": unit,
        "tcl_observed": candidate_value,
        "c0_observed": baseline_value,
        "observed_delta_or_ratio": ratio,
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "frozen_margin": margin,
        "pass_rule": "upper_ci_less_than_margin",
        "result": "PASS" if high < margin else "NOT_ESTABLISHED",
    }


def noninferiority_rows(
    data: dict[str, dict[str, np.ndarray]], indices: np.ndarray,
) -> list[dict[str, object]]:
    tcl, c0 = data["TCL"], data["C0"]
    mean_stat = lambda values, axis: np.mean(values, axis=axis)
    p95_stat = lambda values, axis: np.quantile(values, 0.95, axis=axis)
    p99_stat = lambda values, axis: np.quantile(values, 0.99, axis=axis)
    return [
        delta_ni("e_hull_mean", tcl["e_hull"], c0["e_hull"], indices, 1.0, "eV/atom", 0.025),
        delta_ni("stable", tcl["stable"], c0["stable"], indices, 100.0, "percentage_points", -10.0, True),
        delta_ni("nus", tcl["nus"], c0["nus"], indices, 100.0, "percentage_points", -10.0, True),
        delta_ni("rmsd_mean", tcl["rmsd"], c0["rmsd"], indices, 1.0, "angstrom", 0.020),
        ratio_ni("atomic_force_mean", tcl["atomic_force_mean"], c0["atomic_force_mean"], indices, mean_stat, "ratio", 1.20),
        ratio_ni("structure_max_force_mean", tcl["structure_max_force"], c0["structure_max_force"], indices, mean_stat, "ratio", 1.20),
        ratio_ni("structure_max_force_p95", tcl["structure_max_force"], c0["structure_max_force"], indices, p95_stat, "ratio", 1.30),
        ratio_ni("structure_max_force_p99", tcl["structure_max_force"], c0["structure_max_force"], indices, p99_stat, "ratio", 1.50),
        delta_ni("force_gt1", tcl["force_gt1"], c0["force_gt1"], indices, 100.0, "percentage_points", 5.0),
        delta_ni("force_gt2", tcl["force_gt2"], c0["force_gt2"], indices, 100.0, "percentage_points", 2.0),
    ]


def tail_rows(data: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows = []
    for method in METHODS:
        values = data[method]
        for index, seed in enumerate(SEEDS):
            flags = []
            if values["rmsd"][index] > 0.5:
                flags.append("rmsd_gt_0.5")
            if values["structure_max_force"][index] > 1.0:
                flags.append("max_force_gt_1")
            if values["structure_max_force"][index] > 2.0:
                flags.append("max_force_gt_2")
            for threshold in (200, 300, 400):
                if values["relaxation_steps"][index] > threshold:
                    flags.append(f"steps_gt_{threshold}")
            if flags:
                rows.append({
                    "method": method,
                    "seed": seed,
                    "formula": values["formula"][index],
                    "e_hull_ev_per_atom": float(values["e_hull"][index]),
                    "stable": float(values["stable"][index]),
                    "nus": float(values["nus"][index]),
                    "novel": float(values["novel"][index]),
                    "unique": float(values["unique"][index]),
                    "rmsd_angstrom": float(values["rmsd"][index]),
                    "atomic_force_mean_ev_per_angstrom": float(values["atomic_force_mean"][index]),
                    "structure_max_force_ev_per_angstrom": float(values["structure_max_force"][index]),
                    "relaxation_steps": int(values["relaxation_steps"][index]),
                    "outlier_flags": ";".join(flags),
                    "dft_verified": False,
                })
    return rows


def direction(value: float, tolerance: float = 1e-12) -> str:
    if value < -tolerance:
        return "down"
    if value > tolerance:
        return "up"
    return "flat"


def stage_rows(path: Path, comparison_file: str) -> pd.DataFrame:
    table = pd.read_csv(path / comparison_file)
    return table[table["comparison"] == "TCL-FT0"].set_index("metric")


def replication_rows(paired: pd.DataFrame) -> list[dict[str, object]]:
    p0 = stage_rows(P0_ROOT, "paired_comparisons.csv")
    p1 = stage_rows(P1_ROOT, "paired_statistics.csv")
    p2 = stage_rows(P2_ROOT, "paired_statistics.csv")
    formal = paired[paired["comparison"] == "TCL-FT0"].set_index("metric")
    metrics = (
        "e_hull", "stable", "nus", "novel", "unique", "rmsd",
        "atomic_force_mean", "structure_max_force", "relaxation_steps",
    )
    rows = []
    for metric in metrics:
        p0_delta = float(p0.loc[metric, "mean_delta_candidate_minus_baseline"])
        if metric in ("stable", "nus", "novel", "unique"):
            p0_delta *= 100.0
        deltas = {
            "p0": p0_delta,
            "p1": float(p1.loc[metric, "mean_delta_candidate_minus_baseline"]),
            "p2": float(p2.loc[metric, "mean_delta_candidate_minus_baseline"]),
            "formal": float(formal.loc[metric, "mean_delta_candidate_minus_baseline"]),
        }
        directions = {stage: direction(value) for stage, value in deltas.items()}
        target = "down" if METRICS[metric][0] == "lower" else "up"
        if all(directions[stage] == target for stage in ("p0", "p1", "p2", "formal")):
            replicated = "yes_all_four"
        elif all(directions[stage] == target for stage in ("p1", "p2", "formal")):
            replicated = "yes_p1_p2_formal"
        elif all(directions[stage] == target for stage in ("p2", "formal")):
            replicated = "yes_p2_to_formal_only"
        else:
            replicated = "no"
        rows.append({
            "metric": metric,
            "preferred_direction": METRICS[metric][0],
            "p0_delta": deltas["p0"],
            "p0_direction": directions["p0"],
            "p1_delta": deltas["p1"],
            "p1_direction": directions["p1"],
            "p2_delta": deltas["p2"],
            "p2_direction": directions["p2"],
            "formal_delta": deltas["formal"],
            "formal_direction": directions["formal"],
            "formal_ci95_low": float(formal.loc[metric, "bootstrap_ci95_low"]),
            "formal_ci95_high": float(formal.loc[metric, "bootstrap_ci95_high"]),
            "formal_holm_adjusted_pvalue": float(formal.loc[metric, "holm_adjusted_pvalue"]),
            "replicated": replicated,
        })
    return rows


def main() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    if len(generation) != len(METHODS) * len(SEEDS):
        raise RuntimeError("Formal256 generation row count mismatch")
    if generation["dft_verified"].astype(str).str.lower().ne("false").any():
        raise RuntimeError("DFT flag unexpectedly true")
    loader = load_frozen_loader()
    data = {method: loader.load_method(method, generation) for method in METHODS}
    for method, values in data.items():
        finite_or_raise(method, values)
        values["force_gt1"] = (values["structure_max_force"] > 1.0).astype(float)
        values["force_gt2"] = (values["structure_max_force"] > 2.0).astype(float)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(SEEDS), size=(N_BOOTSTRAP, len(SEEDS)))

    quality = pd.DataFrame([summary_row(method, data[method]) for method in METHODS])
    quality.to_csv(ROOT / "quality_results.csv", index=False)

    paired = add_holm(pd.DataFrame(paired_rows(data, indices)))
    paired.to_csv(ROOT / "paired_statistics.csv", index=False)

    superiority = pd.DataFrame(superiority_rows(paired, data, indices))
    superiority.to_csv(ROOT / "superiority_results.csv", index=False)

    noninferiority = pd.DataFrame(noninferiority_rows(data, indices))
    noninferiority.to_csv(ROOT / "noninferiority_results.csv", index=False)

    tails = pd.DataFrame(tail_rows(data))
    if tails.empty:
        tails = pd.DataFrame(columns=(
            "method", "seed", "formula", "e_hull_ev_per_atom", "stable", "nus",
            "novel", "unique", "rmsd_angstrom", "atomic_force_mean_ev_per_angstrom",
            "structure_max_force_ev_per_angstrom", "relaxation_steps",
            "outlier_flags", "dft_verified",
        ))
    tails.to_csv(ROOT / "tail_analysis.csv", index=False)

    replication = pd.DataFrame(replication_rows(paired))
    replication.to_csv(ROOT / "replication_table.csv", index=False)

    print("TABLE_A_ABSOLUTE_QUALITY", flush=True)
    print(quality.to_string(index=False), flush=True)
    print("TABLE_B_TCL_VS_FT0_SUPERIORITY", flush=True)
    print(superiority.to_string(index=False), flush=True)
    print("TABLE_C_TCL_VS_C0_NONINFERIORITY", flush=True)
    print(noninferiority.to_string(index=False), flush=True)
    print("P0_P1_P2_FORMAL_REPLICATION", flush=True)
    print(replication.to_string(index=False), flush=True)
    print(f"severe_outlier_rows={len(tails)}", flush=True)


if __name__ == "__main__":
    main()
