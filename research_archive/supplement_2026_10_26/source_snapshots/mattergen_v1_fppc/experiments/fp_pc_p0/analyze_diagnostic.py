"""Aggregate the frozen 64-seed B0 diagnostic and apply its hard gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.io import read
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/fp_pc_p0"
HISTORICAL = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_async_freeze/experiments/shared_field_dynamics"
)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def group_comparison(frame: pd.DataFrame, flag: str) -> dict[str, Any]:
    violated = frame[frame[flag]]
    clean = frame[~frame[flag]]
    result: dict[str, Any] = {
        "violation_n": len(violated),
        "no_violation_n": len(clean),
        "comparison_estimable": bool(len(violated) and len(clean)),
        "metrics": {},
    }
    for metric in ("final_maxF", "final_RMSD", "final_Ehull"):
        v_mean = float(violated[metric].mean()) if len(violated) else float("nan")
        n_mean = float(clean[metric].mean()) if len(clean) else float("nan")
        v_median = float(violated[metric].median()) if len(violated) else float("nan")
        n_median = float(clean[metric].median()) if len(clean) else float("nan")
        result["metrics"][metric] = {
            "violation_mean": finite_or_none(v_mean),
            "no_violation_mean": finite_or_none(n_mean),
            "violation_median": finite_or_none(v_median),
            "no_violation_median": finite_or_none(n_median),
            "relative_mean_worse": finite_or_none(v_mean / n_mean - 1.0)
            if math.isfinite(v_mean) and math.isfinite(n_mean) and abs(n_mean) > 1e-15
            else None,
            "absolute_mean_difference": finite_or_none(v_mean - n_mean)
            if math.isfinite(v_mean) and math.isfinite(n_mean) else None,
        }
    return result


def validate_replays(seeds: range) -> dict[str, Any]:
    max_cell = max_frac = max_cart = 0.0
    numbers_exact = True
    formula_exact = True
    for seed in seeds:
        current = read(ROOT / "diagnostic_runs" / str(seed) / "generated_crystals.extxyz")
        historical = read(HISTORICAL / "trajectories" / str(seed) / "generated_crystals.extxyz")
        numbers_exact &= bool(np.array_equal(current.numbers, historical.numbers))
        formula_exact &= current.get_chemical_formula() == historical.get_chemical_formula()
        max_cell = max(max_cell, float(np.max(np.abs(current.cell.array - historical.cell.array))))
        frac_delta = current.get_scaled_positions() - historical.get_scaled_positions()
        frac_delta = np.remainder(frac_delta + 0.5, 1.0) - 0.5
        max_frac = max(max_frac, float(np.max(np.abs(frac_delta))))
        max_cart = max(max_cart, float(np.max(np.abs(current.positions - historical.positions))))
    equivalent = numbers_exact and formula_exact and max_cell <= 1e-5 and max_frac <= 1e-7
    return {
        "trajectories": len(seeds),
        "atomic_numbers_exact": numbers_exact,
        "formula_exact": formula_exact,
        "max_abs_cell_difference_angstrom": max_cell,
        "max_abs_periodic_fractional_position_difference": max_frac,
        "max_abs_cartesian_position_difference_angstrom": max_cart,
        "numerical_equivalence_tolerance": {"cell_angstrom": 1e-5, "fractional_position": 1e-7},
        "all_numerically_equivalent": equivalent,
    }


def burden_correlations(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for burden in ("violation_count", "predictor_violation_count", "corrector_violation_count"):
        result[burden] = {}
        for outcome in ("final_maxF", "final_RMSD", "final_Ehull"):
            result[burden][outcome] = {
                "spearman": float(spearmanr(frame[burden], frame[outcome]).statistic),
                "pearson": float(pearsonr(frame[burden], frame[outcome]).statistic),
                "n": len(frame),
                "gate_metric": False,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    seeds = range(99000, 99064)
    paths = [ROOT / "diagnostic_runs" / str(seed) / "proposal_trace.csv" for seed in seeds]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"missing {len(missing)} diagnostic traces")
    paths = [path for path in paths if path.is_file()]
    trace = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    bool_columns = [
        c for c in trace.columns if c.endswith("violation") or c in {
            "before_finite", "candidate_finite", "accepted_unmodified"
        }
    ]
    for column in bool_columns:
        trace[column] = as_bool(trace[column])
    if not args.allow_incomplete:
        expected = 64 * 2000
        if len(trace) != expected:
            raise RuntimeError(f"trace rows {len(trace)} != {expected}")
        if not bool(trace["accepted_unmodified"].all()):
            raise RuntimeError("diagnostic changed at least one proposal")

    quality = pd.read_csv(HISTORICAL / "diagnostic_quality/per_structure.csv").rename(
        columns={
            "sample_seed": "seed", "pre_relaxation_max_force": "final_maxF",
            "rmsd_from_relaxation": "final_RMSD",
            "energy_above_hull_per_atom": "final_Ehull",
        }
    )
    outcomes = quality[["seed", "final_maxF", "final_RMSD", "final_Ehull"]].copy()
    if args.allow_incomplete:
        outcomes = outcomes[outcomes.seed.isin(trace.seed.unique())].copy()
    if args.allow_incomplete:
        outcomes = outcomes[outcomes.seed.isin(sorted(trace.seed.unique()))].copy()
    trace = trace.merge(outcomes, on="seed", how="left", validate="many_to_one")
    if trace[["final_maxF", "final_RMSD", "final_Ehull"]].isna().any().any():
        raise RuntimeError("missing historical outcome after join")
    trace.to_csv(ROOT / "proposal_violation_trace.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    trajectory = outcomes.copy()
    trajectory["subset"] = np.where(trajectory.seed <= 99031, "calibration", "validation")
    for stage in ("predictor", "corrector"):
        stage_any = (
            trace[trace.stage == stage].groupby("seed")["proposal_entered_any_violation"].any()
        )
        stage_count = (
            trace[trace.stage == stage].groupby("seed")["proposal_entered_any_violation"].sum()
        )
        trajectory[f"{stage}_violation"] = (
            trajectory.seed.map(stage_any).astype("boolean").fillna(False).astype(bool)
        )
        trajectory[f"{stage}_violation_count"] = trajectory.seed.map(stage_count).fillna(0).astype(int)
    trajectory["any_violation"] = trajectory.predictor_violation | trajectory.corrector_violation
    trajectory["violation_count"] = (
        trajectory.predictor_violation_count + trajectory.corrector_violation_count
    )

    timing_frame = trace[trace.proposal_entered_any_violation].copy()
    timing_frame["diffusion_stage"] = pd.cut(
        timing_frame.reverse_progress, bins=[-1e-12, 1/3, 2/3, 1 + 1e-12],
        labels=["early", "mid", "late"], include_lowest=True, right=False,
    )
    timing = {}
    for (proposal_stage, diffusion_stage), group in timing_frame.groupby(
        ["stage", "diffusion_stage"], observed=True
    ):
        timing[f"{proposal_stage}_{diffusion_stage}"] = {
            "event_count": int(len(group)), "trajectory_count": int(group.seed.nunique())
        }

    subsets: dict[str, Any] = {}
    burden_association: dict[str, Any] = {}
    for label, cohort in [("all", trajectory),
                          ("calibration", trajectory[trajectory.subset == "calibration"]),
                          ("validation", trajectory[trajectory.subset == "validation"])]:
        subsets[label] = {
            "n": len(cohort),
            "any_violation_rate": float(cohort.any_violation.mean()),
            "predictor_violation_rate": float(cohort.predictor_violation.mean()),
            "corrector_violation_rate": float(cohort.corrector_violation.mean()),
            "comparison": group_comparison(cohort, "any_violation"),
        }
        burden_association[label] = burden_correlations(cohort)

    overall = subsets["all"]
    comparison = overall["comparison"]
    maxf_worse = comparison["metrics"]["final_maxF"]["relative_mean_worse"]
    rmsd_worse = comparison["metrics"]["final_RMSD"]["relative_mean_worse"]
    ehull_worse = comparison["metrics"]["final_Ehull"]["absolute_mean_difference"]
    association = bool(
        comparison["comparison_estimable"] and (
            (maxf_worse is not None and maxf_worse >= 0.25)
            or (rmsd_worse is not None and rmsd_worse >= 0.25)
            or (ehull_worse is not None and ehull_worse >= 0.02)
        )
    )
    rate = float(overall["any_violation_rate"])
    motivated = rate >= 0.20 and association
    status = "PASS" if motivated else "FAIL"
    if rate < 0.10:
        reason = "actionable violation rate below 10%"
    elif not comparison["comparison_estimable"]:
        reason = "violation/no-violation outcome comparison is not estimable"
    elif not association:
        reason = "violation group has no preregistered final-quality degradation"
    else:
        reason = "rate and final-quality association both pass"

    replay = validate_replays(seeds) if not args.allow_incomplete else None
    if replay is not None and not replay["all_numerically_equivalent"]:
        raise RuntimeError("replayed structures are not numerically equivalent to historical cohort")
    region = json.loads((ROOT / "feasible_region.json").read_text())
    summary = {
        "schema_version": 1,
        "diagnostic": status,
        "FP_PC_MOTIVATED": "YES" if motivated else "NO",
        "gate_reason": reason,
        "gate": {
            "rate_required": 0.20, "rate_observed": rate,
            "quality_association_required": True,
            "quality_association_observed": association,
            "maxF_relative_worse_required": 0.25,
            "RMSD_relative_worse_required": 0.25,
            "Ehull_absolute_worse_required_ev_atom": 0.02,
        },
        "thresholds": {
            "d_min_floor_angstrom": region["d_min_floor_angstrom"],
            "VN_low_angstrom3_per_atom": region["VN_low_angstrom3_per_atom"],
            "VN_high_angstrom3_per_atom": region["VN_high_angstrom3_per_atom"],
            "frozen_before_outcomes": region["frozen_before_diagnostic_outcomes"],
        },
        "trajectory_results": subsets,
        "proposal_entry_events": {
            "total": int(trace.proposal_entered_any_violation.sum()),
            "distance": int(trace.proposal_entered_distance_violation.sum()),
            "volume": int(trace.proposal_entered_volume_violation.sum()),
            "predictor": int(trace.loc[trace.stage == "predictor", "proposal_entered_any_violation"].sum()),
            "corrector": int(trace.loc[trace.stage == "corrector", "proposal_entered_any_violation"].sum()),
        },
        "timing": timing,
        "absolute_candidate_infeasibility": {
            "row_fraction": float(trace.candidate_any_violation.mean()),
            "trajectory_fraction": float(trace.groupby("seed").candidate_any_violation.any().mean()),
            "not_used_for_gate": True,
        },
        "violation_burden": {
            "min": int(trajectory.violation_count.min()),
            "median": float(trajectory.violation_count.median()),
            "mean": float(trajectory.violation_count.mean()),
            "max": int(trajectory.violation_count.max()),
        },
        "secondary_burden_correlations": burden_association,
        "replay_equivalence": replay,
        "diagnostic_only_no_proposals_modified": bool(trace.accepted_unmodified.all()),
        "adaptive_cfg_used": False,
        "surrogate_property_eval": True,
        "dft_verified": False,
        "p0_status": "ELIGIBLE" if motivated else "NOT_RUN_DIAGNOSTIC_FAIL",
    }
    (ROOT / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trajectory.to_csv(ROOT / "diagnostic_per_trajectory.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
