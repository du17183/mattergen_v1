"""Compute paired schedule metrics, frozen gates, bootstrap inference, and plots."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from ase.io import read
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from mattergen.diffusion.sampling.stage_bounded_cfg import schedule_id
from experiments.stage_calibrated_bounded_cfg.run_generation import phase_specs


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/stage_calibrated_bounded_cfg"
PLOTS = ROOT / "plots"
EPS = 1e-12
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEEDS = {"calibration": 2026091511, "p0": 2026091512, "formal256": 2026091513}
LAMBDA = 1.0


def json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")


def quality_enriched(phase: str) -> pd.DataFrame:
    phase_root = ROOT / phase
    values = pd.read_csv(phase_root / "property_metrics.csv")
    for column, default in (("e_hull", np.nan), ("stable", False), ("nus", False), ("novel", False), ("unique", False), ("quality_success", False)):
        values[column] = default
    for spec in phase_specs(phase):
        group = spec["schedule_id"]
        structure_path = phase_root / "evaluation_structures" / f"{group}.extxyz"
        quality_root = phase_root / "quality" / group
        if not structure_path.exists():
            continue
        if not (quality_root / "official_detailed.json.gz").exists():
            raise FileNotFoundError(f"quality result missing: {phase}/{group}")
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
            values.loc[target, "quality_success"] = True
    for column in ("structure_valid", "stable", "nus", "novel", "unique", "quality_success"):
        values[column] = values[column].fillna(False).astype(bool)
    values["evaluation_valid"] = values["structure_valid"] & values["quality_success"]
    values.to_csv(phase_root / "evaluated_results.csv", index=False)
    return values


def build_paired(values: pd.DataFrame) -> pd.DataFrame:
    base = values[values["schedule_id"] == "C0"].set_index("seed")
    if len(base) != values["seed"].nunique():
        raise RuntimeError("C0 matrix incomplete")
    rows: list[dict] = []
    for _, row in values.iterrows():
        c0 = base.loc[int(row["seed"])]
        delta = float(row["property_absolute_error"] - c0["property_absolute_error"])
        ehull_delta = float(row["e_hull"] - c0["e_hull"]) if np.isfinite(row["e_hull"]) and np.isfinite(c0["e_hull"]) else float("nan")
        rows.append(
            {
                **row.to_dict(),
                "c0_property_absolute_error": float(c0["property_absolute_error"]),
                "property_delta_vs_c0": delta,
                "property_gain_vs_c0": -delta,
                "property_harm_vs_c0": bool(row["property_absolute_error"] > 1.05 * c0["property_absolute_error"]),
                "positive_property_degradation": max(delta, 0.0),
                "e_hull_delta_vs_c0": ehull_delta,
                "e_hull_harm_vs_c0": bool(np.isfinite(ehull_delta) and ehull_delta > 0.02),
                "stable_loss_vs_c0": bool(c0["stable"] and not row["stable"]),
                "validity_loss_vs_c0": bool(c0["evaluation_valid"] and not row["evaluation_valid"]),
                "nus_loss_vs_c0": bool(c0["nus"] and not row["nus"]),
            }
        )
    paired = pd.DataFrame(rows)
    return paired


def summarize(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    c0 = paired[paired["schedule_id"] == "C0"]
    c0_mean = float(c0["property_absolute_error"].mean())
    c0_median = float(c0["property_absolute_error"].median())
    c0_rates = {name: float(c0[name].mean()) for name in ("evaluation_valid", "stable", "nus", "novel", "unique")}
    c0_ehull = float(c0["e_hull"].mean())
    metrics, harms, tails = [], [], []
    for identifier, frame in paired.groupby("schedule_id", sort=True):
        gain = frame["property_gain_vs_c0"].to_numpy(float)
        degradation = frame["positive_property_degradation"].to_numpy(float)
        n = len(frame)
        worst_n = max(1, int(math.ceil(0.25 * n)))
        mean_gain = float(gain.mean())
        median_gain = float(np.median(gain))
        p90 = float(np.quantile(degradation, 0.90))
        robust_score = median_gain - LAMBDA * p90
        valid_drop = c0_rates["evaluation_valid"] - float(frame["evaluation_valid"].mean())
        stable_drop = c0_rates["stable"] - float(frame["stable"].mean())
        nus_drop = c0_rates["nus"] - float(frame["nus"].mean())
        mean_ehull = float(frame["e_hull"].mean())
        ehull_worsening = mean_ehull - c0_ehull
        guardrail = bool(valid_drop <= 0.05 + EPS and stable_drop <= 0.05 + EPS and nus_drop <= 0.05 + EPS and (np.isfinite(ehull_worsening) and ehull_worsening <= 0.02 + EPS))
        first = frame.iloc[0]
        metrics.append(
            {
                "schedule_id": identifier,
                "schedule_kind": first["schedule_kind"],
                "active_cfg": float(first["active_cfg"]),
                "delta_g": float(first["delta_g"]),
                "start": int(first["start"]),
                "duration": int(first["duration"]),
                "n": n,
                "property_mae_mean": float(frame["property_absolute_error"].mean()),
                "property_mae_median": float(frame["property_absolute_error"].median()),
                "mean_property_gain": mean_gain,
                "median_property_gain": median_gain,
                "relative_mean_gain": mean_gain / max(c0_mean, EPS),
                "relative_median_gain": median_gain / max(c0_median, EPS),
                "wins": int((gain > EPS).sum()),
                "ties": int((np.abs(gain) <= EPS).sum()),
                "losses": int((gain < -EPS).sum()),
                "robust_score": robust_score,
                "evaluation_valid_rate": float(frame["evaluation_valid"].mean()),
                "stable_rate": float(frame["stable"].mean()),
                "nus_rate": float(frame["nus"].mean()),
                "novel_rate": float(frame["novel"].mean()),
                "unique_rate": float(frame["unique"].mean()),
                "mean_e_hull": mean_ehull,
                "validity_drop": valid_drop,
                "stable_drop": stable_drop,
                "nus_drop": nus_drop,
                "mean_e_hull_worsening": ehull_worsening,
                "guardrail_pass": guardrail,
            }
        )
        harms.append(
            {
                "schedule_id": identifier,
                "n": n,
                "property_harm_count": int(frame["property_harm_vs_c0"].sum()),
                "property_harm_rate": float(frame["property_harm_vs_c0"].mean()),
                "e_hull_comparable_n": int(frame["e_hull_delta_vs_c0"].notna().sum()),
                "e_hull_harm_count": int(frame["e_hull_harm_vs_c0"].sum()),
                "e_hull_harm_rate": float(frame["e_hull_harm_vs_c0"].mean()),
                "stable_loss_count": int(frame["stable_loss_vs_c0"].sum()),
                "stable_loss_rate": float(frame["stable_loss_vs_c0"].mean()),
                "validity_loss_count": int(frame["validity_loss_vs_c0"].sum()),
                "nus_loss_count": int(frame["nus_loss_vs_c0"].sum()),
            }
        )
        tails.append(
            {
                "schedule_id": identifier,
                "n": n,
                "p75_positive_property_degradation": float(np.quantile(degradation, 0.75)),
                "p90_positive_property_degradation": p90,
                "worst_quartile_mean_property_degradation": float(np.sort(degradation)[-worst_n:].mean()),
                "maximum_property_degradation": float(degradation.max()),
            }
        )
    return pd.DataFrame(metrics), pd.DataFrame(harms), pd.DataFrame(tails)


def bootstrap_ci(values: np.ndarray, seed: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    boot = array[indices].mean(axis=1)
    return {"n": len(array), "mean": float(array.mean()), "ci95_low": float(np.quantile(boot, 0.025)), "ci95_high": float(np.quantile(boot, 0.975)), "resamples": BOOTSTRAP_RESAMPLES}


def row_lookup(frame: pd.DataFrame, identifier: str) -> pd.Series:
    matched = frame[frame["schedule_id"] == identifier]
    if len(matched) != 1:
        raise RuntimeError(f"summary lookup failed: {identifier}")
    return matched.iloc[0]


def pair_values(paired: pd.DataFrame, left: str, right: str, column: str) -> np.ndarray:
    pivot = paired.pivot(index="seed", columns="schedule_id", values=column)
    return pivot[left].to_numpy(float) - pivot[right].to_numpy(float)


def rank_key(row: pd.Series) -> tuple:
    duration = int(row["duration"])
    start = int(row["start"])
    return (-float(row["robust_score"]), -float(row["mean_property_gain"]), float(row["property_harm_rate"]), duration, abs(float(row["delta_g"])), start)


def write_common(phase: str, paired: pd.DataFrame, metrics: pd.DataFrame, harms: pd.DataFrame, tails: pd.DataFrame) -> pd.DataFrame:
    phase_root = ROOT / phase
    merged = metrics.merge(harms, on=["schedule_id", "n"], validate="one_to_one").merge(tails, on=["schedule_id", "n"], validate="one_to_one")
    paired.to_csv(phase_root / "paired_results.csv", index=False)
    metrics.to_csv(phase_root / "metrics.csv", index=False)
    harms.to_csv(phase_root / "harm_metrics.csv", index=False)
    tails.to_csv(phase_root / "tail_metrics.csv", index=False)
    return merged


def plot_calibration(summary: pd.DataFrame, selected: str | None, b1: str | None, b2: str | None) -> None:
    bounded = summary[summary["schedule_kind"] == "bounded"].sort_values("mean_property_gain")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(bounded["schedule_id"], bounded["mean_property_gain"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Mean paired property gain", title="Bounded-pulse calibration gain")
    fig.tight_layout(); fig.savefig(PLOTS / "schedule_mean_gain.png", dpi=180); plt.close(fig)
    bounded = bounded.sort_values("property_harm_rate")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(bounded["schedule_id"], bounded["property_harm_rate"])
    ax.set(xlabel="Property harm rate", title="Bounded-pulse calibration harm")
    fig.tight_layout(); fig.savefig(PLOTS / "schedule_harm_rate.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for kind, frame in summary.groupby("schedule_kind"):
        if kind != "c0": ax.scatter(frame["property_harm_rate"], frame["median_property_gain"], label=kind)
    ax.axhline(0, color="black", linewidth=0.8); ax.set(xlabel="Property harm rate", ylabel="Median paired property gain", title="Gain versus harm"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "schedule_gain_vs_harm.png", dpi=180); plt.close(fig)
    identifiers = [value for value in ("C0", b1, b2, selected) if value is not None]
    view = summary.set_index("schedule_id").loc[identifiers]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(range(len(view)), view["mean_property_gain"]); axes[0].set_title("Mean property gain")
    axes[1].bar(range(len(view)), view["property_harm_rate"]); axes[1].set_title("Property harm rate")
    for ax in axes: ax.set_xticks(range(len(view)), identifiers, rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(PLOTS / "bounded_vs_permanent.png", dpi=180); plt.close(fig)


def calibration_gate(paired: pd.DataFrame, summary: pd.DataFrame) -> dict:
    constants = summary[summary["schedule_kind"] == "constant"]
    guarded_constants = constants[constants["guardrail_pass"]]
    b1_pool = guarded_constants if len(guarded_constants) else constants
    b1 = sorted((row for _, row in b1_pool.iterrows()), key=rank_key)[0]
    candidates: list[dict] = []
    ranking_rows: list[dict] = []
    for _, candidate in summary[summary["schedule_kind"] == "bounded"].iterrows():
        permanent_id = schedule_id({"kind": "permanent", "start": int(candidate["start"]), "delta_g": float(candidate["delta_g"])})
        permanent = row_lookup(summary, permanent_id)
        if float(permanent["property_harm_rate"]) > 0:
            harm_reduction = (float(permanent["property_harm_rate"]) - float(candidate["property_harm_rate"])) / float(permanent["property_harm_rate"])
            robustness = harm_reduction >= 0.25 - EPS
            robustness_basis = "harm_rate"
        else:
            denominator = float(permanent["p90_positive_property_degradation"])
            tail_reduction = (denominator - float(candidate["p90_positive_property_degradation"])) / denominator if denominator > 0 else float("-inf")
            harm_reduction = tail_reduction
            robustness = bool(float(candidate["property_harm_rate"]) == 0 and tail_reduction >= 0.25 - EPS)
            robustness_basis = "p90_fallback"
        positive = max(float(candidate["relative_mean_gain"]), float(candidate["relative_median_gain"])) >= 0.05 - EPS
        wins = int(candidate["wins"]) > int(candidate["losses"])
        beat_b1 = float(candidate["robust_score"]) > float(b1["robust_score"]) + EPS
        beat_b2 = float(candidate["robust_score"]) > float(permanent["robust_score"]) + EPS
        eligible = bool(candidate["guardrail_pass"] and positive and wins and robustness and beat_b1 and beat_b2)
        record = {**candidate.to_dict(), "permanent_id": permanent_id, "b1_id": str(b1["schedule_id"]), "relative_harm_or_tail_reduction_vs_permanent": harm_reduction, "robustness_basis": robustness_basis, "property_5pct_pass": positive, "wins_pass": wins, "permanent_robustness_pass": robustness, "beats_b1_robust_score": beat_b1, "beats_b2_robust_score": beat_b2, "calibration_eligible": eligible}
        ranking_rows.append(record)
        if eligible: candidates.append(record)
    ranking = pd.DataFrame(ranking_rows).sort_values(by=["calibration_eligible", "robust_score", "mean_property_gain", "property_harm_rate"], ascending=[False, False, False, True])
    ranking.to_csv(ROOT / "calibration/schedule_ranking.csv", index=False)
    selected = sorted(candidates, key=lambda row: (-float(row["robust_score"]), -float(row["mean_property_gain"]), float(row["property_harm_rate"]), int(row["duration"]), abs(float(row["delta_g"])), int(row["start"])))[0] if candidates else None
    decision = {
        "STAGE0_DIAGNOSTIC": "COMPLETE",
        "STAGE_SCHEDULE_CALIBRATION": "GO" if selected else "FAIL",
        "STAGE_CFG_P0": "NOT_RUN",
        "STAGE_BOUNDED_CFG_FORMAL256": "NOT_RUN",
        "STAGE_SPECIFIC_EFFECT": "NOT_RUN",
        "BOUNDED_PULSE_EFFECT": "NOT_RUN",
        "INNOVATION1_FINAL_STATUS": "MIXED",
        "calibration_n": int(paired["seed"].nunique()),
        "eligible_bounded_schedules": len(candidates),
        "selected_schedule_id": selected["schedule_id"] if selected else None,
        "best_constant_id": str(b1["schedule_id"]),
        "permanent_counterpart_id": selected["permanent_id"] if selected else None,
        "stop_reason": None if selected else "No bounded schedule satisfied all pre-registered property, robustness, comparison, and guardrail gates.",
    }
    if selected:
        frozen_path = ROOT / "implementation/frozen_stage_cfg.yaml"
        if frozen_path.exists():
            raise FileExistsError(frozen_path)
        frozen_payload = {
            "schema_version": 1,
            "frozen_after_calibration_before_p0": True,
            "schedule_id": str(selected["schedule_id"]),
            "start": int(selected["start"]),
            "delta_g": float(selected["delta_g"]),
            "duration": int(selected["duration"]),
            "active_cfg": float(selected["active_cfg"]),
            "b1_schedule_id": str(b1["schedule_id"]),
            "b1_constant_cfg": float(b1["active_cfg"]),
            "b2_schedule_id": str(selected["permanent_id"]),
            "selection_rule": "guardrails then maximum median_gain - 1.0*p90_positive_degradation",
            "calibration_decision_sha256": "filled_below",
        }
        decision_bytes = (json.dumps(decision, sort_keys=True) + "\n").encode()
        frozen_payload["calibration_decision_sha256"] = hashlib.sha256(decision_bytes).hexdigest()
        frozen_path.write_text(yaml.safe_dump(frozen_payload, sort_keys=False))
        config_digest = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        (ROOT / "implementation/frozen_stage_cfg_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "frozen_before_p0": True,
                    "frozen_stage_cfg_sha256": config_digest,
                    "selected_schedule_id": str(selected["schedule_id"]),
                    "best_constant_id": str(b1["schedule_id"]),
                    "permanent_counterpart_id": str(selected["permanent_id"]),
                },
                indent=2,
            )
            + "\n"
        )
    selected_id = str(selected["schedule_id"]) if selected else None
    selected_b2 = str(selected["permanent_id"]) if selected else None
    plot_calibration(summary, selected_id, str(b1["schedule_id"]), selected_b2)
    if selected:
        comparisons = {}
        for other in ("C0", str(b1["schedule_id"]), selected_b2):
            gain = -pair_values(paired, selected_id, other, "property_absolute_error")
            comparisons[f"{selected_id}_vs_{other}"] = bootstrap_ci(gain, BOOTSTRAP_SEEDS["calibration"] + len(comparisons))
        json_write(ROOT / "calibration/bootstrap_results.json", comparisons)
    return decision


def aliases() -> tuple[str, str, str]:
    frozen = yaml.safe_load((ROOT / "implementation/frozen_stage_cfg.yaml").read_text())
    return str(frozen["b1_schedule_id"]), str(frozen["b2_schedule_id"]), str(frozen["schedule_id"])


def robustness_pass(summary: pd.DataFrame, s0_id: str, b2_id: str) -> tuple[bool, str]:
    s0, b2 = row_lookup(summary, s0_id), row_lookup(summary, b2_id)
    harm_better = float(s0["property_harm_rate"]) < float(b2["property_harm_rate"]) - EPS
    b2_tail = float(b2["p90_positive_property_degradation"])
    tail_better = b2_tail > 0 and float(s0["p90_positive_property_degradation"]) <= 0.75 * b2_tail + EPS
    return bool(harm_better or tail_better), "harm_rate" if harm_better else "p90_tail" if tail_better else "neither"


def fresh_gate(phase: str, paired: pd.DataFrame, summary: pd.DataFrame) -> dict:
    b1_id, b2_id, s0_id = aliases()
    c0, b1, b2, s0 = (row_lookup(summary, identifier) for identifier in ("C0", b1_id, b2_id, s0_id))
    positive = max(float(s0["relative_mean_gain"]), float(s0["relative_median_gain"])) >= 0.05 - EPS
    wins = int(s0["wins"]) > int(s0["losses"])
    robust, robust_basis = robustness_pass(summary, s0_id, b2_id)
    property_better_b1 = bool(float(s0["property_mae_mean"]) < float(b1["property_mae_mean"]) - EPS or float(s0["property_mae_median"]) < float(b1["property_mae_median"]) - EPS)
    b1_composite = bool(property_better_b1 and float(s0["robust_score"]) > float(b1["robust_score"]) + EPS)
    guardrail = bool(s0["guardrail_pass"])
    property_comparisons = {}
    for index, other in enumerate(("C0", b1_id, b2_id)):
        gain = -pair_values(paired, s0_id, other, "property_absolute_error")
        property_comparisons[f"{s0_id}_vs_{other}_property_gain"] = bootstrap_ci(gain, BOOTSTRAP_SEEDS[phase] + index)
    if phase == "p0":
        go = positive and wins and robust and b1_composite and guardrail
        decision = {
            "STAGE0_DIAGNOSTIC": "COMPLETE",
            "STAGE_SCHEDULE_CALIBRATION": "GO",
            "STAGE_CFG_P0": "GO" if go else "FAIL",
            "STAGE_BOUNDED_CFG_FORMAL256": "NOT_RUN",
            "STAGE_SPECIFIC_EFFECT": "SUPPORTED" if b1_composite else "NOT_SUPPORTED",
            "BOUNDED_PULSE_EFFECT": "SUPPORTED" if robust else "NOT_SUPPORTED",
            "INNOVATION1_FINAL_STATUS": "MIXED",
            "p0_n": int(paired["seed"].nunique()),
            "property_5pct_pass": positive,
            "wins_pass": wins,
            "permanent_robustness_pass": robust,
            "permanent_robustness_basis": robust_basis,
            "constant_composite_pass": b1_composite,
            "guardrail_pass": guardrail,
            "stop_reason": None if go else "Fresh P0 failed at least one pre-registered gate; Formal256 is prohibited.",
        }
    else:
        primary_ci = property_comparisons[f"{s0_id}_vs_C0_property_gain"]
        primary = float(primary_ci["ci95_low"]) > 0
        secondary = {}
        e_delta = pair_values(paired, s0_id, "C0", "e_hull")
        secondary["e_hull_delta_s0_minus_c0"] = bootstrap_ci(e_delta, BOOTSTRAP_SEEDS[phase] + 10)
        for offset, metric in enumerate(("stable", "nus", "evaluation_valid"), start=11):
            gain = pair_values(paired, s0_id, "C0", metric)
            secondary[f"{metric}_gain_s0_minus_c0"] = bootstrap_ci(gain, BOOTSTRAP_SEEDS[phase] + offset)
        no_adverse = bool(
            not (float(secondary["e_hull_delta_s0_minus_c0"]["ci95_low"]) > 0)
            and all(not (float(secondary[f"{metric}_gain_s0_minus_c0"]["ci95_high"]) < 0) for metric in ("stable", "nus", "evaluation_valid"))
        )
        confirmed = primary and robust and b1_composite and guardrail and no_adverse
        decision = {
            "STAGE0_DIAGNOSTIC": "COMPLETE",
            "STAGE_SCHEDULE_CALIBRATION": "GO",
            "STAGE_CFG_P0": "GO",
            "STAGE_BOUNDED_CFG_FORMAL256": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
            "STAGE_SPECIFIC_EFFECT": "SUPPORTED" if b1_composite else "NOT_SUPPORTED",
            "BOUNDED_PULSE_EFFECT": "SUPPORTED" if robust else "NOT_SUPPORTED",
            "INNOVATION1_FINAL_STATUS": "CONFIRMED" if confirmed else "MIXED",
            "formal_n": int(paired["seed"].nunique()),
            "primary_bootstrap_ci_excludes_zero_favorable": primary,
            "permanent_robustness_pass": robust,
            "permanent_robustness_basis": robust_basis,
            "constant_composite_pass": b1_composite,
            "guardrail_pass": guardrail,
            "no_adverse_secondary_ci": no_adverse,
            "secondary_bootstrap": secondary,
            "stop_reason": None if confirmed else "Formal256 did not satisfy every pre-registered confirmation condition.",
        }
    json_write(ROOT / phase / "bootstrap_results.json", {"property": property_comparisons, "secondary": decision.get("secondary_bootstrap", {})})
    plot_fresh(phase, paired, summary, ("C0", b1_id, b2_id, s0_id), property_comparisons)
    return decision


def plot_fresh(phase: str, paired: pd.DataFrame, summary: pd.DataFrame, identifiers: tuple[str, ...], bootstrap: dict) -> None:
    prefix = "p0" if phase == "p0" else "formal"
    pivot = paired.pivot(index="seed", columns="schedule_id", values="property_absolute_error")
    _, _, _, s0_id = identifiers
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in pivot[["C0", s0_id]].iterrows(): ax.plot([0, 1], row, color="0.75", linewidth=0.6)
    ax.set_xticks([0, 1], ["C0", "S0"]); ax.set_ylabel("Property absolute error"); ax.set_title(f"{phase.upper()} paired property")
    fig.tight_layout(); fig.savefig(PLOTS / f"{prefix}_property_paired.png", dpi=180); plt.close(fig)
    view = summary.set_index("schedule_id").loc[list(identifiers)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(4), view["property_harm_rate"]); ax.set_xticks(range(4), ["C0", "B1", "B2", "S0"]); ax.set_ylabel("Property harm rate"); ax.set_title(f"{phase.upper()} harm comparison")
    fig.tight_layout(); fig.savefig(PLOTS / f"{prefix}_harm_comparison.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([pivot[value].to_numpy(float) for value in identifiers], labels=["C0", "B1", "B2", "S0"], showmeans=True); ax.set_ylabel("Property absolute error"); ax.set_title(f"{phase.upper()} C0/B1/B2/S0")
    fig.tight_layout(); fig.savefig(PLOTS / f"{prefix}_c0_b1_b2_s0.png", dpi=180); plt.close(fig)
    if phase == "formal256":
        names, means, low, high = [], [], [], []
        for name, result in bootstrap.items():
            names.append(name.replace(f"{s0_id}_vs_", "")); means.append(result["mean"]); low.append(result["mean"] - result["ci95_low"]); high.append(result["ci95_high"] - result["mean"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(range(len(names)), means, yerr=[low, high], fmt="o", capsize=5); ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks(range(len(names)), names, rotation=20, ha="right"); ax.set_ylabel("Paired property gain, 95% bootstrap CI")
        fig.tight_layout(); fig.savefig(PLOTS / "formal_bootstrap.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(4), view["p90_positive_property_degradation"]); ax.set_xticks(range(4), ["C0", "B1", "B2", "S0"]); ax.set_ylabel("P90 positive degradation"); ax.set_title("Formal tail comparison")
        fig.tight_layout(); fig.savefig(PLOTS / "formal_tail_comparison.png", dpi=180); plt.close(fig)


def report(phase: str, decision: dict, summary: pd.DataFrame) -> None:
    phase_root = ROOT / phase
    status_key = {"calibration": "STAGE_SCHEDULE_CALIBRATION", "p0": "STAGE_CFG_P0", "formal256": "STAGE_BOUNDED_CFG_FORMAL256"}[phase]
    lines = [f"# {phase} final report", "", "All outcomes use the frozen paired protocol and surrogate property/quality evaluators; no DFT verification is claimed.", "", "## Decision", "", f"- {status_key} = {decision[status_key]}", f"- Seeds: {paired_n(summary)}", f"- Selected schedule: {decision.get('selected_schedule_id', aliases()[2] if (ROOT / 'implementation/frozen_stage_cfg.yaml').exists() else 'none')}", f"- Stop reason: {decision.get('stop_reason') or 'none'}", "", "## Frozen status", ""]
    for key in ("STAGE0_DIAGNOSTIC", "STAGE_SCHEDULE_CALIBRATION", "STAGE_CFG_P0", "STAGE_BOUNDED_CFG_FORMAL256", "STAGE_SPECIFIC_EFFECT", "BOUNDED_PULSE_EFFECT", "INNOVATION1_FINAL_STATUS"):
        lines.append(f"{key} = {decision[key]}")
    (phase_root / "final_report.md").write_text("\n".join(lines) + "\n")


def paired_n(summary: pd.DataFrame) -> int:
    return int(summary["n"].max())


def main(phase: str) -> None:
    PLOTS.mkdir(exist_ok=True)
    paired = build_paired(quality_enriched(phase))
    metrics, harms, tails = summarize(paired)
    summary = write_common(phase, paired, metrics, harms, tails)
    if phase == "calibration":
        decision = calibration_gate(paired, summary)
    else:
        decision = fresh_gate(phase, paired, summary)
    json_write(ROOT / phase / "decision_summary.json", decision)
    report(phase, decision, summary)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "p0", "formal256"), required=True)
    arguments = parser.parse_args()
    main(arguments.phase)
