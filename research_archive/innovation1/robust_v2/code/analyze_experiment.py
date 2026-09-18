"""Paired statistics, harm/tail analysis, decisions, plots, and reports."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

from ase.io import read
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/robust_adaptive_cfg_v2"
PLOTS = ROOT / "plots"
METHODS = {
    "p0": ("C0", "A_OLD", "A_NORM", "A_ROBUST"),
    "formal256": ("C0", "A_OLD", "A_ROBUST"),
}
BOOTSTRAP_SEEDS = {"p0": 20260916, "formal256": 20260917}
N_BOOTSTRAP = 20_000
TARGET = 0.1


def root_for(cohort: str) -> Path:
    return ROOT / cohort


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def seeds_for(cohort: str) -> tuple[int, ...]:
    return tuple(map(int, json.loads((root_for(cohort) / "seeds.json").read_text())["seeds"]))


def load_method(cohort: str, method: str) -> list[dict]:
    root = root_for(cohort)
    seeds = seeds_for(cohort)
    generation = sorted(
        (row for row in read_csv(root / "generation_manifest.csv") if row["method"] == method),
        key=lambda row: int(row["seed"]),
    )
    properties = sorted(
        (row for row in read_csv(root / "property_metrics.csv") if row["method"] == method),
        key=lambda row: int(row["seed"]),
    )
    with gzip.open(root / "p0_quality" / method / "official_detailed.json.gz", "rt") as stream:
        detail = json.load(stream)
    relaxation = json.loads(
        (root / "p0_relaxation" / method / "relaxation_summary.json").read_text()
    )
    initial = read(
        root / "p0_relaxation" / method / "initial_with_properties.extxyz", index=":"
    )
    expected = list(seeds)
    if [int(row["seed"]) for row in generation] != expected:
        raise RuntimeError(f"{cohort}/{method}: generation seeds mismatch")
    if [int(row["seed"]) for row in properties] != expected:
        raise RuntimeError(f"{cohort}/{method}: property seeds mismatch")
    if list(map(int, relaxation["sample_seeds"])) != expected or relaxation["success"] is not True:
        raise RuntimeError(f"{cohort}/{method}: relaxation mismatch")
    lengths = {len(generation), len(properties), len(initial)}
    lengths.update(
        len(detail[key])
        for key in (
            "energy_above_hull_per_atom",
            "stable",
            "novel_unique_stable",
            "novel",
            "unique",
            "rmsd_from_relaxation",
        )
    )
    if lengths != {len(seeds)}:
        raise RuntimeError(f"{cohort}/{method}: metric lengths={lengths}")
    rows = []
    for index, seed in enumerate(seeds):
        force_norms = np.linalg.norm(np.asarray(initial[index].get_forces()), axis=1)
        row = {
            "method": method,
            "seed": seed,
            "formula": generation[index]["formula"],
            "num_atoms": int(generation[index]["num_atoms"]),
            "maxF_ev_per_a": float(force_norms.max()),
            "atomic_force_mean_ev_per_a": float(force_norms.mean()),
            "rmsd_a": float(detail["rmsd_from_relaxation"][index]),
            "e_hull_ev_per_atom": float(detail["energy_above_hull_per_atom"][index]),
            "stable": bool(detail["stable"][index]),
            "nus": bool(detail["novel_unique_stable"][index]),
            "novel": bool(detail["novel"][index]),
            "unique": bool(detail["unique"][index]),
            "mag_density_a3": float(properties[index]["chgnet_mag_density"]),
            "mag_absolute_error_a3": float(properties[index]["mag_absolute_error"]),
            "mag_hit": truth(properties[index]["mag_hit"]),
            "valid": truth(properties[index]["structure_valid"]),
            "minimum_periodic_distance_a": float(
                properties[index]["minimum_periodic_distance_angstrom"]
            ),
            "generation_seconds": float(generation[index]["elapsed_seconds"]),
            "mattergen_score_calls": int(generation[index]["mattergen_score_calls"]),
            "relaxation_steps": int(relaxation["relaxation_steps"][index]),
            "surrogate_property_eval": True,
            "dft_verified": False,
        }
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                if key == "minimum_periodic_distance_a" and len(initial[index]) == 1:
                    continue
                raise RuntimeError(f"{cohort}/{method}/{seed}: nonfinite {key}")
        rows.append(row)
    return rows


def summarize(method: str, rows: list[dict]) -> dict:
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=float)

    return {
        "method": method,
        "n": len(rows),
        "e_hull_mean_ev_per_atom": float(values("e_hull_ev_per_atom").mean()),
        "e_hull_median_ev_per_atom": float(np.median(values("e_hull_ev_per_atom"))),
        "stable_fraction": float(values("stable").mean()),
        "novel_fraction": float(values("novel").mean()),
        "unique_fraction": float(values("unique").mean()),
        "nus_fraction": float(values("nus").mean()),
        "validity_fraction": float(values("valid").mean()),
        "mag_mae_a3": float(values("mag_absolute_error_a3").mean()),
        "mag_hit_fraction": float(values("mag_hit").mean()),
        "maxF_mean_ev_per_a": float(values("maxF_ev_per_a").mean()),
        "atomic_force_mean_ev_per_a": float(values("atomic_force_mean_ev_per_a").mean()),
        "rmsd_mean_a": float(values("rmsd_a").mean()),
        "runtime_mean_seconds": float(values("generation_seconds").mean()),
        "mattergen_score_calls_mean": float(values("mattergen_score_calls").mean()),
        "surrogate_property_eval": True,
        "dft_verified": False,
    }


METRICS = {
    "E-hull": ("e_hull_ev_per_atom", "lower"),
    "Stable": ("stable", "higher"),
    "NUS": ("nus", "higher"),
    "Novel": ("novel", "higher"),
    "Unique": ("unique", "higher"),
    "Validity": ("valid", "higher"),
    "Mag MAE": ("mag_absolute_error_a3", "lower"),
    "Mag Hit": ("mag_hit", "higher"),
    "MaxF": ("maxF_ev_per_a", "lower"),
    "Atomic Force": ("atomic_force_mean_ev_per_a", "lower"),
    "RMSD": ("rmsd_a", "lower"),
    "Runtime": ("generation_seconds", "lower"),
}


def bootstrap_comparisons(
    cohort: str, data: dict[str, list[dict]], comparisons: tuple[tuple[str, str], ...]
) -> list[dict]:
    n = len(seeds_for(cohort))
    rng = np.random.default_rng(BOOTSTRAP_SEEDS[cohort])
    indices = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    results = []
    for baseline_name, method_name in comparisons:
        for metric, (key, direction) in METRICS.items():
            baseline = np.asarray([row[key] for row in data[baseline_name]], dtype=float)
            method = np.asarray([row[key] for row in data[method_name]], dtype=float)
            delta = method - baseline
            samples = delta[indices].mean(axis=1)
            tolerance = 1e-12
            favorable = delta < -tolerance if direction == "lower" else delta > tolerance
            unfavorable = delta > tolerance if direction == "lower" else delta < -tolerance
            results.append(
                {
                    "comparison": f"{method_name}-{baseline_name}",
                    "baseline": baseline_name,
                    "method": method_name,
                    "metric": metric,
                    "preferred_direction": direction,
                    "n": n,
                    "baseline_mean": float(baseline.mean()),
                    "method_mean": float(method.mean()),
                    "mean_delta_method_minus_baseline": float(delta.mean()),
                    "median_paired_delta": float(np.median(delta)),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "wins": int(favorable.sum()),
                    "ties": int((~(favorable | unfavorable)).sum()),
                    "losses": int(unfavorable.sum()),
                    "resamples": N_BOOTSTRAP,
                    "bootstrap_seed": BOOTSTRAP_SEEDS[cohort],
                }
            )
    return results


def harm_tables(data: dict[str, list[dict]], methods: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    baseline = data["C0"]
    summaries = []
    per_seed = []
    for method in methods:
        if method == "C0":
            continue
        e_delta = np.asarray(
            [candidate["e_hull_ev_per_atom"] - base["e_hull_ev_per_atom"] for base, candidate in zip(baseline, data[method])]
        )
        property_delta = np.asarray(
            [candidate["mag_absolute_error_a3"] - base["mag_absolute_error_a3"] for base, candidate in zip(baseline, data[method])]
        )
        harm_e = e_delta > 0.02
        harm_s = np.asarray([base["stable"] and not candidate["stable"] for base, candidate in zip(baseline, data[method])])
        harm_p = np.asarray(
            [candidate["mag_absolute_error_a3"] > 1.1 * base["mag_absolute_error_a3"] for base, candidate in zip(baseline, data[method])]
        )
        worst_count = max(1, math.ceil(len(e_delta) * 0.25))
        summaries.append(
            {
                "method": method,
                "baseline": "C0",
                "n": len(e_delta),
                "harm_rate_ehull": float(harm_e.mean()),
                "harm_rate_stable": float(harm_s.mean()),
                "harm_rate_property": float(harm_p.mean()),
                "ehull_degradation_p75": float(np.quantile(e_delta, 0.75)),
                "ehull_degradation_p90": float(np.quantile(e_delta, 0.90)),
                "worst_quartile_ehull_degradation_mean": float(np.sort(e_delta)[-worst_count:].mean()),
                "property_degradation_p75": float(np.quantile(property_delta, 0.75)),
                "property_degradation_p90": float(np.quantile(property_delta, 0.90)),
            }
        )
        for index, (base, candidate) in enumerate(zip(baseline, data[method])):
            per_seed.append(
                {
                    "method": method,
                    "seed": candidate["seed"],
                    "ehull_delta_vs_C0": e_delta[index],
                    "stable_delta_vs_C0": int(candidate["stable"]) - int(base["stable"]),
                    "nus_delta_vs_C0": int(candidate["nus"]) - int(base["nus"]),
                    "property_error_delta_vs_C0": property_delta[index],
                    "harm_ehull": bool(harm_e[index]),
                    "harm_stable": bool(harm_s[index]),
                    "harm_property": bool(harm_p[index]),
                }
            )
    return summaries, per_seed


def paired_rows(data: dict[str, list[dict]], methods: tuple[str, ...]) -> list[dict]:
    rows = []
    for index, seed in enumerate(row["seed"] for row in data["C0"]):
        row = {"seed": seed}
        for method in methods:
            for metric, (key, _) in METRICS.items():
                clean = metric.lower().replace("-", "_").replace(" ", "_")
                row[f"{method}_{clean}"] = data[method][index][key]
                if method != "C0":
                    row[f"delta_{method}_minus_C0_{clean}"] = (
                        float(data[method][index][key]) - float(data["C0"][index][key])
                    )
        rows.append(row)
    return rows


def control_summary(root: Path, methods: tuple[str, ...]) -> list[dict]:
    trace = pd.read_csv(root / "control_trace.csv")
    rows = []
    for method in methods:
        schedule = {
            "C0": "constant",
            "A_OLD": "adaptive",
            "A_NORM": "normalized_adaptive",
            "A_ROBUST": "robust_adaptive",
        }[method]
        part = trace[trace["guidance_schedule"] == schedule].copy()
        cfg = part["final_guidance"].astype(float)
        confidence = pd.to_numeric(part.get("confidence"), errors="coerce")
        disagreement = pd.to_numeric(part.get("field_std"), errors="coerce")
        fallback = part.get("fallback_to_base", pd.Series(index=part.index, dtype=object)).astype(str).str.lower() == "true"
        slew = part.get("slew_limited", pd.Series(index=part.index, dtype=object)).astype(str).str.lower() == "true"
        rows.append(
            {
                "method": method,
                "events": int(len(part)),
                "cfg_mean": float(cfg.mean()),
                "cfg_median": float(cfg.median()),
                "cfg_p05": float(cfg.quantile(0.05)),
                "cfg_p95": float(cfg.quantile(0.95)),
                "cfg_min": float(cfg.min()),
                "cfg_max": float(cfg.max()),
                "max_cfg_excursion": float((cfg - 2.0).abs().max()),
                "time_away_from_baseline_rate": float(((cfg - 2.0).abs() > 0.01).mean()),
                "confidence_active_rate": None if confidence.isna().all() else float((confidence > 0).mean()),
                "fallback_to_g0_rate": None if method != "A_ROBUST" else float(fallback.mean()),
                "field_disagreement_mean": None if disagreement.isna().all() else float(disagreement.mean()),
                "field_disagreement_p95": None if disagreement.isna().all() else float(disagreement.quantile(0.95)),
                "slew_limit_trigger_count": None if method != "A_ROBUST" else int(slew.sum()),
            }
        )
    write_csv(root / "controller_behavior.csv", rows)
    return rows


def relative_reduction(old: float, robust: float) -> float | None:
    return None if old <= 0 else (old - robust) / old


def make_decision(
    cohort: str,
    summaries: dict[str, dict],
    harms: dict[str, dict],
    bootstraps: list[dict],
) -> dict:
    old = harms["A_OLD"]
    robust = harms["A_ROBUST"]
    reductions = {
        name: relative_reduction(old[f"harm_rate_{name}"], robust[f"harm_rate_{name}"])
        for name in ("ehull", "stable", "property")
    }
    at_least_one_30 = any(value is not None and value >= 0.30 for value in reductions.values())
    no_harm_reverse = all(
        robust[f"harm_rate_{name}"] - old[f"harm_rate_{name}"] <= 0.05 + 1e-12
        for name in reductions
    )
    c0 = summaries["C0"]
    r0 = summaries["A_ROBUST"]
    changes = {
        "ehull": r0["e_hull_mean_ev_per_atom"] - c0["e_hull_mean_ev_per_atom"],
        "stable": r0["stable_fraction"] - c0["stable_fraction"],
        "nus": r0["nus_fraction"] - c0["nus_fraction"],
    }
    average_positive = changes["ehull"] < -1e-12 or changes["stable"] > 1e-12 or changes["nus"] > 1e-12
    guardrails = {
        "mag_mae_worsening_le_10pct": (
            (r0["mag_mae_a3"] - c0["mag_mae_a3"]) / max(c0["mag_mae_a3"], 1e-12)
            <= 0.10 + 1e-12
        ),
        "validity_drop_le_5pp": c0["validity_fraction"] - r0["validity_fraction"] <= 0.05 + 1e-12,
        "nus_drop_le_5pp": c0["nus_fraction"] - r0["nus_fraction"] <= 0.05 + 1e-12,
    }
    no_severe_tail = c0["stable_fraction"] - r0["stable_fraction"] <= 0.20 + 1e-12
    common = {
        "harm_relative_reduction_vs_A_OLD": reductions,
        "at_least_one_harm_relative_reduction_ge_30pct": at_least_one_30,
        "no_other_harm_increase_gt_5pp": no_harm_reverse,
        "mean_quality_changes_A_ROBUST_minus_C0": changes,
        "at_least_one_mean_quality_direction_favorable": average_positive,
        "guardrails": guardrails,
        "no_severe_stable_collapse_gt_20pp": no_severe_tail,
    }
    if cohort == "p0":
        go = at_least_one_30 and no_harm_reverse and average_positive and all(guardrails.values()) and no_severe_tail
        neutral = changes["ehull"] <= 0.01 and changes["stable"] >= -0.05 and changes["nus"] >= -0.05
        borderline = (
            not go
            and at_least_one_30
            and no_harm_reverse
            and neutral
            and all(guardrails.values())
            and no_severe_tail
        )
        verdict = "GO" if go else "BORDERLINE" if borderline else "FAIL"
        return {
            "ROBUST_ADAPTIVE_CFG_P0": verdict,
            "formal256_allowed": verdict == "GO",
            "next": "RUN_FRESH_FORMAL256" if verdict == "GO" else "STOP_NO_RESCUE_TUNING",
            **common,
        }

    headline = [
        row
        for row in bootstraps
        if row["comparison"] == "A_ROBUST-C0" and row["metric"] in ("E-hull", "Stable", "NUS")
    ]
    favorable_ci = []
    reverse_ci = []
    for row in headline:
        if row["preferred_direction"] == "lower":
            favorable = row["ci95_high"] < 0
            reverse = row["ci95_low"] > 0
        else:
            favorable = row["ci95_low"] > 0
            reverse = row["ci95_high"] < 0
        if favorable:
            favorable_ci.append(row["metric"])
        if reverse:
            reverse_ci.append(row["metric"])
    confirmed = (
        bool(favorable_ci)
        and not reverse_ci
        and at_least_one_30
        and no_harm_reverse
        and no_severe_tail
    )
    return {
        "ROBUST_ADAPTIVE_CFG_FORMAL256": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
        "INNOVATION1_FINAL_STATUS": "CONFIRMED" if confirmed else "MIXED",
        "headline_metrics_with_favorable_ci_excluding_zero": favorable_ci,
        "headline_metrics_with_significant_reverse_ci": reverse_ci,
        "stop": True,
        **common,
    }


def plots(cohort: str, data: dict[str, list[dict]], harm_rows: list[dict], behavior: list[dict], bootstrap: list[dict]) -> None:
    PLOTS.mkdir(exist_ok=True)
    suffix = "" if cohort == "p0" else "formal256_"
    old_cfg = []
    robust_cfg = []
    trace = pd.read_csv(root_for(cohort) / "control_trace.csv")
    old_cfg = trace.loc[trace["guidance_schedule"] == "adaptive", "final_guidance"].astype(float)
    robust_cfg = trace.loc[trace["guidance_schedule"] == "robust_adaptive", "final_guidance"].astype(float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(old_cfg, bins=60, density=True, alpha=0.5, label="A-old")
    ax.hist(robust_cfg, bins=60, density=True, alpha=0.5, label="A-robust")
    ax.axvline(2.0, color="black", linestyle="--")
    ax.set(xlabel="CFG", ylabel="density", title="V1 vs Robust V2 CFG distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / ("v1_vs_v2_cfg_distribution.png" if cohort == "p0" else "formal256_cfg_distribution.png"), dpi=180)
    plt.close(fig)

    harm_frame = pd.DataFrame(harm_rows).set_index("method")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(harm_frame))
    width = 0.25
    for offset, key, label in [(-width, "harm_rate_ehull", "E-hull"), (0, "harm_rate_stable", "Stable"), (width, "harm_rate_property", "Property")]:
        ax.bar(x + offset, harm_frame[key], width=width, label=label)
    ax.set_xticks(x, harm_frame.index)
    ax.set(ylabel="harm rate", title="Harm-rate comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / ("harm_rate_comparison.png" if cohort == "p0" else "formal256_harm_rate.png"), dpi=180)
    plt.close(fig)

    base = np.asarray([row["e_hull_ev_per_atom"] for row in data["C0"]])
    robust = np.asarray([row["e_hull_ev_per_atom"] for row in data["A_ROBUST"]])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(base, robust, s=14, alpha=0.7)
    limit = max(base.max(), robust.max())
    ax.plot([0, limit], [0, limit], color="black", linestyle="--")
    ax.set(xlabel="C0 E-hull", ylabel="A-robust E-hull", title="Paired E-hull")
    fig.tight_layout()
    fig.savefig(PLOTS / ("ehull_paired.png" if cohort == "p0" else "formal256_paired_ehull.png"), dpi=180)
    plt.close(fig)

    summary = {method: summarize(method, rows) for method, rows in data.items()}
    for metric, key, filename in [
        ("Stable", "stable_fraction", "stable_comparison.png"),
        ("NUS", "nus_fraction", "nus_comparison.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        names = list(data)
        ax.bar(names, [summary[name][key] for name in names])
        ax.set(ylim=(0, 1), ylabel="fraction", title=f"{metric} comparison")
        fig.tight_layout()
        fig.savefig(PLOTS / ((suffix + filename) if cohort != "p0" else filename), dpi=180)
        plt.close(fig)

    frame = pd.DataFrame(harm_rows).set_index("method")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(frame.index, frame["worst_quartile_ehull_degradation_mean"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set(ylabel="mean ΔE-hull in worst 25%", title="Worst-quartile degradation")
    fig.tight_layout()
    fig.savefig(PLOTS / ((suffix + "worst_quartile_comparison.png") if cohort != "p0" else "worst_quartile_comparison.png"), dpi=180)
    plt.close(fig)

    if cohort == "formal256":
        head = [row for row in bootstrap if row["comparison"] == "A_ROBUST-C0" and row["metric"] in ("E-hull", "Stable", "NUS")]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        means = np.asarray([row["mean_delta_method_minus_baseline"] for row in head])
        lower = means - np.asarray([row["ci95_low"] for row in head])
        upper = np.asarray([row["ci95_high"] for row in head]) - means
        ax.errorbar(range(len(head)), means, yerr=[lower, upper], fmt="o", capsize=4)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xticks(range(len(head)), [row["metric"] for row in head])
        ax.set(ylabel="paired mean delta (A-robust − C0)", title="20k paired bootstrap 95% CI")
        fig.tight_layout()
        fig.savefig(PLOTS / "formal256_bootstrap.png", dpi=180)
        plt.close(fig)


def main(cohort: str) -> None:
    root = root_for(cohort)
    methods = METHODS[cohort]
    data = {method: load_method(cohort, method) for method in methods}
    per_structure = [row for method in methods for row in data[method]]
    write_csv(root / "per_structure_metrics.csv", per_structure)
    summaries = [summarize(method, data[method]) for method in methods]
    write_csv(root / "quality_metrics.csv", summaries)
    harm_summary, harm_per_seed = harm_tables(data, methods)
    write_csv(root / "harm_metrics.csv", harm_summary)
    write_csv(root / "harm_per_seed.csv", harm_per_seed)
    pairs = paired_rows(data, methods)
    write_csv(root / "paired_results.csv", pairs)
    comparisons = tuple(("C0", method) for method in methods if method != "C0")
    if "A_OLD" in methods and "A_ROBUST" in methods:
        comparisons += (("A_OLD", "A_ROBUST"),)
    bootstrap = bootstrap_comparisons(cohort, data, comparisons)
    (root / "bootstrap_results.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    behavior = control_summary(root, methods)
    summary_by_method = {row["method"]: row for row in summaries}
    harm_by_method = {row["method"]: row for row in harm_summary}
    decision = make_decision(cohort, summary_by_method, harm_by_method, bootstrap)
    decision.update(
        {
            "cohort": cohort,
            "n_paired": len(seeds_for(cohort)),
            "all_registered_seeds_retained": True,
            "bootstrap_resamples": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEEDS[cohort],
            "SURROGATE_PROPERTY_EVAL": True,
            "DFT_VERIFIED": False,
            "parameters_retuned_after_results": False,
        }
    )
    (root / "decision_summary.json").write_text(json.dumps(decision, indent=2) + "\n")
    plots(cohort, data, harm_summary, behavior, bootstrap)

    title = "Fresh P0" if cohort == "p0" else "Fresh Formal256"
    report = f"""# Robust Adaptive CFG V2 — {title}

All {len(seeds_for(cohort))} preregistered paired seeds were retained. The methods, controller, calibration, confidence rule, bounds, slew cap, EMA, metrics, bootstrap count, and decision thresholds were frozen before this cohort was generated. No rescue tuning was performed.

## Quality metrics

{pd.DataFrame(summaries).to_markdown(index=False)}

## Harm and tail metrics

{pd.DataFrame(harm_summary).to_markdown(index=False)}

## Controller behavior

{pd.DataFrame(behavior).to_markdown(index=False)}

## Decision

```json
{json.dumps(decision, indent=2, ensure_ascii=False)}
```

All energy/stability and magnetic-property measurements are frozen surrogate evaluations. `SURROGATE_PROPERTY_EVAL=True`; `DFT_VERIFIED=False`. A P0 GO is only permission to run Formal256, not confirmatory evidence. Formal256 confirmation requires a favorable headline bootstrap CI excluding zero, no significant reverse headline, reduced harm, and no severe Stable tail collapse.
"""
    (root / "final_report.md").write_text(report)
    status = "COMPLETED"
    (root / "pipeline_status.json").write_text(json.dumps({"status": status, "stage": "decision", "decision": decision}, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("p0", "formal256"), required=True)
    args = parser.parse_args()
    main(args.cohort)
