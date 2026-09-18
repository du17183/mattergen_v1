"""Evaluate fresh P0/Formal256 with paired statistics, 20k bootstrap, and hard gates."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from ase.io import read
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_offline_analysis import clopper_pearson, json_write


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/calibrated_safe_selection_cfg"
POLICIES = ("C0", "T0", "S0")
BOOTSTRAP_RESAMPLES = 20_000


def enrich_quality(cohort_root: Path) -> pd.DataFrame:
    values = pd.read_csv(cohort_root / "property_metrics.csv")
    for column, default in (
        ("e_hull", np.nan), ("stable", False), ("nus", False), ("novel", False),
        ("unique", False), ("rmsd", np.nan), ("quality_success", False),
    ):
        values[column] = default
    for policy in POLICIES:
        structure_path = cohort_root / "evaluation_structures" / f"{policy}.extxyz"
        quality_root = cohort_root / "quality" / policy
        if not structure_path.exists():
            continue
        if not (quality_root / "official_detailed.json.gz").exists():
            raise FileNotFoundError(f"quality result missing: {policy}")
        atoms = read(structure_path, index=":")
        per_structure = pd.read_csv(quality_root / "per_structure.csv")
        with gzip.open(quality_root / "official_detailed.json.gz", "rt") as stream:
            detailed = json.load(stream)
        for output_index, row in per_structure.reset_index(drop=True).iterrows():
            identifier = str(atoms[int(row["index"])].info["branch_id"])
            match = values.index[values["branch_id"] == identifier]
            if len(match) != 1:
                raise RuntimeError(f"quality branch lookup failed: {identifier}")
            target = match[0]
            values.loc[target, "e_hull"] = float(detailed["energy_above_hull_per_atom"][output_index])
            values.loc[target, "stable"] = bool(detailed["stable"][output_index])
            values.loc[target, "nus"] = bool(detailed["novel_unique_stable"][output_index])
            values.loc[target, "novel"] = bool(detailed["novel"][output_index])
            values.loc[target, "unique"] = bool(detailed["unique"][output_index])
            values.loc[target, "rmsd"] = float(detailed["rmsd_from_relaxation"][output_index])
            values.loc[target, "quality_success"] = True
    for column in ("stable", "nus", "novel", "unique", "quality_success"):
        values[column] = values[column].fillna(False).astype(bool)
    values["evaluation_valid"] = values["structure_valid"].astype(bool) & values["quality_success"]
    return values


def paired_table(values: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    trace_by_seed = traces.set_index("seed")
    for seed, group in values.groupby("seed", sort=False):
        if len(group) != 3 or set(group["policy"]) != set(POLICIES):
            raise RuntimeError(f"paired policy mismatch: {seed}")
        policy = {name: group[group["policy"] == name].iloc[0] for name in POLICIES}
        row = {"seed": int(seed), "adaptive_action": str(trace_by_seed.loc[int(seed), "selected_action"])}
        for name in POLICIES:
            for metric in (
                "property_absolute_error", "e_hull", "stable", "nus", "novel",
                "unique", "evaluation_valid",
            ):
                row[f"{metric}_{name}"] = policy[name][metric]
        row["adapted"] = row["adaptive_action"] != "KEEP"
        row["property_benefit_S0_vs_C0"] = float(row["property_absolute_error_C0"]) - float(row["property_absolute_error_S0"])
        row["property_benefit_S0_vs_T0"] = float(row["property_absolute_error_T0"]) - float(row["property_absolute_error_S0"])
        row["property_harm"] = bool(
            row["adapted"]
            and float(row["property_absolute_error_S0"]) > 1.05 * float(row["property_absolute_error_C0"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    chunk = 1000
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return {
        "n": len(values), "mean": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "resamples": BOOTSTRAP_RESAMPLES, "seed": seed,
    }


def plots(cohort: str, frame: pd.DataFrame, bootstrap_result: dict, metrics: pd.DataFrame) -> None:
    cohort_root = ROOT / cohort
    prefix = "p0" if cohort == "p0" else "formal"
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(frame["property_absolute_error_C0"], frame["property_absolute_error_S0"], alpha=0.7)
    maximum = float(max(frame["property_absolute_error_C0"].max(), frame["property_absolute_error_S0"].max()))
    ax.plot([0, maximum], [0, maximum], "k--"); ax.set(xlabel="C0 property error", ylabel="S0 property error")
    fig.tight_layout(); fig.savefig(cohort_root / f"{prefix}_property_paired.png", dpi=180); plt.close(fig)

    adapted = frame[frame["adapted"]]
    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.bar(["non-harm", "harm"], [int((~adapted["property_harm"]).sum()), int(adapted["property_harm"].sum())])
    ax.set(ylabel="Adaptive samples")
    fig.tight_layout(); fig.savefig(cohort_root / f"{prefix}_harm.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.bar(["KEEP", "ADAPT"], [int((~frame["adapted"]).sum()), int(frame["adapted"].sum())])
    ax.set(ylabel="Samples")
    fig.tight_layout(); fig.savefig(cohort_root / ("p0_coverage.png" if cohort == "p0" else "formal_risk_coverage.png"), dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5)); ax.bar(metrics["policy"], metrics["property_mae"])
    ax.set(ylabel="Property MAE (lower is better)")
    fig.tight_layout(); fig.savefig(cohort_root / f"{prefix}_c0_t0_s0.png", dpi=180); plt.close(fig)

    if cohort == "formal256":
        point = bootstrap_result["mean"]; lower, upper = bootstrap_result["ci95"]
        fig, ax = plt.subplots(figsize=(5, 4.5)); ax.errorbar(["C0-S0 benefit"], [point], yerr=[[point - lower], [upper - point]], fmt="o", capsize=5)
        ax.axhline(0, color="black", linestyle="--"); ax.set(ylabel="Paired property benefit")
        fig.tight_layout(); fig.savefig(cohort_root / "formal_bootstrap.png", dpi=180); plt.close(fig)


def main(cohort: str) -> None:
    cohort_root = ROOT / cohort
    values = enrich_quality(cohort_root)
    traces = pd.read_csv(cohort_root / "controller_trace.csv")
    frame = paired_table(values, traces)
    frame.to_csv(cohort_root / "paired_results.csv", index=False)
    metric_rows = []
    for policy in POLICIES:
        metric_rows.append(
            {
                "policy": policy, "n": len(frame),
                "property_mae": float(frame[f"property_absolute_error_{policy}"].mean()),
                "mean_e_hull": float(frame[f"e_hull_{policy}"].mean()),
                "stable": float(frame[f"stable_{policy}"].astype(float).mean()),
                "nus": float(frame[f"nus_{policy}"].astype(float).mean()),
                "novel": float(frame[f"novel_{policy}"].astype(float).mean()),
                "unique": float(frame[f"unique_{policy}"].astype(float).mean()),
                "validity": float(frame[f"evaluation_valid_{policy}"].astype(float).mean()),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(cohort_root / "metrics.csv", index=False)
    adapted = frame["adapted"].astype(bool)
    adaptive_count = int(adapted.sum())
    harm_count = int(frame.loc[adapted, "property_harm"].sum())
    selective_harm = harm_count / adaptive_count if adaptive_count else 1.0
    harm_ci = clopper_pearson(harm_count, adaptive_count)
    pd.DataFrame([
        {
            "n": len(frame), "adaptive_count": adaptive_count, "harm_count": harm_count,
            "overall_property_harm_rate": float(frame["property_harm"].mean()),
            "selective_property_harm_rate": selective_harm,
            "selective_harm_ci95_lower": harm_ci[0], "selective_harm_ci95_upper": harm_ci[1],
        }
    ]).to_csv(cohort_root / "harm_metrics.csv", index=False)
    action_counts = frame["adaptive_action"].value_counts()
    pd.DataFrame([
        {
            "n": len(frame), "adaptive_count": adaptive_count, "coverage": float(adapted.mean()),
            "down_count": int(action_counts.get("DOWN", 0)), "keep_count": int(action_counts.get("KEEP", 0)),
            "up_count": int(action_counts.get("UP", 0)),
        }
    ]).to_csv(cohort_root / "coverage_metrics.csv", index=False)
    wins = int((frame["property_benefit_S0_vs_C0"] > 0).sum())
    losses = int((frame["property_benefit_S0_vs_C0"] < 0).sum())
    ties = int((frame["property_benefit_S0_vs_C0"] == 0).sum())
    property_bootstrap = bootstrap(frame["property_benefit_S0_vs_C0"].to_numpy(float), 20261010 if cohort == "p0" else 20261011)
    secondary = {
        "e_hull_lower_is_better": bootstrap((frame["e_hull_C0"] - frame["e_hull_S0"]).to_numpy(float), 20261020),
        "stable_higher_is_better": bootstrap((frame["stable_S0"].astype(float) - frame["stable_C0"].astype(float)).to_numpy(), 20261021),
        "nus_higher_is_better": bootstrap((frame["nus_S0"].astype(float) - frame["nus_C0"].astype(float)).to_numpy(), 20261022),
        "validity_higher_is_better": bootstrap((frame["evaluation_valid_S0"].astype(float) - frame["evaluation_valid_C0"].astype(float)).to_numpy(), 20261023),
    }
    bootstrap_payload = {"property_C0_minus_S0": property_bootstrap, "secondary": secondary}
    json_write(cohort_root / "bootstrap_results.json", bootstrap_payload)
    c0 = metrics.set_index("policy").loc["C0"]
    t0 = metrics.set_index("policy").loc["T0"]
    s0 = metrics.set_index("policy").loc["S0"]
    relative_improvement = float((c0["property_mae"] - s0["property_mae"]) / max(c0["property_mae"], 1e-12))
    coverage = float(adapted.mean())
    prior = json.loads((ROOT / "final_status.json").read_text())
    if cohort == "p0":
        checks = {
            "relative_property_mae_improvement_ge_5pct": relative_improvement >= 0.05,
            "coverage_ge_15pct": coverage >= 0.15,
            "selective_property_harm_le_10pct": selective_harm <= 0.10,
            "s0_property_mae_better_than_t0": s0["property_mae"] < t0["property_mae"],
            "stable_drop_le_5pp": c0["stable"] - s0["stable"] <= 0.05,
            "nus_drop_le_5pp": c0["nus"] - s0["nus"] <= 0.05,
            "validity_drop_le_5pp": c0["validity"] - s0["validity"] <= 0.05,
            "e_hull_worsening_le_0.02": s0["mean_e_hull"] - c0["mean_e_hull"] <= 0.02,
        }
        gate = "GO" if all(checks.values()) else "FAIL"
        prior.update(
            SAFE_SELECTION_P0=gate, SAFE_SELECTION_FORMAL256="NOT_RUN",
            SAMPLE_ADAPTIVITY="SUPPORTED" if checks["s0_property_mae_better_than_t0"] else "NOT_SUPPORTED",
            INNOVATION1_FINAL_STATUS="MIXED",
            next="RUN_FRESH_FORMAL256" if gate == "GO" else "STOP",
        )
    else:
        checks = {
            "property_bootstrap_ci95_favorable_excludes_zero": property_bootstrap["ci95"][0] > 0,
            "coverage_ge_15pct": coverage >= 0.15,
            "selective_property_harm_le_10pct": selective_harm <= 0.10,
            "s0_property_mae_better_than_t0": s0["property_mae"] < t0["property_mae"],
            "e_hull_no_clear_adverse_ci": secondary["e_hull_lower_is_better"]["ci95"][1] >= 0,
            "stable_no_clear_adverse_ci": secondary["stable_higher_is_better"]["ci95"][1] >= 0,
            "nus_no_clear_adverse_ci": secondary["nus_higher_is_better"]["ci95"][1] >= 0,
            "validity_no_clear_adverse_ci": secondary["validity_higher_is_better"]["ci95"][1] >= 0,
        }
        gate = "CONFIRMED" if all(checks.values()) else "NOT_CONFIRMED"
        prior.update(
            SAFE_SELECTION_FORMAL256=gate,
            SAMPLE_ADAPTIVITY="SUPPORTED" if checks["s0_property_mae_better_than_t0"] else "NOT_SUPPORTED",
            INNOVATION1_FINAL_STATUS="CONFIRMED" if gate == "CONFIRMED" else "MIXED",
            next="COMPLETE",
        )
    prior[f"{cohort}_gate_checks"] = checks
    prior[f"{cohort}_summary"] = {
        "n": len(frame), "coverage": coverage, "adaptive_count": adaptive_count,
        "selective_harm": selective_harm, "selective_harm_ci95": list(harm_ci),
        "relative_property_mae_improvement": relative_improvement,
        "c0_property_mae": float(c0["property_mae"]), "t0_property_mae": float(t0["property_mae"]),
        "s0_property_mae": float(s0["property_mae"]), "wins": wins, "ties": ties, "losses": losses,
        "property_bootstrap": property_bootstrap,
    }
    json_write(ROOT / "final_status.json", prior)
    report_lines = [
        f"# Calibrated Safe-Selection — {cohort.upper()}", "",
        f"- Gate: `{gate}`", f"- N: `{len(frame)}`", f"- Coverage: `{coverage:.4f}` ({adaptive_count}/{len(frame)})",
        f"- Selective property harm: `{selective_harm:.4f}`; exact 95% CI `{harm_ci[0]:.4f}, {harm_ci[1]:.4f}`",
        f"- C0/T0/S0 property MAE: `{c0['property_mae']:.6f} / {t0['property_mae']:.6f} / {s0['property_mae']:.6f}`",
        f"- Relative S0 improvement vs C0: `{relative_improvement:.4%}`",
        f"- Wins/ties/losses: `{wins}/{ties}/{losses}`",
        f"- 20k paired bootstrap C0-S0 benefit: `{property_bootstrap['mean']:.6f}`; 95% CI `{property_bootstrap['ci95'][0]:.6f}, {property_bootstrap['ci95'][1]:.6f}`",
        "", "## Frozen gate checks", "",
    ]
    report_lines.extend(f"- `{key}`: `{value}`" for key, value in checks.items())
    report_lines += ["", "No result-driven retuning or rescue was performed.\n"]
    report = "\n".join(report_lines)
    (cohort_root / "final_report.md").write_text(report)
    root_status_lines = "\n".join(f"- `{key} = {value}`" for key, value in prior.items() if key.isupper())
    (ROOT / "final_report.md").write_text(
        "# Calibrated Safe-Selection Adaptive CFG\n\n" + root_status_lines + "\n\n" + report
    )
    plots(cohort, frame, property_bootstrap, metrics)
    print(json.dumps(prior, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("p0", "formal256"), required=True)
    args = parser.parse_args()
    main(args.cohort)
