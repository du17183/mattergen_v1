"""Apply the pre-registered paired P0 gate without rescue tuning."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from ase.io import read
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/risk_calibrated_adaptive_cfg"
P0 = ROOT / "p0"
PLOTS = ROOT / "plots"
POLICIES = ("C0", "T0", "A0")
BOOTSTRAP_SEED = 20261001
RESAMPLES = 20_000


def enrich() -> pd.DataFrame:
    values = pd.read_csv(P0 / "property_metrics.csv")
    for column, default in (("e_hull", np.nan), ("stable", False), ("nus", False),
                            ("novel", False), ("unique", False), ("quality_success", False)):
        values[column] = default
    for policy in POLICIES:
        structure_path = P0 / "evaluation_structures" / f"{policy}.extxyz"
        if not structure_path.exists(): continue
        atoms = read(structure_path, index=":"); quality = P0 / "quality" / policy
        per_structure = pd.read_csv(quality / "per_structure.csv")
        with gzip.open(quality / "official_detailed.json.gz", "rt") as stream: detailed = json.load(stream)
        for output_index, row in per_structure.reset_index(drop=True).iterrows():
            input_index = int(row["index"]); seed = int(atoms[input_index].info["sample_seed"])
            match = values.index[(values["seed"] == seed) & (values["policy"] == policy)]
            if len(match) != 1: raise RuntimeError(f"P0 quality lookup failed: {seed}/{policy}")
            target = match[0]
            values.loc[target, "e_hull"] = float(detailed["energy_above_hull_per_atom"][output_index])
            values.loc[target, "stable"] = bool(detailed["stable"][output_index])
            values.loc[target, "nus"] = bool(detailed["novel_unique_stable"][output_index])
            values.loc[target, "novel"] = bool(detailed["novel"][output_index])
            values.loc[target, "unique"] = bool(detailed["unique"][output_index])
            values.loc[target, "quality_success"] = True
    for column in ("stable", "nus", "novel", "unique", "quality_success"):
        values[column] = values[column].fillna(False).astype(bool)
    values["evaluation_valid"] = values["structure_valid"].astype(bool) & values["quality_success"]
    values.to_csv(P0 / "metrics.csv", index=False)
    return values


def bootstrap(delta: np.ndarray, metric: str, comparison: str, rng: np.random.Generator) -> dict:
    draws = rng.choice(delta, size=(RESAMPLES, len(delta)), replace=True).mean(axis=1)
    return {"metric": metric, "comparison": comparison, "n": len(delta), "mean_delta": float(delta.mean()),
            "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)),
            "resamples": RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED}


def main() -> None:
    metrics = enrich(); wide = metrics.pivot(index="seed", columns="policy")
    trace = pd.read_csv(P0 / "controller_trace.csv").set_index("seed")
    seeds = wide.index.to_numpy(); paired_rows, harm_rows = [], []
    for seed in seeds:
        row = {"seed": int(seed), "a0_action": str(trace.loc[seed, "selected_action"])}
        for metric in ("property_absolute_error", "e_hull", "stable", "nus", "evaluation_valid"):
            for policy in POLICIES: row[f"{metric}_{policy}"] = wide.loc[seed, (metric, policy)]
        for comparison, left, right in (("A0_minus_C0", "A0", "C0"), ("T0_minus_C0", "T0", "C0"), ("A0_minus_T0", "A0", "T0")):
            row[f"delta_property_{comparison}"] = float(row[f"property_absolute_error_{left}"]) - float(row[f"property_absolute_error_{right}"])
            row[f"delta_ehull_{comparison}"] = float(row[f"e_hull_{left}"]) - float(row[f"e_hull_{right}"])
        paired_rows.append(row)
        adapted = row["a0_action"] != "KEEP"
        delta_property = row["delta_property_A0_minus_C0"]
        delta_ehull = row["delta_ehull_A0_minus_C0"]
        harm_rows.append({"seed": int(seed), "adapted": adapted,
                          "property_harm": bool(delta_property > .05 * max(float(row["property_absolute_error_C0"]), 1e-12)),
                          "ehull_harm": bool(np.isfinite(delta_ehull) and delta_ehull > .02),
                          "stable_harm": bool(row["stable_C0"] and not row["stable_A0"])})
    paired = pd.DataFrame(paired_rows); harm = pd.DataFrame(harm_rows)
    paired.to_csv(P0 / "paired_results.csv", index=False); harm.to_csv(P0 / "harm_metrics.csv", index=False)
    adapted = harm["adapted"]
    coverage = float(adapted.mean()); adaptive_count = int(adapted.sum())
    coverage_frame = pd.DataFrame([
        {"scope": "all", "n": len(harm), "adaptive_count": adaptive_count, "coverage": coverage,
         "property_harm_rate_all": float(harm["property_harm"].mean()),
         "selective_property_harm_rate": float(harm.loc[adapted, "property_harm"].mean()) if adaptive_count else 1.0,
         "selective_ehull_harm_rate": float(harm.loc[adapted, "ehull_harm"].mean()) if adaptive_count else 1.0,
         "selective_stable_harm_rate": float(harm.loc[adapted, "stable_harm"].mean()) if adaptive_count else 1.0}
    ])
    coverage_frame.to_csv(P0 / "coverage_metrics.csv", index=False)
    rng = np.random.default_rng(BOOTSTRAP_SEED); boot = []
    for comparison in ("A0_minus_C0", "T0_minus_C0", "A0_minus_T0"):
        boot.append(bootstrap(paired[f"delta_property_{comparison}"].to_numpy(float), "property_absolute_error", comparison, rng))
        finite = paired[f"delta_ehull_{comparison}"].dropna().to_numpy(float)
        if len(finite): boot.append(bootstrap(finite, "e_hull", comparison, rng))
    (P0 / "bootstrap_results.json").write_text(json.dumps({"results": boot}, indent=2) + "\n")
    means = metrics.groupby("policy").agg(property_mae=("property_absolute_error", "mean"), e_hull=("e_hull", "mean"),
                                           stable=("stable", "mean"), nus=("nus", "mean"), validity=("evaluation_valid", "mean"),
                                           novel=("novel", "mean"), unique=("unique", "mean")).to_dict("index")
    property_improvement = (means["C0"]["property_mae"] - means["A0"]["property_mae"]) / max(means["C0"]["property_mae"], 1e-12)
    checks = {"property_improvement_ge_10pct": property_improvement >= .10,
              "property_harm_rate_le_15pct": float(harm["property_harm"].mean()) <= .15,
              "coverage_ge_15pct": coverage >= .15,
              "a0_better_than_t0": means["A0"]["property_mae"] < means["T0"]["property_mae"],
              "stable_drop_le_5pp": means["A0"]["stable"] - means["C0"]["stable"] >= -.05,
              "nus_drop_le_5pp": means["A0"]["nus"] - means["C0"]["nus"] >= -.05,
              "validity_drop_le_5pp": means["A0"]["validity"] - means["C0"]["validity"] >= -.05,
              "mean_ehull_worsening_le_0.02": means["A0"]["e_hull"] - means["C0"]["e_hull"] <= .02}
    decision_value = "GO" if all(checks.values()) else "FAIL"
    adaptivity = "SUPPORTED" if checks["a0_better_than_t0"] else "NOT_SUPPORTED"
    decision = {"FULL_OUTCOME_DATASET": "COMPLETE", "OFFLINE_SELECTIVE_GATE": "GO",
                "OFFLINE_PREDICTABILITY": "GO", "RISK_CALIBRATED_CFG_P0": decision_value,
                "RISK_CALIBRATED_CFG_FORMAL256": "NOT_RUN", "SAMPLE_ADAPTIVITY": adaptivity,
                "INNOVATION1_FINAL_STATUS": "MIXED", "property_improvement_fraction": property_improvement,
                "policy_means": means, "coverage": coverage, "adaptive_count": adaptive_count,
                "property_harm_rate": float(harm["property_harm"].mean()), "gate_checks": checks,
                "next": "RUN_FRESH_FORMAL256" if decision_value == "GO" else "STOP",
                "rescue_tuning": False, "parameters_changed_after_p0": False}
    (P0 / "decision_summary.json").write_text(json.dumps(decision, indent=2, default=float) + "\n")
    (ROOT / "final_status.json").write_text(json.dumps(decision, indent=2, default=float) + "\n")
    lines = ["# Fresh P0 Report", "", f"`RISK_CALIBRATED_CFG_P0 = {decision_value}`", "",
             f"Property improvement A0 vs C0: {property_improvement:.2%}", f"Coverage: {coverage:.2%}",
             f"Property harm rate: {decision['property_harm_rate']:.2%}", f"Sample adaptivity: `{adaptivity}`", "", "## Gate checks", ""]
    lines += [f"- `{key}`: `{value}`" for key, value in checks.items()]
    lines += ["", "No rescue tuning is permitted after this decision."]
    (P0 / "final_report.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(paired["property_absolute_error_C0"], paired["property_absolute_error_A0"])
    limit = max(paired["property_absolute_error_C0"].max(), paired["property_absolute_error_A0"].max()); ax.plot([0, limit], [0, limit], "k--")
    ax.set(xlabel="C0 property error", ylabel="A0 property error"); fig.tight_layout(); fig.savefig(PLOTS / "p0_property_paired.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(["Property", "E-hull", "Stable"], [harm["property_harm"].mean(), harm["ehull_harm"].mean(), harm["stable_harm"].mean()]); ax.set_ylabel("Harm rate")
    fig.tight_layout(); fig.savefig(PLOTS / "p0_harm_rate.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(POLICIES, [means[p]["property_mae"] for p in POLICIES]); ax.set_ylabel("Property MAE")
    fig.tight_layout(); fig.savefig(PLOTS / "p0_c0_vs_t0_vs_a0.png", dpi=180); plt.close(fig)
    print(json.dumps(decision, indent=2, default=float))


if __name__ == "__main__": main()
