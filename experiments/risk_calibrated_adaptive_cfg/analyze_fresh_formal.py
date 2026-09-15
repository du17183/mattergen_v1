"""Evaluate Formal256 confirmation using only pre-registered paired criteria."""

from __future__ import annotations

import json
from pathlib import Path

import analyze_fresh_p0 as shared
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / "formal256"
PLOTS = ROOT / "plots"
RESAMPLES = 20_000
BOOTSTRAP_SEED = 20261002


def paired_bootstrap(values: np.ndarray, higher_is_better: bool, rng: np.random.Generator) -> dict:
    draws = rng.choice(values, size=(RESAMPLES, len(values)), replace=True).mean(axis=1)
    return {"n": len(values), "mean_delta": float(values.mean()),
            "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)),
            "higher_is_better": higher_is_better, "resamples": RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED}


def main() -> None:
    shared.P0 = FORMAL
    metrics = shared.enrich(); wide = metrics.pivot(index="seed", columns="policy")
    trace = pd.read_csv(FORMAL / "controller_trace.csv").set_index("seed")
    rows, harms = [], []
    for seed in wide.index:
        action = str(trace.loc[seed, "selected_action"]); row = {"seed": int(seed), "a0_action": action}
        for metric in ("property_absolute_error", "e_hull", "stable", "nus", "evaluation_valid"):
            for policy in shared.POLICIES: row[f"{metric}_{policy}"] = wide.loc[seed, (metric, policy)]
        for comparison, left, right in (("A0_minus_C0", "A0", "C0"), ("T0_minus_C0", "T0", "C0"), ("A0_minus_T0", "A0", "T0")):
            row[f"delta_property_{comparison}"] = float(row[f"property_absolute_error_{left}"]) - float(row[f"property_absolute_error_{right}"])
            row[f"delta_ehull_{comparison}"] = float(row[f"e_hull_{left}"]) - float(row[f"e_hull_{right}"])
            for metric in ("stable", "nus", "evaluation_valid"):
                row[f"delta_{metric}_{comparison}"] = float(row[f"{metric}_{left}"]) - float(row[f"{metric}_{right}"])
        rows.append(row)
        harms.append({"seed": int(seed), "adapted": action != "KEEP",
                      "property_harm": row["delta_property_A0_minus_C0"] > .05 * max(float(row["property_absolute_error_C0"]), 1e-12),
                      "ehull_harm": np.isfinite(row["delta_ehull_A0_minus_C0"]) and row["delta_ehull_A0_minus_C0"] > .02,
                      "stable_harm": bool(row["stable_C0"] and not row["stable_A0"])})
    paired, harm = pd.DataFrame(rows), pd.DataFrame(harms)
    paired.to_csv(FORMAL / "paired_results.csv", index=False); harm.to_csv(FORMAL / "harm_metrics.csv", index=False)
    adapted = harm["adapted"]; coverage = float(adapted.mean())
    selective_harm = float(harm.loc[adapted, "property_harm"].mean()) if adapted.any() else 1.0
    pd.DataFrame([{"n": len(harm), "adaptive_count": int(adapted.sum()), "coverage": coverage,
                   "property_harm_rate_all": float(harm["property_harm"].mean()),
                   "selective_property_harm_rate": selective_harm,
                   "selective_ehull_harm_rate": float(harm.loc[adapted, "ehull_harm"].mean()) if adapted.any() else 1.0,
                   "selective_stable_harm_rate": float(harm.loc[adapted, "stable_harm"].mean()) if adapted.any() else 1.0}]).to_csv(FORMAL / "coverage_metrics.csv", index=False)
    rng = np.random.default_rng(BOOTSTRAP_SEED); results = {}
    for comparison in ("A0_minus_C0", "A0_minus_T0"):
        results[f"property_{comparison}"] = paired_bootstrap(paired[f"delta_property_{comparison}"].to_numpy(float), False, rng)
        results[f"ehull_{comparison}"] = paired_bootstrap(paired[f"delta_ehull_{comparison}"].dropna().to_numpy(float), False, rng)
    for metric in ("stable", "nus", "evaluation_valid"):
        results[f"{metric}_A0_minus_C0"] = paired_bootstrap(paired[f"delta_{metric}_A0_minus_C0"].to_numpy(float), True, rng)
    (FORMAL / "bootstrap_results.json").write_text(json.dumps(results, indent=2) + "\n")
    means = metrics.groupby("policy").agg(property_mae=("property_absolute_error", "mean"), e_hull=("e_hull", "mean"),
                                           stable=("stable", "mean"), nus=("nus", "mean"), validity=("evaluation_valid", "mean"),
                                           novel=("novel", "mean"), unique=("unique", "mean")).to_dict("index")
    checks = {"primary_property_ci_favorable": results["property_A0_minus_C0"]["ci95_high"] < 0,
              "a0_better_than_t0": means["A0"]["property_mae"] < means["T0"]["property_mae"],
              "selective_property_harm_le_10pct": selective_harm <= .10,
              "coverage_ge_15pct": coverage >= .15,
              "ehull_no_clear_adverse_ci": results["ehull_A0_minus_C0"]["ci95_low"] <= 0,
              "stable_no_clear_adverse_ci": results["stable_A0_minus_C0"]["ci95_high"] >= 0,
              "nus_no_clear_adverse_ci": results["nus_A0_minus_C0"]["ci95_high"] >= 0,
              "validity_no_clear_adverse_ci": results["evaluation_valid_A0_minus_C0"]["ci95_high"] >= 0}
    confirmed = all(checks.values()); adaptivity = "SUPPORTED" if checks["a0_better_than_t0"] else "NOT_SUPPORTED"
    status = {"FULL_OUTCOME_DATASET": "COMPLETE", "OFFLINE_SELECTIVE_GATE": "GO", "OFFLINE_PREDICTABILITY": "GO",
              "RISK_CALIBRATED_CFG_P0": "GO", "RISK_CALIBRATED_CFG_FORMAL256": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
              "SAMPLE_ADAPTIVITY": adaptivity, "INNOVATION1_FINAL_STATUS": "CONFIRMED" if confirmed and adaptivity == "SUPPORTED" else "MIXED",
              "coverage": coverage, "selective_property_harm_rate": selective_harm, "policy_means": means,
              "gate_checks": checks, "parameters_changed_after_formal": False, "rescue_tuning": False, "next": "COMPLETE"}
    (FORMAL / "decision_summary.json").write_text(json.dumps(status, indent=2, default=float) + "\n")
    (ROOT / "final_status.json").write_text(json.dumps(status, indent=2, default=float) + "\n")
    lines = ["# Formal256 Report", "", f"`RISK_CALIBRATED_CFG_FORMAL256 = {status['RISK_CALIBRATED_CFG_FORMAL256']}`",
             f"`SAMPLE_ADAPTIVITY = {adaptivity}`", f"`INNOVATION1_FINAL_STATUS = {status['INNOVATION1_FINAL_STATUS']}`", "",
             f"Coverage: {coverage:.2%}; selective property harm: {selective_harm:.2%}.", "", "## Gate checks", ""]
    lines += [f"- `{key}`: `{value}`" for key, value in checks.items()]
    (FORMAL / "final_report.md").write_text("\n".join(lines) + "\n")
    (ROOT / "final_report.md").write_text("\n".join(lines) + "\n")
    delta = paired["delta_property_A0_minus_C0"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(delta, bins=30); ax.axvline(0, color="black", linestyle="--"); ax.set(xlabel="A0 - C0 property error", ylabel="seeds")
    fig.tight_layout(); fig.savefig(PLOTS / "formal256_property_paired.png", dpi=180); plt.close(fig)
    primary = results["property_A0_minus_C0"]; fig, ax = plt.subplots(figsize=(6, 3)); ax.errorbar([primary["mean_delta"]], [0], xerr=[[primary["mean_delta"]-primary["ci95_low"]], [primary["ci95_high"]-primary["mean_delta"]]], fmt="o"); ax.axvline(0, color="black", linestyle="--"); ax.set_yticks([]); ax.set_xlabel("Paired property delta, 95% bootstrap CI")
    fig.tight_layout(); fig.savefig(PLOTS / "formal256_bootstrap.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); ax.bar(["coverage", "selective harm"], [coverage, selective_harm]); ax.axhline(.15, color="black", linestyle="--")
    fig.tight_layout(); fig.savefig(PLOTS / "formal256_risk_coverage.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(shared.POLICIES, [means[p]["property_mae"] for p in shared.POLICIES]); ax.set_ylabel("Property MAE")
    fig.tight_layout(); fig.savefig(PLOTS / "formal256_c0_vs_t0_vs_a0.png", dpi=180); plt.close(fig)
    print(json.dumps(status, indent=2, default=float))


if __name__ == "__main__": main()
