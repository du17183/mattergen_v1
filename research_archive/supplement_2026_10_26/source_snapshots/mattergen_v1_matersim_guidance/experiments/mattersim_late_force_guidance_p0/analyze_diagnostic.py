"""Freeze Stage A0 statistics and enforce the preregistered gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
TIMES = (0.10, 0.05, 0.02, 0.01)
PREDICTORS = ("maxF_t", "mean_force_t", "energy_per_atom_t")
OUTCOMES = ("final_maxF", "final_RMSD", "final_Ehull")


def _correlation(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    values = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3 or values["x"].nunique() < 2 or values["y"].nunique() < 2:
        return float("nan"), float("nan"), len(values)
    result = spearmanr(values["x"], values["y"]) if method == "spearman" else pearsonr(values["x"], values["y"])
    return float(result.statistic), float(result.pvalue), len(values)


def main() -> None:
    physics = pd.read_csv(ROOT / "physics_single_point.csv")
    quality = pd.read_csv(ROOT / "diagnostic_final_quality/per_structure.csv")
    if len(physics) != 160 or len(quality) != 32:
        raise RuntimeError(f"unexpected diagnostic sizes: physics={len(physics)} quality={len(quality)}")
    if "sample_seed" in quality and quality["sample_seed"].nunique() == 32:
        quality["seed"] = quality["sample_seed"].astype(int)
    else:
        quality["seed"] = np.arange(710000, 710032)
    final = physics.loc[np.isclose(physics["t_norm"], 0.0)].copy()
    final = final[["seed", "energy_t", "mean_force_t", "maxF_t"]].rename(columns={
        "energy_t": "final_energy", "mean_force_t": "final_mean_force",
        "maxF_t": "final_maxF",
    })
    quality = quality[["seed", "rmsd_from_relaxation", "energy_above_hull_per_atom"]].rename(columns={
        "rmsd_from_relaxation": "final_RMSD",
        "energy_above_hull_per_atom": "final_Ehull",
    })
    diagnostic = physics.merge(final, on="seed", validate="many_to_one").merge(
        quality, on="seed", validate="many_to_one"
    )
    required_order = [
        "seed", "subset", "t_norm", "energy_t", "mean_force_t", "maxF_t",
        "stress_norm_t", "min_distance_t", "volume_per_atom_t", "final_energy",
        "final_mean_force", "final_maxF", "final_RMSD", "final_Ehull",
        "actual_model_t", "state_kind", "num_atoms", "energy_per_atom_t",
        "finite_valid", "obvious_ood", "failure_reason", "evaluation_seconds",
    ]
    diagnostic[required_order].to_csv(ROOT / "physics_diagnostic.csv", index=False)

    correlations = []
    for subset in ("all", "calibration", "validation"):
        cohort = diagnostic if subset == "all" else diagnostic[diagnostic["subset"] == subset]
        for t_norm in TIMES:
            group = cohort[np.isclose(cohort["t_norm"], t_norm)]
            for predictor in PREDICTORS:
                for outcome in OUTCOMES:
                    for method in ("spearman", "pearson"):
                        statistic, pvalue, n = _correlation(group[predictor], group[outcome], method)
                        correlations.append({
                            "subset": subset, "t_norm": t_norm,
                            "predictor": predictor, "outcome": outcome,
                            "method": method, "statistic": statistic,
                            "pvalue": pvalue, "n": n,
                        })
    correlation_frame = pd.DataFrame(correlations)
    correlation_frame.to_csv(ROOT / "diagnostic_correlations.csv", index=False)

    stability = []
    for t_norm in (*TIMES, 0.0):
        group = diagnostic[np.isclose(diagnostic["t_norm"], t_norm)].copy()
        positive = group[(group["maxF_t"] > 0) & (group["final_maxF"] > 0)]
        rho, pvalue, n = _correlation(group["maxF_t"], group["final_maxF"], "spearman")
        stability.append({
            "t_norm": t_norm,
            "n": len(group),
            "finite_valid_fraction": float(group["finite_valid"].mean()),
            "obvious_ood_fraction": float(group["obvious_ood"].mean()),
            "median_maxF": float(group["maxF_t"].median()),
            "median_abs_maxF_delta_to_final": float((group["maxF_t"] - group["final_maxF"]).abs().median()),
            "median_abs_log10_maxF_ratio_to_final": float(
                np.abs(np.log10(positive["maxF_t"] / positive["final_maxF"])).median()
            ),
            "maxF_final_spearman": rho,
            "maxF_final_spearman_pvalue": pvalue,
            "correlation_n": n,
        })
    stability_frame = pd.DataFrame(stability)
    stability_frame.to_csv(ROOT / "diagnostic_stability.csv", index=False)

    primary = correlation_frame[
        (correlation_frame["subset"] == "all")
        & (correlation_frame["predictor"] == "maxF_t")
        & (correlation_frame["outcome"] == "final_maxF")
        & (correlation_frame["method"] == "spearman")
        & (correlation_frame["t_norm"] <= 0.05)
    ].sort_values("statistic", ascending=False)
    best = primary.iloc[0]
    valid_fraction = float(
        diagnostic[np.isclose(diagnostic["t_norm"], best["t_norm"])]["finite_valid"].mean()
    )
    auxiliary = correlation_frame[
        (correlation_frame["subset"] == "all")
        & (correlation_frame["predictor"] == "energy_per_atom_t")
        & (correlation_frame["outcome"] == "final_Ehull")
        & (correlation_frame["method"] == "spearman")
        & np.isclose(correlation_frame["t_norm"], best["t_norm"])
    ].iloc[0]
    if float(best["statistic"]) >= 0.50 and valid_fraction >= 0.90:
        verdict, reliable, next_step = "GO", True, "RUN_16_PAIRED_P0"
    elif float(best["statistic"]) < 0.30 or valid_fraction < 0.90:
        verdict, reliable, next_step = "FAIL", False, "STOP_A"
    else:
        verdict, reliable, next_step = "BORDERLINE", False, "STOP_A"
    summary = {
        "A_DIAGNOSTIC": verdict,
        "LATE_PHYSICS_SIGNAL_RELIABLE": reliable,
        "next": next_step,
        "best_late_t_norm": float(best["t_norm"]),
        "best_maxF_final_spearman": float(best["statistic"]),
        "best_maxF_final_spearman_pvalue": float(best["pvalue"]),
        "valid_fraction_at_best_t": valid_fraction,
        "energy_per_atom_final_Ehull_spearman_at_best_t": float(auxiliary["statistic"]),
        "auxiliary_energy_support": bool(float(auxiliary["statistic"]) >= 0.30),
        "diagnostic_n": 32,
        "surrogate_property_eval": True,
        "dft_verified": False,
    }
    (ROOT / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = [
        "# MatterSim late clean-estimate diagnostic",
        "",
        f"**A_DIAGNOSTIC={verdict}**; **LATE_PHYSICS_SIGNAL_RELIABLE={str(reliable)}**; **NEXT={next_step}**.",
        "",
        "The primary gate excludes the actual final structure (`t_norm=0.00`). That row is the state after the final predictor at the sampler epsilon `model_t=0.001`, not a model evaluation at mathematical t=0.",
        "",
        "## Primary evidence",
        "",
        f"Best non-final `t<=0.05` MaxF-to-final-MaxF Spearman: **{float(best['statistic']):.6f}** at t={float(best['t_norm']):.2f} (n={int(best['n'])}, p={float(best['pvalue']):.4g}).",
        f"Finite/valid fraction at that time: **{valid_fraction:.2%}**.",
        f"Energy/atom-to-final-E-hull Spearman at that time: **{float(auxiliary['statistic']):.6f}**.",
        "",
        "## Q1: late-signal stability",
        "",
        stability_frame.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Q2: predictive correlations",
        "",
        "All requested Spearman and Pearson combinations, including calibration/validation splits, are in `diagnostic_correlations.csv`. The decision uses the preregistered full 32-seed MaxF Spearman only.",
        "",
        "## Q3: finite/OOD audit",
        "",
        f"Across 160 evaluations, finite rate={diagnostic['finite_valid'].mean():.2%}; obvious-OOD rate={diagnostic['obvious_ood'].mean():.2%}. OOD was frozen as nonfinite, |energy/atom|>20 eV, MaxF>100 eV/A, minimum distance<0.5 A, or V/N outside [1,200] A^3.",
        "",
        "All energy, force, stress, relaxation, and E-hull quantities are MatterSim-5M surrogate results. `DFT_VERIFIED=False`.",
    ]
    (ROOT / "diagnostic_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
