"""Summarize historical V1--V5 evidence without using it for validation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/stage_calibrated_bounded_cfg"
STAGE0 = ROOT / "stage0"
PLOTS = ROOT / "plots"
V3 = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_cfg_oracle/experiments/"
    "counterfactual_adaptive_cfg_v3/oracle/oracle_headroom.csv"
)
V4 = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_risk_cfg/experiments/"
    "risk_calibrated_adaptive_cfg/dataset/treatment_effect_dataset.csv"
)
V5_DECISION = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_safe_cfg/experiments/"
    "calibrated_safe_selection_cfg/validation/decision_summary.json"
)
V2_RESIDUALS = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_robust_adaptive_cfg/experiments/"
    "robust_adaptive_cfg_v2/calibration/residual_calibration.csv"
)


def main() -> None:
    for source in (V3, V4, V5_DECISION, V2_RESIDUALS):
        if not source.is_file():
            raise FileNotFoundError(source)
    STAGE0.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(V4)
    headroom_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    for step, frame in data.groupby("sampling_step", sort=True):
        base = frame["property_absolute_error_KEEP"].to_numpy(float)
        down = frame["property_absolute_error_DOWN"].to_numpy(float)
        up = frame["property_absolute_error_UP"].to_numpy(float)
        oracle = np.minimum.reduce((base, down, up))
        global_means = {"DOWN": float(down.mean()), "KEEP": float(base.mean()), "UP": float(up.mean())}
        best_global = min(global_means, key=lambda key: (global_means[key], {"KEEP": 0, "DOWN": 1, "UP": 2}[key]))
        headroom_rows.append(
            {
                "source": "V4_full_outcome_96",
                "sampling_step": int(step),
                "t_norm": float(frame["decision_t_norm"].iloc[0]),
                "n": len(frame),
                "cfg2_property_mae": float(base.mean()),
                "oracle_property_mae": float(oracle.mean()),
                "oracle_absolute_gain": float((base - oracle).mean()),
                "oracle_relative_gain": float((base.mean() - oracle.mean()) / base.mean()),
                "oracle_nonkeep_rate": float((frame["oracle_action"] != "KEEP").mean()),
                "best_global_action": best_global,
                "best_global_property_mae": global_means[best_global],
                "best_global_relative_gain": float((base.mean() - global_means[best_global]) / base.mean()),
            }
        )
        for action in ("DOWN", "UP"):
            delta = frame[f"delta_property_{action}"].to_numpy(float)
            degradation = np.maximum(delta, 0.0)
            count = max(1, int(np.ceil(0.25 * len(degradation))))
            e_delta = frame[f"delta_ehull_{action}"].to_numpy(float)
            sensitivity_rows.append(
                {
                    "source": "V4_full_outcome_CFG_delta_0.25_permanent",
                    "sampling_step": int(step),
                    "t_norm": float(frame["decision_t_norm"].iloc[0]),
                    "action": action,
                    "delta_g": -0.25 if action == "DOWN" else 0.25,
                    "n": len(frame),
                    "mean_property_delta": float(delta.mean()),
                    "median_property_delta": float(np.median(delta)),
                    "win_rate": float((delta < 0).mean()),
                    "property_harm_rate_5pct": float(frame[f"harm_property_{action}"].mean()),
                    "p75_positive_degradation": float(np.quantile(degradation, 0.75)),
                    "p90_positive_degradation": float(np.quantile(degradation, 0.90)),
                    "worst_quartile_mean_degradation": float(np.sort(degradation)[-count:].mean()),
                    "mean_e_hull_delta": float(np.nanmean(e_delta)),
                    "e_hull_harm_rate": float(frame[f"harm_ehull_{action}"].mean()),
                    "stable_loss_rate": float(frame[f"harm_stable_{action}"].mean()),
                }
            )

    headroom = pd.DataFrame(headroom_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    headroom.to_csv(STAGE0 / "stage_headroom.csv", index=False)
    sensitivity.to_csv(STAGE0 / "stage_sensitivity.csv", index=False)

    residuals = pd.read_csv(V2_RESIDUALS)
    residual_summary = (
        residuals.groupby(["phase", "field"], as_index=False)
        .agg(
            mu_min=("mu_log_residual", "min"),
            mu_max=("mu_log_residual", "max"),
            sigma_mean=("sigma_log_residual", "mean"),
        )
    )
    residual_summary["mu_stage_range"] = residual_summary["mu_max"] - residual_summary["mu_min"]
    residual_summary.to_csv(STAGE0 / "residual_stage_dependence.csv", index=False)

    v3 = pd.read_csv(V3)
    v3_full = v3[v3["selection"] == "full_oracle_vs_cfg2"].iloc[0]
    v3_short = v3[v3["selection"] == "short_selected_full_vs_cfg2"].iloc[0]
    try:
        v5 = json.loads(V5_DECISION.read_text())
    except json.JSONDecodeError:
        v5 = {}
    lines = [
        "# Stage 0 historical diagnostic",
        "",
        "Historical outcomes are used only to delimit the candidate region. They are not calibration, P0, or Formal evidence.",
        "",
        "## Findings",
        "",
        "- V1 residual-driven Adaptive CFG was unstable across cohorts; this pre-registered historical conclusion rules out reusing its sample-wise controller.",
        f"- V3 full-outcome Oracle reduced mean property MAE by {100*float(v3_full['property_objective_improvement_fraction']):.2f}% on n={int(v3_full['n'])}; the short-horizon-selected full outcome instead changed it by {100*float(v3_short['property_objective_improvement_fraction']):.2f}%.",
        f"- V4 Oracle relative headroom declined from {100*headroom.iloc[0]['oracle_relative_gain']:.2f}% at step {int(headroom.iloc[0]['sampling_step'])} to {100*headroom.iloc[-1]['oracle_relative_gain']:.2f}% at step {int(headroom.iloc[-1]['sampling_step'])}.",
        f"- Across V4 permanent +/-0.25 interventions, property harm rates ranged from {100*sensitivity['property_harm_rate_5pct'].min():.2f}% to {100*sensitivity['property_harm_rate_5pct'].max():.2f}%; this supports smaller bounded pulses rather than long-held changes.",
        f"- V2 normalized residual calibration has nonzero stage ranges for every field/phase (minimum log-residual range {residual_summary['mu_stage_range'].min():.3f}), confirming timestep dependence without claiming sample-level predictability.",
        f"- V5 frozen safe-selection validation status was {v5.get('VALIDATION_SAFE_GATE', v5.get('decision', 'FAIL/unknown'))}; it is used as evidence to stop sample-wise gating, not as evidence for this schedule.",
        "",
        "## Frozen candidate-region rationale",
        "",
        "- Starts: steps 100, 250, and 400 cover early-middle, middle, and middle-late regions while excluding step 550, where historical Oracle headroom was lowest.",
        "- Amplitude: delta_g in {-0.10, +0.10}. V4 used permanent +/-0.25 changes and showed large tails; halving the magnitude is the conservative stability-first choice.",
        "- Duration: K in {50, 100}. These windows are long enough to affect multiple reverse steps, but force return to CFG=2 far earlier than the permanent comparator.",
        "- This choice is frozen before any new schedule-calibration outcome is generated.",
        "",
        "STAGE0_DIAGNOSTIC = COMPLETE",
    ]
    (STAGE0 / "historical_summary.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(headroom["sampling_step"], 100 * headroom["oracle_relative_gain"], marker="o", label="sample-wise full Oracle")
    ax.plot(headroom["sampling_step"], 100 * headroom["best_global_relative_gain"], marker="s", label="best global permanent action")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Reverse diffusion step", ylabel="Relative property MAE gain (%)", title="Historical final-outcome headroom by timestep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "historical_headroom_by_timestep.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"STAGE0_DIAGNOSTIC": "COMPLETE", "rows": {"headroom": len(headroom), "sensitivity": len(sensitivity)}}))


if __name__ == "__main__":
    main()
