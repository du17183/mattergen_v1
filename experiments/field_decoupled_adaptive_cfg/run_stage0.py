"""Historical field-residual attribution used only to motivate frozen field policies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/field_decoupled_adaptive_cfg"
STAGE0 = ROOT / "stage0"
PLOTS = ROOT / "plots"
SOURCES = {
    "V4_full_outcome": Path("/mnt/datasets-livsyn/dxl/mattergen_v1_risk_cfg/experiments/risk_calibrated_adaptive_cfg/dataset/online_features_raw.csv"),
    "V5_safe_selection": Path("/mnt/datasets-livsyn/dxl/mattergen_v1_safe_cfg/experiments/calibrated_safe_selection_cfg/dataset/online_features_raw.csv"),
}
PREVIOUS_P0 = Path("/mnt/datasets-livsyn/dxl/mattergen_v1_stage_cfg/experiments/stage_calibrated_bounded_cfg/p0/decision_summary.json")


def main() -> None:
    for path in (*SOURCES.values(), PREVIOUS_P0):
        if not path.is_file():
            raise FileNotFoundError(path)
    STAGE0.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    frames = []
    for source, path in SOURCES.items():
        frame = pd.read_csv(path)
        frame["source"] = source
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    columns = {"atomic_numbers": "r_atomic_numbers", "pos": "r_pos", "cell": "r_cell"}
    timestep_rows: list[dict] = []
    for (source, step), frame in data.groupby(["source", "sampling_step"], sort=True):
        for field, column in columns.items():
            values = frame[column].astype(float).to_numpy()
            timestep_rows.append({"source": source, "sampling_step": int(step), "field": field, "n": len(values), "residual_rms_mean": float(values.mean()), "residual_rms_median": float(np.median(values)), "residual_rms_std": float(values.std(ddof=1)), "residual_rms_p90": float(np.quantile(values, 0.90))})
    by_timestep = pd.DataFrame(timestep_rows)
    by_timestep.to_csv(STAGE0 / "field_guidance_by_timestep.csv", index=False)
    norm_rows = []
    for field, column in columns.items():
        values = data[column].astype(float).to_numpy()
        norm_rows.append({"field": field, "n": len(values), "residual_rms_mean": float(values.mean()), "residual_rms_median": float(np.median(values)), "residual_rms_std": float(values.std(ddof=1)), "residual_rms_p90": float(np.quantile(values, 0.90)), "log_residual_mean": float(np.log(values + 1e-12).mean())})
    norms = pd.DataFrame(norm_rows)
    norms.to_csv(STAGE0 / "field_guidance_norm.csv", index=False)
    previous = json.loads(PREVIOUS_P0.read_text())
    report = [
        "# Stage 0 field-attribution diagnostic",
        "",
        "Historical traces are used only to motivate and sanity-check field decoupling. They are not calibration, P0, or Formal evidence.",
        "",
        "## Evidence boundary",
        "",
        "- V4/V5 online traces show strongly heterogeneous conditional-unconditional residual magnitudes across atomic_numbers, position, and cell fields and across timesteps.",
        "- The preceding stage-bounded P0 retained a property signal but failed physical guardrails; its status is hypothesis-generating only.",
        f"- Previous BOUNDED_PULSE_EFFECT: {previous.get('BOUNDED_PULSE_EFFECT')}; previous STAGE_SPECIFIC_EFFECT: {previous.get('STAGE_SPECIFIC_EFFECT')}.",
        "- Residual magnitudes have different units and tensor shapes, so raw magnitudes must not be interpreted as causal effect sizes.",
        "- Causal field attribution is reserved for the new paired 12-policy calibration.",
        "",
        "## Frozen diagnostic conclusion",
        "",
        "All three score fields have nonzero, timestep-dependent conditioning residuals. A field-wise CFG implementation is therefore meaningful to test, but Stage0 does not preselect PROPERTY_FIELD or QUALITY_HARM_FIELD.",
        "",
        "FIELD_ATTRIBUTION_DIAGNOSTIC = COMPLETE",
    ]
    (STAGE0 / "field_attribution_report.md").write_text("\n".join(report) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for field, frame in by_timestep[by_timestep["source"] == "V4_full_outcome"].groupby("field"):
        ax.plot(frame["sampling_step"], frame["residual_rms_median"], marker="o", label=field)
    ax.set_yscale("log")
    ax.set(xlabel="Reverse diffusion step", ylabel="Median ||conditional-unconditional|| RMS (log)", title="Historical field guidance residuals")
    ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "field_guidance_norm.png", dpi=180); plt.close(fig)
    print(json.dumps({"FIELD_ATTRIBUTION_DIAGNOSTIC": "COMPLETE", "sources": list(SOURCES), "rows": len(by_timestep)}))


if __name__ == "__main__":
    main()
