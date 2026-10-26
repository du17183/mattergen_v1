"""Build the once-frozen Robust Adaptive CFG V2 residual calibration.

All numerical rules in this file are preregistered before the replay finishes:
20 progress bins; phase/field-specific mean and sample standard deviation of
log(residual + 1e-12); z clipping at 3; V1's phase-specific EMA beta=0.95;
the 75th and 95th absolute-smoothed-signal quantiles define the dead zone and
scale; median baseline disagreement defines exponential agreement decay.
There is no endpoint-quality input and no parameter sweep.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "experiments/robust_adaptive_cfg_v2"
OUT = ROOT / "calibration"
IMPLEMENTATION = ROOT / "implementation"
TRACE = OUT / "baseline_cfg_trace.csv"
FIELDS = {
    "atomic": "delta_atomic_numbers",
    "position": "delta_pos",
    "cell": "delta_cell",
}

TIME_BINS = 20
LOG_EPS = 1e-12
SIGMA_FLOOR = 1e-6
Z_CLIP = 3.0
EMA_BETA = 0.95
DEADZONE_QUANTILE = 0.75
SCALE_QUANTILE = 0.95
CFG_BASE = 2.0
CFG_MIN = 1.5
CFG_MAX = 2.5
ADAPTATION_AMPLITUDE = 0.5
SLEW_CAP = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def phase_ema(frame: pd.DataFrame, column: str) -> pd.Series:
    result = pd.Series(index=frame.index, dtype=float)
    for (_, phase), indices in frame.groupby(["sample_seed", "phase"], sort=False).groups.items():
        previous = None
        for index in sorted(indices, key=lambda item: int(frame.at[item, "score_call_index"])):
            value = float(frame.at[index, column])
            previous = value if previous is None else EMA_BETA * previous + (1 - EMA_BETA) * value
            result.at[index] = previous
    return result


def main() -> None:
    completion = json.loads((OUT / "baseline_replay_completion.json").read_text())
    if completion["status"] != "COMPLETE" or not completion["all_archived_structures_matched"]:
        raise RuntimeError("baseline replay is incomplete or failed deterministic matching")
    trace = pd.read_csv(TRACE)
    if len(trace) != 128000 or set(trace["sample_seed"]) != set(range(744000, 744064)):
        raise RuntimeError("calibration trace cohort mismatch")
    trace["time_bin"] = np.minimum((trace["progress"] * TIME_BINS).astype(int), TIME_BINS - 1)

    rows = []
    lookup: dict[tuple[str, str, int], tuple[float, float]] = {}
    for (phase, time_bin), group in trace.groupby(["phase", "time_bin"], sort=True):
        for field, column in FIELDS.items():
            values = np.log(group[column].astype(float).clip(lower=0) + LOG_EPS)
            mu = float(values.mean())
            sigma_raw = float(values.std(ddof=1))
            sigma = max(sigma_raw, SIGMA_FLOOR)
            row = {
                "phase": phase,
                "field": field,
                "time_bin": int(time_bin),
                "progress_low": int(time_bin) / TIME_BINS,
                "progress_high": (int(time_bin) + 1) / TIME_BINS,
                "mu_log_residual": mu,
                "sigma_log_residual": sigma,
                "sigma_before_floor": sigma_raw,
                "sample_count": int(len(values)),
            }
            rows.append(row)
            lookup[(phase, field, int(time_bin))] = (mu, sigma)
    calibration = pd.DataFrame(rows)
    if len(calibration) != 2 * 3 * TIME_BINS or calibration["sample_count"].min() < 100:
        raise RuntimeError("unstable calibration bin count")
    calibration.to_csv(OUT / "residual_calibration.csv", index=False)

    for field, column in FIELDS.items():
        z_values = []
        for phase, time_bin, value in zip(trace["phase"], trace["time_bin"], trace[column]):
            mu, sigma = lookup[(str(phase), field, int(time_bin))]
            z_values.append(np.clip((math.log(max(float(value), 0.0) + LOG_EPS) - mu) / sigma, -Z_CLIP, Z_CLIP))
        trace[f"z_{field}"] = z_values
    z_columns = [f"z_{field}" for field in FIELDS]
    trace["z_mean"] = trace[z_columns].mean(axis=1)
    trace["z_median"] = trace[z_columns].median(axis=1)
    trace["z_field_std"] = trace[z_columns].std(axis=1, ddof=0)
    trace["z_field_range"] = trace[z_columns].max(axis=1) - trace[z_columns].min(axis=1)
    trace["z_ema"] = phase_ema(trace, "z_median")

    absolute_signal = trace["z_ema"].abs()
    signal_deadzone = float(absolute_signal.quantile(DEADZONE_QUANTILE))
    signal_p95 = float(absolute_signal.quantile(SCALE_QUANTILE))
    signal_scale = max(signal_p95 - signal_deadzone, 0.25)
    median_disagreement = float(trace["z_field_std"].median())
    agreement_k = math.log(2.0) / max(median_disagreement, 1e-6)
    signal_conf = np.clip((absolute_signal - signal_deadzone) / signal_scale, 0.0, 1.0)
    agreement_conf = np.exp(-agreement_k * trace["z_field_std"])
    trace["signal_confidence"] = signal_conf
    trace["agreement_confidence"] = agreement_conf
    trace["confidence"] = signal_conf * agreement_conf
    trace["offline_v2_target_cfg"] = np.clip(
        CFG_BASE + trace["confidence"] * ADAPTATION_AMPLITUDE * np.tanh(trace["z_ema"]),
        CFG_MIN,
        CFG_MAX,
    )

    diagnostic_columns = [
        "sample_seed",
        "sampling_step",
        "score_call_index",
        "phase",
        "progress",
        "time_bin",
        *[column for pair in zip(FIELDS.values(), z_columns) for column in pair],
        "z_mean",
        "z_median",
        "z_ema",
        "z_field_std",
        "z_field_range",
        "signal_confidence",
        "agreement_confidence",
        "confidence",
        "offline_v2_target_cfg",
    ]
    trace[diagnostic_columns].to_csv(OUT / "normalized_baseline_trace.csv", index=False)

    calibration_json = {
        "version": "robust_adaptive_cfg_v2_calibration_v1",
        "source": {
            "purpose": "historical fixed-CFG=2 deterministic replay; calibration only",
            "target_dft_mag_density": 0.2,
            "seeds": [744000, 744063],
            "seed_count": 64,
            "trace_rows": len(trace),
            "trace_sha256": sha256(TRACE),
            "all_archived_structures_matched": True,
        },
        "transform": {
            "name": "log_zscore_by_phase_field_time_bin",
            "time_bins": TIME_BINS,
            "log_eps": LOG_EPS,
            "sigma_floor": SIGMA_FLOOR,
            "z_clip": Z_CLIP,
            "mu_estimator": "arithmetic mean of log residual",
            "sigma_estimator": "sample standard deviation (ddof=1)",
        },
        "aggregation": "median(z_atomic,z_position,z_cell)",
        "online_statistics": {
            "phase_specific_ema": True,
            "ema_beta": EMA_BETA,
            "first_observation": "initialize EMA to current z_median",
            "control_signal": "phase-specific EMA of z_median",
        },
        "confidence": {
            "signal_deadzone": signal_deadzone,
            "signal_deadzone_rule": f"baseline q{DEADZONE_QUANTILE}(|z_ema|)",
            "signal_scale": signal_scale,
            "signal_scale_rule": f"max(q{SCALE_QUANTILE}(|z_ema|)-deadzone,0.25)",
            "signal_confidence": "clip((abs(z_ema)-deadzone)/signal_scale,0,1)",
            "median_field_disagreement": median_disagreement,
            "agreement_k": agreement_k,
            "agreement_k_rule": "ln(2)/baseline_median(std(z_fields))",
            "agreement_confidence": "exp(-agreement_k*std(z_fields))",
            "final_confidence": "signal_confidence*agreement_confidence",
        },
        "calibration_table_sha256": sha256(OUT / "residual_calibration.csv"),
        "endpoint_quality_used": False,
        "parameter_sweep_used": False,
    }
    (OUT / "residual_calibration.json").write_text(
        json.dumps(calibration_json, indent=2, ensure_ascii=False) + "\n"
    )

    frozen = {
        "method": "Robust Multi-field Residual-driven Online Adaptive CFG V2",
        "frozen": True,
        "freeze_boundary": "before any fresh P0 generation",
        "git_head": git_value("rev-parse", "HEAD"),
        "source_sha256": {
            "guidance_schedule.py": sha256(PROJECT / "mattergen/diffusion/sampling/guidance_schedule.py"),
            "robust_guidance.py": sha256(PROJECT / "mattergen/diffusion/sampling/robust_guidance.py"),
            "classifier_free_guidance.py": sha256(PROJECT / "mattergen/diffusion/sampling/classifier_free_guidance.py"),
            "residual_calibration.json": sha256(OUT / "residual_calibration.json"),
            "residual_calibration.csv": sha256(OUT / "residual_calibration.csv"),
            "p0_seeds.json": sha256(ROOT / "p0/seeds.json"),
            "formal256_seeds.json": sha256(ROOT / "formal256/seeds.json"),
        },
        "base_guidance": CFG_BASE,
        "residual_calibration_json": str((OUT / "residual_calibration.json").resolve()),
        "residual_calibration_csv": str((OUT / "residual_calibration.csv").resolve()),
        "residual_transform": calibration_json["transform"],
        "aggregation": calibration_json["aggregation"],
        "online_statistics": calibration_json["online_statistics"],
        "confidence": calibration_json["confidence"],
        "adaptation": {
            "raw_delta": f"{ADAPTATION_AMPLITUDE}*tanh(z_ema)",
            "target": "g0 + confidence*raw_delta",
            "amplitude": ADAPTATION_AMPLITUDE,
            "cfg_min": CFG_MIN,
            "cfg_max": CFG_MAX,
            "slew_rate_cap_per_score_call": SLEW_CAP,
            "fallback": "CFG=2.0 exactly when signal confidence is zero or residual is invalid",
        },
        "predictor_corrector_handling": "separate z_median EMA; shared frozen calibration rule with phase-specific bins",
        "stage_gating": "OFF",
        "p0": {
            "methods": ["C0", "A_OLD", "A_NORM", "A_ROBUST"],
            "paired_seed_count": 32,
            "target_dft_mag_density": 0.1,
            "bootstrap_resamples": 20000,
            "bootstrap_seed": 20260916,
            "primary_robustness_comparison": "A_ROBUST vs A_OLD",
            "go_criteria": {
                "harm_reduction": "at least one of Ehull/Stable/Property has >=30% relative HarmRate reduction vs A_OLD",
                "other_harm_guardrail": "no other HarmRate increases by >5 percentage points vs A_OLD",
                "average_quality": "vs C0 at least one of mean Ehull lower, Stable higher, or NUS higher",
                "mag_mae_worsening_max_fraction": 0.10,
                "validity_drop_max_percentage_points": 5.0,
                "nus_drop_max_percentage_points": 5.0,
                "severe_stable_collapse_max_percentage_points": 20.0,
            },
        },
        "formal256": {
            "allowed_only_if": "ROBUST_ADAPTIVE_CFG_P0=GO",
            "methods": ["C0", "A_OLD", "A_ROBUST"],
            "paired_seed_count": 256,
            "target_dft_mag_density": 0.1,
            "bootstrap_resamples": 20000,
            "bootstrap_seed": 20260917,
            "primary_comparison": "C0 vs A_ROBUST",
            "confirmation_criteria": {
                "headline": "at least one favorable E-hull/Stable/NUS paired bootstrap 95% CI excludes zero",
                "other_headlines": "no other E-hull/Stable/NUS paired bootstrap 95% CI significantly reverses",
                "harm_reduction": "at least one HarmRate has >=30% relative reduction vs fresh A_OLD and no other increases >5pp",
                "severe_stable_collapse_max_percentage_points": 20.0,
            },
        },
        "forbidden_after_freeze": [
            "CFG range",
            "confidence threshold/rule",
            "residual calibration rule",
            "aggregation method",
            "slew cap",
            "EMA",
            "seed",
            "primary metric",
            "GO threshold",
        ],
    }
    IMPLEMENTATION.mkdir(parents=True, exist_ok=True)
    config_path = IMPLEMENTATION / "robust_adaptive_cfg_v2_config.yaml"
    config_path.write_text(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True))
    frozen["config_sha256"] = sha256(config_path)
    (IMPLEMENTATION / "frozen_manifest.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n"
    )
    description = f"""# Robust Adaptive CFG V2 algorithm

The frozen training-free controller applies the following chain independently at every predictor/corrector score call:

1. Compute RMS conditional–unconditional residuals for atomic numbers, positions, and cell.
2. Transform each residual with phase/field/{TIME_BINS}-bin `log(r+{LOG_EPS})` calibration and clip z to ±{Z_CLIP}.
3. Aggregate the three standardized fields with their median.
4. Update a predictor- or corrector-specific EMA with beta={EMA_BETA}.
5. Compute signal confidence outside the baseline-derived dead zone and multiply it by exponential field-agreement confidence.
6. When signal confidence is zero, target the trusted fixed anchor CFG={CFG_BASE}; otherwise use `g0 + c*{ADAPTATION_AMPLITUDE}*tanh(z_ema)`.
7. Clip to [{CFG_MIN}, {CFG_MAX}] and apply a per-score-call slew cap of {SLEW_CAP}.

Stage gating is OFF. There is one global CFG shared by all fields. No learned weights, MLP, RL, gradient guidance, adapter, LoRA, or diffusion-model retraining is used. `A_NORM` is the normalization+median ablation without confidence, EMA control, or slew limiting; `A_OLD` is the original frozen V1.
"""
    (IMPLEMENTATION / "algorithm_description.md").write_text(description)

    bins = trace.groupby("time_bin")[["z_atomic", "z_position", "z_cell"]].median()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = (bins.index.to_numpy() + 0.5) / TIME_BINS
    for column, label in [("z_atomic", "atomic"), ("z_position", "position"), ("z_cell", "cell")]:
        ax.plot(x, bins[column], label=label)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="reverse-diffusion progress",
        ylabel="median normalized residual z",
        title="Timestep-normalized fixed-CFG residual",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "plots/normalized_residual_by_timestep.png", dpi=180)
    plt.close(fig)
    print(json.dumps(frozen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
