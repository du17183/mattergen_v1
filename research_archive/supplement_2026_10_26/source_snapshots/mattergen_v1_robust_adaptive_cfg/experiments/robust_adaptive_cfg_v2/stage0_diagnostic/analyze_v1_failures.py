"""Diagnose frozen Adaptive CFG V1 without generating new samples.

This script intentionally consumes only historical cohorts.  It joins every
saved controller decision to the paired final outcome for its seed, freezes the
harm definitions requested by the protocol, and emits event-, bin-, and
sample-level audit tables plus the four Stage-0 figures.
"""

from __future__ import annotations

import json
import math
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "experiments" / "robust_adaptive_cfg_v2"
OUT = ROOT / "stage0_diagnostic"
PLOTS = ROOT / "plots"
MAIN = Path("/mnt/datasets-livsyn/dxl/mattergen_v1")
F0 = Path("/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance")

COHORTS = {
    "A5_64": {
        "trace_glob": F0
        / "experiments/adaptive_cfg_force_guidance_compatibility/generation/A0/*/adaptive_cfg_trace.csv",
        "metrics": F0
        / "experiments/adaptive_cfg_force_guidance_compatibility/per_structure_metrics.csv",
        "seed_min": 744000,
        "seed_max": 744063,
    },
    "minimal8": {
        "trace_glob": MAIN
        / "experiments/final_innovation1_audit/minimal_ablation/generation/A0/*/adaptive_cfg_trace.csv",
        "metrics": MAIN
        / "experiments/final_innovation1_audit/minimal_ablation/per_structure_metrics.csv",
        "seed_min": 746000,
        "seed_max": 746007,
    },
}

FORMAL = (
    MAIN
    / "experiments/final_thesis_results/sources"
    / "a367f85efe8027bf8ef2db41cc48ec2fd3d60582"
    / "thesis_archive/data/innovation1/source/formal_per_seed_records.csv"
)

FIELD_COLUMNS = {
    "atomic": "delta_atomic_numbers",
    "position": "delta_pos",
    "cell": "delta_cell",
}


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def finite(value: float | int | np.number) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def load_outcomes(path: Path, cohort: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw[raw["method"].isin(["C0", "A0"])].copy()
    raw["stable"] = bool_series(raw["stable"])
    raw["nus"] = bool_series(raw["nus"])
    raw["valid"] = bool_series(raw["valid"])
    fields = [
        "e_hull_ev_per_atom",
        "stable",
        "nus",
        "novel",
        "unique",
        "valid",
        "mag_absolute_error_a3",
    ]
    wide = raw.pivot(index="seed", columns="method", values=fields)
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["cohort"] = cohort
    wide["ehull_delta"] = wide["e_hull_ev_per_atom_A0"] - wide["e_hull_ev_per_atom_C0"]
    wide["stable_delta"] = wide["stable_A0"].astype(int) - wide["stable_C0"].astype(int)
    wide["nus_delta"] = wide["nus_A0"].astype(int) - wide["nus_C0"].astype(int)
    wide["property_delta"] = (
        wide["mag_absolute_error_a3_A0"] - wide["mag_absolute_error_a3_C0"]
    )
    wide["harm_ehull"] = wide["ehull_delta"] > 0.02
    wide["harm_stable"] = wide["stable_C0"].astype(bool) & ~wide["stable_A0"].astype(bool)
    wide["harm_property"] = wide["mag_absolute_error_a3_A0"] > (
        1.1 * wide["mag_absolute_error_a3_C0"]
    )
    wide["harm_any"] = wide[["harm_ehull", "harm_stable", "harm_property"]].any(axis=1)
    return wide


def load_trace(pattern: Path, cohort: str) -> pd.DataFrame:
    paths = [Path(value) for value in sorted(glob.glob(str(pattern)))]
    if not paths:
        raise FileNotFoundError(pattern)
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_path"] = str(path)
        frames.append(frame)
    trace = pd.concat(frames, ignore_index=True)
    trace["cohort"] = cohort
    trace["seed"] = trace["sample_seed"].astype(int)
    trace = trace.sort_values(["seed", "score_call_index"], kind="stable").reset_index(drop=True)
    trace["time_bin"] = np.minimum((trace["progress"] * 20).astype(int), 19)
    trace["stage"] = pd.cut(
        trace["progress"],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["early", "middle", "late"],
        include_lowest=True,
    ).astype(str)
    trace["cfg_delta_from_g0"] = trace["final_guidance"] - 2.0
    trace["cfg_abs_delta_from_g0"] = trace["cfg_delta_from_g0"].abs()
    trace["cfg_step_change"] = trace.groupby("seed")["final_guidance"].diff().fillna(0.0)
    trace["extreme_cfg"] = (trace["final_guidance"] < 1.0) | (trace["final_guidance"] > 3.0)
    values = trace[list(FIELD_COLUMNS.values())].rename(
        columns={column: field for field, column in FIELD_COLUMNS.items()}
    )
    trace["residual_mean"] = values.mean(axis=1)
    trace["residual_median"] = values.median(axis=1)
    trace["field_range"] = values.max(axis=1) - values.min(axis=1)
    trace["field_std"] = values.std(axis=1, ddof=0)
    for field, column in FIELD_COLUMNS.items():
        trace[f"log_residual_{field}"] = np.log(trace[column].clip(lower=0) + 1e-12)
    return trace


def distribution_summary(values: pd.Series) -> dict[str, float | None]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return {key: None for key in ("mean", "median", "p75", "p90", "p95", "max")}
    return {
        "mean": finite(clean.mean()),
        "median": finite(clean.median()),
        "p75": finite(clean.quantile(0.75)),
        "p90": finite(clean.quantile(0.90)),
        "p95": finite(clean.quantile(0.95)),
        "max": finite(clean.max()),
    }


def formal_harm_summary() -> dict[str, object]:
    frame = pd.read_csv(FORMAL)
    e_delta = frame["A0_minus_C0_ehull"]
    stable_harm = frame["stable_C0"].astype(bool) & ~frame["stable_A0"].astype(bool)
    nus_harm = frame["NUS_C0"].astype(bool) & ~frame["NUS_A0"].astype(bool)
    return {
        "n": int(len(frame)),
        "traces_available": False,
        "harm_rate_ehull": finite((e_delta > 0.02).mean()),
        "harm_rate_stable": finite(stable_harm.mean()),
        "harm_rate_nus": finite(nus_harm.mean()),
        "ehull_degradation": distribution_summary(e_delta),
        "note": "Formal256 endpoints are retained, but controller traces were not archived.",
    }


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traces = []
    outcomes = []
    for cohort, spec in COHORTS.items():
        trace = load_trace(spec["trace_glob"], cohort)
        outcome = load_outcomes(spec["metrics"], cohort)
        expected = set(range(spec["seed_min"], spec["seed_max"] + 1))
        assert set(trace["seed"].unique()) == expected
        assert set(outcome["seed"].unique()) == expected
        traces.append(trace)
        outcomes.append(outcome)
    trace = pd.concat(traces, ignore_index=True)
    outcome = pd.concat(outcomes, ignore_index=True)
    trace = trace.merge(
        outcome[
            [
                "cohort",
                "seed",
                "ehull_delta",
                "stable_delta",
                "nus_delta",
                "property_delta",
                "harm_ehull",
                "harm_stable",
                "harm_property",
                "harm_any",
            ]
        ],
        on=["cohort", "seed"],
        how="left",
        validate="many_to_one",
    )

    event_columns = [
        "cohort",
        "seed",
        "sampling_step",
        "score_call_index",
        "phase",
        "t",
        "progress",
        "time_bin",
        "stage",
        "delta_atomic_numbers",
        "delta_pos",
        "delta_cell",
        "residual_mean",
        "residual_median",
        "ema",
        "ratio",
        "final_guidance",
        "cfg_delta_from_g0",
        "cfg_step_change",
        "field_range",
        "field_std",
        "extreme_cfg",
        "harm_ehull",
        "harm_stable",
        "harm_property",
        "harm_any",
    ]
    trace[event_columns].to_csv(OUT / "cfg_trace_analysis.csv", index=False)

    field_rows = []
    for (cohort, phase, time_bin), group in trace.groupby(["cohort", "phase", "time_bin"]):
        for field, column in FIELD_COLUMNS.items():
            values = group[column].astype(float)
            logs = np.log(values.clip(lower=0) + 1e-12)
            field_rows.append(
                {
                    "cohort": cohort,
                    "phase": phase,
                    "time_bin": int(time_bin),
                    "field": field,
                    "sample_count": int(len(values)),
                    "residual_mean": finite(values.mean()),
                    "residual_std": finite(values.std(ddof=1)),
                    "residual_p50": finite(values.median()),
                    "residual_p95": finite(values.quantile(0.95)),
                    "log_residual_mean": finite(logs.mean()),
                    "log_residual_std": finite(logs.std(ddof=1)),
                    "field_range_mean": finite(group["field_range"].mean()),
                    "field_std_mean": finite(group["field_std"].mean()),
                }
            )
    field = pd.DataFrame(field_rows)
    field.to_csv(OUT / "field_residual_analysis.csv", index=False)

    seed_rows = []
    for (cohort, seed), group in trace.groupby(["cohort", "seed"], sort=True):
        source = outcome[(outcome["cohort"] == cohort) & (outcome["seed"] == seed)].iloc[0]
        row = source.to_dict()
        row.update(
            {
                "cfg_mean": finite(group["final_guidance"].mean()),
                "cfg_median": finite(group["final_guidance"].median()),
                "cfg_min": finite(group["final_guidance"].min()),
                "cfg_max": finite(group["final_guidance"].max()),
                "cfg_p05": finite(group["final_guidance"].quantile(0.05)),
                "cfg_p95": finite(group["final_guidance"].quantile(0.95)),
                "max_cfg_excursion": finite(group["cfg_abs_delta_from_g0"].max()),
                "mean_cfg_excursion": finite(group["cfg_abs_delta_from_g0"].mean()),
                "time_away_gt_0_1": finite((group["cfg_abs_delta_from_g0"] > 0.1).mean()),
                "extreme_cfg_rate": finite(group["extreme_cfg"].mean()),
                "max_step_change": finite(group["cfg_step_change"].abs().max()),
                "mean_field_disagreement": finite(group["field_std"].mean()),
                "p95_field_disagreement": finite(group["field_std"].quantile(0.95)),
            }
        )
        for stage, stage_group in group.groupby("stage"):
            row[f"{stage}_mean_abs_cfg_delta"] = finite(stage_group["cfg_abs_delta_from_g0"].mean())
            row[f"{stage}_extreme_cfg_rate"] = finite(stage_group["extreme_cfg"].mean())
        seed_rows.append(row)
    samples = pd.DataFrame(seed_rows)
    samples.to_csv(OUT / "bad_sample_analysis.csv", index=False)
    return trace, field, samples


def plot_diagnostics(trace: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    binned = trace.groupby("time_bin")["final_guidance"].agg(["mean", "median"])
    q05 = trace.groupby("time_bin")["final_guidance"].quantile(0.05)
    q95 = trace.groupby("time_bin")["final_guidance"].quantile(0.95)
    x = (binned.index.to_numpy() + 0.5) / 20
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(x, q05.to_numpy(), q95.to_numpy(), alpha=0.2, label="P5–P95")
    ax.plot(x, binned["mean"], label="mean")
    ax.plot(x, binned["median"], label="median")
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1, label="fixed anchor")
    ax.set(xlabel="reverse-diffusion progress", ylabel="CFG", title="Adaptive CFG V1 trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "v1_cfg_trajectory.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for harmful, label in [(False, "no recorded harm"), (True, "harm")]:
        part = trace[trace["harm_any"] == harmful]
        curve = part.groupby("time_bin")["cfg_abs_delta_from_g0"].mean()
        ax.plot((curve.index.to_numpy() + 0.5) / 20, curve, label=label)
    ax.set(
        xlabel="reverse-diffusion progress",
        ylabel="mean |CFG − 2|",
        title="V1 excursion for good vs harmful paired outcomes",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "good_vs_bad_cfg_trajectory.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for phase, ax in zip(["corrector", "predictor"], axes):
        part = trace[trace["phase"] == phase]
        for field, column in FIELD_COLUMNS.items():
            curve = part.groupby("time_bin")[column].median()
            ax.plot((curve.index.to_numpy() + 0.5) / 20, curve, label=field)
        ax.set_yscale("log")
        ax.set_title(phase)
        ax.set_xlabel("reverse-diffusion progress")
    axes[0].set_ylabel("median raw residual (log scale)")
    axes[1].legend()
    fig.suptitle("Field residuals are strongly timestep- and field-dependent")
    fig.tight_layout()
    fig.savefig(PLOTS / "field_residual_by_timestep.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = [
        trace.loc[~trace["harm_any"], "field_std"],
        trace.loc[trace["harm_any"], "field_std"],
    ]
    ax.boxplot(values, labels=["no recorded harm", "harm"], showfliers=False)
    ax.set(ylabel="cross-field raw residual std", title="Field disagreement by outcome")
    fig.tight_layout()
    fig.savefig(PLOTS / "field_disagreement.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trace, field, samples = build_tables()
    plot_diagnostics(trace)

    cohort_summary = {}
    for cohort, group in samples.groupby("cohort"):
        cohort_summary[cohort] = {
            "n": int(len(group)),
            "harm_rate_ehull": finite(group["harm_ehull"].mean()),
            "harm_rate_stable": finite(group["harm_stable"].mean()),
            "harm_rate_property": finite(group["harm_property"].mean()),
            "harm_rate_any": finite(group["harm_any"].mean()),
            "ehull_degradation": distribution_summary(group["ehull_delta"]),
            "property_degradation": distribution_summary(group["property_delta"]),
            "stable_degradation": distribution_summary(group["stable_delta"]),
            "cfg": {
                "minimum": finite(group["cfg_min"].min()),
                "maximum": finite(group["cfg_max"].max()),
                "mean_max_excursion": finite(group["max_cfg_excursion"].mean()),
                "mean_extreme_rate": finite(group["extreme_cfg_rate"].mean()),
            },
        }

    harmful = samples[samples["harm_any"]]
    safe = samples[~samples["harm_any"]]
    stage_columns = [f"{stage}_mean_abs_cfg_delta" for stage in ("early", "middle", "late")]
    stage_excess = {
        column.removesuffix("_mean_abs_cfg_delta"): finite(harmful[column].mean() - safe[column].mean())
        for column in stage_columns
    }
    stage_gate = "OFF"
    finite_excess = {key: value for key, value in stage_excess.items() if value is not None and value > 0}
    if finite_excess:
        total = sum(finite_excess.values())
        winner = max(finite_excess, key=finite_excess.get)
        # A relative concentration caused by numerically tiny differences is not
        # "clear" evidence for a gate.  Require both concentration and an
        # interpretable >=0.02 CFG absolute excess before adding complexity.
        if (
            total > 0
            and finite_excess[winner] / total >= 0.60
            and finite_excess[winner] >= 0.02
        ):
            stage_gate = f"ON_EXCLUDE_{winner.upper()}"

    scale_shift = []
    for phase in ["corrector", "predictor"]:
        for field_name in FIELD_COLUMNS:
            rows = field[(field["phase"] == phase) & (field["field"] == field_name)]
            pivot = rows.pivot(index="time_bin", columns="cohort", values="log_residual_mean")
            if set(COHORTS).issubset(pivot.columns):
                difference = (pivot["minimal8"] - pivot["A5_64"]).abs()
                scale_shift.append(
                    {
                        "phase": phase,
                        "field": field_name,
                        "mean_abs_log_shift": finite(difference.mean()),
                        "max_abs_log_shift": finite(difference.max()),
                    }
                )

    summary = {
        "status": "COMPLETE_NO_NEW_GENERATION",
        "event_count": int(len(trace)),
        "seed_count_with_trace": int(len(samples)),
        "cohorts": cohort_summary,
        "formal256_endpoint_only": formal_harm_summary(),
        "harm_definitions": {
            "Harm_Ehull": "Adaptive E-hull - Base E-hull > 0.02 eV/atom",
            "Harm_Stable": "Base Stable=1 and Adaptive Stable=0",
            "Harm_Property": "Adaptive property absolute error > 1.10 * Base error",
        },
        "questions": {
            "Q1_extreme_cfg": {
                "harm_mean_extreme_rate": finite(harmful["extreme_cfg_rate"].mean()),
                "safe_mean_extreme_rate": finite(safe["extreme_cfg_rate"].mean()),
                "harm_mean_max_excursion": finite(harmful["max_cfg_excursion"].mean()),
                "safe_mean_max_excursion": finite(safe["max_cfg_excursion"].mean()),
            },
            "Q2_field_disagreement": {
                "harm_mean": finite(harmful["mean_field_disagreement"].mean()),
                "safe_mean": finite(safe["mean_field_disagreement"].mean()),
            },
            "Q3_timestep_dependence": {
                "max_over_min_median_residual_ratio_by_field_phase": {
                    f"{phase}_{field_name}": finite(
                        rows["residual_p50"].max() / max(rows["residual_p50"].min(), 1e-12)
                    )
                    for phase in ["corrector", "predictor"]
                    for field_name in FIELD_COLUMNS
                    for rows in [field[(field["phase"] == phase) & (field["field"] == field_name)]]
                }
            },
            "Q4_cohort_scale_shift": scale_shift,
            "Q5_predictor_corrector": trace.groupby("phase")[
                ["residual_mean", "ema", "final_guidance"]
            ].mean().to_dict(orient="index"),
            "Q6_harm_stage": {
                "harm_minus_safe_mean_abs_cfg_delta": stage_excess,
                "stage_gate_decision": stage_gate,
            },
        },
        "stage0_frozen_design_choices": {
            "cfg_bounds": [1.5, 2.5],
            "slew_rate_cap": 0.05,
            "stage_gating": stage_gate,
            "reason": "Use protocol-default conservative bounds; enable a stage gate only if one stage has >=60% of positive excess and >=0.02 absolute excess mean |CFG-2|.",
        },
        "limitations": [
            "Formal256 controller trajectories were not archived; only its paired endpoints are included.",
            "Trace-linked diagnosis uses two previously observed cohorts (64 + 8) and is calibration/design evidence only.",
            "Association between controller behavior and final surrogate outcomes is not causal evidence.",
            "All quality/property values are surrogate evaluations; DFT_VERIFIED=False.",
        ],
    }
    dump_json(OUT / "diagnostic_summary.json", summary)

    report = f"""# Adaptive CFG V1 failure-pattern report

Status: `COMPLETE_NO_NEW_GENERATION`

The audit joins {len(trace):,} saved V1 decisions from {len(samples)} historical paired seeds to their final outcomes. Formal256 contributes all 256 paired endpoints but no controller traces because those traces were not archived. No sample was regenerated or excluded in Stage 0.

## Frozen harm definitions

- Harm_Ehull: Adaptive minus Base E-hull > 0.02 eV/atom.
- Harm_Stable: Base Stable=1 and Adaptive Stable=0.
- Harm_Property: Adaptive magnetic-property absolute error > Base error +10%.

## Cohort results

```json
{json.dumps(cohort_summary, indent=2, ensure_ascii=False)}
```

## Diagnostic answers

1. Extreme CFG: harmful seeds have mean extreme-event rate {summary['questions']['Q1_extreme_cfg']['harm_mean_extreme_rate']}; non-harm seeds {summary['questions']['Q1_extreme_cfg']['safe_mean_extreme_rate']}. The corresponding mean maximum excursions are {summary['questions']['Q1_extreme_cfg']['harm_mean_max_excursion']} and {summary['questions']['Q1_extreme_cfg']['safe_mean_max_excursion']}.
2. Field disagreement: harmful-event-weighted seed mean {summary['questions']['Q2_field_disagreement']['harm_mean']}; non-harm {summary['questions']['Q2_field_disagreement']['safe_mean']}.
3. Raw residual is timestep-dependent; the field/phase max:min ratios are recorded in `diagnostic_summary.json` and visualized in `field_residual_by_timestep.png`.
4. Cohort drift is present to the degree quantified by absolute shifts in time-binned log residual means in `diagnostic_summary.json`. This motivates frozen-CFG calibration rather than raw cross-field averaging.
5. Predictor and corrector distributions are reported separately; calibration will also be phase-specific.
6. Harm-associated excess CFG excursion by stage is `{stage_excess}`. Frozen stage-gating decision: `{stage_gate}`.

## V2 design consequence

The once-only V2 uses log-residual timestep calibration, a median across the three standardized fields, confidence fallback to CFG=2, CFG bounds [1.5, 2.5], and a per-score-call slew cap of 0.05. A stage gate is used only if the explicit 60% concentration diagnostic selected one; otherwise it remains off. No learned controller, field-specific CFG, property gradient, or parameter sweep is introduced.

## Limitations

This is retrospective failure diagnosis, not independent validation. The fresh P0 seeds must remain unseen until the V2 configuration is frozen. All stability and property outcomes are surrogate-based (`DFT_VERIFIED=False`).
"""
    (OUT / "failure_pattern_report.md").write_text(report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
