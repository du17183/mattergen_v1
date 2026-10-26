#!/usr/bin/env python3
"""Aggregate the preregistered adaptive-neighborhood motivation gate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/adaptive_neighborhood_p0"
FIELDS = ("atomic", "pos", "cell")
BANDS = ("near", "mid", "far")
TIMESTEPS = (0.80, 0.50, 0.25, 0.10, 0.05, 0.02)
STAGE = {
    0.80: "early",
    0.50: "early",
    0.25: "mid",
    0.10: "mid",
    0.05: "late",
    0.02: "late",
}


def relative_change(early: float, late: float) -> float:
    return abs(late - early) / max(abs(early), 1.0e-12)


def main() -> None:
    shard_paths = sorted((OUT / "diagnostic_shards").glob("band_sensitivity_*.csv"))
    if not shard_paths:
        raise RuntimeError("no diagnostic shards")
    per_seed = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)
    if per_seed.duplicated(["seed", "requested_t_norm", "block", "distance_band"]).any():
        raise RuntimeError("duplicate diagnostic rows")
    if sorted(per_seed.seed.unique()) != list(range(94000, 94032)):
        raise RuntimeError("diagnostic cohort is incomplete")
    if len(per_seed) != 32 * 6 * 4 * 3:
        raise RuntimeError("diagnostic row contract mismatch")
    per_seed.to_csv(OUT / "band_sensitivity_per_seed.csv", index=False)

    keys = [
        "block",
        "requested_t_norm",
        "actual_t",
        "sampling_step",
        "distance_band",
        "band_lower_angstrom",
        "band_upper_angstrom",
    ]
    aggregations = {
        f"{field}_sensitivity": ["mean", "std", "median"] for field in FIELDS
    }
    aggregated = per_seed.groupby(keys, as_index=False).agg(aggregations)
    aggregated.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in aggregated.columns
    ]
    aggregated["n"] = 32
    aggregated.to_csv(OUT / "band_sensitivity.csv", index=False)

    stage_data = per_seed.copy()
    stage_data["stage"] = stage_data["requested_t_norm"].map(STAGE)
    stage_rows = []
    for block in range(1, 5):
        for field in FIELDS:
            column = f"{field}_sensitivity"
            for band in BANDS:
                subset = stage_data[
                    (stage_data.block == block) & (stage_data.distance_band == band)
                ]
                means = subset.groupby("stage")[column].mean()
                early = float(means["early"])
                late = float(means["late"])
                stage_rows.append(
                    {
                        "block": block,
                        "field": field,
                        "distance_band": band,
                        "early_mean": early,
                        "mid_mean": float(means["mid"]),
                        "late_mean": late,
                        "early_to_late_relative_change": relative_change(early, late),
                        "direction": "increase" if late > early else "decrease",
                    }
                )
    stage_frame = pd.DataFrame(stage_rows)
    stage_frame.to_csv(OUT / "stage_sensitivity_summary.csv", index=False)

    active = np.maximum(stage_frame.early_mean, stage_frame.late_mean) >= 1.0e-6
    condition_1_rows = stage_frame[
        active & (stage_frame.early_to_late_relative_change >= 0.20)
    ]
    condition_1 = not condition_1_rows.empty

    ranking_changes = []
    for block in range(1, 5):
        for field in FIELDS:
            group = stage_frame[
                (stage_frame.block == block) & (stage_frame.field == field)
            ].set_index("distance_band")
            early = group.early_mean.sort_values(ascending=False)
            late = group.late_mean.sort_values(ascending=False)
            early_sep = (
                (early.iloc[0] - early.iloc[1]) / max(abs(early.iloc[0]), 1.0e-12)
            )
            late_sep = (
                (late.iloc[0] - late.iloc[1]) / max(abs(late.iloc[0]), 1.0e-12)
            )
            if (
                early.index[0] != late.index[0]
                and early_sep >= 0.05
                and late_sep >= 0.05
            ):
                ranking_changes.append(
                    {
                        "block": block,
                        "field": field,
                        "early_top_band": early.index[0],
                        "late_top_band": late.index[0],
                        "early_top_separation": early_sep,
                        "late_top_separation": late_sep,
                    }
                )
    condition_2 = bool(ranking_changes)

    monotonic_rows = []
    mean_per_t = per_seed.groupby(
        ["block", "requested_t_norm", "distance_band"], as_index=False
    )[[f"{field}_sensitivity" for field in FIELDS]].mean()
    for block in range(1, 5):
        far = mean_per_t[
            (mean_per_t.block == block) & (mean_per_t.distance_band == "far")
        ].set_index("requested_t_norm").loc[list(TIMESTEPS)]
        for field in FIELDS:
            rho, pvalue = spearmanr(
                np.asarray(TIMESTEPS), far[f"{field}_sensitivity"].to_numpy(float)
            )
            monotonic_rows.append(
                {"block": block, "field": field, "spearman_rho": rho, "pvalue": pvalue}
            )
    monotonic = pd.DataFrame(monotonic_rows)
    monotonic.to_csv(OUT / "far_monotonicity.csv", index=False)
    monotonic_fields = []
    for field in FIELDS:
        rows = monotonic[
            (monotonic.field == field) & (monotonic.spearman_rho.abs() >= 0.80)
        ]
        signs = np.sign(rows.spearman_rho.to_numpy(float))
        if max(int(np.sum(signs > 0)), int(np.sum(signs < 0))) >= 3:
            monotonic_fields.append(field)
    condition_3 = bool(monotonic_fields)

    stable_field_rows = []
    for field in FIELDS:
        field_rows = condition_1_rows[condition_1_rows.field == field]
        for band in BANDS:
            for direction in ("increase", "decrease"):
                rows = field_rows[
                    (field_rows.distance_band == band)
                    & (field_rows.direction == direction)
                ]
                if rows.block.nunique() >= 2:
                    stable_field_rows.append(
                        {"field": field, "band": band, "direction": direction}
                    )
    stable_fields = sorted({row["field"] for row in stable_field_rows})
    condition_4 = len(stable_fields) >= 2
    motivated = bool(condition_1 or condition_2 or condition_3 or condition_4)

    summaries = []
    for path in sorted((OUT / "diagnostic_shards").glob("run_summaries_*.json")):
        summaries.extend(json.loads(path.read_text()))
    if len(summaries) != 32:
        raise RuntimeError("run summary cohort mismatch")
    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "sample_count": 32,
        "trajectory_states": 32 * 6,
        "sensitivity_rows": len(per_seed),
        "generation_seconds_total": float(sum(row["generation_seconds"] for row in summaries)),
        "diagnostic_seconds_total": float(sum(row["diagnostic_seconds"] for row in summaries)),
        "peak_allocated_bytes_max": int(max(row["peak_allocated_bytes"] for row in summaries)),
        "conditions": {
            "early_late_change_ge_20pct": condition_1,
            "near_mid_far_top_rank_changes": condition_2,
            "stable_far_monotonic_trend": condition_3,
            "at_least_two_fields_stable_stage_difference": condition_4,
        },
        "condition_1_matching_rows": condition_1_rows.to_dict(orient="records"),
        "condition_2_ranking_changes": ranking_changes,
        "condition_3_monotonic_fields": monotonic_fields,
        "condition_4_stable_fields": stable_fields,
        "ADAPTIVE_NEIGHBORHOOD_MOTIVATED": motivated,
        "next_phase": "IDENTITY_IMPLEMENTATION_AND_GATE_TRAINING" if motivated else "STOP_A",
        "adaptive_cfg_used": False,
        "new_parameters_trained": False,
        "dft_verified": False,
    }
    (OUT / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    create_plots(mean_per_t, stage_frame)
    print(json.dumps(summary, sort_keys=True), flush=True)


def create_plots(mean_per_t: pd.DataFrame, stage_frame: pd.DataFrame) -> None:
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    colors = {"near": "tab:blue", "mid": "tab:orange", "far": "tab:green"}
    for axis, field in zip(axes, FIELDS, strict=True):
        for band in BANDS:
            values = mean_per_t[mean_per_t.distance_band == band].groupby(
                "requested_t_norm"
            )[f"{field}_sensitivity"].mean().loc[list(TIMESTEPS)]
            axis.plot(TIMESTEPS, values, marker="o", color=colors[band], label=band)
        axis.set_title(field)
        axis.set_xlabel("diffusion t")
        axis.invert_xaxis()
    axes[0].set_ylabel("normalized score sensitivity")
    axes[-1].legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "sensitivity_vs_timestep.png", dpi=180)
    plt.close(figure)

    stage_order = ["early", "mid", "late"]
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), sharey=True)
    for axis, field in zip(axes, FIELDS, strict=True):
        table = stage_frame[stage_frame.field == field].groupby("distance_band")[[
            "early_mean", "mid_mean", "late_mean"
        ]].mean().loc[list(BANDS)]
        image = axis.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
        axis.set_xticks(range(3), stage_order)
        axis.set_yticks(range(3), BANDS)
        axis.set_title(field)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(plot_dir / "radial_band_importance.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
