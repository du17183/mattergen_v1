"""Freeze calibration thresholds and evaluate field-convergence heterogeneity."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(99000, 99064))
CALIBRATION_SEEDS = tuple(range(99000, 99032))
VALIDATION_SEEDS = tuple(range(99032, 99064))
K = 3
MINIMUM_FREEZE_T = 0.50

FIELD_RULES = {
    "A": {
        "upper": (
            "atomic_entropy",
            "atomic_argmax_change_rate",
        ),
        "lower": ("atomic_mean_max_probability",),
        "exact": {"atomic_mask_fraction": 0.0},
    },
    "X": {
        "upper": (
            "position_rms_cart_displacement",
            "position_max_cart_displacement",
        ),
        "lower": (),
        "exact": {},
    },
    "L": {
        "upper": (
            "cell_metric_ltl_relative_change",
            "cell_volume_relative_change",
            "cell_length_rms_relative_change",
            "cell_length_max_relative_change",
            "cell_angle_rms_change_deg",
            "cell_angle_max_change_deg",
        ),
        "lower": (),
        "exact": {},
    },
}


def _finite_pearson(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(tuple(left), dtype=float)
    y = np.asarray(tuple(right), dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def _read_traces(*, require_all: bool = True) -> pd.DataFrame:
    frames = []
    failures = []
    for seed in SEEDS:
        trace_path = ROOT / "trajectories" / str(seed) / "field_dynamics.csv"
        summary_path = trace_path.with_name("run_summary.json")
        if not trace_path.is_file() or not summary_path.is_file():
            failures.append(seed)
            continue
        trace = pd.read_csv(trace_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if len(trace) != 55 or summary.get("success") is not True:
            raise RuntimeError(f"incomplete diagnostic trajectory for seed {seed}")
        if int(summary["mattergen_score_calls"]) != 2000:
            raise RuntimeError(f"score-call contract changed for seed {seed}")
        trace["formula"] = summary["formula"]
        trace["runtime_seconds"] = float(summary["elapsed_seconds"])
        trace["peak_cuda_memory_mib"] = float(summary["peak_allocated_bytes"]) / 2**20
        frames.append(trace)
    if failures and require_all:
        raise RuntimeError(f"missing diagnostic seeds: {failures}")
    frame = pd.concat(frames, ignore_index=True)
    if require_all and len(frame) != len(SEEDS) * 55:
        raise RuntimeError("unexpected aggregate trace length")
    return frame


def _calibrate(frame: pd.DataFrame) -> dict:
    tail = frame[
        frame["seed"].isin(CALIBRATION_SEEDS) & (frame["model_t"] <= 0.0500001)
    ]
    thresholds: dict[str, dict[str, float | str]] = {}
    for field, rule in FIELD_RULES.items():
        thresholds[field] = {}
        for name in rule["upper"]:
            values = tail[name].to_numpy(float)
            values = values[np.isfinite(values)]
            thresholds[field][name] = float(np.quantile(values, 0.90))
        for name in rule["lower"]:
            values = tail[name].to_numpy(float)
            values = values[np.isfinite(values)]
            thresholds[field][name] = float(np.quantile(values, 0.10))
        for name, value in rule["exact"].items():
            thresholds[field][name] = float(value)
    return {
        "frozen": True,
        "calibration_subset": {
            "seed_start": CALIBRATION_SEEDS[0],
            "seed_end": CALIBRATION_SEEDS[-1],
            "trajectory_count": len(CALIBRATION_SEEDS),
            "stable_tail_model_t_max": 0.05,
        },
        "quantiles": {"upper_residual": 0.90, "lower_confidence": 0.10},
        "consecutive_observations": K,
        "minimum_active_horizon": {"freeze_forbidden_while_model_t_gt": MINIMUM_FREEZE_T},
        "atomic_safety_guard": "atomic_mask_fraction must equal 0",
        "cell_primary_metric": "L^T L (requested); L L^T retained as row-lattice audit only",
        "thresholds": thresholds,
    }


def _stable_mask(group: pd.DataFrame, field: str, frozen: dict) -> np.ndarray:
    thresholds = frozen["thresholds"][field]
    rule = FIELD_RULES[field]
    stable = group["finite_clean_estimates"].astype(bool).to_numpy()
    stable &= group["model_t"].to_numpy(float) <= MINIMUM_FREEZE_T + 1e-8
    for name in rule["upper"]:
        values = group[name].to_numpy(float)
        stable &= np.isfinite(values) & (values <= float(thresholds[name]))
    for name in rule["lower"]:
        values = group[name].to_numpy(float)
        stable &= np.isfinite(values) & (values >= float(thresholds[name]))
    for name, value in rule["exact"].items():
        values = group[name].to_numpy(float)
        stable &= np.isfinite(values) & np.isclose(values, float(value), atol=1e-12)
    return stable


def _first_k_stable_time(group: pd.DataFrame, stable: np.ndarray) -> tuple[float, int]:
    run = 0
    for index, is_stable in enumerate(stable):
        run = run + 1 if is_stable else 0
        if run >= K:
            onset = index - K + 1
            return float(group.iloc[onset]["model_t"]), int(group.iloc[onset]["sampling_step"])
    return float("nan"), -1


def _load_quality() -> pd.DataFrame | None:
    candidates = (
        ROOT / "diagnostic_quality" / "per_structure.csv",
        ROOT / "quality" / "per_structure.csv",
    )
    for path in candidates:
        if path.is_file():
            frame = pd.read_csv(path)
            if "sample_seed" not in frame:
                raise RuntimeError(f"quality file lacks sample_seed: {path}")
            return frame.rename(
                columns={
                    "pre_relaxation_max_force": "final_max_force_ev_ang",
                    "energy_above_hull_per_atom": "final_ehull_ev_atom",
                }
            )
    return None


def _trajectory_table(frame: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    rows = []
    for seed, group in frame.groupby("seed", sort=True):
        group = group.sort_values("sampling_step").reset_index(drop=True)
        row = {
            "seed": int(seed),
            "subset": str(group.iloc[0]["subset"]),
            "formula": str(group.iloc[0]["formula"]),
            "num_atoms": int(group.iloc[0]["num_atoms"]),
            "runtime_seconds": float(group.iloc[0]["runtime_seconds"]),
            "score_calls": 2000,
            "peak_cuda_memory_mib": float(group.iloc[0]["peak_cuda_memory_mib"]),
        }
        stable_masks = {}
        for field in FIELD_RULES:
            stable = _stable_mask(group, field, frozen)
            stable_masks[field] = stable
            convergence_t, convergence_step = _first_k_stable_time(group, stable)
            row[f"T_{field}"] = convergence_t
            row[f"convergence_step_{field}"] = convergence_step
            row[f"converged_{field}"] = bool(np.isfinite(convergence_t))
            row[f"stable_observation_fraction_{field}"] = float(stable.mean())
        finite_times = {field: row[f"T_{field}"] for field in FIELD_RULES
                        if np.isfinite(row[f"T_{field}"])}
        if len(finite_times) == 3:
            maximum = max(finite_times.values())
            first = [name for name, value in finite_times.items()
                     if math.isclose(value, maximum, abs_tol=1e-8)]
            row["first_converged_field"] = "=".join(first)
            row["convergence_order"] = ">".join(
                sorted(finite_times, key=lambda name: (-finite_times[name], name))
            )
            pair_gaps = [abs(finite_times[a] - finite_times[b])
                         for a, b in (("A", "X"), ("A", "L"), ("X", "L"))]
            row["has_pair_gap_ge_0p10"] = bool(max(pair_gaps) >= 0.10 - 1e-8)
            row["maximum_pair_gap"] = float(max(pair_gaps))
        else:
            row["first_converged_field"] = "NC"
            row["convergence_order"] = "NC"
            row["has_pair_gap_ge_0p10"] = False
            row["maximum_pair_gap"] = float("nan")
        # Endpoint-censored sensitivity: a field that never completes K stable
        # observations is assigned T=0.  This is deliberately more permissive
        # for the heterogeneity gate than dropping incomplete field pairs.
        endpoint_times = np.asarray([
            row[f"T_{field}"] if np.isfinite(row[f"T_{field}"]) else 0.0
            for field in FIELD_RULES
        ])
        endpoint_gap = float(endpoint_times.max() - endpoint_times.min())
        row["maximum_pair_gap_endpoint_censored"] = endpoint_gap
        row["has_pair_gap_ge_0p10_endpoint_censored"] = bool(
            endpoint_gap >= 0.10 - 1e-8
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    quality = _load_quality()
    if quality is not None:
        keep = [name for name in (
            "sample_seed", "final_max_force_ev_ang", "final_ehull_ev_atom",
            "rmsd_from_relaxation", "relaxation_steps", "final_energy_ev",
        ) if name in quality]
        result = result.merge(
            quality[keep].rename(columns={"sample_seed": "seed"}), on="seed", how="left",
            validate="one_to_one",
        )
    return result


def _summary_table(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset in ("calibration", "validation", "all"):
        selected = trajectories if subset == "all" else trajectories[trajectories["subset"] == subset]
        for field in FIELD_RULES:
            values = selected[f"T_{field}"].to_numpy(float)
            finite = values[np.isfinite(values)]
            row = {
                "subset": subset,
                "field": field,
                "n": len(selected),
                "n_converged": len(finite),
                "convergence_rate": float(len(finite) / len(selected)),
                "mean_T": float(np.mean(finite)) if len(finite) else np.nan,
                "median_T": float(np.median(finite)) if len(finite) else np.nan,
                "p25_T": float(np.quantile(finite, 0.25)) if len(finite) else np.nan,
                "p75_T": float(np.quantile(finite, 0.75)) if len(finite) else np.nan,
                "p90_T": float(np.quantile(finite, 0.90)) if len(finite) else np.nan,
                "pearson_T_vs_num_atoms": _finite_pearson(values, selected["num_atoms"]),
            }
            for source, destination in (
                ("final_max_force_ev_ang", "pearson_T_vs_final_max_force"),
                ("final_ehull_ev_atom", "pearson_T_vs_final_ehull"),
            ):
                row[destination] = (
                    _finite_pearson(values, selected[source]) if source in selected else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_gate(trajectories: pd.DataFrame) -> dict:
    validation = trajectories[trajectories["subset"] == "validation"].copy()
    medians = {field: float(np.nanmedian(validation[f"T_{field}"].to_numpy(float)))
               for field in FIELD_RULES}
    pair_differences = {
        f"{a}-{b}": abs(medians[a] - medians[b])
        for a, b in (("A", "X"), ("A", "L"), ("X", "L"))
    }
    finite_pair_differences = [value for value in pair_differences.values() if np.isfinite(value)]
    max_median_difference = (
        max(finite_pair_differences) if finite_pair_differences else float("nan")
    )
    strict_fraction = float(validation["has_pair_gap_ge_0p10"].mean())
    trajectory_fraction = float(
        validation["has_pair_gap_ge_0p10_endpoint_censored"].mean()
    )
    passed_by_median = bool(
        np.isfinite(max_median_difference) and max_median_difference >= 0.10 - 1e-8
    )
    passed_by_trajectory = bool(trajectory_fraction >= 0.60 - 1e-8)
    return {
        "FIELD_CONVERGENCE_HETEROGENEITY": passed_by_median or passed_by_trajectory,
        "evaluation_subset": "validation seeds 99032-99063 only",
        "validation_n": len(validation),
        "median_T": medians,
        "median_pair_absolute_differences": pair_differences,
        "maximum_median_pair_difference": max_median_difference,
        "median_criterion_threshold": 0.10,
        "passed_by_median_criterion": passed_by_median,
        "trajectory_pair_gap_fraction": trajectory_fraction,
        "trajectory_pair_gap_n": int(
            validation["has_pair_gap_ge_0p10_endpoint_censored"].sum()
        ),
        "trajectory_pair_gap_strict_all_finite_fraction": strict_fraction,
        "trajectory_pair_gap_strict_all_finite_n": int(
            validation["has_pair_gap_ge_0p10"].sum()
        ),
        "trajectory_pair_gap_threshold": 0.10,
        "trajectory_fraction_threshold": 0.60,
        "passed_by_trajectory_criterion": passed_by_trajectory,
        "nonconverged_policy": (
            "primary trajectory criterion assigns endpoint-censored nonconvergence T=0; "
            "strict all-three-finite sensitivity is also reported"
        ),
    }


def _make_plots(frame: pd.DataFrame, trajectories: pd.DataFrame) -> None:
    plot_frame = frame.copy()
    residuals = {
        "Atomic entropy": "atomic_entropy",
        "Position RMS (A)": "position_rms_cart_displacement",
        "Cell metric residual": "cell_metric_ltl_relative_change",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for axis, (label, column) in zip(axes, residuals.items()):
        grouped = plot_frame.groupby("model_t")[column]
        median = grouped.median().sort_index()
        q25 = grouped.quantile(0.25).sort_index()
        q75 = grouped.quantile(0.75).sort_index()
        axis.plot(median.index, median.values, lw=2)
        axis.fill_between(median.index, q25.values, q75.values, alpha=0.25)
        axis.set_xlabel("normalized model timestep t")
        axis.set_ylabel(label)
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
    fig.suptitle("Field clean-estimate dynamics (median and IQR, n=64)")
    fig.tight_layout()
    fig.savefig(ROOT / "field_residual_vs_timestep.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    bins = np.linspace(0, 0.51, 27)
    for field, color in zip(FIELD_RULES, ("#4477AA", "#EE6677", "#228833")):
        axis.hist(trajectories[f"T_{field}"].dropna(), bins=bins, alpha=0.48,
                  label=field, color=color)
    axis.set_xlabel("convergence timestep T (larger = earlier)")
    axis.set_ylabel("trajectory count")
    axis.legend(title="field")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "field_convergence_time_distribution.png", dpi=180)
    plt.close(fig)

    counts = Counter(trajectories["convergence_order"])
    labels, values = zip(*sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    fig, axis = plt.subplots(figsize=(max(7.2, len(labels) * 1.0), 4.8))
    axis.bar(labels, values, color="#4477AA")
    axis.set_ylabel("trajectory count")
    axis.set_xlabel("field order (larger T = earlier convergence)")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "field_convergence_order.png", dpi=180)
    plt.close(fig)


def _write_report(frame: pd.DataFrame, trajectories: pd.DataFrame, summary: pd.DataFrame,
                  frozen: dict, gate: dict) -> None:
    validation = trajectories[trajectories["subset"] == "validation"]
    first_counts = {
        str(key): int(value)
        for key, value in validation["first_converged_field"].value_counts().items()
    }
    first_percentages = {
        key: float(value / len(validation)) for key, value in first_counts.items()
    }
    order_counts = {
        str(key): int(value)
        for key, value in validation["convergence_order"].value_counts().items()
    }
    (ROOT / "convergence_order_summary.json").write_text(
        json.dumps({
            "subset": "validation",
            "n": len(validation),
            "exclusive_first_field_counts": first_counts,
            "exclusive_first_field_fractions": first_percentages,
            "ordering_counts": order_counts,
            "tie_policy": "ties retained as a separate '=' category; NC means not all three T are finite",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    quality_state = "已合并" if "final_max_force_ev_ang" in trajectories else "尚未生成，相关系数暂为 NaN"
    quality_path = ROOT / "diagnostic_quality" / "quality_summary.json"
    quality_text = "尚未生成"
    if quality_path.is_file():
        quality_summary = json.loads(quality_path.read_text(encoding="utf-8"))
        official = quality_summary["official_metrics"]
        quality_text = (
            f"E-hull={official['avg_energy_above_hull_per_atom']:.6f} eV/atom，"
            f"Stable={official['frac_stable_structures']:.2%}，"
            f"Novel={official['frac_novel_structures']:.2%}，"
            f"Unique={official['frac_unique_structures']:.2%}，"
            f"NUS={official['frac_novel_unique_stable_structures']:.2%}，"
            f"RMSD={official['avg_rmsd_from_relaxation']:.6f}，"
            f"MaxF mean/median/P95={quality_summary['pre_relaxation_max_force_mean']:.6f}/"
            f"{quality_summary['pre_relaxation_max_force_median']:.6f}/"
            f"{quality_summary['pre_relaxation_max_force_p95']:.6f} eV/Å"
        )
    property_path = ROOT / "diagnostic_property_summary.json"
    property_text = "尚未生成"
    if property_path.is_file():
        property_summary = json.loads(property_path.read_text(encoding="utf-8"))
        property_text = (
            f"Mag MAE={property_summary['mag_mae']:.6f} Å^-3，"
            f"Hit={property_summary['mag_hit_count']}/{property_summary['n']}，"
            f"valid={property_summary['valid_count']}/{property_summary['n']}"
        )
    gram_difference = np.abs(
        frame["cell_metric_ltl_relative_change"].to_numpy(float)
        - frame["cell_metric_llt_relative_change"].to_numpy(float)
    )
    finite_gram_difference = gram_difference[np.isfinite(gram_difference)]
    lines = [
        "# Shared Field Dynamics Diagnostic FINAL",
        "",
        "## 实验口径",
        "",
        "- 64 条 fresh trajectories；calibration 32、validation 32，拆分单位为完整 trajectory。",
        "- Official dft_mag_density Full FT；target=0.2 Å^-3；CFG=2；1000 steps；corrector=1；Adaptive CFG=OFF。",
        "- 每条 55 个 observation points，位于 corrector 之后、predictor 更新之前；诊断不增加 GemNet forward，均为 2000 score calls。",
        "- 网格覆盖 model_t=1.0→0.001；请求的 t=0.00 映射到官方最后一次模型调用 t=0.001，不虚构额外 t=0 forward。",
        "- 属性 evaluator 为冻结 CHGNet magnetic-density surrogate（held-out MAE=0.001449，Pearson=0.9499，tau=0.01）；SURROGATE_PROPERTY_EVAL=True，DFT_VERIFIED=False。",
        f"- 诊断统计共 {len(frame)} 行，clean estimates 有限 {int(frame['finite_clean_estimates'].astype(bool).sum())}/{len(frame)}；64 轨迹平均采样 {trajectories['runtime_seconds'].mean():.3f} 秒，合计 128000 score calls。",
        f"- 当前 symmetric-lattice gauge 下 L^T L 与 row-vector physical Gram L L^T 的 residual 差异很小：mean abs diff={finite_gram_difference.mean():.3e}，max={finite_gram_difference.max():.3e}；仍按预注册使用 L^T L 作 gate。",
        "",
        "## 收敛定义",
        "",
        "- Atomic：CFG 合并后的 p(x0|xt) logits 的平均 entropy、argmax change、mean max probability，并强制 mask_fraction=0。",
        "- Position：相邻 observation 的 clean-x0 分数坐标最小像位移，使用前一 clean cell 映射到笛卡尔空间。",
        "- Cell：相邻 clean-x0 的 L^T L 相对 Frobenius 变化为主，同时约束 volume、length、angle 变化；L L^T 仅作 row-lattice audit。",
        "- 阈值只由 calibration 32 在 t<=0.05 的尾部冻结：残差 q90、confidence q10；首次连续 K=3 个 observation 满足即得到 T_field；t>0.5 禁止判为可冻结。",
        "",
        "## 冻结阈值",
        "",
        "```json",
        json.dumps(frozen["thresholds"], indent=2, sort_keys=True),
        "```",
        "",
        "## Validation convergence statistics",
        "",
        "| Field | converged | mean T | median T | P25 | P75 | P90 | r(T,num_atoms) | r(T,MaxF) | r(T,E-hull) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary[summary["subset"] == "validation"].iterrows():
        values = [row[name] for name in (
            "mean_T", "median_T", "p25_T", "p75_T", "p90_T",
            "pearson_T_vs_num_atoms", "pearson_T_vs_final_max_force",
            "pearson_T_vs_final_ehull",
        )]
        rendered = ["NaN" if not np.isfinite(value) else f"{value:.4f}" for value in values]
        lines.append(
            f"| {row['field']} | {int(row['n_converged'])}/{int(row['n'])} | "
            + " | ".join(rendered) + " |"
        )
    lines += [
        "",
        f"最终 CHGNet property/validity：{property_text}。",
        "",
        f"最终 MatterSim 质量：{quality_state}；{quality_text}。",
        "",
        "## Convergence order（validation only）",
        "",
        f"First-field（并列与 NC 独立成类，分母 32）计数：`{first_counts}`。",
        "",
        f"First-field 比例：`{first_percentages}`。即 Atomic={first_percentages.get('A', 0.0):.2%}、Position={first_percentages.get('X', 0.0):.2%}、Cell={first_percentages.get('L', 0.0):.2%}；其余为 tie 或 NC。",
        "",
        f"完整 ordering 分布：`{order_counts}`。",
        "",
        "## Heterogeneity gate（validation only）",
        "",
        f"- validation median T：`{gate['median_T']}`",
        f"- 最大 median pair difference：`{gate['maximum_median_pair_difference']:.6f}`（阈值 0.10）",
        f"- 单轨存在 >=0.10 pair gap：{gate['trajectory_pair_gap_n']}/32 = {gate['trajectory_pair_gap_fraction']:.2%}（阈值 60%）",
        f"- 更严格的三字段 T 均有限口径：{gate['trajectory_pair_gap_strict_all_finite_n']}/32 = {gate['trajectory_pair_gap_strict_all_finite_fraction']:.2%}。主口径将未完成 K-stable 的字段按终点 T=0 截尾，因此更有利于检出异质性。",
        f"- **FIELD_CONVERGENCE_HETEROGENEITY={str(gate['FIELD_CONVERGENCE_HETEROGENEITY']).upper()}**",
        "",
        "若 gate=False，A/B 均按预注册直接记为 FAIL_DIAGNOSTIC，禁止强行实现；若 gate=True，才进入两个相互独立的 P0。",
    ]
    (ROOT / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Freeze thresholds once all 32 calibration trajectories exist.")
    parser.add_argument("--allow-refreeze-thresholds", action="store_true",
                        help="Only for exact deterministic reruns before P0 begins.")
    args = parser.parse_args()
    frame = _read_traces(require_all=not args.calibrate_only)
    threshold_path = ROOT / "convergence_thresholds.json"
    newly_calibrated = _calibrate(frame)
    if args.calibrate_only:
        present = set(frame["seed"].astype(int).unique())
        missing = sorted(set(CALIBRATION_SEEDS) - present)
        if missing:
            raise RuntimeError(f"cannot freeze thresholds; missing calibration seeds: {missing}")
        if threshold_path.exists():
            raise FileExistsError(f"thresholds already frozen: {threshold_path}")
        threshold_path.write_text(
            json.dumps(newly_calibrated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(newly_calibrated, indent=2, sort_keys=True))
        return
    if threshold_path.exists() and not args.allow_refreeze_thresholds:
        frozen = json.loads(threshold_path.read_text(encoding="utf-8"))
    else:
        frozen = newly_calibrated
        threshold_path.write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    trajectories = _trajectory_table(frame, frozen)
    trajectories.to_csv(ROOT / "trajectory_field_metrics.csv", index=False,
                        quoting=csv.QUOTE_MINIMAL)
    summary = _summary_table(trajectories)
    summary.to_csv(ROOT / "convergence_summary.csv", index=False)
    gate = _evaluate_gate(trajectories)
    (ROOT / "heterogeneity_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _make_plots(frame, trajectories)
    _write_report(frame, trajectories, summary, frozen, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
