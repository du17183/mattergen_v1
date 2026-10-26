"""Full-cohort, paired 20k-bootstrap Timing analysis; never drops a failed seed."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.mattersim_late_force_guidance_p0 import analyze_p0 as frozen


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "timing_formal32"
THESIS = Path("/mnt/datasets-livsyn/dxl/mattergen_v1_thesis_final")
RESULTS = THESIS / "results/rcnfgd/timing_ablation"
DOCS = THESIS / "docs"
METRICS = {
    "max_force": ("maxF_ev_per_a", "lower", "eV/A"),
    "mean_force": ("atomic_force_mean_ev_per_a", "lower", "eV/A"),
    "property_mae": ("mag_absolute_error_a3", "lower", "A^-3"),
    "validity": ("valid", "higher", "fraction"),
    "stable": ("stable", "higher", "fraction"),
    "e_hull": ("e_hull_ev_per_atom", "lower", "eV/atom"),
    "nus": ("nus", "higher", "fraction"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv_new(path: Path, rows: list[dict]) -> None:
    if not rows or path.exists():
        raise RuntimeError(f"empty or existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def archive_placeholder(path: Path) -> None:
    if not path.exists():
        return
    archive = path.with_name(path.stem + "_NOT_RUN_archived" + path.suffix)
    if archive.exists():
        raise FileExistsError(f"existing placeholder archive: {archive}")
    content = path.read_text(encoding="utf-8")
    if "NOT_RUN" not in content:
        raise RuntimeError(f"refusing to replace non-placeholder result: {path}")
    shutil.copy2(path, archive)
    path.unlink()


def values(rows: list[dict], key: str) -> np.ndarray:
    value = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if not np.isfinite(value).all():
        raise RuntimeError(f"nonfinite {key}")
    return value


def fmt(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    protocol = json.loads((ROOT / "timing_protocol.json").read_text())
    seeds = list(range(protocol["seed_start"], protocol["seed_stop_exclusive"]))
    methods = protocol["methods"]
    status = json.loads((OUT / "generation_status_attempt3.json").read_text())
    if status["status"] != "COMPLETE" or status["completed_paired_seeds"] != seeds:
        raise RuntimeError("generation cohort is incomplete")
    for method in methods:
        if not (OUT / "p0_quality" / method / "official_detailed.json.gz").is_file():
            raise RuntimeError(f"quality evaluation incomplete: {method}")
        if not (OUT / "p0_relaxation" / method / "relaxation_summary.json").is_file():
            raise RuntimeError(f"MatterSim relaxation incomplete: {method}")
    generation = read_csv(OUT / "generation_manifest.csv")
    properties = read_csv(OUT / "p0_property_metrics.csv")
    frozen.ROOT = OUT
    frozen.SEEDS = tuple(seeds)
    frozen.METHODS = tuple(methods)
    data = {method: frozen.load_method(method, generation, properties) for method in methods}
    if any([row["seed"] for row in data[method]] != seeds for method in methods):
        raise RuntimeError("paired seed order mismatch")
    genmap = {(row["method"], int(row["seed"])): row for row in generation}
    summary_rows: list[dict] = []
    for method in methods:
        for row in data[method]:
            aux = genmap[(method, row["seed"])]
            summary_rows.append({
                "method": method, "seed": row["seed"], "injection_stage": method,
                "injection_t_max": protocol["guidance_t_max"][method],
                "force_calls": int(aux["actual_mattersim_calls"]),
                "failed_force_calls": int(aux["failed_mattersim_calls"]),
                "force_opportunities": int(aux["guidance_eligible"]),
                "force_accepted": int(aux["guidance_accepted"]),
                "force_fallbacks": int(aux["guidance_fallbacks"]),
                "property_mae": row["mag_absolute_error_a3"],
                "max_force": row["maxF_ev_per_a"],
                "mean_force": row["atomic_force_mean_ev_per_a"],
                "stable": row["stable"], "validity": row["valid"],
                "e_hull": row["e_hull_ev_per_atom"], "nus": row["nus"],
                "wall_time_seconds": float(aux["end_to_end_seconds"]),
                "sampling_wall_seconds": float(aux["elapsed_seconds"]),
                "cuda_event_elapsed_seconds": float(aux["cuda_event_elapsed_seconds"]),
                "mattergen_score_calls": int(aux["mattergen_score_calls"]),
                "physical_gpu": aux["physical_gpu"],
                "surrogate_property_eval": True, "dft_verified": False,
            })
    if len(summary_rows) != len(seeds) * len(methods):
        raise RuntimeError("not all 128 method-seed records present")

    bcount = int(protocol["paired_bootstrap_resamples"])
    rng = np.random.default_rng(int(protocol["paired_bootstrap_seed"]))
    indices = rng.integers(0, len(seeds), size=(bcount, len(seeds)))
    method_summary = []
    for method in methods:
        rows = data[method]
        aux = [genmap[(method, seed)] for seed in seeds]
        output = {"method": method, "n": len(seeds),
                  "injection_t_max": protocol["guidance_t_max"][method]}
        for name, (key, _, _) in METRICS.items():
            x = values(rows, key)
            ci = np.quantile(x[indices].mean(axis=1), (0.025, 0.975))
            output.update({f"{name}_mean": float(x.mean()),
                           f"{name}_std": float(x.std(ddof=1)),
                           f"{name}_mean_ci95_low": float(ci[0]),
                           f"{name}_mean_ci95_high": float(ci[1])})
        for name, key in (("force_calls", "actual_mattersim_calls"),
                          ("force_opportunities", "guidance_eligible"),
                          ("force_accepted", "guidance_accepted"),
                          ("force_fallbacks", "guidance_fallbacks"),
                          ("wall_time_seconds", "end_to_end_seconds"),
                          ("cuda_event_elapsed_seconds", "cuda_event_elapsed_seconds")):
            x = np.asarray([float(row[key]) for row in aux], dtype=float)
            output[f"{name}_mean"] = float(x.mean())
            output[f"{name}_std"] = float(x.std(ddof=1))
            output[f"{name}_total"] = float(x.sum())
        method_summary.append(output)

    paired = []
    for baseline, candidate in (("C0", "Early"), ("C0", "Middle"), ("C0", "Late"),
                                ("Early", "Middle"), ("Early", "Late"), ("Middle", "Late")):
        for name, (key, direction, unit) in METRICS.items():
            x, y = values(data[baseline], key), values(data[candidate], key)
            delta = y - x
            ci = np.quantile(delta[indices].mean(axis=1), (0.025, 0.975))
            favorable = delta < -1e-12 if direction == "lower" else delta > 1e-12
            unfavorable = delta > 1e-12 if direction == "lower" else delta < -1e-12
            paired.append({
                "baseline": baseline, "candidate": candidate, "metric": name,
                "direction": direction, "unit": unit, "n_pairs": len(seeds),
                "baseline_mean": float(x.mean()), "candidate_mean": float(y.mean()),
                "candidate_minus_baseline": float(delta.mean()),
                "paired_ci95_low": float(ci[0]), "paired_ci95_high": float(ci[1]),
                "candidate_favorable_pairs": int(favorable.sum()),
                "ties": int((~(favorable | unfavorable)).sum()),
                "candidate_unfavorable_pairs": int(unfavorable.sum()),
                "bootstrap_resamples": bcount,
                "bootstrap_seed": protocol["paired_bootstrap_seed"],
            })

    def contrast(a: str, b: str, metric: str) -> dict:
        return next(row for row in paired if row["baseline"] == a and row["candidate"] == b
                    and row["metric"] == metric)

    late_early = contrast("Early", "Late", "max_force")
    late_middle = contrast("Middle", "Late", "max_force")
    early_late_force = late_early["paired_ci95_low"] > 0
    middle_late_force = late_middle["paired_ci95_low"] > 0
    late_force_better = late_early["paired_ci95_high"] < 0 and late_middle["paired_ci95_high"] < 0
    if late_force_better:
        timing_statement = "Late has lower MaxF than both Early and Middle on this cohort; both unadjusted paired intervals exclude zero."
    elif early_late_force or middle_late_force:
        timing_statement = "An earlier-start arm has lower MaxF than Late with an unadjusted paired interval excluding zero; the late-best hypothesis is not supported."
    else:
        timing_statement = "The force ranking among Early, Middle and Late is not resolved by both unadjusted paired intervals; do not claim a unique best start stage."

    means = {row["method"]: row for row in method_summary}
    pareto = []
    for method in methods:
        own = means[method]
        dominated = any(
            other != method and
            means[other]["property_mae_mean"] <= own["property_mae_mean"] and
            means[other]["max_force_mean"] <= own["max_force_mean"] and
            (means[other]["property_mae_mean"] < own["property_mae_mean"] or
             means[other]["max_force_mean"] < own["max_force_mean"])
            for other in methods
        )
        if not dominated:
            pareto.append(method)

    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS / "timing_ablation_summary.csv"
    report_path = RESULTS / "timing_ablation_report.md"
    archive_placeholder(summary_path)
    archive_placeholder(report_path)
    write_csv_new(summary_path, summary_rows)
    write_csv_new(RESULTS / "timing_method_summary.csv", method_summary)
    write_csv_new(RESULTS / "timing_paired_bootstrap.csv", paired)
    write_csv_new(OUT / "per_structure_metrics.csv", [row for method in methods for row in data[method]])
    (RESULTS / "timing_protocol_snapshot.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# RC-NFGD Force Injection Timing Ablation", "",
        "**Status: COMPLETE; 32 fresh paired seeds (750000–750031), four arms, no dropped pairs.**", "",
        "Only force-feedback start time changed: C0 none; Early t≤1.0 (1000 predictor opportunities); Middle t≤0.5 (500); Late t≤0.02 (20, original RC-NFGD). The original MatterGen/MatterSim checkpoints, frozen sampler, force formula and safety fallback were unchanged. Each method uses 2000 MatterGen score calls per sample. Results are **not equal force-compute comparisons**; changing the start time also changes the cumulative force-call budget.", "",
        "The pre-relaxation MatterSim MaxF is primary; mean force, CHGNet magnetic-density property MAE, validity, Stable, E-hull and NUS are secondary surrogate/official pipeline metrics. No DFT calculation was performed. Means use n=32; std is sample standard deviation. Paired intervals use 20,000 percentile bootstrap resamples with seed 20260929, candidate−baseline sign; reported intervals are exploratory and not multiplicity-adjusted.", "",
        "| Method | Force calls/sample mean±std | End-to-end wall s mean±std | CUDA-event interval s mean±std | MaxF mean±std (eV/Å) | Property MAE mean±std | Stable | NUS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_summary:
        lines.append(
            f"| {row['method']} | {fmt(row['force_calls_mean'],1)}±{fmt(row['force_calls_std'],1)} | "
            f"{fmt(row['wall_time_seconds_mean'],1)}±{fmt(row['wall_time_seconds_std'],1)} | "
            f"{fmt(row['cuda_event_elapsed_seconds_mean'],1)}±{fmt(row['cuda_event_elapsed_seconds_std'],1)} | "
            f"{fmt(row['max_force_mean'])}±{fmt(row['max_force_std'])} | "
            f"{fmt(row['property_mae_mean'])}±{fmt(row['property_mae_std'])} | "
            f"{fmt(row['stable_mean'],3)} | {fmt(row['nus_mean'],3)} |"
        )
    lines += ["", "CUDA events bracket the sampling call and may include CPU waiting; they are **not** a measure of active GPU-SM utilization. Potential-loading time is included in end-to-end wall time but not in the CUDA-event interval.", "", "## Paired MaxF contrasts", "", "| Candidate − baseline | Mean delta (eV/Å) | Paired 95% CI | Better/tie/worse pairs |", "|---|---:|---:|---:|"]
    for row in paired:
        if row["metric"] != "max_force":
            continue
        lines.append(f"| {row['candidate']} − {row['baseline']} | {fmt(row['candidate_minus_baseline'])} | [{fmt(row['paired_ci95_low'])}, {fmt(row['paired_ci95_high'])}] | {row['candidate_favorable_pairs']}/{row['ties']}/{row['candidate_unfavorable_pairs']} |")
    lines += ["", "## Stage interpretation", "", timing_statement,
              "This is a joint change of injection timing and cumulative force-computation budget, so it cannot isolate a pure timing effect at fixed compute. Force opportunities, actual calls, accepted/fallback events, wall time and CUDA-event intervals are preserved per seed. Negative outcomes and all Early/Middle failures remain included.", "",
              f"On the cohort mean (Property MAE, MaxF) plane, nondominated methods are: {', '.join(pareto)}. This descriptive Pareto check does not include POST from another seed cohort and does not establish statistical dominance.", "",
              "Full 128-row per-seed table: `timing_ablation_summary.csv`; per-method mean/std/mean bootstrap interval: `timing_method_summary.csv`; every paired 20k interval: `timing_paired_bootstrap.csv`. Existing NOT_RUN placeholders were preserved as `*_NOT_RUN_archived.*`.", "",
              "No claim of DFT-level stability, universal late optimality, or superiority over all post-processing methods is supported.", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    impact = ["# RC-NFGD Timing Ablation：创新点2最终影响", "",
              "状态：正式 32 paired seeds 四臂实验完成。", "",
              "## 是否支持方法动机", "", timing_statement,
              "无论排名，比较同时改变注入阶段和 MatterSim 调用预算，不能把结果写成固定预算下的时机因果证明。Late 对应原方法，不根据本次结果调整任何超参数。", "",
              "## 标题与正文", "",
              "推荐保留‘面向冻结材料扩散模型的神经力场反馈引导方法’。仅当本次数据支持时，可在第4章增加‘当前设定下后期注入的属性—代理力折中’这一限定性消融结论；若 Early/Middle 更好，必须如实写明原 late 设计并非最佳。摘要不应把时机分析上升为普适最优性。", "",
              "## 与 Pareto 分析的关系", "",
              f"本次同一 32-seed cohort 的均值二维 Pareto 前沿包含：{', '.join(pareto)}。既有 C0/RC-NFGD/POST Pareto 来自独立历史 cohort，不能与本次四臂均值合并为同一配对前沿；POST 在其原 cohort 的代理力效果更强这一负结果应保留。", "",
              "## 证据边界", "",
              "MatterSim 与 CHGNet 都是代理模型；Stable/E-hull/NUS 为既有质量管线指标。无 DFT 验证、无跨生成模型普适性、无与 Fixed-K2 联合增益。报告计算成本，不声称同预算。", ""]
    target = DOCS / "rcnfgd_timing_analysis_final.md"
    if target.exists():
        raise FileExistsError(target)
    target.write_text("\n".join(impact), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "paired_seeds": len(seeds),
                      "methods": methods, "timing_statement": timing_statement,
                      "pareto_methods": pareto, "results": str(RESULTS)}), flush=True)


if __name__ == "__main__":
    main()
