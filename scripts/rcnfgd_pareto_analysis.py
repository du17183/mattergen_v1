"""Existing-data RC-NFGD Pareto audit. No generation or retraining."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/root/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/final_thesis_results/all_per_seed_metrics.csv")
OUT = ROOT / "results/rcnfgd/pareto_analysis"
FIG = ROOT / "figures/rcnfgd"
N_BOOT = 20000
COLORS = {"C0": "#697684", "F0": "#166F86", "POST": "#C17A35"}
MARKERS = {"C0": "o", "F0": "s", "POST": "^"}
LABELS = {"C0": "MatterGen / C0", "F0": "RC-NFGD", "POST": "POST"}


def read_data():
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for cohort, methods, n in (("A6", ("C0", "F0", "POST"), 32), ("A1", ("C0", "F0"), 256)):
        grouped[cohort] = {}
        for method in methods:
            records = [r for r in rows if r["cohort"] == cohort and r["method"] == method]
            if len(records) != n or len({int(r["seed"]) for r in records}) != n:
                raise ValueError(f"Incomplete or duplicate {cohort}/{method}: {len(records)}")
            grouped[cohort][method] = {int(r["seed"]): r for r in records}
        if len({tuple(sorted(grouped[cohort][m])) for m in methods}) != 1:
            raise ValueError(f"Unpaired seeds in {cohort}")
        if any(r["dft_verified"].lower() != "false" for m in methods for r in grouped[cohort][m].values()):
            raise ValueError("Unexpected DFT verification flag")
    return grouped


def array(group, method, metric):
    values = np.asarray([float(group[method][seed][metric]) for seed in sorted(group[method])], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite {method}/{metric}")
    return values


def mean_ci(values, seed):
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[sample].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def paired_ci(group, a, b, metric, seed):
    diff = array(group, a, metric) - array(group, b, metric)
    return float(diff.mean()), mean_ci(diff, seed)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def figure_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 6.5,
        "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.dpi": 300,
    })


def export(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(FIG / f"{name}.alignment.json"),
                                       overlay_svg=str(FIG / f"{name}.alignment.svg"), strict=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.055)
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", pad_inches=0.055)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.055, dpi=300)
    plt.close(fig)


def plot_scatter(group):
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.25, top=0.96)
    for method in ("C0", "F0", "POST"):
        x, y = array(group, method, "Property MAE"), array(group, method, "MaxF")
        ax.scatter(x, y, s=16, marker=MARKERS[method], color=COLORS[method],
                   alpha=0.65, linewidths=0.2, edgecolors="white", label=LABELS[method])
        ax.scatter([x.mean()], [y.mean()], s=48, marker=MARKERS[method],
                   color=COLORS[method], edgecolors="black", linewidths=0.55, zorder=5)
    ax.set(xlabel="Property MAE (proxy units)", ylabel="MatterSim MaxF (eV Å⁻¹)")
    ax.grid(color="#E6E8EA", lw=0.45, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.55, 0.00), handletextpad=0.30, columnspacing=0.8)
    export(fig, "property_force_pareto")


def plot_frontier(summary):
    rows = [r for r in summary if r["cohort"] == "A6"]
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.25, top=0.96)
    for r in rows:
        method = r["method"]
        x, y = float(r["property_mae_mean"]), float(r["maxf_mean"])
        xerr = [[x - float(r["property_mae_ci_low"])], [float(r["property_mae_ci_high"]) - x]]
        yerr = [[y - float(r["maxf_ci_low"])], [float(r["maxf_ci_high"]) - y]]
        ax.errorbar([x], [y], xerr=xerr, yerr=yerr, fmt=MARKERS[method],
                    color=COLORS[method], ecolor=COLORS[method], markersize=5,
                    capsize=2, elinewidth=0.85, label=LABELS[method], zorder=4)
    front = sorted((r for r in rows if r["point_estimate_pareto_front"] == "True"),
                   key=lambda r: float(r["property_mae_mean"]))
    ax.plot([float(r["property_mae_mean"]) for r in front],
            [float(r["maxf_mean"]) for r in front], color="#2C3E50", lw=0.85, ls="--", zorder=2)
    ax.set(xlabel="Mean property MAE (proxy units)", ylabel="Mean MatterSim MaxF (eV Å⁻¹)")
    ax.grid(color="#E6E8EA", lw=0.45, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.55, 0.00), handletextpad=0.30, columnspacing=0.8)
    export(fig, "pareto_frontier")


def main():
    groups = read_data()
    figure_style()
    rows, raw = [], []
    for cohort, group in groups.items():
        for method in group:
            metric_stats = {}
            for metric, prefix in (("Property MAE", "property_mae"), ("MaxF", "maxf"),
                                   ("Atomic Force", "mean_force")):
                values = array(group, method, metric)
                ci = mean_ci(values, 20260929 + len(values) * 10 + len(method) + len(metric))
                metric_stats.update({f"{prefix}_mean": values.mean(), f"{prefix}_std": values.std(ddof=1),
                                     f"{prefix}_ci_low": ci[0], f"{prefix}_ci_high": ci[1]})
            for seed, record in group[method].items():
                raw.append({"cohort": cohort, "seed": seed, "method": method,
                            "property_mae": record["Property MAE"], "maxf_ev_a": record["MaxF"],
                            "mean_force_ev_a": record["Atomic Force"]})
            rows.append({"cohort": cohort, "method": method, "n": len(group[method]), **metric_stats})
    for r in rows:
        peers = [p for p in rows if p["cohort"] == r["cohort"] and p["method"] != r["method"]]
        dominators = [p["method"] for p in peers if
                      float(p["property_mae_mean"]) <= float(r["property_mae_mean"]) and
                      float(p["maxf_mean"]) <= float(r["maxf_mean"]) and
                      (float(p["property_mae_mean"]) < float(r["property_mae_mean"]) or
                       float(p["maxf_mean"]) < float(r["maxf_mean"]))]
        r["point_estimate_pareto_front"] = not bool(dominators)
        r["dominated_by"] = ";".join(dominators)
    write_csv(OUT / "pareto_summary.csv", rows, list(rows[0]))
    write_csv(OUT / "pareto_per_structure.csv", raw, list(raw[0]))
    plot_scatter(groups["A6"])
    plot_frontier(rows)
    a6 = groups["A6"]
    lines = ["# RC-NFGD Property–Force Pareto 分析", "",
             "状态：COMPLETED_EXISTING_DATA；只重分析历史 A6、A1 结果，未重新训练或生成。",
             "", "**定义**：横轴为冻结属性代理的磁密度 Property MAE；纵轴为终态 MatterSim MaxF，均越小越好。A6 三臂同为 seeds 745000–745031，n=32，逐 seed 配对；A1 两臂 n=256，单独分析，绝不与 A6 混成前沿。前沿是二维**方法均值点估计**，不是显著性判定，也不是逐结构选择器。", "",
             "| Cohort | Method | n | Property MAE mean ± SD | MatterSim MaxF mean ± SD (eV/Å) | Point-estimate front |",
             "|---|---|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['cohort']} | {LABELS[r['method']]} | {r['n']} | {float(r['property_mae_mean']):.6f} ± {float(r['property_mae_std']):.6f} | {float(r['maxf_mean']):.6f} ± {float(r['maxf_std']):.6f} | {'yes' if r['point_estimate_pareto_front'] else 'no'} |")
    lines += ["", "方法均值的 20,000 次 seed bootstrap 95% CI 位于 `pareto_summary.csv`，同一批配对索引用于差值的 CI。以下差值定义为 **前者−后者**，正数表示后者更低；没有多重比较校正，解释为补充性分析。", "",
              "| Cohort | Comparison | Metric | Mean difference | Paired bootstrap 95% CI |", "|---|---|---|---:|---:|"]
    for cohort, comparisons in (("A6", (("C0", "F0"), ("C0", "POST"), ("F0", "POST"))),
                                ("A1", (("C0", "F0"),))):
        for a, b in comparisons:
            for metric in ("Property MAE", "MaxF", "Atomic Force"):
                delta, ci = paired_ci(groups[cohort], a, b, metric, 20260929 + len(cohort) + len(a) + len(b) + len(metric))
                lines.append(f"| {cohort} | {a}−{b} | {metric} | {delta:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] |")
    lines += ["", "## 解释与边界", "",
              "A6 的 POST 在同一 MatterSim MaxF 上具有明显优势；不能声称 RC-NFGD 全面优于后处理。A6 方法均值上，RC-NFGD 与 POST 形成描述性折中，但极小的 MAE 差异必须结合配对 CI 看，不能把点估计前沿升格为确认性性能排序。A1 Formal256 的 Property MAE 方向与 A6 不完全一致，应按独立 cohort 如实保留。", "",
              "这些力和属性指标都来自代理模型。MatterSim 是 RC-NFGD 所用势，存在 verifier/self-consistency 偏倚；已有 CHGNet 方向一致证据不等于独立 DFT 真值。此 Pareto 分析不能证明材料物理真实稳定性、全域最优性或联合创新点收益。", "",
              "图件：`figures/rcnfgd/property_force_pareto.pdf` 为 A6 逐结构散点（大描边符号为均值）；`figures/rcnfgd/pareto_frontier.pdf` 为均值及 95% CI，虚线仅连接描述性非支配点。原始来源与逐结构复核可由 `pareto_per_structure.csv` 追溯。", ""]
    (OUT / "pareto_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(rows)} method summaries and {len(raw)} per-structure records")


if __name__ == "__main__":
    main()
