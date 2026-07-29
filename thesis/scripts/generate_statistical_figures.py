#!/usr/bin/env python3
"""Generate skill-guided statistical figures solely from ``thesis_archive``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "figures" / "source" / "styles"))

from common.paths import ARCHIVE_ROOT, FIGURE_OUTPUTS, FIGURE_SOURCE_DATA, ensure_output_directories
from paper_style import COLORS, clean_axes, panel_label, paper_context, save_figure

RNG = np.random.default_rng(20260729)


def load_csv(group: str) -> pd.DataFrame:
    return pd.read_csv(ARCHIVE_ROOT / "data" / group / "per_seed_metrics.csv")


def load_json(relative: str):
    with (ARCHIVE_ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def source_csv(stem: str, data: pd.DataFrame) -> None:
    data.to_csv(FIGURE_SOURCE_DATA / f"{stem}.csv", index=False)


def fig05():
    stem = "fig05_adaptive_cfg_results"
    data = load_csv("innovation1")
    report = load_json("reports/innovation1/formal_final_report.json")
    diff_e = data["a0_ehull"] - data["c0_ehull"]
    changes = pd.DataFrame(
        {
            "seed": data["seed"],
            "ehull_difference_ev_atom": diff_e,
            "stable_difference": data["a0_stable"].astype(int) - data["c0_stable"].astype(int),
            "nus_difference": data["a0_nus"].astype(int) - data["c0_nus"].astype(int),
        }
    )
    source_csv(stem, changes)
    stats_map = {row["metric"]: row for row in report["innovation1"]["paired_statistics"]}
    with paper_context():
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), layout="constrained")
        ax = axes[0]
        jitter = RNG.normal(0, 0.035, len(diff_e))
        ax.scatter(jitter, diff_e, s=8, alpha=0.24, color=COLORS["blue"], edgecolors="none")
        e_row = stats_map["energy_above_hull_per_atom"]; ci = [e_row["bootstrap_95_ci_low"], e_row["bootstrap_95_ci_high"]]
        mean = float(diff_e.mean())
        ax.errorbar(0, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], fmt="D", color=COLORS["black"], capsize=4)
        ax.axhline(0, color=COLORS["gray"], lw=0.9, ls="--")
        ax.set(xlim=(-0.25, 0.25), xticks=[], ylabel=r"Paired $\Delta E_\mathrm{hull}$ (eV atom$^{-1}$)")
        ax.set_title("A0 Adaptive CFG − C0")
        clean_axes(ax)
        panel_label(ax, "A")
        ax.text(
            0.04,
            0.96,
            f"mean={mean:+.4f}\n95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]\nHolm p=1.00",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )

        ax = axes[1]
        labels = ["Stable", "NUS"]
        values = [changes["stable_difference"].mean() * 100, changes["nus_difference"].mean() * 100]
        cis = [
            [stats_map["stable"]["bootstrap_95_ci_low"], stats_map["stable"]["bootstrap_95_ci_high"]],
            [stats_map["novel_unique_stable"]["bootstrap_95_ci_low"], stats_map["novel_unique_stable"]["bootstrap_95_ci_high"]],
        ]
        low = [v - c[0] * 100 for v, c in zip(values, cis)]
        high = [c[1] * 100 - v for v, c in zip(values, cis)]
        ax.bar(labels, values, color=[COLORS["green"], COLORS["purple"]], edgecolor=COLORS["black"], zorder=2)
        ax.errorbar(range(2), values, yerr=[low, high], fmt="none", color=COLORS["black"], capsize=4)
        ax.axhline(0, color=COLORS["gray"], lw=0.9, ls="--")
        ax.set_ylabel("Paired rate change (percentage points)")
        ax.set_title("Quality-rate changes")
        clean_axes(ax)
        panel_label(ax, "B")
        for i, v in enumerate(values):
            ax.text(i, v + 0.8, f"{v:+.2f} pp", ha="center", fontsize=7)
        fig.suptitle("Adaptive CFG formal evaluation (n=256; paired effects not statistically significant)", fontsize=9)
        save_figure(fig, stem, FIGURE_OUTPUTS)


def _formal_force_row(comparison: str):
    path = ARCHIVE_ROOT / "reports" / "innovation2" / "formal_paired_statistics.csv"
    frame = pd.read_csv(path)
    return frame[(frame.metric == "pre_relax_max_force_ev_ang") & (frame.comparison == comparison)].iloc[0]


def fig06():
    stem = "fig06_e3pcr_force_formal256"
    data = load_csv("innovation2")
    row = _formal_force_row("E3-G vs C0")
    source_csv(
        stem,
        data[["seed", "c0_max_force", "e3a_max_force", "e3g_max_force", "gate_on", "exact_fallback"]].copy(),
    )
    with paper_context():
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), layout="constrained")
        ax = axes[0]
        series = [data.c0_max_force, data.e3a_max_force, data.e3g_max_force]
        positions = [1, 2, 3]
        bp = ax.boxplot(series, positions=positions, widths=0.48, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], [COLORS["gray"], COLORS["orange"], COLORS["green"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for i, values in enumerate(series, 1):
            x = i + RNG.normal(0, 0.075, len(values))
            ax.scatter(x, values, s=5, alpha=0.18, color=COLORS["black"], edgecolors="none")
            ax.scatter(i, np.mean(values), marker="D", s=25, color=COLORS["black"], zorder=4)
        ax.set(xticks=positions, xticklabels=["C0", "E3-A", "E3-G"], ylabel=r"Pre-relax max force (eV $\AA^{-1}$)")
        ax.set_yscale("log")
        ax.set_title("Three-arm distributions")
        clean_axes(ax)
        panel_label(ax, "A")

        ax = axes[1]
        diff = data.e3g_max_force - data.c0_max_force
        order = np.argsort(diff.to_numpy())
        colors = np.where(diff.to_numpy()[order] <= 0, COLORS["green"], COLORS["vermillion"])
        ax.scatter(np.arange(len(diff)), diff.to_numpy()[order], c=colors, s=9, alpha=0.7)
        ax.axhline(0, color=COLORS["gray"], lw=0.9, ls="--")
        mean = float(row.mean_difference)
        ax.axhline(mean, color=COLORS["black"], lw=1.3)
        ax.fill_between(
            [0, len(diff) - 1],
            float(row.bootstrap_95_ci_low),
            float(row.bootstrap_95_ci_high),
            color=COLORS["sky"],
            alpha=0.18,
        )
        ax.set(xlabel="Paired seed (sorted by effect)", ylabel=r"$\Delta$ max force: E3-G − C0 (eV $\AA^{-1}$)")
        ax.set_title("Learned-gated paired effects")
        clean_axes(ax)
        panel_label(ax, "B")
        ax.text(
            0.03,
            0.04,
            "−23.28%\n95% CI [−0.1450, −0.0325]\nHolm-adjusted p=4.19×10⁻¹⁰\nWin/Tie/Loss=163/0/93",
            transform=ax.transAxes,
            va="bottom",
            fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
        fig.suptitle("E3-PCR formal independent evaluation (n=256)", fontsize=9)
        save_figure(fig, stem, FIGURE_OUTPUTS)


def fig07():
    stem = "fig07_gate_safety_ablation"
    summary = load_json("reports/innovation2/final_summary.json")
    mech = summary["mechanism"]
    behavior = mech["behavior"]
    rates = pd.DataFrame(
        [
            ["Refinement rate", behavior["E3-A"]["refinement_rate"], behavior["E3-G"]["refinement_rate"], "fraction"],
            ["Harm rate", mech["e3a_harm_rate"], mech["e3g_harm_rate"], "fraction"],
            ["Low-force harm rate", mech["low_force_e3a_harm_rate"], mech["low_force_e3g_harm_rate"], "fraction"],
            ["Gain retention", 1.0, mech["gain_retention"], "fraction"],
            ["Mean displacement", behavior["E3-A"]["mean_displacement_angstrom"], behavior["E3-G"]["mean_displacement_angstrom"], "angstrom"],
            ["P95 displacement", behavior["E3-A"]["p95_displacement_angstrom"], behavior["E3-G"]["p95_displacement_angstrom"], "angstrom"],
        ],
        columns=["metric", "always_on", "learned_gated", "unit"],
    )
    rates["source"] = "thesis_archive/reports/innovation2/final_summary.json"
    source_csv(stem, rates)
    with paper_context():
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45), layout="constrained")
        ax = axes[0]
        rate_data = rates[rates.unit == "fraction"]
        y = np.arange(len(rate_data))
        h = 0.34
        ax.barh(y + h / 2, rate_data.always_on * 100, h, label="Always-on", color=COLORS["orange"], edgecolor="black")
        ax.barh(y - h / 2, rate_data.learned_gated * 100, h, label="Learned-gated", color=COLORS["green"], edgecolor="black")
        ax.set(yticks=y, yticklabels=rate_data.metric, xlabel="Rate (%)", xlim=(0, 108))
        ax.invert_yaxis()
        clean_axes(ax, "x")
        panel_label(ax, "A")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        ax.text(0.98, 0.48, "Harm McNemar p=0.000534", transform=ax.transAxes, ha="right", fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88})

        ax = axes[1]
        disp = rates[rates.unit == "angstrom"]
        y = np.arange(len(disp))
        ax.barh(y + h / 2, disp.always_on, h, color=COLORS["orange"], edgecolor="black")
        ax.barh(y - h / 2, disp.learned_gated, h, color=COLORS["green"], edgecolor="black")
        ax.set(yticks=y, yticklabels=disp.metric, xlabel=r"Displacement ($\AA$)")
        ax.invert_yaxis()
        clean_axes(ax, "x")
        panel_label(ax, "B")
        ax.set_title("Smaller intervention")
        fig.suptitle("Learned Gate safety mechanism (n=256): less coverage and harm; 80.66% gain retained", fontsize=9)
        save_figure(fig, stem, FIGURE_OUTPUTS)


def fig08():
    stem = "fig08_gate_confidence_force_gain"
    data = load_csv("innovation2")
    plot = data[["seed", "gate_confidence", "gate_on", "c0_max_force", "e3g_max_force"]].copy()
    plot["force_gain"] = plot.c0_max_force - plot.e3g_max_force
    rho, p = stats.spearmanr(plot.gate_confidence, plot.force_gain)
    plot["spearman_rho"] = rho
    plot["spearman_p"] = p
    source_csv(stem, plot)
    with paper_context():
        fig, ax = plt.subplots(figsize=(5.3, 3.65), layout="constrained")
        off = ~plot.gate_on.astype(bool)
        ax.scatter(
            plot.loc[off, "gate_confidence"],
            plot.loc[off, "force_gain"],
            marker="x",
            s=24,
            color=COLORS["gray"],
            label=f"Gate-off (n={off.sum()}; exact fallback)",
        )
        ax.scatter(
            plot.loc[~off, "gate_confidence"],
            plot.loc[~off, "force_gain"],
            marker="o",
            s=15,
            alpha=0.65,
            facecolors="none",
            edgecolors=COLORS["green"],
            label=f"Gate-on (n={(~off).sum()})",
        )
        coeff = np.polyfit(plot.gate_confidence, plot.force_gain, 1)
        xs = np.linspace(plot.gate_confidence.min(), plot.gate_confidence.max(), 100)
        ax.plot(xs, np.polyval(coeff, xs), color=COLORS["blue"], lw=1.2, label="Linear trend (descriptive)")
        ax.axhline(0, color=COLORS["vermillion"], lw=0.9, ls="--")
        ax.axvline(0.5, color=COLORS["gray"], lw=0.9, ls=":")
        ax.set(xlabel="Gate confidence", ylabel=r"Force gain: C0 − E3-G (eV $\AA^{-1}$)")
        clean_axes(ax)
        ax.legend(frameon=False, loc="upper left")
        ax.text(0.98, 0.97, f"Spearman ρ={rho:.3f}\np={p:.3g}", transform=ax.transAxes, ha="right", va="top")
        ax.set_title("Gate confidence versus realized force improvement (n=256)")
        save_figure(fig, stem, FIGURE_OUTPUTS)


def fig09():
    stem = "fig09_combination_replication_forest"
    rows = []
    for label, report_path, seeds, rel in [
        ("Cohort 1", "reports/compatibility/paired_statistics.csv", "41000–41063", -0.2709842977928487),
        ("Cohort 2", "reports/replication/paired_statistics.csv", "50000–50063", -0.19017613183171514),
    ]:
        frame = pd.read_csv(ARCHIVE_ROOT / report_path)
        row = frame[frame.metric == "pre_relax_max_force_ev_ang"].iloc[0]
        rows.append(
            {
                "cohort": label,
                "seed_range": seeds,
                "n": int(row.paired_count),
                "mean_difference": row.mean_difference,
                "ci_low": row.bootstrap_95_ci_low,
                "ci_high": row.bootstrap_95_ci_high,
                "relative_change": rel,
                "p_value": row.p_value,
            }
        )
    frame = pd.DataFrame(rows)
    source_csv(stem, frame)
    with paper_context():
        fig, ax = plt.subplots(figsize=(6.2, 3.1), layout="constrained")
        y = np.arange(len(frame))[::-1]
        x = frame.mean_difference.to_numpy()
        ax.errorbar(
            x,
            y,
            xerr=[x - frame.ci_low.to_numpy(), frame.ci_high.to_numpy() - x],
            fmt="D",
            color=COLORS["blue"],
            ecolor=COLORS["black"],
            capsize=4,
            markersize=5,
        )
        ax.axvline(0, color=COLORS["vermillion"], lw=0.9, ls="--")
        ax.set(yticks=y, yticklabels=frame.cohort, xlabel=r"Paired mean difference (eV $\AA^{-1}$)")
        ax.set_ylim(-0.6, 1.6)
        clean_axes(ax, "x")
        for yi, row in zip(y, rows):
            ax.text(
                0.99,
                yi,
                f"{row['seed_range']} · n={row['n']} · {row['relative_change']*100:.2f}% · p={row['p_value']:.3g}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="right",
                fontsize=7,
            )
        ax.set_title("Independent combination cohorts (not pooled)")
        save_figure(fig, stem, FIGURE_OUTPUTS)


def fig10():
    stem = "fig10_independent64_pairplot"
    data = load_csv("compatibility_2")
    plot = data[["seed", "a0_max_force", "a0_e3g_max_force", "gate_on", "exact_fallback"]].copy()
    plot["algorithmic_outcome"] = np.where(plot.gate_on.astype(bool), "Gate-on", "Gate-off exact tie")
    source_csv(stem, plot)
    with paper_context():
        fig, ax = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
        for _, row in plot.iterrows():
            off = not bool(row.gate_on)
            color = COLORS["gray"] if off else (COLORS["green"] if row.a0_e3g_max_force <= row.a0_max_force else COLORS["vermillion"])
            ax.plot([0, 1], [row.a0_max_force, row.a0_e3g_max_force], color=color, alpha=0.35, lw=0.7, ls=":" if off else "-")
        ax.scatter(np.zeros(len(plot)), plot.a0_max_force, s=13, color=COLORS["blue"], alpha=0.65, label="A0")
        gate_on = plot.gate_on.astype(bool)
        ax.scatter(np.ones(gate_on.sum()), plot.loc[gate_on, "a0_e3g_max_force"], s=14, facecolors="none", edgecolors=COLORS["green"], label="A0+E3-G gate-on")
        ax.scatter(np.ones((~gate_on).sum()), plot.loc[~gate_on, "a0_e3g_max_force"], s=22, marker="x", color=COLORS["gray"], label="Gate-off exact tie")
        ax.set(xticks=[0, 1], xticklabels=["A0", "A0 + E3-G"], ylabel=r"Pre-relax max force (eV $\AA^{-1}$)", xlim=(-0.35, 1.35))
        clean_axes(ax)
        ax.legend(frameon=False, fontsize=6.5)
        ax.set_title("Independent cohort 2 paired outcomes (seeds 50000–50063; n=64)")
        save_figure(fig, stem, FIGURE_OUTPUTS)


def fig11():
    stem = "fig11_leakage_diagnostic"
    data = load_csv("leakage_diagnostic")
    plot = data.copy()
    plot["force_gain"] = plot.a0_max_force - plot.a0_e3g_max_force
    source_csv(stem, plot)
    groups = ["training_overlap", "held_out"]
    labels = ["Training-overlap", "Held-out"]
    harms = [int(plot.loc[plot.cohort == g, "refinement_harm"].sum()) for g in groups]
    ns = [int((plot.cohort == g).sum()) for g in groups]
    _, fisher_p = stats.fisher_exact([[harms[0], ns[0] - harms[0]], [harms[1], ns[1] - harms[1]]], alternative="less")
    with paper_context():
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), layout="constrained")
        ax = axes[0]
        for i, (group, label, color) in enumerate(zip(groups, labels, [COLORS["orange"], COLORS["blue"]])):
            values = plot.loc[plot.cohort == group, "force_gain"]
            ax.scatter(i + RNG.normal(0, 0.065, len(values)), values, s=7, alpha=0.25, color=color, edgecolors="none")
            ax.errorbar(i, values.mean(), yerr=values.sem() * 1.96, fmt="D", color=COLORS["black"], capsize=4)
        ax.axhline(0, color=COLORS["gray"], lw=0.9, ls="--")
        ax.set(xticks=[0, 1], xticklabels=labels, ylabel=r"Force gain: A0 − A0+E3-G (eV $\AA^{-1}$)")
        clean_axes(ax)
        panel_label(ax, "A")
        ax.set_title("Mean-effect diagnostic")

        ax = axes[1]
        rates = np.array(harms) / np.array(ns)
        ax.bar(labels, rates * 100, color=[COLORS["orange"], COLORS["blue"]], edgecolor="black")
        ax.set(ylabel="Harm rate (%)", ylim=(0, max(22, rates.max() * 125)))
        clean_axes(ax)
        panel_label(ax, "B")
        for i, (h, n, r) in enumerate(zip(harms, ns, rates)):
            ax.text(i, r * 100 + 0.9, f"{h}/{n}\n{r*100:.2f}%", ha="center", fontsize=7)
        ax.text(0.50, 0.93, f"Fisher one-sided p={fisher_p:.3g}", transform=ax.transAxes, ha="center")
        ax.set_title("Safety inflation under overlap")
        fig.suptitle("Diagnostic only · Mixed 256 excluded from independent claims", color=COLORS["vermillion"], fontsize=9)
        save_figure(fig, stem, FIGURE_OUTPUTS)


NEGATIVE_ROUTES = [
    ("Residual Reuse", "Reduce CFG work", "Best throughput gain only +1.16%", "No-Go"),
    ("Corrector Gating", "Skip physical forwards", "~1.5× speed; Stable/NUS and E-hull worsened", "No-Go"),
    ("Budget-aware Gating", "Safer corrector skip", "Quality gates or speed target failed", "No-Go"),
    ("FN-PRA", "Representation alignment", "Composition and Stable −6.25 pp", "No-Go"),
    ("CrystalREPA", "Unconditional REPA repro", "E-hull +0.09424; RMSD worsened", "No-Go"),
    ("RP-QTFG", "Training-free physics guidance", "Offline direction positive; online RMSD worsened", "No-Go"),
    ("CG-TDR", "Teacher residual correction", "Residual no better than zero; safe gate near no effect", "No-Go"),
    ("Q1 UQ-PQR", "Post-generation quality", "Novel −14.86 pp", "No-Go"),
    ("Q2 RFR", "Force refinement", "Novel −30.25 pp; Unique −8.45 pp", "No-Go"),
    ("Q4 CPRC", "Constrained correction", "Novel −17.08 pp", "No-Go"),
    ("Q5 CQPS", "Quality-preserving selection", "Novel −15.63 pp", "No-Go"),
    ("Q6 NS-SetRank", "Candidate ranking", "Novel −12.50 pp", "No-Go"),
    ("GPU acceleration routes", "Bitwise-safe throughput", "Batch changed quality; compile/static/MPS below gates", "No-Go"),
]


def fig12():
    stem = "fig12_negative_routes_summary"
    frame = pd.DataFrame(NEGATIVE_ROUTES, columns=["method", "goal", "observed_failure", "final_status"])
    frame["source"] = "thesis_archive/EXPERIMENT_LINEAGE.md"
    source_csv(stem, frame)
    with paper_context():
        fig, ax = plt.subplots(figsize=(7.1, 6.4), layout="constrained")
        ax.axis("off")
        table = ax.table(
            cellText=frame[["method", "goal", "observed_failure", "final_status"]].values,
            colLabels=["Route", "Goal", "Observed failure", "Status"],
            colWidths=[0.18, 0.20, 0.48, 0.10],
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.3)
        table.scale(1, 1.55)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(COLORS["light_gray"])
            if r == 0:
                cell.set_facecolor(COLORS["blue"])
                cell.set_text_props(color="white", weight="bold")
            elif c == 3:
                cell.set_facecolor("#FCE8E3")
                cell.set_text_props(color=COLORS["vermillion"], weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#F7F7F7")
        ax.set_title("Representative No-Go routes and the evidence that stopped them", pad=10)
        ax.text(
            0,
            0.01,
            "Source: frozen experiment lineage. Descriptive summary only; no synthetic unified score.",
            transform=ax.transAxes,
            fontsize=6.5,
        )
        save_figure(fig, stem, FIGURE_OUTPUTS)


GENERATORS = {
    "fig05": fig05,
    "fig06": fig06,
    "fig07": fig07,
    "fig08": fig08,
    "fig09": fig09,
    "fig10": fig10,
    "fig11": fig11,
    "fig12": fig12,
}


def main(selected: list[str] | None = None) -> None:
    ensure_output_directories()
    selected = selected or list(GENERATORS)
    for name in selected:
        GENERATORS[name]()
        print(f"generated {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("figures", nargs="*", choices=sorted(GENERATORS))
    args = parser.parse_args()
    main(args.figures or None)
