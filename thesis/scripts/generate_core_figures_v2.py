#!/usr/bin/env python3
"""Redraw the seven main-text thesis figures with a cleaner publication layout.

This module intentionally leaves every source-data CSV unchanged.  It is called
after the legacy generators, so it only replaces the visual exports for Figures
1, 2, 3, 5, 6, 7, and 9.  The remaining figures keep their reproducible draft
layouts for supplementary or appendix use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "figures" / "source" / "styles"))

from common.paths import ARCHIVE_ROOT, FIGURE_OUTPUTS, FIGURE_SOURCE_DATA, ensure_output_directories
from paper_style import COLORS, clean_axes, panel_label, paper_context, save_figure


RNG = np.random.default_rng(20260730)
NEUTRAL = "#F4F5F7"
PALE_BLUE = "#EAF4FA"
PALE_GREEN = "#E8F5F0"
PALE_ORANGE = "#FFF3D9"
PALE_PURPLE = "#F6EAF2"
PALE_RED = "#FCEBE6"


def _rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str = NEUTRAL,
    edgecolor: str = COLORS["black"],
    fontsize: float = 7.2,
    linewidth: float = 0.9,
    linestyle: str = "-",
    weight: str = "normal",
    zorder: int = 3,
) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        zorder=zorder + 1,
    )


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["black"],
    linestyle: str = "-",
    connectionstyle: str = "arc3",
    linewidth: float = 1.0,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linestyle": linestyle,
            "linewidth": linewidth,
            "connectionstyle": connectionstyle,
            "shrinkA": 1,
            "shrinkB": 1,
        },
        zorder=2,
    )


def _stage_band(ax, x: float, y: float, w: float, h: float, label: str, color: str) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
    )
    ax.text(x + 0.12, y + h - 0.22, label, va="top", fontsize=7.2, weight="bold")


def _diamond(ax, cx: float, cy: float, w: float, h: float, text: str, color: str) -> None:
    vertices = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(
        patches.Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor=COLORS["black"],
            linewidth=0.9,
            zorder=3,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=6.7, zorder=4)


def fig01() -> None:
    """Main-text overview: method stages and strict evaluation boundary."""

    stem = "fig01_full_method_architecture"
    with paper_context():
        fig, ax = plt.subplots(figsize=(7.1, 3.35), layout="constrained")
        ax.set(xlim=(0, 12.1), ylim=(0, 5.0))
        ax.axis("off")

        _stage_band(ax, 0.15, 1.45, 5.85, 3.25, "Innovation 1 · sampling-stage control", PALE_BLUE)
        _stage_band(ax, 6.15, 1.45, 5.75, 3.25, "Innovation 2 · safe post-generation refinement", PALE_GREEN)

        _rounded_box(ax, 0.42, 2.45, 1.20, 0.82, "Target\n$dft\\_mag\\_density$", facecolor=PALE_ORANGE)
        _rounded_box(ax, 1.95, 2.45, 1.55, 0.82, "MatterGen\nfull P/C sampler", facecolor=PALE_BLUE)
        _rounded_box(ax, 3.82, 2.45, 1.43, 0.82, "Residual-adaptive\nCFG", facecolor=PALE_ORANGE)
        _rounded_box(ax, 5.52, 2.45, 1.28, 0.82, "Generated\ncrystal", facecolor=NEUTRAL)
        _diamond(ax, 7.57, 2.86, 1.15, 1.05, "Learned\nGate", PALE_PURPLE)

        _rounded_box(ax, 8.42, 3.34, 1.28, 0.70, "Gate-on\nE3-PCR", facecolor=PALE_GREEN)
        _rounded_box(
            ax,
            8.42,
            1.72,
            1.28,
            0.70,
            "Gate-off /\nrejected",
            facecolor=NEUTRAL,
            linestyle="--",
        )
        _rounded_box(ax, 10.18, 2.45, 1.42, 0.82, "Final crystal\nspecies/cell fixed", facecolor=PALE_BLUE)
        _rounded_box(
            ax,
            9.65,
            0.25,
            1.95,
            0.72,
            "MatterSim-5M\nsurrogate evaluation",
            facecolor=PALE_ORANGE,
            linestyle="--",
        )

        for start, end in [
            ((1.62, 2.86), (1.95, 2.86)),
            ((3.50, 2.86), (3.82, 2.86)),
            ((5.25, 2.86), (5.52, 2.86)),
            ((6.80, 2.86), (7.00, 2.86)),
            ((8.14, 3.05), (8.42, 3.67)),
            ((8.14, 2.67), (8.42, 2.07)),
            ((9.70, 3.69), (10.18, 3.03)),
            ((9.70, 2.07), (10.18, 2.69)),
        ]:
            _arrow(ax, start, end)
        _arrow(ax, (10.89, 2.45), (10.62, 0.97), linestyle="--", color=COLORS["gray"])

        ax.text(9.03, 4.20, "5 bounded position steps", ha="center", fontsize=6.7, color=COLORS["green"])
        ax.text(9.03, 1.46, "Exact fallback to input", ha="center", fontsize=6.7, color=COLORS["gray"])
        ax.text(
            0.35,
            0.35,
            "Evaluation boundary: surrogate force, relaxation, E-hull and stability; no DFT verification",
            fontsize=6.7,
            color=COLORS["vermillion"],
        )
        save_figure(fig, stem, FIGURE_OUTPUTS)


def fig02() -> None:
    """Adaptive CFG: show the implemented scalar controller precisely."""

    stem = "fig02_adaptive_cfg_mechanism"
    with paper_context():
        fig, ax = plt.subplots(figsize=(7.1, 3.75), layout="constrained")
        ax.set(xlim=(0, 12.0), ylim=(0, 5.3))
        ax.axis("off")

        _rounded_box(ax, 0.25, 3.65, 1.45, 0.72, "Conditional\nscore $s^{cond}$", facecolor=PALE_BLUE)
        _rounded_box(ax, 0.25, 2.22, 1.45, 0.72, "Unconditional\nscore $s^{uncond}$", facecolor=NEUTRAL)

        residual_y = [4.05, 3.03, 2.01]
        residual_labels = [
            r"cell: $\delta_c=\operatorname{RMS}(r_c)$",
            r"position: $\delta_x=\operatorname{RMS}(r_x)$",
            r"atom: $\delta_a=\operatorname{RMS}(r_a)$",
        ]
        for y, label in zip(residual_y, residual_labels):
            _rounded_box(ax, 2.20, y - 0.34, 1.95, 0.68, label, facecolor=PALE_ORANGE, fontsize=6.6)
            _arrow(ax, (1.70, 3.99), (2.20, y + 0.08), connectionstyle="arc3,rad=-0.08")
            _arrow(ax, (1.70, 2.58), (2.20, y - 0.08), connectionstyle="arc3,rad=0.08")

        _rounded_box(ax, 4.65, 2.69, 1.35, 0.82, "Mean valid\nfield RMS $\\delta_t$", facecolor=PALE_ORANGE)
        for y in residual_y:
            _arrow(ax, (4.15, y), (4.65, 3.10), connectionstyle="arc3,rad=0.05")

        _rounded_box(ax, 6.50, 3.48, 1.55, 0.78, "Phase-specific EMA\n$m_{t,p}$", facecolor=PALE_PURPLE)
        _rounded_box(ax, 6.50, 2.04, 1.55, 0.78, "Ratio\n$q_t=\\delta_t/(m_{t,p}+\\epsilon)$", facecolor=PALE_PURPLE, fontsize=6.4)
        _arrow(ax, (6.00, 3.10), (6.50, 3.82))
        _arrow(ax, (7.28, 3.48), (7.28, 2.82))

        _rounded_box(
            ax,
            8.58,
            2.69,
            1.42,
            0.82,
            "Adaptive multiplier\nclip to [0.25, 4]",
            facecolor=PALE_ORANGE,
            fontsize=6.6,
        )
        _rounded_box(
            ax,
            10.48,
            2.69,
            1.20,
            0.82,
            "CFG scale\nclip to [0, 5]",
            facecolor=PALE_GREEN,
            fontsize=6.6,
        )
        _arrow(ax, (8.05, 2.43), (8.58, 3.10))
        _arrow(ax, (10.00, 3.10), (10.48, 3.10))

        ax.add_patch(
            patches.FancyBboxPatch(
                (0.28, 0.42),
                11.40,
                0.92,
                boxstyle="round,pad=0.03,rounding_size=0.06",
                facecolor=PALE_BLUE,
                edgecolor=COLORS["blue"],
                linewidth=0.8,
            )
        )
        ax.text(
            5.98,
            1.02,
            r"$m_{t,p}=\beta m_{t-1,p}+(1-\beta)\delta_t,\quad "
            r"u_t=\mathrm{clip}\!\left(1+\alpha(q_t-1),0.25,4\right),\quad "
            r"g_t=\mathrm{clip}(g_0u_t,0,5)$",
            ha="center",
            va="center",
            fontsize=7.2,
        )
        ax.text(
            5.98,
            0.66,
            r"$g_0=2.0,\ \alpha=0.50,\ \beta=0.95,\ \epsilon=10^{-6}$"
            "  ·  predictor and corrector use separate EMA states",
            ha="center",
            va="center",
            fontsize=6.7,
        )
        ax.text(
            5.98,
            4.93,
            "CFG fusion continues through the complete Predictor + Corrector process; no sampling step is removed",
            ha="center",
            fontsize=7.2,
            weight="bold",
            color=COLORS["blue"],
        )
        save_figure(fig, stem, FIGURE_OUTPUTS)


def fig03() -> None:
    """Learned Gate and safe-bounded refiner with an explicit rejection path."""

    stem = "fig03_e3pcr_mechanism"
    with paper_context():
        fig, ax = plt.subplots(figsize=(7.1, 4.35), layout="constrained")
        ax.set(xlim=(0, 12.2), ylim=(0, 6.2))
        ax.axis("off")

        _stage_band(ax, 0.15, 1.00, 3.55, 4.82, "1 · Risk estimation", PALE_PURPLE)
        _stage_band(ax, 3.90, 1.00, 5.00, 4.82, "2 · Bounded equivariant refinement", PALE_GREEN)
        _stage_band(ax, 9.10, 1.00, 2.90, 4.82, "3 · Safety decision", PALE_BLUE)

        _rounded_box(
            ax,
            0.42,
            3.28,
            1.62,
            1.35,
            "14 frozen features\nsize/density (3)\ngeometry/composition (4)\nCHGNet E/F (4)\nstress/magnetism (3)",
            facecolor=PALE_ORANGE,
            fontsize=6.25,
        )
        _rounded_box(
            ax,
            2.30,
            3.47,
            1.05,
            0.96,
            "MLP\n14 → 8 → 1\n129 params",
            facecolor=PALE_PURPLE,
            fontsize=6.5,
        )
        _diamond(ax, 4.38, 3.95, 1.10, 1.15, "$c\\geq0.5$?", PALE_PURPLE)
        _arrow(ax, (2.04, 3.95), (2.30, 3.95))
        _arrow(ax, (3.35, 3.95), (3.83, 3.95))

        _rounded_box(ax, 5.15, 4.45, 1.15, 0.72, "CHGNet forces", facecolor=PALE_GREEN)
        _rounded_box(ax, 6.62, 4.45, 1.38, 0.72, "Position proposal\n$\\eta=0.01$", facecolor=PALE_GREEN)
        _rounded_box(ax, 5.15, 2.94, 1.15, 0.82, "Per-step cap\n0.02 Å", facecolor=PALE_GREEN)
        _rounded_box(ax, 6.62, 2.94, 1.38, 0.82, "Backtrack\n1, 1/2, 1/4", facecolor=PALE_ORANGE)
        _arrow(ax, (4.93, 4.27), (5.15, 4.81))
        _arrow(ax, (6.30, 4.81), (6.62, 4.81))
        _arrow(ax, (7.31, 4.45), (5.73, 3.76), connectionstyle="arc3,rad=-0.15")
        _arrow(ax, (6.30, 3.35), (6.62, 3.35))
        ax.add_patch(
            patches.FancyArrowPatch(
                (7.95, 3.20),
                (7.78, 4.50),
                connectionstyle="arc3,rad=0.45",
                arrowstyle="-|>",
                linewidth=0.9,
                color=COLORS["green"],
            )
        )
        ax.text(7.95, 4.08, "repeat ×5", fontsize=6.5, color=COLORS["green"], rotation=90, va="center")

        _diamond(ax, 9.72, 3.95, 1.28, 1.25, "finite?\n$d_{min}\\geq0.5$ Å?\nenergy ↓?", PALE_ORANGE)
        _rounded_box(ax, 10.55, 4.58, 1.15, 0.72, "Accept refined\npositions", facecolor=PALE_GREEN)
        _rounded_box(
            ax,
            10.55,
            2.30,
            1.15,
            0.72,
            "Exact fallback\nto input",
            facecolor=NEUTRAL,
            linestyle="--",
        )
        _arrow(ax, (8.00, 3.35), (9.08, 3.78))
        _arrow(ax, (10.36, 4.20), (10.55, 4.94))
        _arrow(ax, (10.08, 3.45), (10.55, 2.66))
        _arrow(ax, (4.38, 3.38), (10.55, 2.66), linestyle="--", color=COLORS["gray"], connectionstyle="arc3,rad=0.18")

        ax.text(4.90, 4.68, "on", fontsize=6.5, color=COLORS["green"], weight="bold")
        ax.text(6.80, 2.14, "reject after ≤3 trials", fontsize=6.4, color=COLORS["vermillion"])
        ax.text(6.02, 1.36, "Cumulative wrapped displacement ≤ 0.10 Å", ha="center", fontsize=7.0, weight="bold")
        ax.text(
            6.10,
            0.40,
            "Atomic species unchanged  ·  cell unchanged  ·  inference does not update CHGNet or Gate parameters",
            ha="center",
            fontsize=6.8,
            color=COLORS["blue"],
        )
        save_figure(fig, stem, FIGURE_OUTPUTS)


def _load_json(relative: str) -> dict:
    with (ARCHIVE_ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def fig05() -> None:
    """Adaptive CFG effects: raw paired E-hull values plus binary effect intervals."""

    stem = "fig05_adaptive_cfg_results"
    data = pd.read_csv(FIGURE_SOURCE_DATA / f"{stem}.csv")
    report = _load_json("reports/innovation1/formal_final_report.json")
    stats_map = {row["metric"]: row for row in report["innovation1"]["paired_statistics"]}

    with paper_context():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.1, 3.25),
            gridspec_kw={"width_ratios": [1.08, 1.0]},
            layout="constrained",
        )

        ax = axes[0]
        values = data.ehull_difference_ev_atom.to_numpy(float)
        parts = ax.violinplot(values, positions=[0], widths=0.62, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(COLORS["sky"])
            body.set_edgecolor(COLORS["blue"])
            body.set_alpha(0.35)
        jitter = RNG.normal(0, 0.075, len(values))
        ax.scatter(jitter, values, s=7, alpha=0.22, color=COLORS["blue"], edgecolors="none", rasterized=False)
        row = stats_map["energy_above_hull_per_atom"]
        mean = float(values.mean())
        lo, hi = float(row["bootstrap_95_ci_low"]), float(row["bootstrap_95_ci_high"])
        ax.errorbar(0, mean, yerr=[[mean - lo], [hi - mean]], fmt="D", color=COLORS["black"], capsize=4, zorder=5)
        ax.axhline(0, color=COLORS["vermillion"], linestyle="--", linewidth=0.9)
        ax.set(
            xlim=(-0.48, 0.48),
            xticks=[0],
            xticklabels=["A0 − C0"],
            ylabel=r"Paired $\Delta E_\mathrm{hull}$ (eV atom$^{-1}$)",
        )
        clean_axes(ax)
        panel_label(ax, "A")
        ax.set_title("Per-seed E-hull effects")
        ax.text(
            0.03,
            0.97,
            f"mean {mean:+.4f}\n95% CI [{lo:+.4f}, {hi:+.4f}]\nHolm p=1.00",
            transform=ax.transAxes,
            va="top",
            fontsize=6.8,
        )

        ax = axes[1]
        labels = ["Stable", "NUS"]
        metric_keys = ["stable", "novel_unique_stable"]
        means = np.array([data.stable_difference.mean(), data.nus_difference.mean()]) * 100
        lows = np.array([stats_map[key]["bootstrap_95_ci_low"] for key in metric_keys]) * 100
        highs = np.array([stats_map[key]["bootstrap_95_ci_high"] for key in metric_keys]) * 100
        y = np.array([1, 0])
        ax.errorbar(
            means,
            y,
            xerr=[means - lows, highs - means],
            fmt="D",
            color=COLORS["blue"],
            ecolor=COLORS["black"],
            capsize=4,
            markersize=5,
        )
        ax.axvline(0, color=COLORS["vermillion"], linestyle="--", linewidth=0.9)
        ax.set(yticks=y, yticklabels=labels, xlabel="Paired rate change (percentage points)", ylim=(-0.65, 1.65))
        clean_axes(ax, "x")
        panel_label(ax, "B")
        ax.set_title("Binary quality effects")
        for yi, mean_value, lo_value, hi_value in zip(y, means, lows, highs):
            ax.text(
                0.98,
                yi,
                f"{mean_value:+.2f} pp  [{lo_value:+.2f}, {hi_value:+.2f}]",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=6.6,
            )
        ax.text(0.98, 0.04, "All Holm-adjusted p=1.00", transform=ax.transAxes, ha="right", fontsize=6.6)
        fig.suptitle(
            "Adaptive CFG formal evaluation · n=256 · directional trends, not statistically significant",
            fontsize=9,
        )
        save_figure(fig, stem, FIGURE_OUTPUTS)


def fig06() -> None:
    """E3-PCR formal result: full distributions and paired effect ordering."""

    stem = "fig06_e3pcr_force_formal256"
    data = pd.read_csv(FIGURE_SOURCE_DATA / f"{stem}.csv")
    stats_frame = pd.read_csv(ARCHIVE_ROOT / "reports/innovation2/formal_paired_statistics.csv")
    row = stats_frame[
        (stats_frame.metric == "pre_relax_max_force_ev_ang")
        & (stats_frame.comparison == "E3-G vs C0")
    ].iloc[0]

    with paper_context():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.1, 3.35),
            gridspec_kw={"width_ratios": [1.0, 1.18]},
            layout="constrained",
        )

        ax = axes[0]
        series = [
            ("C0", data.c0_max_force.to_numpy(float), COLORS["gray"]),
            ("E3-A", data.e3a_max_force.to_numpy(float), COLORS["orange"]),
            ("E3-G", data.e3g_max_force.to_numpy(float), COLORS["green"]),
        ]
        for label, values, color in series:
            ordered = np.sort(values)
            cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
            ax.step(ordered, cumulative, where="post", label=label, color=color, linewidth=1.35)
        ax.set_xscale("log")
        ax.set(xlabel=r"Pre-relax max force (eV $\AA^{-1}$; log scale)", ylabel="Empirical cumulative fraction")
        clean_axes(ax)
        panel_label(ax, "A")
        ax.legend(frameon=False, loc="lower right")
        ax.set_title("Three-arm force distributions")

        ax = axes[1]
        diff = (data.e3g_max_force - data.c0_max_force).to_numpy(float)
        ordered = np.sort(diff)
        x = np.arange(len(ordered))
        colors = np.where(ordered <= 0, COLORS["green"], COLORS["vermillion"])
        ax.scatter(x, ordered, c=colors, s=8, alpha=0.72, edgecolors="none")
        ax.axhline(0, color=COLORS["black"], linestyle="--", linewidth=0.9)
        mean = float(row.mean_difference)
        lo, hi = float(row.bootstrap_95_ci_low), float(row.bootstrap_95_ci_high)
        ax.axhspan(lo, hi, color=COLORS["sky"], alpha=0.18)
        ax.axhline(mean, color=COLORS["blue"], linewidth=1.2)
        ax.set(
            xlabel="Paired seeds sorted by E3-G effect",
            ylabel=r"$\Delta F_\mathrm{max}$: E3-G − C0 (eV $\AA^{-1}$)",
        )
        clean_axes(ax)
        panel_label(ax, "B")
        ax.set_title("Paired learned-gated effects")
        ax.text(
            0.03,
            0.97,
            "mean −0.0799 eV Å⁻¹ (−23.28%)\n"
            "95% CI [−0.1450, −0.0325]\n"
            "Holm p=4.19×10⁻¹⁰ · W/T/L=163/0/93",
            transform=ax.transAxes,
            va="top",
            fontsize=6.7,
        )
        fig.suptitle("E3-PCR independent formal evaluation · seeds 40000–40255 · n=256", fontsize=9)
        save_figure(fig, stem, FIGURE_OUTPUTS)


def fig07() -> None:
    """Gate safety ablation using direction-aware dumbbells instead of dense bars."""

    stem = "fig07_gate_safety_ablation"
    data = pd.read_csv(FIGURE_SOURCE_DATA / f"{stem}.csv")
    rate = data[(data.unit == "fraction") & (data.metric != "Gain retention")].reset_index(drop=True)
    displacement = data[data.unit == "angstrom"].reset_index(drop=True)
    retention = float(data.loc[data.metric == "Gain retention", "learned_gated"].iloc[0])

    with paper_context():
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(7.1, 3.15),
            gridspec_kw={"width_ratios": [1.45, 1.0, 0.72]},
            layout="constrained",
        )

        ax = axes[0]
        y = np.arange(len(rate))[::-1]
        always = rate.always_on.to_numpy(float) * 100
        gated = rate.learned_gated.to_numpy(float) * 100
        for yi, left, right in zip(y, always, gated):
            ax.plot([left, right], [yi, yi], color=COLORS["light_gray"], linewidth=2.2, zorder=1)
        ax.scatter(always, y, color=COLORS["orange"], marker="s", s=28, label="Always-on", zorder=3)
        ax.scatter(gated, y, color=COLORS["green"], marker="o", s=28, label="Learned-gated", zorder=3)
        ax.set(
            yticks=y,
            yticklabels=["Refinement coverage", "Overall harm", "Low-force harm"],
            xlabel="Rate (%)",
            xlim=(0, 105),
            ylim=(-0.65, 2.65),
        )
        clean_axes(ax, "x")
        panel_label(ax, "A")
        ax.set_title("Selective intervention")
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=6.5)
        ax.text(0.02, 0.02, "Harm McNemar p=0.000534", transform=ax.transAxes, fontsize=6.4)

        ax = axes[1]
        y2 = np.arange(len(displacement))[::-1]
        a2 = displacement.always_on.to_numpy(float)
        g2 = displacement.learned_gated.to_numpy(float)
        for yi, left, right in zip(y2, a2, g2):
            ax.plot([left, right], [yi, yi], color=COLORS["light_gray"], linewidth=2.2)
        ax.scatter(a2, y2, color=COLORS["orange"], marker="s", s=28)
        ax.scatter(g2, y2, color=COLORS["green"], marker="o", s=28)
        ax.set(
            yticks=y2,
            yticklabels=["Mean", "P95"],
            xlabel=r"Wrapped displacement ($\AA$)",
            ylim=(-0.65, 1.65),
        )
        clean_axes(ax, "x")
        panel_label(ax, "B")
        ax.set_title("Smaller movement")
        ax.text(
            0.02,
            0.02,
            "Summary values only;\nno per-seed CI available",
            transform=ax.transAxes,
            fontsize=6.2,
        )

        ax = axes[2]
        ax.barh([0], [100], color=COLORS["light_gray"], height=0.28)
        ax.barh([0], [retention * 100], color=COLORS["green"], height=0.28)
        ax.scatter([retention * 100], [0], color=COLORS["black"], marker="D", s=24, zorder=3)
        ax.set(xlim=(0, 105), yticks=[], xlabel="Retained mean force gain (%)", ylim=(-0.65, 0.65))
        clean_axes(ax, "x")
        panel_label(ax, "C")
        ax.set_title("Benefit retained")
        ax.text(retention * 100, 0.18, f"{retention*100:.2f}%", ha="center", fontsize=7.0)

        fig.suptitle("Learned Gate safety mechanism · n=256 · less coverage and harm", fontsize=9)
        save_figure(fig, stem, FIGURE_OUTPUTS)


def fig09() -> None:
    """Classic two-cohort forest layout with no pooled effect."""

    stem = "fig09_combination_replication_forest"
    data = pd.read_csv(FIGURE_SOURCE_DATA / f"{stem}.csv")
    y = np.arange(len(data))[::-1]

    with paper_context():
        fig = plt.figure(figsize=(7.1, 2.45), layout="constrained")
        grid = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.70, 2.30])
        left = fig.add_subplot(grid[0, 0])
        ax = fig.add_subplot(grid[0, 1])
        right = fig.add_subplot(grid[0, 2])

        left.axis("off")
        right.axis("off")
        for yi, row in zip(y, data.itertuples(index=False)):
            left.text(0.00, yi, row.cohort, va="center", fontsize=8, weight="bold")
            left.text(0.00, yi - 0.25, f"seeds {row.seed_range} · n={int(row.n)}", va="center", fontsize=6.5)

        means = data.mean_difference.to_numpy(float)
        lows = data.ci_low.to_numpy(float)
        highs = data.ci_high.to_numpy(float)
        ax.errorbar(
            means,
            y,
            xerr=[means - lows, highs - means],
            fmt="D",
            color=COLORS["blue"],
            ecolor=COLORS["black"],
            capsize=4,
            markersize=5,
        )
        ax.axvline(0, color=COLORS["vermillion"], linestyle="--", linewidth=0.9)
        ax.axvspan(-0.14, 0, color=COLORS["green"], alpha=0.045)
        ax.set(
            yticks=[],
            xlabel=r"Paired mean difference: A0+E3-G − A0 (eV $\AA^{-1}$)",
            ylim=(-0.65, 1.65),
            xlim=(min(-0.13, lows.min() - 0.01), 0.018),
        )
        clean_axes(ax, "x")
        ax.text(0.05, 0.95, "favours A0+E3-G", transform=ax.transAxes, va="top", fontsize=6.3, color=COLORS["green"])

        for yi, row in zip(y, data.itertuples(index=False)):
            right.text(
                0.00,
                yi + 0.11,
                f"{row.mean_difference:+.4f}  [{row.ci_low:+.4f}, {row.ci_high:+.4f}] eV Å⁻¹",
                va="center",
                fontsize=6.8,
            )
            right.text(
                0.00,
                yi - 0.18,
                f"{row.relative_change*100:.2f}% · p={row.p_value:.3g}",
                va="center",
                fontsize=6.5,
                color=COLORS["gray"],
            )
        fig.suptitle("Two independent combination cohorts · reported side by side, never pooled", fontsize=9)
        save_figure(fig, stem, FIGURE_OUTPUTS)


GENERATORS = {
    "fig01": fig01,
    "fig02": fig02,
    "fig03": fig03,
    "fig05": fig05,
    "fig06": fig06,
    "fig07": fig07,
    "fig09": fig09,
}


def main(selected: list[str] | None = None) -> None:
    ensure_output_directories()
    selected = selected or list(GENERATORS)
    for key in selected:
        GENERATORS[key]()
        print(f"generated core-v2 {key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("figures", nargs="*")
    args = parser.parse_args()
    unknown = sorted(set(args.figures) - set(GENERATORS))
    if unknown:
        parser.error(f"unknown figure keys: {unknown}")
    main(args.figures or None)
