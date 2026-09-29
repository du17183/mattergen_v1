#!/usr/bin/env python3
"""Rebuild three thesis-integration vector figures from frozen definitions/data.

No experiments, statistics, or source CSVs are modified by this script.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "thesis_integration"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})
INK = "#23364A"
BLUE = "#315D93"
PURPLE = "#71558E"
TEAL = "#247A77"
GRAY = "#6F7882"
LIGHT = "#EFF3F7"
ORANGE = "#AF693B"


def export(fig, stem):
    fig.savefig(OUT / (stem + ".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / (stem + ".svg"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / (stem + ".png"), dpi=600,
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def box(ax, xy, wh, label, edge=INK, face="white", fs=8, lw=1.0):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.008,rounding_size=0.012",
                           linewidth=lw, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, label, ha="center", va="center",
            color=INK, fontsize=fs, linespacing=1.2)
    return patch


def arrow(ax, a, b, color=GRAY, lw=1.1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, color=color,
                                 shrinkA=1, shrinkB=1))


def canvas(width=7.2, height=3.0):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_framework():
    fig, ax = canvas(7.35, 3.0)
    ax.text(.02, .96, "Budget-constrained verifier-guided inference",
            ha="left", va="top", weight="bold", fontsize=10, color=INK)
    box(ax, (.02, .39), (.17, .19), "Frozen model\n+ condition", BLUE, LIGHT)
    box(ax, (.245, .39), (.17, .19), "Shared prefix\nsteps 0–399", BLUE, LIGHT)
    box(ax, (.465, .39), (.15, .19), "Saved state\n+ RNG", BLUE, LIGHT)
    arrow(ax, (.19, .485), (.245, .485), BLUE)
    arrow(ax, (.415, .485), (.465, .485), BLUE)
    branch_y = (.72, .45, .18)
    names = ("C0 exact\nreference", "GPulse\nsuffix", "PPulse\nsuffix")
    for y, name in zip(branch_y, names):
        box(ax, (.665, y), (.13, .16), name, BLUE if y != .72 else GRAY,
            "#EDF3FA" if y != .72 else "#F2F3F4", 7.8)
        arrow(ax, (.615, .485), (.665, y+.08), BLUE)
        arrow(ax, (.795, y+.08), (.84, .485), GRAY)
    box(ax, (.84, .395), (.14, .18), "Frozen SAFE-A\nverifier", TEAL, "#EAF5F2", 7.8)
    ax.text(.91, .31, "Accept valid gain\nor exact C0 fallback",
            ha="center", va="center", color=INK, fontsize=7.7)
    ax.text(.02, .045,
            "Fixed-K2: 4,400 score calls (2.2× C0); 3 terminal evaluations. "
            "Budget is not wall time.",
            fontsize=7.5, color=GRAY, va="bottom")
    export(fig, "budget_verifier_framework")


def draw_best_mechanism():
    fig, ax = canvas(7.35, 3.2)
    ax.text(.02, .96, "Same terminal rule; different candidate sources and costs",
            ha="left", va="top", weight="bold", fontsize=10, color=INK)
    ax.text(.025, .78, "Best-of-2", weight="bold", color=PURPLE,
            ha="left", va="center", fontsize=8)
    box(ax, (.18, .75), (.20, .10), "C0 full trajectory", PURPLE, "#F2EDF6", 7.8)
    box(ax, (.18, .59), (.20, .10), "Sample 2 (full)", PURPLE, "#F2EDF6", 7.4)
    arrow(ax, (.38, .80), (.51, .72), PURPLE)
    arrow(ax, (.38, .64), (.51, .72), PURPLE)
    box(ax, (.52, .66), (.17, .12), "SAFE-A selection\n+ C0 fallback", TEAL, "#EAF5F2", 7.7)
    ax.text(.72, .72, "4,000 score calls\n2 terminal evaluations\n(2.0× C0)",
            ha="left", va="center", fontsize=7.6, color=INK)
    ax.plot([.02, .97], [.50, .50], color="#D2D8DE", lw=.8)
    ax.text(.025, .30, "Fixed-K2", weight="bold", color=BLUE,
            ha="left", va="center", fontsize=8)
    box(ax, (.18, .26), (.15, .12), "Shared prefix\n+ saved RNG", BLUE, "#EDF3FA", 7.7)
    arrow(ax, (.33, .32), (.39, .32), BLUE)
    for y, label in zip((.40, .27, .14),
                        ("C0 suffix", "GPulse suffix", "PPulse suffix")):
        box(ax, (.40, y), (.16, .10), label, BLUE, "#EDF3FA", 7.4)
        arrow(ax, (.56, y+.05), (.62, .32), BLUE)
    box(ax, (.63, .26), (.15, .12), "SAFE-A selection\n+ C0 fallback", TEAL, "#EAF5F2", 7.4)
    ax.text(.80, .32, "4,400 score calls\n3 terminal evaluations\n(2.2× C0)",
            ha="left", va="center", fontsize=7.6, color=INK)
    ax.text(.02, .02, "Mechanism diagram only; no performance ordering is implied.",
            fontsize=7.5, color=GRAY, va="bottom")
    export(fig, "best2_vs_fixedk2_mechanism")


def draw_timing():
    path = ROOT / "results" / "rcnfgd" / "timing_ablation" / "timing_paired_bootstrap.csv"
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    arms = ("Early", "Middle", "Late")
    observed = {}
    for row in rows:
        if row["baseline"] == "C0" and row["candidate"] in arms and row["metric"] == "max_force":
            observed[row["candidate"]] = row
    assert set(observed) == set(arms)
    expected = {"Early": (-.0212681031, -.1291808278, .0859120720),
                "Middle": (-.0491472182, -.14368, .02438),
                "Late": (-.0512252763, -.0683399682, -.0376978543)}
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    yloc = {"Early": 2, "Middle": 1, "Late": 0}
    colors = {"Early": "#75A7A2", "Middle": "#4A928D", "Late": TEAL}
    costs = {"Early": "931.5 calls", "Middle": "500.0 calls",
             "Late": "20.0 calls"}
    for name in arms:
        row = observed[name]
        mean = float(row["candidate_minus_baseline"])
        lo = float(row["paired_ci95_low"])
        hi = float(row["paired_ci95_high"])
        assert abs(mean-expected[name][0]) < 2e-7
        assert abs(lo-expected[name][1]) < 1e-5
        assert abs(hi-expected[name][2]) < 1e-5
        y = yloc[name]
        ax.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt="o",
                    color=colors[name], ecolor=colors[name],
                    markersize=6, capsize=3.5, elinewidth=1.8, zorder=3)
        ax.text(.105, y, costs[name], color=GRAY, ha="left", va="center",
                fontsize=7.5)
    ax.axvline(0, color=GRAY, linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks([2, 1, 0], ["Early", "Middle", "Late"])
    ax.set_xlim(-.16, .20)
    ax.set_ylim(-.65, 2.7)
    ax.set_xlabel("Paired MaxF change vs C0 (eV/Å)")
    ax.set_title("Force-feedback timing: effect and force-call budget",
                 loc="left", pad=10, color=INK)
    ax.grid(axis="x", color="#E5E9EC", linewidth=.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    export(fig, "timing_paired_maxf")


if __name__ == "__main__":
    draw_framework()
    draw_best_mechanism()
    draw_timing()
