#!/usr/bin/env python3
"""Render a non-overwriting RC-NFGD method-figure pilot.

The figure is a schematic of the frozen Chapter 4 method. It performs no
model inference, statistical calculation, or experiment, and does not replace
the currently referenced thesis figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure_pilots" / "rc_nfgd"
STEM = "I2_F1_rc_nfgd_pipeline_pilot_v2"

INK = "#23313D"
MUTED = "#526373"
NEUTRAL = "#657584"
NEUTRAL_EDGE = "#BFCAD4"
NEUTRAL_BG = "#F5F7F9"
TEAL = "#087F78"
TEAL_EDGE = "#95C8C0"
TEAL_BG = "#F0F8F6"
ORANGE = "#AD6830"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
    }
)


def box(ax, x, y, w, h, *, edge, fill, title, detail, accent=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=1.5",
        linewidth=0.95,
        edgecolor=edge,
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(patch)
    if accent:
        ax.plot([x + 1.2, x + 1.2], [y + 2.5, y + h - 2.5],
                color=accent, linewidth=2.5, solid_capstyle="round", zorder=4)
    text_x = x + w / 2 + (0.8 if accent else 0)
    ax.text(text_x, y + h * 0.65, title, ha="center", va="center",
            fontsize=8.55, fontweight="bold", color=INK, zorder=5)
    ax.text(text_x, y + h * 0.28, detail, ha="center", va="center",
            fontsize=7.55, color=MUTED, zorder=5)


def arrow(ax, xy_a, xy_b, *, color, width=1.25, style="-", mutation=9):
    ax.add_patch(
        FancyArrowPatch(
            xy_a, xy_b, arrowstyle="-|>", mutation_scale=mutation,
            linewidth=width, color=color, linestyle=style,
            connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=0, zorder=2,
        )
    )


def render() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.45, 3.16))  # ≈138 mm: USTB A4 text width
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Baseline and intervention are separate visual lanes. Only the position
    # predictor receives the accepted feedback; other fields remain untouched.
    ax.add_patch(FancyBboxPatch((2, 53), 96, 39,
                               boxstyle="round,pad=0.003,rounding_size=2.5",
                               linewidth=0.7, edgecolor="#DAE2E9",
                               facecolor=NEUTRAL_BG, zorder=0))
    ax.add_patch(FancyBboxPatch((2, 11), 96, 38,
                               boxstyle="round,pad=0.003,rounding_size=2.5",
                               linewidth=0.7, edgecolor="#CFE4DF",
                               facecolor=TEAL_BG, zorder=0))

    ax.text(5.5, 86.5, "BASE SAMPLER", fontsize=7.8, fontweight="bold",
            color=NEUTRAL, va="center")
    ax.text(31, 43.5, "RC-NFGD  /  late-stage feedback", fontsize=7.8,
            fontweight="bold", color=TEAL, va="center")

    box(ax, 8, 62, 29, 18, edge=NEUTRAL_EDGE, fill="white",
        title="MatterGen", detail=r"current $x_t$  →  scores $s_t$")
    box(ax, 69, 62, 23, 18, edge=NEUTRAL_EDGE, fill="white",
        title="Position predictor", detail=r"next $f_{t-\Delta t}$")
    arrow(ax, (37.1, 71), (68.8, 71), color=NEUTRAL, width=1.5)
    ax.text(53, 75.5, "original position score", ha="center", va="bottom",
            fontsize=7.25, color=MUTED)

    box(ax, 8, 18, 25, 17, edge=TEAL_EDGE, fill="white",
        title="Clean estimate", detail=r"periodic $\hat{x}_0$", accent=TEAL)
    box(ax, 39, 18, 23, 17, edge=TEAL_EDGE, fill="white",
        title="MatterSim", detail="centered forces", accent=TEAL)
    box(ax, 68, 18, 24, 17, edge=TEAL_EDGE, fill="white",
        title="Bound + map", detail=r"$\delta r$  →  $\delta f$", accent=TEAL)
    arrow(ax, (33.2, 26.5), (38.7, 26.5), color=TEAL)
    arrow(ax, (62.2, 26.5), (67.7, 26.5), color=TEAL)

    # The lower route branches from the frozen model and rejoins only at the
    # predictor. Failed guidance leaves its original score unchanged.
    arrow(ax, (22.5, 61.8), (22.5, 35.2), color=TEAL, width=1.2)
    arrow(ax, (80, 35.2), (80, 61.8), color=TEAL, width=1.2)
    ax.text(76, 56.3, "check → score injection", ha="right", va="center",
            fontsize=7.25, fontweight="bold", color=TEAL)

    ax.plot([5.5, 94.5], [7.8, 7.8], color="#D9E1E6", lw=0.7)
    ax.text(5.5, 4.2,
            r"$t=0.020\ldots0.001$ (20 events)  ·  atomic/cell/corrector unchanged",
            fontsize=7.25, color=MUTED, ha="left", va="center")
    ax.text(94.5, 4.2, "failure → original score", fontsize=7.25,
            color=ORANGE, ha="right", va="center")

    pdf = OUT / f"{STEM}.pdf"
    svg = OUT / f"{STEM}.svg"
    png = OUT / f"{STEM}.png"
    tiff = OUT / f"{STEM}.tiff"
    fig.savefig(pdf, bbox_inches=None, pad_inches=0,
                metadata={"Creator": "MatterGen thesis figure pilot"})
    fig.savefig(svg, bbox_inches=None, pad_inches=0)
    fig.savefig(png, dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(tiff, dpi=600, bbox_inches=None, pad_inches=0,
                pil_kwargs={"compression": "tiff_lzw"})
    for target in (pdf, svg, png, tiff):
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    render()
