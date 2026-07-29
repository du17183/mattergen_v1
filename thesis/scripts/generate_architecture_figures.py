#!/usr/bin/env python3
"""Render four editable architecture/lineage schematics from repository data.

The corresponding Graphviz DOT files are the editable semantic sources.  This
portable CPU renderer mirrors those nodes and edges with Matplotlib, so a
system Graphviz executable is not required on the target laptop.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "figures" / "source" / "styles"))

from common.paths import FIGURE_OUTPUTS, FIGURE_SOURCE_DATA, ensure_output_directories
from paper_style import COLORS, paper_context, save_figure


def box(ax, x, y, w, h, text, color, *, linestyle="-", fontsize=7):
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1,
        edgecolor=COLORS["black"],
        facecolor=color,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def arrow(ax, start, end, *, color=None, linestyle="-", connectionstyle="arc3"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.1,
            "color": color or COLORS["black"],
            "linestyle": linestyle,
            "connectionstyle": connectionstyle,
        },
        zorder=1,
    )


def write_source(stem: str, rows: list[dict]) -> None:
    path = FIGURE_SOURCE_DATA / f"{stem}.csv"
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fig01():
    stem = "fig01_full_method_architecture"
    with paper_context():
        fig, ax = plt.subplots(figsize=(11.6, 4.1), layout="constrained")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.add_patch(patches.Rectangle((0.015, 0.49), 0.56, 0.46, fill=False, lw=1.4, ls="--"))
        ax.text(0.03, 0.91, "Innovation 1 · sampling stage", weight="bold", color=COLORS["blue"])
        ax.add_patch(patches.Rectangle((0.59, 0.14), 0.395, 0.81, fill=False, lw=1.4, ls="--"))
        ax.text(0.605, 0.91, "Innovation 2 · post-generation stage", weight="bold", color=COLORS["green"])
        nodes = [
            (0.03, 0.63, 0.12, 0.15, "Target\n$dft\\_mag\\_density$", COLORS["yellow"]),
            (0.18, 0.63, 0.15, 0.15, "MatterGen\nPredictor/Corrector", COLORS["sky"]),
            (0.36, 0.63, 0.11, 0.15, "Adaptive\nCFG", COLORS["orange"]),
            (0.50, 0.63, 0.11, 0.15, "Generated\ncrystal", "#F7F7F7"),
            (0.64, 0.63, 0.10, 0.15, "Learned\nGate", COLORS["purple"]),
            (0.78, 0.74, 0.09, 0.12, "Gate-on", COLORS["green"]),
            (0.78, 0.52, 0.09, 0.12, "Gate-off", "#F7F7F7"),
            (0.89, 0.74, 0.09, 0.12, "E3-PCR", COLORS["green"]),
            (0.89, 0.52, 0.09, 0.12, "Exact\nfallback", "#F7F7F7"),
            (0.70, 0.28, 0.13, 0.13, "Final crystal", COLORS["sky"]),
            (0.86, 0.28, 0.12, 0.13, "MatterSim-5M\nsurrogate", COLORS["yellow"]),
        ]
        for n in nodes:
            box(ax, *n)
        for a, b in [
            ((0.15, 0.705), (0.18, 0.705)),
            ((0.33, 0.705), (0.36, 0.705)),
            ((0.47, 0.705), (0.50, 0.705)),
            ((0.61, 0.705), (0.64, 0.705)),
            ((0.74, 0.705), (0.78, 0.80)),
            ((0.74, 0.68), (0.78, 0.58)),
            ((0.87, 0.80), (0.89, 0.80)),
            ((0.87, 0.58), (0.89, 0.58)),
            ((0.935, 0.74), (0.79, 0.41)),
            ((0.935, 0.52), (0.77, 0.41)),
            ((0.83, 0.345), (0.86, 0.345)),
        ]:
            arrow(ax, a, b)
        ax.text(0.92, 0.17, "Evaluation only · no DFT verification", ha="center", fontsize=7)
        save_figure(fig, stem, FIGURE_OUTPUTS)
    write_source(stem, [{"node": n[4].replace("\n", " "), "category": "pipeline"} for n in nodes])


def fig02():
    stem = "fig02_adaptive_cfg_mechanism"
    with paper_context():
        fig, ax = plt.subplots(figsize=(9.5, 4.2), layout="constrained")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        box(ax, 0.03, 0.68, 0.15, 0.14, "Conditional\nbranch", COLORS["sky"])
        box(ax, 0.03, 0.35, 0.15, 0.14, "Unconditional\nbranch", "#F7F7F7")
        box(ax, 0.23, 0.52, 0.16, 0.15, "Three-field residuals\ncell · position · atom", COLORS["yellow"])
        box(ax, 0.44, 0.52, 0.12, 0.15, "EMA\nsmoothing", COLORS["orange"])
        box(ax, 0.61, 0.52, 0.16, 0.15, "Residual-driven\nscale update", COLORS["orange"])
        box(ax, 0.81, 0.52, 0.14, 0.15, "Clamp [0, 5]\n+ CFG fusion", COLORS["green"])
        box(ax, 0.68, 0.20, 0.20, 0.14, "Continue full\nPredictor + Corrector", COLORS["sky"])
        for a, b in [
            ((0.18, 0.75), (0.23, 0.61)),
            ((0.18, 0.42), (0.23, 0.57)),
            ((0.39, 0.595), (0.44, 0.595)),
            ((0.56, 0.595), (0.61, 0.595)),
            ((0.77, 0.595), (0.81, 0.595)),
            ((0.88, 0.52), (0.79, 0.34)),
        ]:
            arrow(ax, a, b)
        ax.text(
            0.5,
            0.07,
            "No Predictor skip · No Corrector skip · Not Corrector Gating",
            ha="center",
            weight="bold",
            color=COLORS["vermillion"],
        )
        save_figure(fig, stem, FIGURE_OUTPUTS)
    write_source(
        stem,
        [
            {"stage": i + 1, "component": name}
            for i, name in enumerate(
                ["conditional", "unconditional", "three-field residuals", "EMA", "scale update", "clamp", "CFG fusion"]
            )
        ],
    )


def fig03():
    stem = "fig03_e3pcr_mechanism"
    with paper_context():
        fig, ax = plt.subplots(figsize=(10.4, 5.1), layout="constrained")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        box(ax, 0.03, 0.72, 0.12, 0.13, "Generated\ncrystal", COLORS["sky"])
        box(ax, 0.19, 0.72, 0.14, 0.13, "14-dimensional\nrisk features", COLORS["yellow"])
        box(ax, 0.37, 0.72, 0.13, 0.13, "129-parameter\nGate", COLORS["purple"])
        box(ax, 0.54, 0.72, 0.13, 0.13, "Confidence\nthreshold 0.5", COLORS["purple"])
        box(ax, 0.73, 0.75, 0.10, 0.11, "Gate-on", COLORS["green"])
        box(ax, 0.73, 0.50, 0.10, 0.11, "Gate-off", "#F7F7F7")
        box(ax, 0.03, 0.28, 0.13, 0.13, "5-step equivariant\nrefinement", COLORS["green"])
        box(ax, 0.20, 0.28, 0.13, 0.13, "Per-step radius", COLORS["green"])
        box(ax, 0.37, 0.28, 0.13, 0.13, "Cumulative\ntrust region", COLORS["green"])
        box(ax, 0.54, 0.28, 0.11, 0.13, "Backtracking", COLORS["orange"])
        box(ax, 0.69, 0.28, 0.12, 0.13, "Safety checks", COLORS["orange"])
        box(ax, 0.85, 0.28, 0.12, 0.13, "Accepted refined\nstructure", COLORS["sky"])
        box(ax, 0.85, 0.50, 0.12, 0.11, "Exact fallback", "#F7F7F7")
        for a, b in [
            ((0.15, 0.785), (0.19, 0.785)),
            ((0.33, 0.785), (0.37, 0.785)),
            ((0.50, 0.785), (0.54, 0.785)),
            ((0.67, 0.785), (0.73, 0.805)),
            ((0.67, 0.755), (0.73, 0.555)),
            ((0.78, 0.75), (0.095, 0.41)),
            ((0.83, 0.555), (0.85, 0.555)),
            ((0.16, 0.345), (0.20, 0.345)),
            ((0.33, 0.345), (0.37, 0.345)),
            ((0.50, 0.345), (0.54, 0.345)),
            ((0.65, 0.345), (0.69, 0.345)),
            ((0.81, 0.345), (0.85, 0.345)),
            ((0.75, 0.28), (0.91, 0.61)),
        ]:
            arrow(ax, a, b)
        ax.text(
            0.5,
            0.10,
            "Positions only · Atomic species unchanged · Cell unchanged",
            ha="center",
            weight="bold",
            color=COLORS["blue"],
        )
        save_figure(fig, stem, FIGURE_OUTPUTS)
    write_source(
        stem,
        [
            {"parameter": "gate input", "value": "14 dimensions"},
            {"parameter": "trainable gate", "value": "129 parameters"},
            {"parameter": "threshold", "value": "0.5"},
            {"parameter": "refinement", "value": "5 steps"},
            {"parameter": "fields changed", "value": "positions only"},
        ],
    )


def fig04():
    stem = "fig04_experiment_lineage"
    rows = [
        ("Adaptive CFG formal256", "Formal", "20000–20255", COLORS["blue"], "-"),
        ("E3-PCR formal256", "Formal independent", "40000–40255", COLORS["green"], "-"),
        ("Compatibility cohort 1", "Independent replication", "41000–41063", COLORS["sky"], "--"),
        ("Independent cohort 2", "Independent replication", "50000–50063", COLORS["sky"], "--"),
        ("Leakage diagnostic", "Diagnostic only", "20000–20255", COLORS["orange"], ":"),
        ("Mixed 256 cohort", "INVALID for independent claims", "20000–20255", "#F7F7F7", ":"),
    ]
    with paper_context():
        fig, ax = plt.subplots(figsize=(9.4, 5.4), layout="constrained")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        box(ax, 0.05, 0.43, 0.14, 0.14, "C0\noriginal MatterGen", COLORS["yellow"])
        ys = [0.83, 0.68, 0.53, 0.38, 0.23, 0.08]
        for y, (name, kind, seeds, color, style) in zip(ys, rows):
            box(ax, 0.32, y, 0.28, 0.10, name, color, linestyle=style)
            box(ax, 0.68, y, 0.25, 0.10, f"{kind}\nseeds {seeds}", "#FFFFFF", linestyle=style, fontsize=6.5)
            arrow(ax, (0.19, 0.50), (0.32, y + 0.05), linestyle=style)
            arrow(ax, (0.60, y + 0.05), (0.68, y + 0.05), linestyle=style)
        ax.text(
            0.80,
            0.02,
            "Line style + text label encode evidence class; color is supplementary.",
            ha="center",
            fontsize=6.5,
        )
        save_figure(fig, stem, FIGURE_OUTPUTS)
    write_source(
        stem,
        [
            {"experiment": name, "evidence_class": kind, "seed_range": seeds}
            for name, kind, seeds, _, _ in rows
        ],
    )


GENERATORS = {
    "fig01": fig01,
    "fig02": fig02,
    "fig03": fig03,
    "fig04": fig04,
}


def main(selected: list[str] | None = None) -> None:
    ensure_output_directories()
    selected = selected or list(GENERATORS)
    for name in selected:
        GENERATORS[name]()
        print(f"generated {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("figures", nargs="*")
    args = parser.parse_args()
    unknown = sorted(set(args.figures) - set(GENERATORS))
    if unknown:
        parser.error(f"unknown figure keys: {unknown}")
    main(args.figures or None)
