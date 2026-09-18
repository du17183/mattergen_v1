#!/usr/bin/env python3
"""Render the two frozen thesis research lineages as PNG/PDF/SVG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_archive"

COLORS = {
    "SUPPORTED": "#2e7d32",
    "MIXED": "#ed6c02",
    "FAIL": "#c62828",
    "NOT_SUPPORTED": "#6a1b9a",
    "INCONCLUSIVE": "#546e7a",
}

I1 = [
    ("Adaptive\nCFG V1", "MIXED"),
    ("Robust\nV2", "FAIL"),
    ("Oracle\nV3", "MIXED"),
    ("Risk\nV4", "FAIL"),
    ("Safe\nV5", "FAIL"),
    ("Stage /\nField", "FAIL"),
    ("Branch\nOracle", "SUPPORTED"),
    ("Reference\nBranching", "SUPPORTED"),
    ("Phase B\nLinear", "NOT_SUPPORTED"),
    ("C1: Fixed + /\nLinear -", "MIXED"),
    ("Fixed-K2\nFinal", "SUPPORTED"),
]

I2 = [
    ("Initial\nForce", "INCONCLUSIVE"),
    ("Reliability", "SUPPORTED"),
    ("Clean-x0", "SUPPORTED"),
    ("Coordinate\nMapping", "SUPPORTED"),
    ("P0", "SUPPORTED"),
    ("Formal32", "SUPPORTED"),
    ("Formal256", "SUPPORTED"),
    ("CHGNet", "SUPPORTED"),
    ("Direction", "SUPPORTED"),
    ("Trust /\nBounding", "NOT_SUPPORTED"),
    ("RC-NFGD\nFinal", "SUPPORTED"),
]


def draw_row(ax: plt.Axes, items: list[tuple[str, str]], y: float, title: str) -> None:
    xs = list(range(len(items)))
    for index in range(len(items) - 1):
        ax.add_patch(
            FancyArrowPatch(
                (xs[index] + 0.38, y),
                (xs[index + 1] - 0.38, y),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#607d8b",
                zorder=1,
            )
        )
    for x, (label, status) in zip(xs, items):
        color = COLORS[status]
        box = FancyBboxPatch(
            (x - 0.36, y - 0.22),
            0.72,
            0.44,
            boxstyle="round,pad=0.025,rounding_size=0.05",
            linewidth=1.7,
            edgecolor=color,
            facecolor="white",
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.015, label, ha="center", va="center", fontsize=7.8, color="#182026", zorder=3)
        ax.text(x, y - 0.30, status.replace("_", " "), ha="center", va="top", fontsize=6.8, color=color, weight="bold")
    ax.text(-0.75, y, title, ha="right", va="center", fontsize=11, weight="bold", color="#263238")


def main() -> int:
    fig, ax = plt.subplots(figsize=(16, 4.8))
    draw_row(ax, I1, 1.25, "Innovation 1")
    draw_row(ax, I2, 0.25, "Innovation 2")
    ax.set_xlim(-1.7, max(len(I1), len(I2)) - 0.35)
    ax.set_ylim(-0.45, 1.85)
    ax.axis("off")
    ax.set_title("MatterGen Thesis Research Evolution", fontsize=16, weight="bold", pad=16)
    fig.text(
        0.5,
        0.015,
        "Historical route statuses are preserved; final claims: Fixed-K2 SUPPORTED, Linear-K2 NOT SUPPORTED, RC-NFGD SUPPORTED, DFT_VERIFIED=false.",
        ha="center",
        fontsize=8.5,
        color="#455a64",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.99, 0.96))
    for suffix in ("png", "pdf", "svg"):
        output = OUT / f"research_evolution.{suffix}"
        fig.savefig(output, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        if suffix == "svg":
            # Matplotlib writes a space before many SVG path newlines. The
            # whitespace is semantically irrelevant but fails `git diff --check`.
            text = output.read_text(encoding="utf-8")
            output.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
