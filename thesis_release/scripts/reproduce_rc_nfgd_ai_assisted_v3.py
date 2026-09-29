"""AI-brief-assisted, scientifically redrawn RC-NFGD method figure.

The AI concepts supply composition/palette ideas only; all scientific geometry,
arrows, text and process semantics are created here as editable vectors.
"""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon


sys.path.insert(0, "/root/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "figure_pilots" / "rc_nfgd_redesign"
STEM = OUT / "I2_F1_rc_nfgd_ai_assisted_v3"

WIDTH_MM = 138.4
HEIGHT_MM = 70.0
INK = "#263746"
MUTED = "#5F6E7B"
GUIDE = "#CAD4DC"
BLUE = "#1D6087"
TEAL = "#2A817E"
AMBER = "#A36A20"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.5,
        "axes.linewidth": 0.8,
    }
)


def txt(ax, x, y, value, *, size=7.4, color=INK, weight="normal", ha="left"):
    ax.text(x, y, value, fontsize=size, color=color, weight=weight, ha=ha, va="center", zorder=8)


def arr(ax, start, end, *, color=INK, lw=0.9, ms=6.5):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


ATOM_POSITIONS = [(-6.0, -3.4), (2.8, -4.0), (5.6, 2.9), (-3.1, 4.0), (0.1, 0.0)]


def unit_cell(ax, cx, cy, *, moved=False):
    corners = [(cx - 10, cy - 8), (cx + 7, cy - 8), (cx + 10, cy + 8), (cx - 7, cy + 8)]
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=MUTED, linewidth=0.9, zorder=2))
    for i, (dx, dy) in enumerate(ATOM_POSITIONS):
        x, y = cx + dx, cy + dy
        if moved and i == 1:
            ax.add_patch(Circle((x, y), 1.05, facecolor="white", edgecolor=GUIDE, linewidth=0.85, zorder=3))
            arr(ax, (x + 0.8, y + 0.5), (x + 3.2, y + 1.7), color=BLUE, lw=1.0, ms=5.8)
            x, y = x + 3.4, y + 1.8
        ax.add_patch(
            Circle(
                (x, y),
                1.05,
                facecolor=BLUE if moved and i == 1 else "white",
                edgecolor=BLUE if moved and i == 1 else MUTED,
                linewidth=0.9,
                zorder=4,
            )
        )


def evaluator(ax, cx, cy):
    layers = [
        [(cx - 9, cy - 4), (cx - 9, cy + 4)],
        [(cx, cy - 5), (cx, cy), (cx, cy + 5)],
        [(cx + 9, cy - 4), (cx + 9, cy + 4)],
    ]
    for left, right in zip(layers[:-1], layers[1:]):
        for p in left:
            for q in right:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=GUIDE, lw=0.55, zorder=1)
    for index, layer in enumerate(layers):
        for x, y in layer:
            ax.add_patch(
                Circle(
                    (x, y),
                    1.35,
                    facecolor=TEAL if index == 1 else "white",
                    edgecolor=TEAL,
                    linewidth=0.85,
                    zorder=3,
                )
            )


def draw():
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_MM)
    ax.set_ylim(0, HEIGHT_MM)
    ax.axis("off")

    # Context timeline. A visible break makes the late interval schematic,
    # rather than falsely presenting its width as 20/1000 of this drawn line.
    txt(ax, 6, 64.5, "MatterGen reverse diffusion", size=8.2, weight="bold")
    txt(ax, 132, 64.5, "late predictor · t ≤ 0.02", size=7.4, color=BLUE, ha="right")
    ax.plot([7, 91], [58.1, 58.1], color=GUIDE, lw=1.7, solid_capstyle="round")
    ax.plot([99, 131], [58.1, 58.1], color=BLUE, lw=1.9, solid_capstyle="round")
    ax.plot([93.2, 94.8], [56.7, 59.5], color=MUTED, lw=0.8)
    ax.plot([96.0, 97.6], [56.7, 59.5], color=MUTED, lw=0.8)
    txt(ax, 7, 53.7, "t = 1", size=6.9, color=MUTED)
    txt(ax, 99, 53.7, "t = 0.02", size=6.9, color=BLUE)
    txt(ax, 131, 53.7, "ε", size=7.0, color=BLUE, ha="right")
    txt(ax, 7, 47.9, "One eligible predictor event", size=7.2, color=MUTED)
    ax.plot([7, 131], [45.6, 45.6], color=GUIDE, lw=0.65)

    # AI-assisted motif selection, re-created as deterministic geometry.
    txt(ax, 12, 41.2, "01  Reconstruct", size=7.45, color=BLUE, weight="bold")
    txt(ax, 54, 41.2, "02  Evaluate force", size=7.45, color=TEAL, weight="bold")
    txt(ax, 97, 41.2, "03  Inject score", size=7.45, color=BLUE, weight="bold")
    unit_cell(ax, 24, 29.4)
    evaluator(ax, 68, 29.4)
    unit_cell(ax, 112, 29.4, moved=True)
    arr(ax, (38.8, 29.4), (52.0, 29.4), color=INK)
    arr(ax, (82.0, 29.4), (96.0, 29.4), color=INK)
    txt(ax, 24, 18.2, "predicted clean crystal", size=7.15, ha="center")
    txt(ax, 68, 18.2, "frozen MatterSim-5M", size=7.15, ha="center")
    txt(ax, 112, 18.2, "position predictor", size=7.15, ha="center")

    ax.plot([7, 131], [13.8, 13.8], color=GUIDE, lw=0.65)
    txt(ax, 7, 10.0, "center forces  →  bounded Cartesian δr  →  fractional δf", size=6.85, color=INK)
    txt(ax, 131, 10.0, "position score only", size=6.85, color=BLUE, ha="right")
    ax.plot([7, 131], [5.9, 5.9], color=GUIDE, lw=0.55)
    txt(ax, 7, 2.8, "invalid feedback → original score", size=6.6, color=AMBER)
    txt(ax, 131, 2.8, "atomic + cell scores and corrector unchanged", size=6.5, color=MUTED, ha="right")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(STEM) + ".alignment.json",
        overlay_svg=str(STEM) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(str(STEM) + ".pdf", facecolor="white")
    fig.savefig(str(STEM) + ".svg", facecolor="white")
    fig.savefig(str(STEM) + ".png", dpi=600, facecolor="white")
    fig.savefig(
        str(STEM) + ".tiff",
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    draw()
