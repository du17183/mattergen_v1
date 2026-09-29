"""RC-NFGD computational graph, intentionally free of AI-art motifs.

This single-panel method schematic encodes only verified signal dependencies.
It is a non-overwriting design candidate for the frozen Chapter 4 method.
"""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


sys.path.insert(0, "/root/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "figure_pilots" / "rc_nfgd_redesign"
STEM = OUT / "I2_F1_rc_nfgd_computational_v6"
W, H = 138.4, 59.0

INK = "#24343E"
MID = "#5D6D76"
HAIR = "#B7C4CA"
PALE = "#DCE4E7"
POS = "#245F89"
FORCE = "#2B7B7D"
CAUTION = "#956A36"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "font.size": 7.0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
})


def text(ax, x, y, value, *, size=7, color=INK, bold=False, ha="left"):
    ax.text(x, y, value, ha=ha, va="center", fontsize=size, color=color,
            weight="bold" if bold else "normal", zorder=9)


def line(ax, a, b, *, color=HAIR, lw=0.72, z=2):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw,
            solid_capstyle="round", zorder=z)


def arrow(ax, a, b, *, color=MID, lw=0.9, ms=5.7, z=6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, shrinkA=0,
                                 shrinkB=0, zorder=z))


def draw():
    fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.axis("off")

    text(ax, 6.8, 54.1, "RC-NFGD · one eligible predictor step", size=8.1,
         bold=True)
    text(ax, 131.6, 54.1, "t ≤ 0.02 · 20 steps in total", size=6.9,
         color=MID, ha="right")
    line(ax, (6.8, 50.8), (131.6, 50.8), color=PALE, lw=0.73)

    # Frozen MatterGen scores split into unchanged and position-only lanes.
    text(ax, 8.4, 40.0, "state zₜ", size=7.3)
    arrow(ax, (23.5, 40.0), (29.0, 40.0), color=MID)
    text(ax, 30.5, 42.4, "MatterGen", size=7.8, bold=True)
    text(ax, 30.5, 38.2, "frozen score model", size=6.65, color=MID)
    line(ax, (53.2, 40.0), (58.1, 40.0), color=MID, lw=0.82)
    line(ax, (58.1, 34.5), (58.1, 45.4), color=MID, lw=0.82)

    text(ax, 62.2, 47.3, "atomic + cell scores", size=6.8, color=MID)
    arrow(ax, (58.1, 44.9), (113.6, 44.9), color=MID, lw=0.78)
    text(ax, 62.2, 36.5, "position score", size=6.9, color=POS, bold=True)
    line(ax, (58.1, 34.5), (100.1, 34.5), color=POS, lw=1.03)
    ax.add_patch(Circle((102.5, 34.5), 2.05, facecolor="white",
                        edgecolor=POS, linewidth=0.9, zorder=5))
    text(ax, 102.5, 34.5, "+", size=8.0, color=POS, ha="center")
    arrow(ax, (104.7, 34.5), (113.6, 34.5), color=POS, lw=1.0)

    line(ax, (116.0, 32.7), (116.0, 46.3), color=INK, lw=1.25)
    text(ax, 120.0, 42.9, "official", size=7.0, bold=True)
    text(ax, 120.0, 38.9, "predictor", size=7.0, bold=True)
    text(ax, 120.0, 34.9, "update", size=6.4, color=MID)

    # A compact, inspectable feedback path. It returns to the position-only
    # addition node; it does not modify atomic/cell scores or the corrector.
    line(ax, (42.0, 36.3), (42.0, 27.0), color=FORCE, lw=0.82)
    arrow(ax, (42.0, 27.0), (42.0, 25.0), color=FORCE, lw=0.82)
    text(ax, 33.5, 22.6, "predicted clean x₀", size=6.95,
         color=INK, ha="center")
    text(ax, 33.5, 18.5, "from state + scores", size=6.25,
         color=MID, ha="center")

    arrow(ax, (48.6, 22.6), (53.1, 22.6), color=FORCE, lw=0.82)
    text(ax, 67.0, 22.6, "MatterSim-5M", size=7.0, bold=True, ha="center")
    text(ax, 67.0, 18.5, "Cartesian forces", size=6.25,
         color=MID, ha="center")
    arrow(ax, (81.0, 22.6), (85.7, 22.6), color=FORCE, lw=0.82)
    text(ax, 103.0, 22.6, "center · bound · map", size=6.95,
         color=INK, ha="center")
    text(ax, 103.0, 18.5, "δr → periodic δf → Δs_pos", size=6.45,
         color=POS, ha="center")
    line(ax, (102.5, 26.7), (102.5, 32.1), color=POS, lw=0.96)
    arrow(ax, (102.5, 32.1), (102.5, 32.45), color=POS, lw=0.96, ms=5.5)

    line(ax, (6.8, 13.4), (131.6, 13.4), color=PALE, lw=0.7)
    text(ax, 7.0, 8.8, "Δs_pos = δf / c_pos", size=7.05, color=POS)
    text(ax, 131.4, 8.8, "actual predictor coefficient", size=6.45,
         color=MID, ha="right")
    text(ax, 7.0, 3.9, "Invalid force, cell or geometry → no injection",
         size=6.45, color=CAUTION)
    text(ax, 131.4, 3.9, "Atomic/cell scores and corrector unchanged",
         size=6.45, color=MID, ha="right")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=str(STEM) + ".alignment.json",
        overlay_svg=str(STEM) + ".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(str(STEM) + ".pdf", facecolor="white")
    fig.savefig(str(STEM) + ".svg", facecolor="white")
    fig.savefig(str(STEM) + ".png", dpi=600, facecolor="white")
    fig.savefig(str(STEM) + ".tiff", dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    draw()
