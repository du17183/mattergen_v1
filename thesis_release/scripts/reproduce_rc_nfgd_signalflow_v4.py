"""RC-NFGD score-bus architecture schematic, a non-overwriting trial.

Draws only the actual computational relationships: frozen MatterGen score
fields, a late predictor-only force branch, position-score insertion, and a
no-injection fallback. No decorative crystal or neural-network icons are used.
"""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


sys.path.insert(0, "/root/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "figure_pilots" / "rc_nfgd_redesign"
STEM = OUT / "I2_F1_rc_nfgd_signalflow_v4"
WIDTH_MM = 138.4
HEIGHT_MM = 61.0

INK = "#25323D"
MUTED = "#596771"
RULE = "#BEC9D0"
PALE = "#F5F7F8"
BLUE = "#276A91"
TEAL = "#33827F"
AMBER = "#9F6A25"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.7,
        "axes.linewidth": 0.8,
    }
)


def put(ax, x, y, text, *, size=6.6, color=INK, weight="normal", ha="left"):
    ax.text(x, y, text, fontsize=size, color=color, weight=weight, ha=ha, va="center", zorder=8)


def arrow(ax, start, end, *, color=INK, width=0.8, scale=6.3):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            color=color,
            linewidth=width,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def line(ax, xs, ys, *, color=RULE, width=0.72, style="-"):
    ax.plot(xs, ys, color=color, lw=width, linestyle=style, zorder=3, solid_capstyle="round")


def draw():
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_MM)
    ax.set_ylim(0, HEIGHT_MM)
    ax.axis("off")

    # Frozen base sampler. The three score fields are explicit signal rails.
    ax.add_patch(Rectangle((6, 36), 126, 22, facecolor=PALE, edgecolor="none", zorder=0))
    put(ax, 8, 55.8, "FROZEN MATTERGEN SAMPLER", size=6.55, weight="bold")
    put(ax, 8, 45.0, "state", size=6.5, color=MUTED)
    put(ax, 8, 41.9, "zₜ", size=7.1)
    arrow(ax, (20.0, 45.2), (27.2, 45.2), color=MUTED)

    ax.add_patch(Rectangle((28, 39.3), 22.5, 13.5, facecolor="white", edgecolor=RULE, linewidth=0.8, zorder=1))
    put(ax, 39.25, 48.1, "MatterGen", size=7.0, weight="bold", ha="center")
    put(ax, 39.25, 43.2, "frozen score net", size=6.2, color=MUTED, ha="center")

    line(ax, [50.5, 57], [45.3, 45.3], color=MUTED)
    line(ax, [57, 57], [40.1, 49.3], color=MUTED)

    for y, label in ((49.3, "atomic score"), (44.7, "cell score")):
        line(ax, [57, 109.2], [y, y], color=RULE)
        arrow(ax, (109.2, y), (111.1, y), color=MUTED, width=0.7, scale=4.8)
        put(ax, 62, y + 1.72, label, size=6.3, color=MUTED)

    y_pos = 40.1
    line(ax, [57, 97.5], [y_pos, y_pos], color=BLUE, width=1.05)
    line(ax, [100.7, 109.2], [y_pos, y_pos], color=BLUE, width=1.05)
    arrow(ax, (109.2, y_pos), (111.1, y_pos), color=BLUE, width=0.95, scale=5.0)
    put(ax, 62, y_pos + 1.72, "position score", size=6.45, color=INK, weight="bold")
    ax.add_patch(Circle((99.1, y_pos), 1.6, facecolor="white", edgecolor=BLUE, linewidth=0.95, zorder=6))
    put(ax, 99.1, y_pos + 0.1, "+", size=6.4, color=BLUE, ha="center")

    ax.add_patch(Rectangle((111.5, 38.7), 18.9, 14.8, facecolor="white", edgecolor=RULE, linewidth=0.8, zorder=1))
    put(ax, 120.95, 48.5, "Predictor", size=6.8, weight="bold", ha="center")
    put(ax, 120.95, 43.2, "official update", size=6.2, color=MUTED, ha="center")

    # Late force branch. Vertical connections enter the position rail only.
    line(ax, [39.2, 39.2, 32.5], [39.3, 31.0, 31.0], color=TEAL, width=0.8)
    arrow(ax, (32.5, 31.0), (32.5, 27.7), color=TEAL, width=0.8, scale=5.8)
    put(ax, 54.0, 32.4, "RC-NFGD  ·  t ≤ 0.02", size=6.35, color=INK)

    stage_x = (32.5, 59.0, 84.0, 112.0)
    stage_title = ("predicted clean x₀", "MatterSim-5M", "center + bound", "score map")
    stage_sub = ("from state + scores", "Cartesian forces", "δr → periodic δf", "Δsₚₒₛ = δf / cₚₒₛ")
    for x, title, detail in zip(stage_x, stage_title, stage_sub):
        put(ax, x, 25.5, title, size=6.55, color=INK, weight="bold", ha="center")
        put(ax, x, 19.4, detail, size=6.1, color=MUTED, ha="center")
    arrow(ax, (45.3, 22.4), (48.2, 22.4), color=TEAL, width=0.82, scale=5.8)
    arrow(ax, (68.9, 22.4), (72.2, 22.4), color=TEAL, width=0.82, scale=5.8)
    arrow(ax, (96.0, 22.4), (99.1, 22.4), color=BLUE, width=0.95, scale=5.8)

    line(ax, [112.0, 112.0, 99.1], [28.3, 30.0, 30.0], color=BLUE, width=0.95)
    arrow(ax, (99.1, 30.0), (99.1, 38.4), color=BLUE, width=0.95, scale=6.0)

    line(ax, [7, 131], [13.4, 13.4], color=RULE, width=0.65)
    put(ax, 8, 8.5, "Guard fails  →  no injection; keep original score", size=6.35, color=AMBER)
    put(ax, 8, 3.8, "Atomic and cell scores, weights, CFG and corrector remain unchanged.", size=6.15, color=MUTED)
    put(ax, 130.5, 3.8, "20 eligible steps", size=6.15, color=MUTED, ha="right")

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
