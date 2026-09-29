"""Independent, manuscript-grounded RC-NFGD method schematic.

This is a non-overwriting thesis figure pilot. No earlier figure asset or
plotting source is loaded. Run with the established Python figure environment.
"""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon


SKILL_SCRIPTS = Path("/root/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "figure_pilots" / "rc_nfgd_redesign"
STEM = OUT / "I2_F1_rc_nfgd_skill_redesign"

WIDTH_MM = 138.4
HEIGHT_MM = 91.0

INK = "#202833"
MUTED = "#66727D"
PALE = "#DCE2E6"
BLUE = "#165E88"
BLUE_SOFT = "#DCECF4"
TEAL = "#2E7D78"
TEAL_SOFT = "#E2F1ED"
AMBER = "#A36B1F"
AMBER_SOFT = "#F8EDDB"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.5,
        "axes.linewidth": 0.8,
        "text.color": INK,
    }
)


def label(ax, x, y, text, *, size=7.5, color=INK, weight="normal", ha="left"):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        weight=weight,
        ha=ha,
        va="center",
        zorder=10,
    )


def arrow(ax, start, end, *, color=INK, lw=1.0, ms=7, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            linestyle=style,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def crystal(ax, cx, cy):
    """Schematic periodic cell: geometry illustration, not data."""
    cell = [(cx - 11, cy - 8), (cx + 8, cy - 8), (cx + 11, cy + 8), (cx - 8, cy + 8)]
    ax.add_patch(Polygon(cell, fill=False, edgecolor=MUTED, linewidth=0.85, zorder=3))
    atoms = [(cx - 6.4, cy - 3.8), (cx + 2.7, cy - 4.0), (cx - 2.2, cy + 3.8), (cx + 7.1, cy + 4.5)]
    for i, (x, y) in enumerate(atoms):
        ax.add_patch(
            Circle(
                (x, y),
                radius=1.75 if i == 0 else 1.35,
                facecolor=BLUE if i == 0 else "white",
                edgecolor=BLUE,
                linewidth=0.85,
                zorder=4,
            )
        )


def force_motif(ax, cx, cy):
    atoms = [(cx - 7, cy - 3), (cx + 6, cy - 4), (cx - 1, cy + 5)]
    vecs = [(-2.8, -1.5), (3.0, -1.1), (-0.2, 2.6)]
    for (x, y), (dx, dy) in zip(atoms, vecs):
        ax.add_patch(Circle((x, y), 1.3, facecolor="white", edgecolor=TEAL, linewidth=0.9, zorder=4))
        arrow(ax, (x + 1.4 * dx / 3, y + 1.4 * dy / 3), (x + dx, y + dy), color=TEAL, lw=1.25, ms=6)
    ax.add_patch(Arc((cx, cy), 23, 20, theta1=10, theta2=165, color=PALE, linewidth=0.8))


def draw():
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_MM)
    ax.set_ylim(0, HEIGHT_MM)
    ax.axis("off")

    # A calibrated window, not a uniformly applied force field.
    label(ax, 6, 85.0, "RC-NFGD  |  reliability-gated force feedback", size=9.0, weight="bold")
    label(ax, 6, 78.4, "1,000-step reverse diffusion", size=7.1, color=MUTED)
    label(ax, 61, 78.4, "window enlarged for clarity", size=6.2, color=MUTED)
    ax.plot([7, 131], [71.0, 71.0], color=PALE, lw=2.1, solid_capstyle="round")
    ax.plot([111.3, 131], [71.0, 71.0], color=BLUE, lw=2.8, solid_capstyle="round")
    ax.plot([111.3, 111.3], [68.3, 73.7], color=BLUE, lw=0.85)
    label(ax, 7, 66.1, "t = 1", size=7.0, color=MUTED)
    label(ax, 111.3, 66.1, "t = 0.02", size=7.3, color=BLUE, ha="center", weight="bold")
    label(ax, 131, 66.1, "ε", size=7.4, color=BLUE, ha="right")
    label(ax, 131, 77.0, "last 20 predictors only", size=7.1, color=BLUE, ha="right")

    # One eligible predictor event: three transformations, one continuous path.
    ax.plot([6, 132], [62.8, 62.8], color=PALE, lw=0.85)
    for x in (47.2, 88.4):
        ax.plot([x, x], [22.8, 61.0], color=PALE, lw=0.8)
    label(ax, 7, 58.9, "01   CLEAN STRUCTURE", size=7.15, color=BLUE, weight="bold")
    label(ax, 49.7, 58.9, "02   FORCE FEEDBACK", size=7.15, color=TEAL, weight="bold")
    label(ax, 90.8, 58.9, "03   SCORE UPDATE", size=7.15, color=BLUE, weight="bold")

    crystal(ax, 25.2, 42.0)
    label(ax, 26.0, 27.4, "predicted clean crystal", size=7.2, ha="center")
    label(ax, 26.0, 23.9, "from current MatterGen state", size=6.6, color=MUTED, ha="center")

    force_motif(ax, 66.8, 43.4)
    label(ax, 67.8, 27.4, "frozen MatterSim-5M", size=7.2, ha="center")
    label(ax, 67.8, 23.9, "center forces → bounded δr", size=6.6, color=MUTED, ha="center")

    label(ax, 91.5, 49.7, "Cartesian → fractional", size=7.0, color=MUTED)
    label(ax, 91.5, 44.5, "δf = δr · inv(Ĥ₀)", size=8.7, weight="bold")
    ax.plot([91.5, 128.8], [40.3, 40.3], color=PALE, lw=0.75)
    label(ax, 91.5, 36.2, "position predictor only", size=7.0, color=MUTED)
    label(ax, 91.5, 31.1, "s̃ₜᶠ = sₜᶠ + δf / cₜᶠ", size=8.7, weight="bold")
    label(ax, 91.5, 24.1, "atomic / cell / corrector unchanged", size=6.45, color=MUTED)

    arrow(ax, (41.3, 43.1), (49.3, 43.1), color=INK, ms=7)
    arrow(ax, (82.2, 43.1), (90.2, 43.1), color=INK, ms=7)

    # The fallback is a guardrail, not a second optimization branch.
    ax.plot([6, 132], [19.0, 19.0], color=PALE, lw=0.85)
    ax.add_patch(Circle((9.0, 11.8), 2.1, facecolor=AMBER_SOFT, edgecolor=AMBER, linewidth=0.8))
    label(ax, 14.0, 12.9, "ACCEPT finite output, valid cell / score coefficient, and distance ≥ 0.5 Å", size=6.7, color=INK)
    label(ax, 14.0, 8.5, "Otherwise keep the original MatterGen score; no feedback is applied.", size=6.7, color=AMBER)
    label(ax, 132, 3.2, "η = 0.005 Å · hard cap = 0.01 Å · CFG = 2.0", size=6.0, color=MUTED, ha="right")

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
