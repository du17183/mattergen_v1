"""AI-concept-informed, scientifically redrawn RC-NFGD thesis figure.

The visual grammar is inspired by an internal AI concept only. All atom positions,
force glyphs, arrows, labels and score relations are deterministic vector marks.
The crystals are generic schematics, not measured structures or output samples.
"""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


sys.path.insert(0, "/root/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "figure_pilots" / "rc_nfgd_redesign"
STEM = OUT / "I2_F1_rc_nfgd_editorial_v5"
W, H = 138.4, 82.0  # USTB thesis text width, not a journal-width claim.

INK = "#283741"
MUTED = "#60717B"
SOFT = "#A8B6BE"
FAINT = "#D9E2E6"
PALE = "#F5F8F8"
BLUE = "#2A628A"
TEAL = "#2C8589"
AMBER = "#9A6A31"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


def label(ax, x, y, s, *, size=7.2, color=INK, weight="normal", ha="left"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha,
            va="center", zorder=20)


def path(ax, points, *, color=SOFT, lw=0.72, z=2, ls="-"):
    ax.plot([p[0] for p in points], [p[1] for p in points], color=color,
            linewidth=lw, linestyle=ls, solid_capstyle="round", zorder=z)


def arrow(ax, p, q, *, color=INK, lw=0.94, ms=6.4, z=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, color=color, shrinkA=0,
                                 shrinkB=0, zorder=z))


ATOM_FRACTIONS = (
    (0.14, 0.18, 0.19),
    (0.72, 0.20, 0.53),
    (0.45, 0.52, 0.70),
    (0.20, 0.77, 0.54),
    (0.78, 0.72, 0.26),
)

# These five screen-space cues sum to zero. They are conceptual, not forces from
# any generated sample. The actual method centers full Cartesian force vectors.
FORCE_CUES = ((-2.0, 1.0), (1.5, 1.5), (2.0, -1.0), (-1.0, -2.2), (-0.5, 0.7))


def crystal(ax, cx, cy, *, forces=False):
    origin = (cx - 11.0, cy - 10.1)
    u, v, w = (18.6, 0.0), (1.6, 18.6), (4.5, 4.1)

    def proj(a, b, c):
        return (origin[0] + a*u[0] + b*v[0] + c*w[0],
                origin[1] + a*u[1] + b*v[1] + c*w[1])

    # Pale back faces and darker foreground edges create perspective without
    # asserting atom bonds or a particular chemical structure.
    for c in (1, 0):
        corners = [proj(0, 0, c), proj(1, 0, c), proj(1, 1, c), proj(0, 1, c)]
        tone = FAINT if c else SOFT
        for j in range(4):
            path(ax, [corners[j], corners[(j + 1) % 4]], color=tone,
                 lw=0.66 if c else 0.82, z=1 if c else 3)
    for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
        path(ax, [proj(a, b, 0), proj(a, b, 1)], color=SOFT, lw=0.7, z=2)

    for i, abc in sorted(enumerate(ATOM_FRACTIONS), key=lambda item: -item[1][2]):
        x, y = proj(*abc)
        ax.add_patch(Circle((x, y), 1.56, facecolor="#8FA3B0",
                            edgecolor="#657985", linewidth=0.57, zorder=6))
        ax.add_patch(Circle((x - 0.44, y + 0.51), 0.48,
                            facecolor="#E8F0F1", edgecolor="none", zorder=7))
        if forces:
            dx, dy = FORCE_CUES[i]
            arrow(ax, (x + 1.65*dx/2.4, y + 1.65*dy/2.4),
                  (x + dx*1.84, y + dy*1.84),
                  color=TEAL, lw=1.08, ms=5.4, z=9)


def evaluator(ax, cx, cy):
    layers = (
        ((cx-10.4, cy-6.3), (cx-10.4, cy), (cx-10.4, cy+6.3)),
        ((cx, cy-8.5), (cx, cy-2.8), (cx, cy+2.8), (cx, cy+8.5)),
        ((cx+10.4, cy-6.3), (cx+10.4, cy), (cx+10.4, cy+6.3)),
    )
    for left, right in zip(layers[:-1], layers[1:]):
        for j, p in enumerate(left):
            for k, q in enumerate(right):
                if (j + 2*k) % 4 != 3:
                    path(ax, [p, q], color=FAINT, lw=0.64, z=1)
    for layer in layers:
        for x, y in layer:
            ax.add_patch(Circle((x, y), 1.45, facecolor="#EAF1F2",
                                edgecolor=TEAL, linewidth=0.72, zorder=5))


def draw():
    fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # A quiet timeline supplies context; the length is schematic, not scaled.
    label(ax, 7, 76.6, "FROZEN MATTERGEN  /  LATE PREDICTOR", size=7.05,
          color=INK, weight="bold")
    label(ax, 131.4, 76.6, "20 events · t ≤ 0.02", size=7.05,
          color=BLUE, ha="right")
    path(ax, [(7, 72.9), (131.4, 72.9)], color=FAINT, lw=0.7)

    # Three science modules use the AI concept's open composition, but all
    # structures, force vectors and relations are explicitly redrawn.
    crystal(ax, 24.2, 52.4)
    evaluator(ax, 68.9, 53.5)
    crystal(ax, 111.4, 52.4, forces=True)
    arrow(ax, (43.1, 53.5), (49.4, 53.5), color=MUTED, lw=0.87, ms=6.2)
    arrow(ax, (86.9, 53.5), (93.0, 53.5), color=MUTED, lw=0.87, ms=6.2)

    for x, n, title, sub, accent in (
        (24.2, "01", "Predicted clean", "structure x₀(t)", INK),
        (68.9, "02", "MatterSim-5M", "frozen force evaluation", TEAL),
        (111.4, "03", "Centered forces", "Cartesian force vectors", TEAL),
    ):
        label(ax, x, 37.4, n, size=6.45, color=accent, weight="bold", ha="center")
        label(ax, x, 33.1, title, size=7.5, color=INK, weight="bold", ha="center")
        label(ax, x, 28.7, sub, size=6.85, color=MUTED, ha="center")

    path(ax, [(7, 24.6), (131.4, 24.6)], color=FAINT, lw=0.7)
    label(ax, 7, 21.4, "POSITION-ONLY SCORE INJECTION", size=6.65,
          color=BLUE, weight="bold")

    # The score-increment label carries the critical semantics: this is not a
    # direct geometry replacement or a force applied to other score fields.
    label(ax, 7.5, 14.0, "scale + cap", size=7.3, color=INK)
    arrow(ax, (31.7, 14.0), (38.2, 14.0), color=BLUE, lw=0.88)
    label(ax, 39.6, 14.0, "Cartesian δr  →  periodic δf", size=7.3, color=INK)
    arrow(ax, (88.0, 14.0), (94.4, 14.0), color=BLUE, lw=0.88)
    label(ax, 96.1, 14.0, "Δs_pos = δf / c_pos", size=7.5, color=BLUE)

    path(ax, [(7, 7.7), (131.4, 7.7)], color=FAINT, lw=0.62)
    label(ax, 7, 4.0, "Invalid feedback → original score", size=6.65,
          color=AMBER)
    label(ax, 131.4, 4.0, "Atomic/cell scores and corrector unchanged",
          size=6.55, color=MUTED, ha="right")

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
