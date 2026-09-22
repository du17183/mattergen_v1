#!/usr/bin/env python3
"""Rebuild the conceptual figures used in Chapters 1--2.

The source diagrams are deliberately kept untouched.  This renderer creates a
print-oriented vector set for the polished LaTeX preview: flat panels, a
restrained palette, consistent typography, and no decorative drop shadows.
There are no experimental inputs or learned parameters in these diagrams.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Circle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "concept_figures_polished"

INK = "#263238"
MUTED = "#667585"
GRID = "#C8D2DC"
BLUE = "#2F6F9F"
BLUE_LIGHT = "#EAF2F8"
GREEN = "#17825F"
GREEN_LIGHT = "#E8F5EF"
AMBER = "#C48700"
AMBER_LIGHT = "#FFF4D8"
RED = "#C45151"
RED_LIGHT = "#FBECEC"
PURPLE = "#6557B8"
PURPLE_LIGHT = "#F0EEFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "mattergen-concept-figures-2026",
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.035}
        if ext == "png":
            kwargs["dpi"] = 320
        if ext == "pdf":
            kwargs["metadata"] = {"Creator": "MatterGen thesis conceptual figures", "CreationDate": None, "ModDate": None}
        elif ext == "png":
            kwargs["metadata"] = {"Software": "MatterGen thesis conceptual figures"}
        fig.savefig(OUT / f"{stem}.{ext}", **kwargs)
        if ext == "svg":
            target = OUT / f"{stem}.svg"
            target.write_text("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def canvas(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, *, face: str = "white", edge: str = GRID, lw: float = 1.2, radius: float = 0.018) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=1,
        )
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = BLUE, lw: float = 1.2, rad: float = 0.0, ms: float = 9) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=3))


def kicker(ax: plt.Axes, text: str) -> None:
    ax.text(0.02, 0.97, text, ha="left", va="top", fontsize=7.2, fontweight="bold", color=MUTED)


def label(ax: plt.Axes, x: float, y: float, text: str, *, size: float = 8.2, weight: str = "normal", color: str = INK, ha: str = "center", va: str = "center") -> None:
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, fontweight=weight, color=color, linespacing=1.05)


def figure1() -> None:
    fig, ax = canvas((6.8, 3.65))
    box(ax, 0.28, 0.79, 0.44, 0.12, face=BLUE_LIGHT, edge=BLUE, lw=1.35)
    label(ax, 0.50, 0.855, "Frozen MatterGen baseline", size=9.0, weight="bold")
    label(ax, 0.50, 0.817, "C0 · atomic species + positions + cell · CFG = 2.0", size=6.2, color=MUTED)
    arrow(ax, (0.50, 0.785), (0.25, 0.725), color=BLUE, lw=1.0)
    arrow(ax, (0.50, 0.785), (0.75, 0.725), color=BLUE, lw=1.0)

    for x, title, subtitle, edge, fill in [
        (0.05, "INNOVATION 1 / ATTRIBUTE CONTROL", "residual diagnostics + terminal selection", GREEN, GREEN_LIGHT),
        (0.55, "INNOVATION 2 / LOCAL CONSISTENCY", "stage reliability + force feedback", AMBER, AMBER_LIGHT),
    ]:
        label(ax, x + 0.20, 0.685, title, size=7.0, weight="bold", color=edge)
        label(ax, x + 0.20, 0.642, subtitle, size=6.0, color=MUTED)
        box(ax, x, 0.50, 0.40, 0.095, face=fill, edge=edge)
        box(ax, x, 0.36, 0.40, 0.095, face=fill, edge=edge)
    label(ax, 0.25, 0.548, "reference-preserved\nshared prefix", size=7.0, weight="bold")
    label(ax, 0.25, 0.408, "Fixed-K2 candidate branches", size=7.0, weight="bold")
    label(ax, 0.75, 0.548, "late-stage MatterSim\nforce feedback", size=7.0, weight="bold")
    label(ax, 0.75, 0.408, "periodic mapping +\nbounded correction", size=7.0, weight="bold")
    arrow(ax, (0.25, 0.50), (0.25, 0.46), color=GREEN, lw=1.0)
    arrow(ax, (0.75, 0.50), (0.75, 0.46), color=AMBER, lw=1.0)
    label(ax, 0.25, 0.32, "terminal constraints → exact C0 fallback", size=6.3, color=MUTED)
    label(ax, 0.75, 0.32, "position-only injection → safety fallback", size=6.3, color=MUTED)
    ax.plot([0.25, 0.25, 0.50], [0.30, 0.20, 0.20], color=BLUE, lw=1.0)
    ax.plot([0.75, 0.75, 0.50], [0.30, 0.20, 0.20], color=BLUE, lw=1.0)
    arrow(ax, (0.50, 0.20), (0.50, 0.162), color=BLUE, lw=1.0)
    box(ax, 0.29, 0.045, 0.42, 0.115, face=PURPLE_LIGHT, edge=PURPLE, lw=1.2)
    label(ax, 0.50, 0.115, "Unified frozen evaluation", size=7.8, weight="bold")
    label(ax, 0.50, 0.074, "paired seeds · MatterSim · CHGNet · surrogate boundaries", size=5.7, color=MUTED)
    ax.set_ylim(0.02, 0.94)
    save(fig, "figure1_1_technical_route")


def figure2_1() -> None:
    fig, ax = canvas((6.8, 3.35))
    # A compact lattice sketch.
    origin = (0.07, 0.23)
    ux, uy = 0.13, 0.0
    vx, vy = 0.065, 0.145
    for i in range(4):
        p = (origin[0] + i * ux, origin[1] + i * uy)
        q = (p[0] + 2 * vx, p[1] + 2 * vy)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=GRID, lw=0.65)
    for j in range(3):
        p = (origin[0] + j * vx, origin[1] + j * vy)
        q = (p[0] + 3 * ux, p[1] + 3 * uy)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=GRID, lw=0.65)
    cell = [(0.20, 0.23), (0.46, 0.23), (0.525, 0.375), (0.265, 0.375)]
    ax.add_patch(Polygon(cell, closed=True, facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.35))
    for x, y, c, r in [(0.23, 0.27, RED, 0.012), (0.49, 0.27, RED, 0.012), (0.295, 0.415, RED, 0.012), (0.37, 0.33, BLUE, 0.010), (0.11, 0.33, BLUE, 0.010), (0.43, 0.475, BLUE, 0.010)]:
        ax.add_patch(Circle((x, y), r, facecolor=c, edgecolor="white", linewidth=0.5, zorder=3))
    arrow(ax, (0.20, 0.18), (0.45, 0.18), color=INK, lw=0.95, ms=7)
    arrow(ax, (0.17, 0.23), (0.235, 0.375), color=INK, lw=0.95, ms=7)
    label(ax, 0.325, 0.135, "cell basis in H", size=6.2, color=MUTED)
    label(ax, 0.145, 0.30, "cell\nbasis", size=6.0, color=MUTED)
    label(ax, 0.34, 0.35, "f", size=8.0, weight="bold", color=BLUE)
    # Equation card.
    box(ax, 0.61, 0.17, 0.33, 0.62, face="#FBFCFD", edge=GRID, lw=1.1)
    label(ax, 0.775, 0.70, "Row-vector convention", size=9.2, weight="bold")
    label(ax, 0.775, 0.57, r"$\mathbf{r}=\mathbf{f}\,\mathbf{H}$", size=14.0, weight="bold", color=BLUE)
    label(ax, 0.775, 0.47, "fractional f: dimensionless", size=6.9, color=MUTED)
    label(ax, 0.775, 0.405, "Cartesian r: angstrom", size=6.9, color=MUTED)
    ax.plot([0.67, 0.88], [0.35, 0.35], color=GRID, lw=0.8)
    label(ax, 0.775, 0.28, r"$\mathbf{f}\sim\mathbf{f}+\mathbf{n}$", size=12.0, weight="bold", color=INK)
    label(ax, 0.775, 0.215, "integer lattice n; wrap to [0, 1)", size=6.2, color=MUTED)
    label(ax, 0.37, 0.055, "atomic species + fractional positions + cell matrix define the periodic structure", size=6.2, color=INK)
    ax.set_ylim(0.02, 0.84)
    save(fig, "fig2_1_crystal_representation")


def figure2_2() -> None:
    fig, ax = canvas((6.8, 3.25))
    states = [(0.07, r"$x_0$", "clean crystal", GREEN_LIGHT, GREEN), (0.38, r"$x_t$", "noisy state", AMBER_LIGHT, AMBER), (0.69, r"$x_T$", "simple prior", BLUE_LIGHT, MUTED)]
    for x, top, bottom, fill, edge in states:
        box(ax, x, 0.43, 0.22, 0.27, face=fill, edge=edge, lw=1.2)
        label(ax, x + 0.11, 0.63, top, size=14.0, weight="bold")
        label(ax, x + 0.11, 0.535, bottom, size=7.2, color=MUTED)
    # atoms in state cards
    for x, cols in [(0.07, [RED, BLUE, AMBER]), (0.38, [RED, BLUE, AMBER]), (0.69, ["#9AAAC0"] * 3)]:
        for j, c in enumerate(cols):
            ax.add_patch(Circle((x + 0.06 + 0.055 * j, 0.48), 0.012 if x != 0.38 or j != 1 else 0.016, facecolor=c, edgecolor="none"))
    arrow(ax, (0.30, 0.60), (0.37, 0.60), color=RED, lw=1.2)
    arrow(ax, (0.61, 0.60), (0.68, 0.60), color=RED, lw=1.2)
    label(ax, 0.335, 0.75, "forward noise", size=6.0, color=RED)
    label(ax, 0.645, 0.75, "forward noise", size=6.0, color=RED)
    arrow(ax, (0.68, 0.47), (0.61, 0.47), color=BLUE, lw=1.2)
    arrow(ax, (0.37, 0.47), (0.30, 0.47), color=BLUE, lw=1.2)
    label(ax, 0.645, 0.385, "score-based denoising", size=5.8, color=BLUE)
    label(ax, 0.335, 0.385, "predictor / corrector", size=5.8, color=BLUE)
    box(ax, 0.16, 0.09, 0.68, 0.19, face=PURPLE_LIGHT, edge="#8582DD", lw=1.0)
    label(ax, 0.50, 0.20, "MatterGen uses field-specific corruption", size=8.1, weight="bold")
    label(ax, 0.50, 0.135, "atomic species: discrete mask   |   positions: wrapped VE   |   cell: lattice VP", size=6.4, color=MUTED)
    ax.set_ylim(0.03, 0.82)
    save(fig, "fig2_2_diffusion_process")


def figure2_3() -> None:
    fig, ax = canvas((6.8, 3.2))
    box(ax, 0.05, 0.61, 0.23, 0.18, face="#F8FAFC", edge=MUTED)
    box(ax, 0.05, 0.28, 0.23, 0.18, face=GREEN_LIGHT, edge=GREEN)
    label(ax, 0.165, 0.735, "null condition", size=7.2, color=MUTED)
    label(ax, 0.165, 0.665, r"$s_{\mathrm{uncond}}$", size=11.0, weight="bold")
    label(ax, 0.165, 0.405, "target condition c", size=7.2, color=GREEN)
    label(ax, 0.165, 0.335, r"$s_{\mathrm{cond}}$", size=11.0, weight="bold")
    box(ax, 0.39, 0.43, 0.25, 0.20, face=AMBER_LIGHT, edge=AMBER)
    label(ax, 0.515, 0.56, "conditional direction", size=7.2)
    label(ax, 0.515, 0.49, r"$\Delta=s_{\mathrm{cond}}-s_{\mathrm{uncond}}$", size=8.6, weight="bold")
    box(ax, 0.75, 0.43, 0.20, 0.20, face=BLUE_LIGHT, edge=BLUE, lw=1.35)
    label(ax, 0.85, 0.56, "guided score", size=7.2, color=BLUE)
    label(ax, 0.85, 0.49, r"$s_{\mathrm{CFG}}$", size=11.0, weight="bold")
    label(ax, 0.85, 0.445, r"$s_{\mathrm{uncond}}+g\Delta$", size=7.0, color=MUTED)
    arrow(ax, (0.28, 0.70), (0.39, 0.55), color=INK, lw=1.0)
    arrow(ax, (0.28, 0.36), (0.39, 0.50), color=INK, lw=1.0)
    arrow(ax, (0.64, 0.53), (0.75, 0.53), color=INK, lw=1.0)
    box(ax, 0.15, 0.08, 0.70, 0.14, face=PURPLE_LIGHT, edge="#8582DD", lw=1.0)
    label(ax, 0.50, 0.15, "g = 0: unconditional   |   g = 1: conditional   |   g > 1: extrapolate along Δ", size=6.6, color=INK)
    ax.set_ylim(0.03, 0.84)
    save(fig, "fig2_3_cfg")


def figure2_4() -> None:
    fig, ax = canvas((6.8, 3.45))
    box(ax, 0.04, 0.24, 0.21, 0.54, face="#F8FAFC", edge=MUTED, lw=1.2)
    box(ax, 0.34, 0.16, 0.32, 0.68, face=PURPLE_LIGHT, edge="#6557E8", lw=1.35)
    box(ax, 0.75, 0.24, 0.21, 0.54, face="#F8FAFC", edge=MUTED, lw=1.2)
    label(ax, 0.145, 0.72, "Noisy state zₜ", size=8.5, weight="bold")
    label(ax, 0.855, 0.72, "Field outputs", size=8.5, weight="bold")
    label(ax, 0.50, 0.75, "GemNetT denoiser", size=9.2, weight="bold", color=PURPLE)
    label(ax, 0.50, 0.675, "periodic crystal graph\n+ geometric message passing", size=5.5, color=MUTED)
    for y, text, edge, fill in [(0.60, "atomic_numbers", RED, RED_LIGHT), (0.46, "pos", BLUE, BLUE_LIGHT), (0.32, "cell H", AMBER, AMBER_LIGHT)]:
        box(ax, 0.07, y, 0.15, 0.085, face=fill, edge=edge, lw=0.9)
        label(ax, 0.145, y + 0.043, text, size=6.8)
    for y, text, edge, fill in [(0.60, "atomic logits", RED, RED_LIGHT), (0.46, "position score", BLUE, BLUE_LIGHT), (0.32, "cell score", AMBER, AMBER_LIGHT)]:
        box(ax, 0.78, y, 0.15, 0.085, face=fill, edge=edge, lw=0.9)
        label(ax, 0.855, y + 0.043, text, size=6.8)
    for y, text in [(0.56, "time embedding t"), (0.42, "property condition c / null"), (0.28, "joint multi-field prediction")]:
        box(ax, 0.39, y, 0.22, 0.075, face="white", edge="#8582DD", lw=0.8)
        label(ax, 0.50, y + 0.038, text, size=6.6)
    arrow(ax, (0.25, 0.51), (0.34, 0.51), color=BLUE, lw=1.0)
    arrow(ax, (0.66, 0.51), (0.75, 0.51), color=BLUE, lw=1.0)
    ax.add_patch(FancyArrowPatch((0.85, 0.23), (0.15, 0.23), arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color=BLUE, connectionstyle="arc3,rad=-0.23"))
    label(ax, 0.50, 0.045, "CFG combination + predictor/corrector update + repeated reverse steps", size=6.6, color=MUTED)
    ax.set_ylim(0.02, 0.87)
    save(fig, "fig2_4_mattergen_overview")


def figure2_5() -> None:
    fig, ax = canvas((6.8, 3.25))
    box(ax, 0.04, 0.38, 0.22, 0.32, face=PURPLE_LIGHT, edge=PURPLE, lw=1.2)
    label(ax, 0.15, 0.61, "Crystal", size=10.0, weight="bold", color=PURPLE)
    label(ax, 0.15, 0.54, r"$(a,\,r,\,H)$", size=9.0, color=INK)
    for x, y in [(0.10, 0.46), (0.16, 0.53), (0.21, 0.45)]:
        ax.add_patch(Circle((x, y), 0.012, facecolor=RED, edgecolor="white", linewidth=0.45))
    box(ax, 0.36, 0.29, 0.28, 0.48, face=GREEN_LIGHT, edge=GREEN, lw=1.25)
    label(ax, 0.50, 0.65, "MLIP", size=10.0, weight="bold", color=GREEN)
    label(ax, 0.50, 0.58, "learned potential E", size=7.2)
    # restrained potential curve
    curve_x = [0.41, 0.45, 0.49, 0.54, 0.59]
    curve_y = [0.43, 0.48, 0.46, 0.42, 0.50]
    ax.plot(curve_x, curve_y, color=GREEN, lw=2.0)
    ax.add_patch(Circle((0.50, 0.46), 0.009, facecolor=RED, edgecolor="none"))
    label(ax, 0.50, 0.34, "surrogate potential-energy surface", size=6.2, color=MUTED)
    box(ax, 0.74, 0.54, 0.22, 0.23, face=AMBER_LIGHT, edge=AMBER, lw=1.2)
    box(ax, 0.74, 0.23, 0.22, 0.23, face=RED_LIGHT, edge=RED, lw=1.2)
    label(ax, 0.85, 0.68, "Energy", size=9.0, weight="bold")
    label(ax, 0.85, 0.60, r"$E(r,H,a)$", size=8.0)
    label(ax, 0.85, 0.37, "Atomic forces", size=9.0, weight="bold")
    label(ax, 0.85, 0.29, r"$F_i=-\nabla_{r_i}E$", size=8.0)
    arrow(ax, (0.26, 0.54), (0.36, 0.54), color=INK, lw=1.0)
    arrow(ax, (0.64, 0.60), (0.74, 0.65), color=INK, lw=1.0)
    arrow(ax, (0.64, 0.45), (0.74, 0.35), color=INK, lw=1.0)
    box(ax, 0.16, 0.055, 0.68, 0.105, face="#F8FAFC", edge=GRID, lw=0.95)
    label(ax, 0.50, 0.118, "Lower surrogate force suggests smaller local residual; it does not prove", size=5.8, color=MUTED)
    label(ax, 0.50, 0.085, "thermodynamic stability or DFT validity.", size=5.8, color=MUTED)
    ax.set_ylim(0.03, 0.83)
    save(fig, "fig2_5_mlip_energy_force")


def main() -> None:
    figure1()
    figure2_1()
    figure2_2()
    figure2_3()
    figure2_4()
    figure2_5()
    print("POLISHED_CONCEPT_FIGURE_REPRODUCTION=PASS")


if __name__ == "__main__":
    main()
