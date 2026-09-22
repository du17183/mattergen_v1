#!/usr/bin/env python3
"""Create a publication-oriented preview of the frozen thesis figures.

This is deliberately separate from ``reproduce_figures.py``.  It reads only
the frozen CSV/JSON release artifacts, never starts a model or a GPU job, and
writes to ``figures_polished`` so the original artwork remains available for
comparison and rollback.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
I1 = ROOT / "innovation1"
I2 = ROOT / "innovation2"
BOOT_SEED = 20_260_915
N_BOOT = 20_000
EPS = 1.0e-12

# Okabe--Ito-inspired colours with a neutral reference.  Every quantitative
# comparison also uses position, marker, hatch, or line style, so colour is
# never the only carrier of meaning.
COLORS = {
    "C0": "#5B6573",
    "Fixed-K2": "#0072B2",
    "Linear-K2": "#E69F00",
    "Random-K2": "#CC79A7",
    "RC-NFGD": "#009E73",
    "positive": "#0072B2",
    "negative": "#D55E00",
    "neutral": "#9AA3AD",
    "ink": "#263238",
    "grid": "#D7DDE3",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.4,
        "axes.labelsize": 7.6,
        "axes.titlesize": 7.8,
        "legend.fontsize": 6.6,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "mattergen-thesis-polished-2026",
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def bootstrap_draws(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    return values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)


def mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    draws = bootstrap_draws(values, seed)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        target = directory / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.035}
        if suffix == "png":
            kwargs["dpi"] = 320
        if suffix == "pdf":
            kwargs["metadata"] = {"Creator": "MatterGen thesis polished preview", "CreationDate": None, "ModDate": None}
        elif suffix == "png":
            kwargs["metadata"] = {"Software": "MatterGen thesis polished preview"}
        fig.savefig(target, **kwargs)
        if suffix == "svg":
            # Remove insignificant trailing whitespace so generated SVGs remain
            # stable under git diff/checks.
            lines = target.read_text(encoding="utf-8").splitlines()
            target.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


def finish_axes(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.tick_params(direction="out", length=2.8, pad=2)
    ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=0.45, alpha=0.82, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        color=COLORS["ink"],
        fontsize=7.5,
    )


def add_timeline_node(
    ax: plt.Axes,
    x: float,
    y: float,
    title: str,
    status: str,
    color: str,
    number: int,
    *,
    width: float = 0.135,
    height: float = 0.185,
) -> None:
    """Draw a compact, print-safe timeline node with redundant status text."""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="white",
        edgecolor=color,
        linewidth=1.05,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x - width / 2 + 0.012,
        y + height / 2 - 0.014,
        str(number),
        ha="left",
        va="top",
        fontsize=5.6,
        fontweight="bold",
        color=color,
        zorder=4,
    )
    ax.text(x, y + 0.025, title, ha="center", va="center", fontsize=6.2, linespacing=1.0, color=COLORS["ink"], zorder=4)
    ax.text(
        x,
        y - height / 2 + 0.028,
        status,
        ha="center",
        va="center",
        fontsize=5.2,
        fontweight="bold",
        color=color,
        zorder=4,
    )


def connect_nodes(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, width: float = 0.135, height: float = 0.185) -> None:
    """Connect neighbouring timeline nodes without crossing their boxes."""
    x0, y0 = start
    x1, y1 = end
    if abs(y1 - y0) < 0.02:
        direction = 1 if x1 > x0 else -1
        p0 = (x0 + direction * width / 2, y0)
        p1 = (x1 - direction * width / 2, y1)
    else:
        direction = 1 if y1 > y0 else -1
        p0 = (x0, y0 + direction * height / 2)
        p1 = (x1, y1 - direction * height / 2)
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=0.8,
            color="#7B8794",
            shrinkA=2,
            shrinkB=2,
            zorder=2,
        )
    )


def i1_figures() -> None:
    out = I1 / "figures_polished"
    metrics = {row["method"]: row for row in read_csv(I1 / "results/c1_metrics.csv")}
    paired = read_csv(I1 / "results/c1_paired_results.csv")
    methods = ["C0", "Fixed_K2", "Linear_K2", "Random_K2"]
    labels = ["C0", "Fixed-K2", "Linear-K2", "Random-K2"]

    # I1-F1: compact dot-and-whisker plot at the final single-column width.
    fig, ax = plt.subplots(figsize=(3.55, 2.15))
    means, lows, highs = [], [], []
    for index, method in enumerate(methods):
        values = np.asarray([float(row[method]) for row in paired])
        mean, low, high = mean_ci(values, BOOT_SEED + index)
        means.append(mean)
        lows.append(low)
        highs.append(high)
        ax.errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=COLORS[labels[index]],
            ecolor=COLORS[labels[index]],
            markerfacecolor="white" if index else COLORS["C0"],
            markeredgewidth=1.0,
            markersize=4.6,
            capsize=2.2,
            linewidth=1.0,
            zorder=3,
        )
        ax.text(high + 0.00065, index, f"{mean:.4f}", va="center", fontsize=6.1, color=COLORS["ink"])
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Property MAE ↓")
    ax.set_xlim(0, max(highs) * 1.19)
    fig.text(0.99, 0.012, "mean; paired 95% bootstrap CI; n=128", ha="right", va="bottom", fontsize=5.5, color="#5F6B7A")
    finish_axes(ax)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "I1_F1_c1_property_mae")

    # I1-F2: sorted paired gains retain every seed while avoiding a dense
    # spaghetti plot.  Positive values are favourable by the frozen definition.
    # Exact zero-gain ties are collapsed into a deliberately marked block: in
    # this frozen result 75 seeds have exactly zero gain, so plotting them at
    # ranks 1--75 would create a large, visually empty plateau.  The tie count
    # remains explicit and the non-zero seeds retain their sorted order.
    c0 = np.asarray([float(row["C0"]) for row in paired])
    fixed = np.asarray([float(row["Fixed_K2"]) for row in paired])
    gains = c0 - fixed
    order = np.argsort(gains)
    ordered = gains[order]
    positive = ordered > EPS
    negative = ordered < -EPS
    tied = ~(positive | negative)
    nonzero = ordered[~tied]
    x = np.arange(1, len(nonzero) + 1)
    fig, ax = plt.subplots(figsize=(3.75, 2.25))
    nonzero_positive = nonzero > EPS
    nonzero_negative = nonzero < -EPS
    ax.vlines(x[nonzero_positive], 0, nonzero[nonzero_positive], color=COLORS["positive"], linewidth=0.5, alpha=0.38, zorder=1)
    ax.vlines(x[nonzero_negative], 0, nonzero[nonzero_negative], color=COLORS["negative"], linewidth=0.5, alpha=0.38, zorder=1)
    ax.scatter(x[nonzero_positive], nonzero[nonzero_positive], color=COLORS["positive"], s=10, linewidth=0, zorder=3, label="improved")
    ax.scatter(x[nonzero_negative], nonzero[nonzero_negative], color=COLORS["negative"], s=10, linewidth=0, zorder=3, label="worsened")
    # A narrow, labelled tie block is more legible than 75 overlapping zero
    # stems while preserving the exact number of tied/fallback seeds.
    if tied.any():
        tie_x = np.linspace(-0.72, -0.10, int(tied.sum()))
        ax.axvspan(-0.82, -0.02, color=COLORS["neutral"], alpha=0.11, zorder=0)
        ax.scatter(tie_x, np.zeros_like(tie_x), color=COLORS["neutral"], marker="|", s=18, linewidth=0.7, zorder=3)
        ax.text(-0.42, 0.014, f"{int(tied.sum())} exact ties", ha="center", va="bottom", fontsize=5.2, color="#5F6B7A")
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xlabel("Paired seed rank (sorted by Δ; zero ties collapsed)")
    ax.set_ylabel("Δ property MAE\n(C0 − Fixed-K2)")
    wins = int(positive.sum())
    ties = int(tied.sum())
    losses = int(negative.sum())
    ax.text(0.02, 0.96, f"W/T/L  {wins}/{ties}/{losses}", transform=ax.transAxes, va="top", fontsize=6.2, color=COLORS["ink"])
    ax.set_xlim(-1.1, len(nonzero) + 2)
    ax.set_xticks(np.arange(0, len(nonzero) + 1, 10))
    ax.margins(y=0.08)
    fig.text(0.99, 0.012, "W/T/L counts all 128 paired seeds; tie block shown at Δ=0", ha="right", va="bottom", fontsize=5.3, color="#5F6B7A")
    finish_axes(ax, grid_axis="y")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "I1_F2_fixed_vs_c0_paired")

    # I1-F3: ECDF is more legible than a histogram when the bootstrap mass is
    # concentrated near zero.  It still shows the complete 20k distribution.
    draws = bootstrap_draws(gains, BOOT_SEED)
    sorted_draws = np.sort(draws)
    cdf = np.arange(1, len(sorted_draws) + 1) / len(sorted_draws)
    low, high = np.quantile(draws, [0.025, 0.975]); mean = float(draws.mean())
    fig, ax = plt.subplots(figsize=(3.75, 2.25))
    ax.step(sorted_draws, cdf, where="post", color=COLORS["positive"], linewidth=1.15)
    ax.axvline(0, color=COLORS["negative"], linestyle=(0, (3, 2)), linewidth=0.7)
    ax.axvspan(low, high, color="#56B4E9", alpha=0.13)
    ax.axvline(mean, color=COLORS["ink"], linewidth=0.9)
    ax.annotate("", xy=(high, 0.10), xytext=(low, 0.10), arrowprops=dict(arrowstyle="|-|", color="#4BA3D3", lw=0.9))
    ax.text((low + high) / 2, 0.135, "95% CI", ha="center", va="bottom", fontsize=5.6, color="#2D6F91")
    ax.text(mean + 0.00012, 0.53, f"mean {mean:.4f}", rotation=90, va="center", fontsize=5.7, color=COLORS["ink"])
    ax.text(0.00008, 0.98, "zero", rotation=90, va="top", fontsize=5.4, color=COLORS["negative"])
    ax.set_xlabel("Bootstrap mean Δ MAE (C0 − Fixed-K2)")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0, 1.02)
    finish_axes(ax)
    fig.tight_layout()
    save(fig, out, "I1_F3_fixed_gain_bootstrap")

    # I1-F4: paired mean points and direct labels are easier to read than four
    # independent bar charts with unrelated y scales.
    quality = [
        ("E-hull", "e_hull", "eV/atom", False),
        ("Stable", "stable", "fraction", True),
        ("NUS", "nus", "fraction", True),
        ("Validity", "validity", "fraction", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(3.55, 3.25))
    for index, (ax, (label, key, unit, is_fraction)) in enumerate(zip(axes.flat, quality)):
        values = [float(metrics[method][key]) for method in ("C0", "Fixed_K2")]
        ax.plot([0, 1], values, color="#AAB2BC", linewidth=0.8, zorder=1)
        ax.scatter(0, values[0], s=22, marker="o", facecolor=COLORS["C0"], edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(1, values[1], s=24, marker="s", facecolor="white", edgecolor=COLORS["Fixed-K2"], linewidth=1.0, zorder=3)
        for xpos, value in enumerate(values):
            text = f"{value:.1%}" if is_fraction else f"{value:.4f}"
            ax.annotate(text, (xpos, value), xytext=(3 if xpos == 0 else 0, 7), textcoords="offset points", ha="left" if xpos == 0 else "center", fontsize=5.8)
        ax.set_xticks([0, 1], ["C0", "Fixed-K2"])
        ax.set_title(f"({chr(97 + index)})  {label}", loc="left", pad=3, fontweight="semibold", fontsize=7.4)
        ax.set_ylabel("%" if is_fraction else unit)
        if np.ptp(values) < EPS:
            lower = max(0.0, values[0] - (0.06 if is_fraction else 0.01))
            upper = min(1.05, values[0] + (0.035 if is_fraction else 0.01))
        else:
            pad = max(np.ptp(values) * 0.55, max(values) * 0.04)
            lower = max(0.0, min(values) - pad)
            upper = min(1.05, max(values) + pad)
        ax.set_ylim(lower, upper)
        if is_fraction:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        delta = values[1] - values[0]
        delta_text = f"Δ={delta * 100:+.1f} pp" if is_fraction else f"Δ={delta:+.4f}"
        ax.text(0.98, 0.04, delta_text, transform=ax.transAxes, ha="right", fontsize=5.2, color="#5F6B7A")
        finish_axes(ax, grid_axis="y")
    fig.text(0.99, 0.005, "mean only; n=128", ha="right", fontsize=5.4, color="#5F6B7A")
    fig.tight_layout(rect=(0, 0.025, 1, 1), h_pad=1.0, w_pad=1.0)
    save(fig, out, "I1_F4_quality_guardrails")

    # I1-F5: hatching separates deployment from offline construction in print.
    fig, ax = plt.subplots(figsize=(3.65, 1.95))
    labels = ["C0", "Fixed-K2", "Phase B (offline)"]
    values = [1.0, 2.2, 3.4]
    bars = ax.barh(labels, values, color=[COLORS["C0"], COLORS["Fixed-K2"], "#D8DDE3"], edgecolor=COLORS["ink"], linewidth=0.4, height=0.44)
    bars[2].set_hatch("/")
    for bar, value in zip(bars, values):
        ax.text(value + 0.06, bar.get_y() + bar.get_height() / 2, f"{value:.1f}×", va="center", fontsize=6.2)
    ax.set_xlabel("Compute relative to C0")
    ax.set_xlim(0, 4.00)
    # Keep the explanatory key outside the data region; the previous in-bar
    # annotation collided with the hatch strokes at single-column size.
    fig.text(0.99, 0.012, "hatched bar = offline construction", ha="right", va="bottom", fontsize=5.3, color="#5F6B7A")
    finish_axes(ax, grid_axis="x")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "I1_F5_compute_comparison")

    # I1-F6: serpentine evidence timeline with a clean 90-degree row turn.
    status_rows = read_csv(ROOT / "combined_summary/experiment_status.csv")
    wanted = ["Adaptive CFG V1", "Robust V2", "Counterfactual Oracle V3", "Risk-Calibrated V4", "Safe Selection V5", "Stage-Calibrated CFG", "Field-Decoupled CFG", "Branch-Compatible Oracle", "Phase B Linear-K2", "C1 Fixed-K2", "C1 Linear-K2"]
    status = {row["Experiment"]: row["Status"] for row in status_rows}
    status_color = {"MIXED": "#C28A00", "FAIL": "#C45119", "SUPPORTED_MECHANISM": "#2878A7", "SUPPORTED": "#17825F", "NOT_SUPPORTED": "#A43A4A"}
    status_text = {"MIXED": "MIXED", "FAIL": "FAIL", "SUPPORTED_MECHANISM": "MECHANISM", "SUPPORTED": "SUPPORTED", "NOT_SUPPORTED": "NOT SUPPORTED"}
    short_titles = [
        "Adaptive CFG\nV1", "Robust CFG\nV2", "Counterfactual\nOracle V3", "Risk-calibrated\nV4", "Safe selection\nV5", "Stage-calibrated\nCFG",
        "Field-decoupled\nCFG", "Branch-compatible\nOracle", "Phase B\nLinear-K2", "C1\nFixed-K2", "C1\nLinear-K2",
    ]
    fig, ax = plt.subplots(figsize=(6.8, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top = [(xpos, 0.70) for xpos in np.linspace(0.09, 0.91, 6)]
    bottom = [(xpos, 0.31) for xpos in np.linspace(0.91, 0.09, 5)]
    positions = top + bottom
    for idx in range(len(positions) - 1):
        connect_nodes(ax, positions[idx], positions[idx + 1])
    for idx, (name, title, (xpos, ypos)) in enumerate(zip(wanted, short_titles, positions), start=1):
        state = status[name]
        add_timeline_node(ax, xpos, ypos, title, status_text[state], status_color[state], idx)
    ax.text(0.01, 0.96, "Chronological evidence path", ha="left", va="top", fontsize=6.3, fontweight="semibold", color=COLORS["ink"])
    ax.text(0.99, 0.04, "status denotes frozen evidence, not a monotonic performance score", ha="right", va="bottom", fontsize=5.5, color="#5F6B7A")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.03)
    save(fig, out, "I1_F6_evidence_progression")


def _paired_force_data() -> tuple[list[dict[str, str]], dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    paired = read_csv(I2 / "results/mattersim_late_force_guidance_formal256/paired_results.csv")
    independent = read_csv(I2 / "results/chgnet_formal256/independent_per_structure.csv")
    matter = {"C0": {}, "F0": {}}
    for row in paired:
        seed = int(row["seed"])
        matter["C0"][seed] = float(row["C0_maxF_ev_per_a"])
        matter["F0"][seed] = float(row["F0_maxF_ev_per_a"])
    chg = {"C0": {}, "F0": {}}
    for row in independent:
        chg[row["method"]][int(row["seed"])] = float(row["maxF_ev_a"])
    return paired, matter, chg


def i2_figures() -> None:
    out = I2 / "figures_polished"

    # I2-F1: uniform snake-flow pipeline with explicit baseline/guidance roles.
    steps = [
        "MatterGen\nreverse diffusion",
        "Predict clean\nstructure $x_0$",
        "Evaluate\nneural force",
        "Map Cartesian →\nfractional",
        "Bounded\nposition update",
        "Resume\ndiffusion",
    ]
    roles = ["BASE SAMPLER", "GUIDANCE", "GUIDANCE", "GUIDANCE", "GUIDANCE", "BASE SAMPLER"]
    node_colors = [COLORS["C0"], COLORS["RC-NFGD"], COLORS["RC-NFGD"], COLORS["RC-NFGD"], COLORS["RC-NFGD"], COLORS["C0"]]
    fig, ax = plt.subplots(figsize=(6.8, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    positions = [(0.15, 0.69), (0.50, 0.69), (0.85, 0.69), (0.85, 0.29), (0.50, 0.29), (0.15, 0.29)]
    for idx in range(len(positions) - 1):
        connect_nodes(ax, positions[idx], positions[idx + 1], width=0.205, height=0.19)
    for idx, ((xpos, ypos), title, role, color) in enumerate(zip(positions, steps, roles, node_colors), start=1):
        add_timeline_node(ax, xpos, ypos, title, role, color, idx, width=0.205, height=0.19)
    ax.text(0.5, 0.04, "predictor-only for $t\leq0.02$ · atomic and cell scores unchanged", ha="center", va="bottom", fontsize=5.7, color="#5F6B7A")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.03)
    save(fig, out, "I2_F1_rc_nfgd_pipeline")

    table = {row["Metric"]: row for row in read_csv(I2 / "tables/formal256_main_results.csv")}
    # I2-F2: lightweight paired point panels avoid the visual mass of bars.
    unit_display = {"eV/angstrom": "eV/Å", "angstrom": "Å"}
    fig, axes = plt.subplots(3, 1, figsize=(3.55, 4.25))
    for index, (ax, metric) in enumerate(zip(axes, ["MaxF", "Mean Force", "RMSD"])):
        row = table[metric]
        values = [float(row["Baseline value"]), float(row["Method value"])]
        y = np.asarray([1.0, 0.0])
        ax.plot(values, y, color="#B3BBC5", linewidth=0.8, zorder=1)
        ax.scatter(values[0], y[0], s=26, marker="o", facecolor=COLORS["C0"], edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(values[1], y[1], s=29, marker="s", facecolor="white", edgecolor=COLORS["RC-NFGD"], linewidth=1.05, zorder=3)
        ax.set_yticks(y, ["C0", "RC-NFGD"])
        for ypos, value in zip(y, values):
            ax.annotate(f"{value:.4f}", (value, ypos), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6.0)
        unit = unit_display.get(row["Unit"], row["Unit"])
        ax.set_title(f"({chr(97 + index)})  {metric} ({unit})", loc="left", pad=3, fontweight="semibold", fontsize=7.4)
        ax.set_xlim(0, max(values) * 1.20)
        rel = (values[0] - values[1]) / values[0] * 100
        ax.text(0.99, 0.08, f"−{rel:.1f}%", transform=ax.transAxes, ha="right", fontsize=5.5, color=COLORS["RC-NFGD"])
        finish_axes(ax, grid_axis="x")
    fig.text(0.99, 0.005, "mean; n=256 · lower is better · local x-scales", ha="right", fontsize=5.4, color="#5F6B7A")
    fig.tight_layout(rect=(0, 0.025, 1, 1), h_pad=0.9)
    save(fig, out, "I2_F2_formal256_primary_metrics")

    # I2-F3: a slim full-range rug preserves every tail observation; the larger
    # lower panel gives the central 99% enough room and full axis ticks.
    paired = read_csv(I2 / "results/mattersim_late_force_guidance_formal256/paired_results.csv")
    metrics = [("MaxF", "maxF_ev_per_a", "eV/Å"), ("Mean force", "atomic_force_mean_ev_per_a", "eV/Å"), ("RMSD", "rmsd_a", "Å")]
    fig = plt.figure(figsize=(6.8, 3.25))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.34, 1.0], hspace=0.10, wspace=0.36)
    for index, (label, column, unit) in enumerate(metrics):
        tail_ax = fig.add_subplot(grid[0, index])
        ax = fig.add_subplot(grid[1, index])
        delta = np.asarray([float(row[f"C0_{column}"]) - float(row[f"F0_{column}"]) for row in paired])
        lo, hi = np.quantile(delta, [0.005, 0.995])
        lo = min(float(lo), 0.0)
        hi = max(float(hi), 0.0)
        central = (delta >= lo) & (delta <= hi)

        tail_ax.scatter(delta, np.zeros_like(delta), marker="|", s=22, linewidth=0.55, color=COLORS["RC-NFGD"], alpha=0.45)
        tail_ax.axvline(0, color=COLORS["negative"], linestyle=(0, (3, 2)), linewidth=0.65)
        tail_ax.set_ylim(-0.22, 0.22)
        full_pad = max(np.ptp(delta) * 0.03, 1e-4)
        tail_ax.set_xlim(delta.min() - full_pad, delta.max() + full_pad)
        tail_ax.set_yticks([])
        tail_ax.set_xticks([])
        tail_ax.tick_params(axis="x", length=0)
        tail_ax.spines["left"].set_visible(False)
        tail_ax.spines["right"].set_visible(False)
        tail_ax.spines["top"].set_visible(False)
        tail_ax.set_title(f"({chr(97 + index)})  {label}", loc="left", pad=2, fontweight="semibold", fontsize=7.4)
        tail_ax.text(0.99, 0.78, f"full range  [{delta.min():.2g}, {delta.max():.2g}]", transform=tail_ax.transAxes, ha="right", fontsize=5.2, color="#5F6B7A")

        ax.hist(delta[central], bins=24, range=(lo, hi), color="#54B99A", edgecolor="white", linewidth=0.35)
        ax.axvline(0, color=COLORS["negative"], linestyle=(0, (3, 2)), linewidth=0.75)
        ax.axvline(delta.mean(), color=COLORS["ink"], linewidth=0.8)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(f"paired improvement ({unit})")
        ax.set_ylabel("count")
        ax.text(0.98, 0.94, f"central 99% · outside={int((~central).sum())}", transform=ax.transAxes, ha="right", va="top", fontsize=5.2, color="#5F6B7A")
        finish_axes(ax, grid_axis="y")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.15)
    save(fig, out, "I2_F3_paired_improvement_distributions")

    # I2-F4: horizontal forest plot makes cohort labels and interval widths
    # readable without the large vertical whitespace of the earlier version.
    scale = read_csv(I2 / "tables/effect_scale_consistency.csv")
    labels = [row["Cohort"] for row in scale]
    means = np.asarray([float(row["Relative MaxF reduction"]) for row in scale]) * 100
    lows = np.asarray([float(row["CI low"]) for row in scale]) * 100
    highs = np.asarray([float(row["CI high"]) for row in scale]) * 100
    y = np.arange(len(scale))[::-1]
    fig, ax = plt.subplots(figsize=(4.25, 2.2))
    ax.errorbar(means, y, xerr=np.vstack([means - lows, highs - means]), fmt="s", markerfacecolor="white", markeredgecolor=COLORS["RC-NFGD"], markeredgewidth=1.0, ecolor="#4BA3D3", capsize=2.8, markersize=4.8, linewidth=0.9)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_yticks(y, [f"{label}  (n={scale[index]['N']})" for index, label in enumerate(labels)])
    ax.set_xlabel("Relative MaxF reduction (%)")
    ax.set_xlim(min(0, lows.min() - 2), highs.max() + 8)
    for ypos, mean in zip(y, means):
        ax.annotate(f"{mean:.1f}%", (mean, ypos), xytext=(5, 0), textcoords="offset points", va="center", fontsize=5.8)
    ax.text(0.99, 0.96, "95% CI · independent cohorts", transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color="#5F6B7A")
    finish_axes(ax, grid_axis="x")
    fig.tight_layout()
    save(fig, out, "I2_F4_effect_consistency")

    # I2-F5: show the complete cross-proxy range beside a properly labelled
    # central-core view; the identity line makes agreement direction visible.
    # The central view uses the intersection of independent axis-wise
    # q0.5--q99.5 bounds (not a new model/data selection rule).
    _, matter, chg = _paired_force_data()
    seeds = sorted(set(matter["C0"]) & set(matter["F0"]) & set(chg["C0"]) & set(chg["F0"]))
    x_values = np.asarray([matter["C0"][seed] - matter["F0"][seed] for seed in seeds])
    y_values = np.asarray([chg["C0"][seed] - chg["F0"][seed] for seed in seeds])
    qx = np.quantile(x_values, [0.005, 0.995])
    qy = np.quantile(y_values, [0.005, 0.995])
    keep = (x_values >= qx[0]) & (x_values <= qx[1]) & (y_values >= qy[0]) & (y_values <= qy[1])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.30})
    ranges = [(x_values, y_values, "full range", len(seeds)), (x_values[keep], y_values[keep], "central core", int(keep.sum()))]
    for index, (ax, (xv, yv, title, count)) in enumerate(zip(axes, ranges)):
        ax.scatter(xv, yv, s=12 if index == 0 else 14, alpha=0.55 if index == 0 else 0.68, color=COLORS["RC-NFGD"], edgecolors="white", linewidth=0.18, zorder=3)
        ax.axhline(0, color="#9AA3AD", linewidth=0.6)
        ax.axvline(0, color="#9AA3AD", linewidth=0.6)
        diagonal_lo = min(float(np.min(xv)), float(np.min(yv)), 0.0)
        diagonal_hi = max(float(np.max(xv)), float(np.max(yv)), 0.0)
        ax.plot([diagonal_lo, diagonal_hi], [diagonal_lo, diagonal_hi], color="#6B7280", linestyle=(0, (3, 2)), linewidth=0.65, zorder=1)
        ax.set_title(f"({chr(97 + index)})  {title}", loc="left", pad=3, fontweight="semibold", fontsize=7.4)
        if index == 1:
            note = f"axis-wise q0.5–q99.5 · n={count}"
        else:
            note = f"n={count}"
        # Keep sample-size/filter metadata off the extreme observations and
        # give it a quiet backing so dense points cannot reduce legibility.
        ax.text(
            0.03,
            0.95,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.1 if index == 1 else 5.6,
            color="#5F6B7A",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.7),
        )
        ax.set_xlabel("MatterSim Δ MaxF (eV/Å)")
        if index == 0:
            ax.set_ylabel("CHGNet Δ MaxF (eV/Å)")
            pad_x = max(np.ptp(x_values) * 0.04, 0.05)
            pad_y = max(np.ptp(y_values) * 0.04, 0.05)
            ax.set_xlim(x_values.min() - pad_x, x_values.max() + pad_x)
            ax.set_ylim(y_values.min() - pad_y, y_values.max() + pad_y)
        else:
            pad_x = max(np.ptp(xv) * 0.08, 0.02)
            pad_y = max(np.ptp(yv) * 0.08, 0.02)
            ax.set_xlim(xv.min() - pad_x, xv.max() + pad_x)
            ax.set_ylim(yv.min() - pad_y, yv.max() + pad_y)
        # Both axes are in the same Δ MaxF unit; equal aspect prevents the
        # identity reference from being visually distorted.
        ax.set_aspect("equal", adjustable="box")
        finish_axes(ax)
    fig.text(0.99, 0.01, "dashed = equal improvement · central core: axis-wise q0.5–q99.5", ha="right", fontsize=5.3, color="#5F6B7A")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.19, wspace=0.30)
    save(fig, out, "I2_F5_mattersim_vs_chgnet")

    # I2-F6: paired dumbbells keep the equal guardrails visible and show the
    # small Property-MAE trade-off without exaggerating it with bar lengths.
    fig, axes = plt.subplots(3, 1, figsize=(3.55, 4.25))
    for index, (ax, metric) in enumerate(zip(axes, ["Stable", "Validity", "Property MAE"])):
        row = table[metric]
        values = [float(row["Baseline value"]), float(row["Method value"])]
        y = np.asarray([1.0, 0.0])
        method_color = COLORS["negative"] if metric == "Property MAE" else COLORS["RC-NFGD"]
        ax.plot(values, y, color="#B3BBC5", linewidth=0.8, zorder=1)
        ax.scatter(values[0], y[0], s=26, marker="o", facecolor=COLORS["C0"], edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(values[1], y[1], s=29, marker="s", facecolor="white", edgecolor=method_color, linewidth=1.05, zorder=3)
        ax.set_yticks(y, ["C0", "RC-NFGD"])
        for ypos, value in zip(y, values):
            label = f"{value:.4f}" if metric == "Property MAE" else f"{value:.1%}"
            ax.annotate(label, (value, ypos), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6.0)
        delta = values[1] - values[0]
        ax.set_title(f"({chr(97 + index)})  {metric}{' ↓' if metric == 'Property MAE' else ' ↑'}", loc="left", pad=3, fontweight="semibold", fontsize=7.4)
        span = max(np.ptp(values), 0.025 if metric != "Property MAE" else 0.00035)
        lower = min(values) - span * 0.65
        upper = max(values) + span * 0.85
        if metric != "Property MAE":
            lower = max(0.0, lower)
            upper = min(1.05, upper)
        else:
            lower = max(0.0, lower)
        ax.set_xlim(lower, upper)
        delta_text = f"Δ={delta:+.4f}" if metric == "Property MAE" else f"Δ={delta:+.1%}"
        ax.text(0.99, 0.08, delta_text, transform=ax.transAxes, ha="right", fontsize=5.5, color=method_color)
        finish_axes(ax, grid_axis="x")
    fig.text(0.99, 0.005, "mean; n=256 · Δ = RC-NFGD − C0 · local x-scales", ha="right", fontsize=5.4, color="#5F6B7A")
    fig.tight_layout(rect=(0, 0.025, 1, 1), h_pad=0.9)
    save(fig, out, "I2_F6_quality_guardrails")


def main() -> None:
    i1_figures()
    i2_figures()
    print("POLISHED_FIGURE_REPRODUCTION=PASS")


if __name__ == "__main__":
    main()
