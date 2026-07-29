"""Shared publication style derived from the four audited scientific skills."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette. Red is reserved for harm/invalidity.
COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#7A7A7A",
    "light_gray": "#E6E6E6",
}

SINGLE_COLUMN_IN = 3.45
DOUBLE_COLUMN_IN = 7.10


@contextmanager
def paper_context():
    """Apply scoped, reproducible Matplotlib settings."""
    settings = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "mattergen-thesis-20260729",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
    with mpl.rc_context(settings):
        yield


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def clean_axes(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=COLORS["light_gray"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig, stem: str, output_root: Path) -> None:
    """Save one opaque, vector-first figure in all required formats."""
    creator = "MatterGen thesis reproducible CPU pipeline"
    fixed_time = datetime(2026, 7, 29, 0, 0, 0)
    pdf_metadata = {"Creator": creator, "CreationDate": fixed_time, "ModDate": fixed_time}
    svg_metadata = {"Creator": creator, "Date": "2026-07-29"}
    fig.savefig(
        output_root / "pdf" / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata=pdf_metadata,
    )
    svg_path = output_root / "svg" / f"{stem}.svg"
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
        metadata=svg_metadata,
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    fig.savefig(
        output_root / "png" / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": creator},
    )
    plt.close(fig)

