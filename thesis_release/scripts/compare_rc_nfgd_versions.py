"""Make non-manuscript RC-NFGD review sheets at an identical print width.

The source figures are included unchanged; only their physical display width
and page placement are standardized for visual comparison.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure_pilots" / "rc_nfgd_redesign"
REDESIGN = OUT / "I2_F1_rc_nfgd_skill_redesign.png"
PAIRS = (
    (
        ROOT / "figure_pilots" / "rc_nfgd" / "I2_F1_rc_nfgd_pipeline_pilot_v2.png",
        "A  Previous corrected pilot (v2)",
        "B  Independent skill redraw",
        "previous_pilot_v2_vs_skill_redraw",
    ),
    (
        ROOT / "innovation2" / "figures_polished" / "I2_F1_rc_nfgd_pipeline.png",
        "A  Figure currently cited in thesis",
        "B  Independent skill redraw",
        "current_thesis_vs_skill_redraw",
    ),
)

SHEET_W_MM = 300.0
SHEET_H_MM = 108.0
IMAGE_W_MM = 138.4
IMAGE_TOP_MM = 96.0

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.2,
    }
)


def make_sheet(left_path: Path, left_title: str, right_title: str, stem: str):
    fig = plt.figure(figsize=(SHEET_W_MM / 25.4, SHEET_H_MM / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, SHEET_W_MM)
    ax.set_ylim(0, SHEET_H_MM)
    ax.axis("off")

    for path, x, title in ((left_path, 6.0, left_title), (REDESIGN, 155.6, right_title)):
        pixels = mpimg.imread(path)
        height_mm = IMAGE_W_MM * pixels.shape[0] / pixels.shape[1]
        ax.imshow(
            pixels,
            extent=(x, x + IMAGE_W_MM, IMAGE_TOP_MM - height_mm, IMAGE_TOP_MM),
            interpolation="nearest",
            zorder=2,
        )
        ax.text(x, 103.0, title, ha="left", va="center", fontsize=8.6, weight="bold", color="#202833")

    ax.plot([150, 150], [5, 98], color="#DCE2E6", lw=0.8, zorder=1)
    fig.canvas.draw()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    for item in PAIRS:
        make_sheet(*item)
