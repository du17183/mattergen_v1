"""Skill-guided statistical export for Figure 10.

Applied skills: scientific-visualization, matplotlib, statistical-analysis.
Source data: thesis_archive/data/compatibility_2/per_seed_metrics.csv. Purpose:
show the newest independent A0 versus A0+E3-G paired outcomes. Statistical
annotations: seed range 50000–50063, n=64, gate-off algorithmic exact ties,
eV/Å units; formal cohort statistics are supplied in the caption.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_statistical_figures import main

if __name__ == "__main__":
    main(["fig10"])
