"""Skill-guided statistical export for Figure 8.

Applied skills: scientific-visualization, matplotlib, statistical-analysis.
Source data: thesis_archive/data/innovation2/per_seed_metrics.csv. Purpose:
relate genuine gate confidence to realized force gain. Statistical annotations:
Spearman rho and exact p-value, y=0 reference, threshold=0.5, n=256, eV/Å.
The trend line is descriptive rather than a causal calibration claim.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_statistical_figures import main

if __name__ == "__main__":
    main(["fig08"])
