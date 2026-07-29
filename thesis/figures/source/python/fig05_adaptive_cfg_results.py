"""Skill-guided statistical export for Figure 5.

Applied skills: scientific-visualization, matplotlib, statistical-analysis.
Source data: thesis_archive/data/innovation1/per_seed_metrics.csv and the frozen
formal report JSON. Purpose: report paired Adaptive CFG effects without
overclaiming non-significant trends. Statistical annotations: paired bootstrap
95% CI, Holm-corrected p-values, n=256, and physical units.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_statistical_figures import main

if __name__ == "__main__":
    main(["fig05"])
