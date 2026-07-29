"""Skill-guided statistical export for Figure 6.

Applied skills: scientific-visualization, matplotlib, statistical-analysis.
Source data: thesis_archive/data/innovation2/per_seed_metrics.csv and formal
paired statistics. Purpose: compare C0, Always-on, and Learned-gated E3-PCR.
Statistical annotations: paired bootstrap 95% CI, Holm-adjusted Wilcoxon p,
formal raw-difference Win/Tie/Loss, n=256, and eV/Å units.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_statistical_figures import main

if __name__ == "__main__":
    main(["fig06"])
