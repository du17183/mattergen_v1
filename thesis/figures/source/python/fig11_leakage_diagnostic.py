"""Skill-guided statistical export for Figure 11.

Applied skills: scientific-visualization, matplotlib, statistical-analysis.
Source data: thesis_archive/data/leakage_diagnostic/per_seed_metrics.csv.
Purpose: contrast mean effect and safety under training overlap versus held-out
data. Statistical annotations: 0/64 versus 31/192 harms, one-sided Fisher exact
p=6.87e-5, raw paired points, and diagnostic-only validity warning.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_statistical_figures import main

if __name__ == "__main__":
    main(["fig11"])
