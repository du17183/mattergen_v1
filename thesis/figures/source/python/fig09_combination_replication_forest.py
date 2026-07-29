"""Skill-guided statistical export for Figure 9.

Applied skills: scientific-visualization, matplotlib, statistical-analysis.
Source data: frozen compatibility and replication paired-statistics CSV files.
Purpose: show two independent cohorts without pooling. Statistical annotations:
absolute paired mean difference, bootstrap 95% CI, p, relative change, n=64
per cohort, seed ranges, and eV/Å units.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_statistical_figures import main

if __name__ == "__main__":
    main(["fig09"])
