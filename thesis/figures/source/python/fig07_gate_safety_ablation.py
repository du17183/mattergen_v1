"""Skill-guided statistical export for Figure 7.

Applied skills: scientific-visualization, matplotlib, statistical-analysis.
Source data: thesis_archive/reports/innovation2/final_summary.json. Purpose:
compare intervention coverage, harm, low-force harm, displacement, and retained
gain. Statistical annotations: McNemar exact p=0.000534, rates, Å units, n=256.
Per-seed mean displacement is unavailable and is not reconstructed.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_statistical_figures import main

if __name__ == "__main__":
    main(["fig07"])
