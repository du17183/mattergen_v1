"""Skill-guided descriptive export for Figure 12.

Applied skills: scientific-schematics, scientific-visualization, matplotlib.
Source data: thesis_archive/EXPERIMENT_LINEAGE.md only. Purpose: map representative
No-Go routes to observed stopping evidence. Statistical annotations: preserved
reported effects where available; no synthetic score, ranking, or inferred
per-seed distribution is created.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_statistical_figures import main

if __name__ == "__main__":
    main(["fig12"])
