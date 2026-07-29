"""Skill-guided export for Figure 3.

Applied skills: scientific-schematics, scientific-visualization, matplotlib.
Source: thesis_archive/configs/e3_pcr_final.yaml.
Purpose: document learned gating, bounded refinement, safety checks, and fallback.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_architecture_figures import main

if __name__ == "__main__":
    main(["fig03"])
