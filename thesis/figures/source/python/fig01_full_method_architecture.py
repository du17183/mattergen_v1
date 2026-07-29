"""Skill-guided export for Figure 1.

Applied skills: scientific-schematics, scientific-visualization, matplotlib.
Source: thesis_archive configs and frozen experiment lineage.
Purpose: show the two-stage method architecture and evaluation boundary.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_architecture_figures import main

if __name__ == "__main__":
    main(["fig01"])
