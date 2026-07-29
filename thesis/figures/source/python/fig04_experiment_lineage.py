"""Skill-guided export for Figure 4.

Applied skills: scientific-schematics, scientific-visualization, matplotlib.
Source: thesis_archive/EXPERIMENT_LINEAGE.md and evaluation_final.yaml.
Purpose: separate formal, independent, supplementary, diagnostic, and invalid evidence.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_architecture_figures import main

if __name__ == "__main__":
    main(["fig04"])
