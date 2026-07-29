"""Skill-guided export for Figure 2.

Applied skills: scientific-schematics, scientific-visualization, matplotlib.
Source: thesis_archive/configs/adaptive_cfg_final.yaml.
Purpose: expose residual-driven Adaptive CFG without timestep/corrector skipping.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from generate_architecture_figures import main

if __name__ == "__main__":
    main(["fig02"])
