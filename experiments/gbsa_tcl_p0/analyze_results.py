"""Reuse the P1 descriptive analysis at the eight-seed P0 scale."""
from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_p0"
FROZEN_ANALYSIS = PROJECT_ROOT / "experiments/tcl_p1/analyze_results.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_tcl_analysis", FROZEN_ANALYSIS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.METHODS = ("C0", "TCL", "GBSA-TCL")
    module.SEEDS = tuple(range(84000, 84008))
    module.COMPARISONS = (
        ("TCL", "C0"),
        ("GBSA-TCL", "C0"),
        ("GBSA-TCL", "TCL"),
    )
    module.BOOTSTRAP_SEED = 20260918
    module.main()


if __name__ == "__main__":
    main()
