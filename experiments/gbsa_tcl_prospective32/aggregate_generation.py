"""Materialize two method-wise structure files in frozen seed order."""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_prospective32"
FROZEN_AGGREGATOR = PROJECT_ROOT / "experiments/tcl_p1/aggregate_generation.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_tcl_aggregator", FROZEN_AGGREGATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.METHODS = ("C0", "GBSA-TCL")
    module.SEEDS = tuple(range(87000, 87032))
    module.main()


if __name__ == "__main__":
    main()
