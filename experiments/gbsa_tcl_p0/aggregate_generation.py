"""Materialize method-wise structure files for the eight P0 paired seeds."""
from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_p0"
FROZEN_AGGREGATOR = PROJECT_ROOT / "experiments/tcl_p1/aggregate_generation.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_tcl_aggregator", FROZEN_AGGREGATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.METHODS = ("C0", "TCL", "GBSA-TCL")
    module.SEEDS = tuple(range(84000, 84008))
    module.main()


if __name__ == "__main__":
    main()
