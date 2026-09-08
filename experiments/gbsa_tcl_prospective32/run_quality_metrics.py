"""Run the frozen official shared-reference evaluator for the two arms."""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_prospective32"
FROZEN_EVALUATOR = PROJECT_ROOT / "experiments/tcl_p1/run_quality_metrics.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_tcl_quality", FROZEN_EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.METHODS = ("C0", "GBSA-TCL")
    module.main()


if __name__ == "__main__":
    main()
