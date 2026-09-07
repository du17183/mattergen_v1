"""Reuse the frozen P1 MatterSim-5M relaxation pipeline for Formal256."""
from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "experiments/tcl_p1/run_relaxation.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_p1_relaxation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = PROJECT_ROOT / "experiments/tcl_formal256"
    module.main()


if __name__ == "__main__":
    main()
