"""Reuse the frozen deterministic generator for the GBSA-TCL method name."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN_GENERATOR = PROJECT_ROOT / "experiments/tcl_dml_p0/generate_one.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_tcl_generator", FROZEN_GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.METHODS = ("C0", "TCL", "GBSA-TCL")
    module.main()


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
