"""Run the frozen shared-reference quality evaluator for B0/S-off/S-on."""
from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/multifield_self_conditioning_p0"
FROZEN_EVALUATOR = PROJECT_ROOT / "experiments/tcl_p1/run_quality_metrics.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_sc_quality", FROZEN_EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.METHODS = ("B0", "S-off", "S-on")
    module.REFERENCE = Path(
        "/mnt/datasets-livsyn/dxl/mattergen_v1/data-release/"
        "alex-mp/reference_MP2020correction.gz"
    )
    module.POTENTIAL = Path(
        "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/"
        "mattersim/mattersim-v1.0.0-5M.pth"
    )
    module.main()


if __name__ == "__main__":
    main()
