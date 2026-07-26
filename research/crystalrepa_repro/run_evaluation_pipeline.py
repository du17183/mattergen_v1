"""Resume-safe post-training evaluation for the CrystalREPA reproduction."""

from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

from research.crystalrepa_repro.common import (
    PROJECT,
    REPORTS,
    atomic_json,
    now,
)


PYTHON = Path(sys.executable)


def run(module: str, *arguments: str) -> None:
    command = [str(PYTHON), "-m", module, *arguments]
    subprocess.run(command, cwd=PROJECT, check=True)


def main() -> None:
    started = now()
    try:
        run("research.crystalrepa_repro.export_inference")
        run("research.crystalrepa_repro.run_generation", "eight")
        run("research.crystalrepa_repro.run_generation", "full")
        run("research.crystalrepa_repro.run_mattersim_relaxation", "launch")
        run("research.crystalrepa_repro.analyze_repro")
        run("research.crystalrepa_repro.finalize_report")
        run("research.crystalrepa_repro.collect_publishable_artifacts")
        payload = {
            "created_at": now(),
            "started_at": started,
            "success": True,
            "stages": [
                "export_inference",
                "eight_seed_smoke",
                "sixty_four_generation",
                "sixty_four_relax",
                "metrics",
                "paired_statistics",
                "repro_go_no_go",
                "finalize_report",
                "collect_publishable_artifacts",
            ],
        }
        atomic_json(REPORTS / "evaluation_pipeline.json", payload)
    except BaseException:
        payload = {
            "created_at": now(),
            "started_at": started,
            "success": False,
            "traceback": traceback.format_exc(),
        }
        atomic_json(REPORTS / "evaluation_pipeline.json", payload)
        raise


if __name__ == "__main__":
    main()
