"""Resume-safe serial orchestrator for SPG-MatterGen Fast Gate."""

from __future__ import annotations

import subprocess

from research.spg_fastgate.common import (
    PROJECT,
    PYTHON,
    STOP_MARKER,
    base_environment,
    file_lock,
    initialize_progress,
    read_json,
)


PIPELINE = (
    ("performance_baseline", "research.spg_fastgate.run_performance_baseline"),
    ("nsight_profile", "research.spg_fastgate.profiler_probe"),
    ("bf16_state_probe", "research.spg_fastgate.bf16_probe"),
    ("bf16_eight_seed", "research.spg_fastgate.run_bf16_eight_seed"),
    ("compile_audit", "research.spg_fastgate.compile_audit"),
    ("b4_generation", "research.spg_fastgate.run_quality_generation"),
    ("b4_relaxation", "research.spg_fastgate.run_mattersim_relaxation"),
    ("b4_quality_metrics", "research.spg_fastgate.analyze_quality"),
    ("amdahl_analysis", "research.spg_fastgate.finalize_fastgate"),
)


def main() -> int:
    initialize_progress()
    with file_lock(PROJECT / ".spg_fastgate_launcher.lock"):
        for stage, module in PIPELINE:
            if STOP_MARKER.exists():
                return 0
            progress = read_json(
                PROJECT.parent / "results/spg_fastgate/progress/master_progress.json",
                {},
            )
            if (
                progress.get("stages", {})
                .get(stage, {})
                .get("status")
                == "success"
            ):
                continue
            result = subprocess.run(
                [str(PYTHON), "-m", module],
                cwd=PROJECT,
                env=base_environment(),
            )
            if result.returncode != 0:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
