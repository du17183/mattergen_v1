"""Hash all outcome-sensitive files before the first fresh generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/stage_calibrated_bounded_cfg"
FILES = (
    "mattergen/diffusion/sampling/risk_calibrated_cfg.py",
    "mattergen/diffusion/sampling/stage_bounded_cfg.py",
    "experiments/stage_calibrated_bounded_cfg/preregistered_schedule_grid.yaml",
    "experiments/stage_calibrated_bounded_cfg/calibration/schedule_grid.yaml",
    "experiments/stage_calibrated_bounded_cfg/preregistered_protocol.yaml",
    "experiments/stage_calibrated_bounded_cfg/calibration/seeds.json",
    "experiments/stage_calibrated_bounded_cfg/future_seed_reservations.json",
    "experiments/stage_calibrated_bounded_cfg/seed_overlap_audit.json",
    "experiments/stage_calibrated_bounded_cfg/stage0/stage_headroom.csv",
    "experiments/stage_calibrated_bounded_cfg/stage0/stage_sensitivity.csv",
    "experiments/stage_calibrated_bounded_cfg/stage0/historical_summary.md",
    "experiments/stage_calibrated_bounded_cfg/stage0/residual_stage_dependence.csv",
    "experiments/stage_calibrated_bounded_cfg/register_seeds.py",
    "experiments/stage_calibrated_bounded_cfg/run_stage0.py",
    "experiments/stage_calibrated_bounded_cfg/run_generation.py",
    "experiments/stage_calibrated_bounded_cfg/prepare_phase.py",
    "experiments/stage_calibrated_bounded_cfg/run_quality.py",
    "experiments/stage_calibrated_bounded_cfg/analyze_phase.py",
    "experiments/stage_calibrated_bounded_cfg/run_pipeline.py",
    "experiments/stage_calibrated_bounded_cfg/initial_audit.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    destination = ROOT / "frozen_manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    hashes = {}
    for relative in FILES:
        path = PROJECT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = sha256(path)
    manifest = {
        "schema_version": 1,
        "frozen_before_generation": True,
        "git_base_commit": "a4fda8a573ee0e372d3b53064e235ae87f0e670c",
        "branch": "experiment/stage-calibrated-bounded-cfg",
        "sha256": hashes,
        "calibration_seeds": 32,
        "p0_seeds_reserved": 32,
        "formal256_seeds_reserved": 256,
        "bounded_schedules": 12,
        "constant_candidates": 4,
        "permanent_counterparts": 6,
        "total_including_c0": 23,
        "bootstrap_resamples": 20000,
        "future_cohorts_reserved_before_calibration_outcomes": True,
        "allowed_after_freeze": "execution and technical bug fixes that do not alter scientific decisions",
        "forbidden_after_freeze": "grid thresholds score lambda metrics seeds selection rules rescue tuning or two-pulse expansion",
        "surrogate_property_eval": True,
        "dft_verified": False,
    }
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    status = {
        "STAGE0_DIAGNOSTIC": "COMPLETE",
        "STAGE_SCHEDULE_CALIBRATION": "RUNNING",
        "STAGE_CFG_P0": "NOT_RUN",
        "STAGE_BOUNDED_CFG_FORMAL256": "NOT_RUN",
        "STAGE_SPECIFIC_EFFECT": "NOT_RUN",
        "BOUNDED_PULSE_EFFECT": "NOT_RUN",
        "INNOVATION1_FINAL_STATUS": "MIXED",
        "stage": "FROZEN_READY",
        "active_phase": "calibration",
        "phase_complete": 0,
        "phase_total": 32,
    }
    (ROOT / "pipeline_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
