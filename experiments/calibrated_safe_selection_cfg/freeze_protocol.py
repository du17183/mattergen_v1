"""Seal every outcome-sensitive protocol and implementation file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/calibrated_safe_selection_cfg"
FILES = (
    "mattergen/diffusion/sampling/risk_calibrated_cfg.py",
    "mattergen/diffusion/sampling/safe_selection_cfg.py",
    "experiments/calibrated_safe_selection_cfg/preregistered_protocol.yaml",
    "experiments/calibrated_safe_selection_cfg/validation/preregistered_candidates.yaml",
    "experiments/calibrated_safe_selection_cfg/dataset/seeds.json",
    "experiments/calibrated_safe_selection_cfg/dataset/seed_overlap_audit.json",
    "experiments/calibrated_safe_selection_cfg/future_seed_reservations.json",
    "experiments/calibrated_safe_selection_cfg/splits/split_manifest.json",
    "experiments/calibrated_safe_selection_cfg/register_seeds.py",
    "experiments/calibrated_safe_selection_cfg/run_dataset_generation.py",
    "experiments/calibrated_safe_selection_cfg/prepare_dataset.py",
    "experiments/calibrated_safe_selection_cfg/run_quality.py",
    "experiments/calibrated_safe_selection_cfg/run_offline_analysis.py",
    "experiments/calibrated_safe_selection_cfg/run_fresh_generation.py",
    "experiments/calibrated_safe_selection_cfg/prepare_fresh.py",
    "experiments/calibrated_safe_selection_cfg/run_fresh_quality.py",
    "experiments/calibrated_safe_selection_cfg/analyze_fresh.py",
    "experiments/calibrated_safe_selection_cfg/run_pipeline.py",
    "experiments/calibrated_safe_selection_cfg/initial_audit.md",
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
        raise FileExistsError("frozen manifest already exists")
    hashes = {}
    for relative in FILES:
        path = PROJECT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = sha256(path)
    manifest = {
        "schema_version": 1,
        "frozen_before_generation": True,
        "git_parent_commit": "a4fda8a573ee0e372d3b53064e235ae87f0e670c",
        "branch": "experiment/calibrated-safe-selection-cfg",
        "sha256": hashes,
        "dataset_trajectories": 128,
        "split_counts": {"train": 64, "calibration": 26, "validation": 19, "test": 19},
        "decision_index": 250,
        "candidate_cfg": [1.75, 2.0, 2.25],
        "future_seed_cohorts_reserved_before_outcomes": True,
        "allowed_after_freeze": "execution and technical bug fixes that do not change any scientific decision",
        "forbidden_after_freeze": "model feature calibration threshold action seed metric harm gate or cohort retuning",
        "surrogate_property_eval": True,
        "dft_verified": False,
    }
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    status = {
        "FULL_OUTCOME_DATASET": "RUNNING",
        "VALIDATION_SAFE_GATE": "NOT_RUN",
        "OFFLINE_SAFE_SELECTION": "NOT_RUN",
        "SAFE_SELECTION_P0": "NOT_RUN",
        "SAFE_SELECTION_FORMAL256": "NOT_RUN",
        "SAMPLE_ADAPTIVITY": "NOT_RUN",
        "INNOVATION1_FINAL_STATUS": "MIXED",
        "stage": "FROZEN_READY",
        "cohort": "dataset",
        "cohort_complete": 0,
        "cohort_total": 128,
    }
    (ROOT / "pipeline_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
