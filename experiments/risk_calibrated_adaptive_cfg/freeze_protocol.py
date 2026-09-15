"""Seal all outcome-sensitive files before the first dataset trajectory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/risk_calibrated_adaptive_cfg"
FILES = (
    "mattergen/diffusion/sampling/risk_calibrated_cfg.py",
    "mattergen/diffusion/tests/test_risk_calibrated_cfg.py",
    "experiments/risk_calibrated_adaptive_cfg/preregistered_protocol.yaml",
    "experiments/risk_calibrated_adaptive_cfg/dataset/candidate_timestep_config.yaml",
    "experiments/risk_calibrated_adaptive_cfg/dataset/seeds.json",
    "experiments/risk_calibrated_adaptive_cfg/future_seed_reservations.json",
    "experiments/risk_calibrated_adaptive_cfg/fresh_seed_audit.json",
    "experiments/risk_calibrated_adaptive_cfg/register_seeds.py",
    "experiments/risk_calibrated_adaptive_cfg/run_dataset_generation.py",
    "experiments/risk_calibrated_adaptive_cfg/prepare_dataset.py",
    "experiments/risk_calibrated_adaptive_cfg/run_quality.py",
    "experiments/risk_calibrated_adaptive_cfg/run_offline_analysis.py",
    "experiments/risk_calibrated_adaptive_cfg/run_fresh_p0_generation.py",
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
    payload = {
        "schema_version": 1,
        "frozen_before_generation": True,
        "git_base_commit": "1af301c5e534ff6b9b9f96ea5dab48036d9c46af",
        "branch": "experiment/risk-calibrated-adaptive-cfg",
        "sha256": hashes,
        "dataset_trajectories": 96,
        "candidate_timesteps": [100, 250, 400, 550],
        "candidate_cfg": [1.75, 2.0, 2.25],
        "future_seed_cohorts_reserved_before_outcomes": True,
        "allowed_after_outcomes": "execution, gate-prescribed P0/Formal evaluation, technical fixes that do not alter frozen decisions",
        "forbidden_after_outcomes": "feature/model/threshold/timestep/action/harm/seed/metric/gate retuning",
        "surrogate_property_eval": True,
        "dft_verified": False,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    (ROOT / "pipeline_status.json").write_text(
        json.dumps(
            {
                "FULL_OUTCOME_DATASET": "RUNNING",
                "OFFLINE_SELECTIVE_GATE": "NOT_RUN",
                "OFFLINE_PREDICTABILITY": "NOT_RUN",
                "RISK_CALIBRATED_CFG_P0": "NOT_RUN",
                "RISK_CALIBRATED_CFG_FORMAL256": "NOT_RUN",
                "SAMPLE_ADAPTIVITY": "NOT_RUN",
                "INNOVATION1_FINAL_STATUS": "MIXED",
                "stage": "dataset_generation",
                "generation_complete": 0,
                "generation_total": 96,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
