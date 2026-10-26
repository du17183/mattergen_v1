"""Freeze the one-shot Oracle diagnostic implementation and preregistration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/counterfactual_adaptive_cfg_v3"
FILES = (
    "mattergen/diffusion/sampling/counterfactual_oracle.py",
    "mattergen/diffusion/tests/test_counterfactual_oracle.py",
    "experiments/counterfactual_adaptive_cfg_v3/oracle/oracle_config.yaml",
    "experiments/counterfactual_adaptive_cfg_v3/oracle/oracle_seeds.json",
    "experiments/counterfactual_adaptive_cfg_v3/preregistered_future_seeds.json",
    "experiments/counterfactual_adaptive_cfg_v3/fresh_seed_audit.json",
    "experiments/counterfactual_adaptive_cfg_v3/run_oracle_generation.py",
    "experiments/counterfactual_adaptive_cfg_v3/prepare_oracle_outputs.py",
    "experiments/counterfactual_adaptive_cfg_v3/run_oracle_quality.py",
    "experiments/counterfactual_adaptive_cfg_v3/analyze_oracle.py",
    "experiments/counterfactual_adaptive_cfg_v3/run_predictability.py",
    "experiments/counterfactual_adaptive_cfg_v3/run_frozen_pipeline.py",
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
        "frozen_before_oracle_generation": True,
        "git_commit": "1af301c5e534ff6b9b9f96ea5dab48036d9c46af",
        "branch": "experiment/counterfactual-adaptive-cfg-v3",
        "sha256": hashes,
        "allowed_post_outcome_changes": "technical reporting only; no method, seed, feature, model, objective, or gate changes",
        "surrogate_property_eval": True,
        "dft_verified": False,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
