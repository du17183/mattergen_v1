"""Serialize the immutable execution-file hash manifest before generation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/frozen_linear_k2_confirmation"
DESTINATION = ROOT / "protocol/frozen_execution_manifest.json"
FIELD_SOURCE = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_field_cfg/"
    "mattergen/diffusion/sampling/field_decoupled_cfg.py"
)
PHASE_B_SOURCE = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/"
    "experiments/reference_preserved_budgeted_cfg/phase_b_sampler.py"
)
FILES = (
    "confirmatory_sampler.py",
    "run_generation.py",
    "prepare_phase.py",
    "run_quality.py",
    "analyze_phase.py",
    "run_pipeline.py",
    "protocol/frozen_method_manifest.json",
    "protocol/protocol.md",
    "protocol/protocol_amendment.md",
    "protocol/reconstruction_audit.json",
    "protocol/frozen_linear_k2_pipeline.joblib",
    "protocol/frozen_linear_k2_parameters.json",
    "protocol/seed_manifest_256.json",
    "protocol/seed_overlap_audit.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=PROJECT, text=True).strip()


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"execution manifest already frozen: {DESTINATION}")
    audit = json.loads((ROOT / "protocol/reconstruction_audit.json").read_text())
    overlap = json.loads((ROOT / "protocol/seed_overlap_audit.json").read_text())
    if not audit.get("success") or audit.get("artifact_status") != "RECONSTRUCTED_ARTIFACT":
        raise RuntimeError("allocator reconstruction did not pass")
    if overlap.get("new_vs_historical_overlap") or overlap.get("c1_c2_overlap"):
        raise RuntimeError("confirmatory seed overlap audit did not pass")
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_CONFIRMATORY_GENERATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": git("branch", "--show-current"),
        "git_base_head": git("rev-parse", "HEAD"),
        "execution_files_sha256": {
            relative: sha256(ROOT / relative) for relative in FILES
        },
        "external_source_sha256": {
            str(FIELD_SOURCE): sha256(FIELD_SOURCE),
            str(PHASE_B_SOURCE): sha256(PHASE_B_SOURCE),
        },
        "reconstructed_artifact_sha256": audit["artifact_sha256"],
        "registered_seed_manifest_sha256": sha256(ROOT / "protocol/seed_manifest_256.json"),
        "registered_seed_count": 256,
        "historical_seed_overlap_count": 0,
        "c1_c2_seed_overlap_count": 0,
    }
    DESTINATION.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
