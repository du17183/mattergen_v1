#!/usr/bin/env python3
"""Validate the compact thesis release without running model inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
HASH_MANIFEST = ROOT / "release_checksums.json"

EXPECTED_STATUS = {
    "Adaptive CFG V1": "MIXED",
    "Robust V2": "FAIL",
    "Counterfactual Oracle V3": "SUPPORTED_MECHANISM",
    "Risk-Calibrated V4": "FAIL",
    "Safe Selection V5": "FAIL",
    "Stage-Calibrated CFG": "MIXED",
    "Field-Decoupled CFG": "FAIL",
    "Branch-Compatible Oracle": "SUPPORTED",
    "Phase B Linear-K2": "NOT_SUPPORTED",
    "C1 Fixed-K2": "SUPPORTED",
    "C1 Linear-K2": "NOT_SUPPORTED",
    "RC-NFGD P0": "PASS",
    "RC-NFGD Formal32": "PASS",
    "RC-NFGD Formal256": "SUPPORTED",
    "CHGNet evaluation": "POSITIVE",
}

EXPECTED_MAIN = {
    ("Fixed-K2", "C1-128", "Property MAE"): (0.03479644444407384, 0.026332355926007578),
    ("RC-NFGD", "Formal256", "MaxF"): (0.22640836794160712, 0.15891148914495132),
    ("RC-NFGD", "Formal256", "Mean Force"): (0.09870230579787798, 0.06937279273450948),
    ("RC-NFGD", "Formal256", "RMSD"): (0.051136946776399364, 0.04389322435605705),
    ("RC-NFGD", "CHGNet Formal256", "MaxF"): (0.1938906372800398, 0.16605730423045764),
}

REQUIRED = [
    "README.md",
    "CLAIMS_AND_LIMITATIONS.md",
    "REPRODUCIBILITY.md",
    "combined_summary/main_results.csv",
    "combined_summary/experiment_status.csv",
    "combined_summary/compute_summary.csv",
    "combined_summary/figure_index.md",
    "innovation1/tables/final_confirmatory_results.csv",
    "innovation1/results/c1_paired_results.csv",
    "innovation1/seeds/confirmatory_seed_manifest_256.json",
    "innovation1/negative_results/reconstruction_audit.json",
    "innovation1/negative_results/frozen_linear_k2_pipeline.joblib",
    "innovation2/tables/formal256_main_results.csv",
    "innovation2/tables/chgnet_validation.csv",
    "innovation2/results/mattersim_late_force_guidance_formal256/paired_results.csv",
    "innovation2/results/chgnet_formal256/independent_per_structure.csv",
    "innovation2/seeds/formal256_seeds.json",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hash_paths() -> list[Path]:
    paths: set[Path] = {REPO / "mattergen/diffusion/sampling/field_decoupled_cfg.py"}
    for innovation in ("innovation1", "innovation2"):
        for group in ("method", "configs", "results", "seeds", "negative_results"):
            base = ROOT / innovation / group
            if base.exists():
                paths.update(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    return sorted(paths)


def write_hash_manifest() -> None:
    files = {path.relative_to(REPO).as_posix(): sha256(path) for path in frozen_hash_paths()}
    payload = {
        "algorithm": "sha256",
        "purpose": "Integrity manifest for frozen method, configuration, seed, and result artifacts",
        "files": files,
    }
    HASH_MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_close(actual: float, expected: float, *, atol: float = 1e-12) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise AssertionError(f"{actual!r} != {expected!r}")


def test_reference_equivalence() -> None:
    """All field scales at 2.0 must exactly reproduce the scalar C0 rule."""
    uncond = {
        "atomic": np.asarray([0.125, -0.5, 1.0], dtype=np.float64),
        "position": np.asarray([[0.25, -0.125, 0.5], [-1.0, 0.75, 0.0]], dtype=np.float64),
        "cell": np.asarray([[1.0, 0.2], [-0.4, 0.8]], dtype=np.float64),
    }
    cond = {name: value + 0.03125 for name, value in uncond.items()}
    scalar = {name: uncond[name] + 2.0 * (cond[name] - uncond[name]) for name in uncond}
    field = {name: uncond[name] + 2.0 * (cond[name] - uncond[name]) for name in uncond}
    if not np.array_equal(scalar["atomic"], field["atomic"]):
        raise AssertionError("atomic reference continuation is not exact")
    if float(np.max(np.abs(scalar["position"] - field["position"]))) != 0.0:
        raise AssertionError("position max abs is not zero")
    if float(np.max(np.abs(scalar["cell"] - field["cell"]))) != 0.0:
        raise AssertionError("cell max abs is not zero")


def test_coordinate_and_force_helpers() -> None:
    """CPU-only checks for the RC-NFGD coordinate map and bounded force step."""
    cell = np.asarray([[4.0, 0.2, 0.0], [0.0, 3.5, 0.1], [0.0, 0.0, 5.0]])
    fractional = np.asarray([[0.1, 0.2, 0.3], [-0.05, 0.04, 0.02]])
    cartesian = fractional @ cell
    recovered = cartesian @ np.linalg.inv(cell)
    if not np.allclose(fractional, recovered, rtol=0.0, atol=1e-14):
        raise AssertionError("Cartesian/fractional round trip failed")

    force = np.asarray([[2.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])
    centered = force - force.mean(axis=0, keepdims=True)
    reference, nominal, hard_cap = 0.07795149218357911, 0.005, 0.01
    scale = nominal / max(float(np.linalg.norm(centered, axis=1).max()), reference)
    correction = centered * scale
    max_norm = float(np.linalg.norm(correction, axis=1).max())
    if max_norm > hard_cap:
        correction *= hard_cap / max_norm
    if correction.shape != force.shape or not np.isfinite(correction).all():
        raise AssertionError("force correction shape/finite check failed")
    if float(np.sum(correction * centered)) <= 0.0:
        raise AssertionError("force correction does not follow the energy-descent force direction")
    if float(np.linalg.norm(correction, axis=1).max()) > hard_cap + 1e-15:
        raise AssertionError("force correction exceeds hard cap")


def check_required_files() -> None:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
    if missing:
        raise AssertionError(f"missing required files: {missing}")


def check_rows_and_metrics() -> None:
    expected_rows = {
        "innovation1/results/c1_paired_results.csv": 128,
        "innovation1/tables/final_confirmatory_results.csv": 4,
        "innovation2/results/mattersim_late_force_guidance_formal256/paired_results.csv": 256,
        "innovation2/results/chgnet_formal256/independent_per_structure.csv": 512,
        "innovation2/tables/formal256_main_results.csv": 6,
        "innovation2/tables/chgnet_validation.csv": 2,
    }
    for relative, expected in expected_rows.items():
        actual = len(read_csv(ROOT / relative))
        if actual != expected:
            raise AssertionError(f"{relative}: expected {expected} rows, got {actual}")

    main = read_csv(ROOT / "combined_summary/main_results.csv")
    for row in main:
        for field in ("n", "baseline_value", "method_value", "absolute_delta", "relative_delta"):
            if not math.isfinite(float(row[field])):
                raise AssertionError(f"non-finite {field} in main_results.csv")
    indexed = {(row["method"], row["cohort"], row["metric"]): row for row in main}
    for key, (baseline, method) in EXPECTED_MAIN.items():
        if key not in indexed:
            raise AssertionError(f"missing main result {key}")
        assert_close(float(indexed[key]["baseline_value"]), baseline)
        assert_close(float(indexed[key]["method_value"]), method)


def check_statuses_and_dft() -> None:
    rows = read_csv(ROOT / "combined_summary/experiment_status.csv")
    status = {row["Experiment"]: row["Status"] for row in rows}
    if status != EXPECTED_STATUS:
        raise AssertionError(f"experiment status mismatch: {status}")

    final = json.loads((ROOT / "innovation2/results/final_decision.json").read_text(encoding="utf-8"))
    if final.get("DFT_VERIFIED") is not False:
        raise AssertionError("DFT_VERIFIED must be false")
    chgnet = read_csv(ROOT / "innovation2/tables/chgnet_validation.csv")
    if any(row["DFT verified"].lower() != "false" for row in chgnet):
        raise AssertionError("CHGNet must be labelled as a surrogate, not DFT")


def check_reconstruction() -> None:
    path = ROOT / "innovation1/negative_results/reconstruction_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    checks = audit["top2_checks"]
    if audit.get("artifact_status") != "RECONSTRUCTED_ARTIFACT" or audit.get("success") is not True:
        raise AssertionError("reconstructed artifact audit is not successful")
    if checks.get("test_exact_matches") != 16 or checks.get("test_seed_count") != 16:
        raise AssertionError("Phase B held-out Top-2 exact reproduction is not 16/16")
    artifact = ROOT / "innovation1/negative_results/frozen_linear_k2_pipeline.joblib"
    if sha256(artifact) != audit["artifact_sha256"]:
        raise AssertionError("reconstructed allocator artifact hash mismatch")


def check_seed_manifests_and_configs() -> None:
    i1 = json.loads((ROOT / "innovation1/seeds/confirmatory_seed_manifest_256.json").read_text(encoding="utf-8"))
    if i1.get("count") != 256 or len(i1.get("seeds", [])) != 256:
        raise AssertionError("Innovation 1 registered seed manifest is incomplete")
    i2 = json.loads((ROOT / "innovation2/seeds/formal256_seeds.json").read_text(encoding="utf-8"))
    paired = i2.get("paired_seeds", [])
    if len(paired) != 256 or i2.get("overlap_count") != 0:
        raise AssertionError("Innovation 2 Formal256 seed manifest is incomplete or overlaps history")
    config_text = (ROOT / "innovation2/configs/formal256_config.yaml").read_text(encoding="utf-8")
    for token in ("experiment:", "frozen: true", "methods:", "seeds:"):
        if token not in config_text:
            raise AssertionError(f"Formal256 config missing {token!r}")


def check_figures() -> None:
    stems = [f"I1_F{index}" for index in range(1, 7)] + [f"I2_F{index}" for index in range(1, 7)]
    files = list((ROOT / "innovation1/figures").glob("*")) + list((ROOT / "innovation2/figures").glob("*"))
    for prefix in stems:
        matched = [path for path in files if path.name.startswith(prefix)]
        suffixes = {path.suffix for path in matched}
        if suffixes != {".png", ".pdf", ".svg"} or any(path.stat().st_size == 0 for path in matched):
            raise AssertionError(f"incomplete figure outputs for {prefix}")


def check_hashes() -> None:
    if not HASH_MANIFEST.is_file():
        raise AssertionError("release_checksums.json is missing")
    payload = json.loads(HASH_MANIFEST.read_text(encoding="utf-8"))
    expected = payload.get("files", {})
    current_paths = frozen_hash_paths()
    current_names = {path.relative_to(REPO).as_posix() for path in current_paths}
    if set(expected) != current_names:
        raise AssertionError("hash manifest file set differs from frozen release inputs")
    mismatches = [name for name, digest in expected.items() if sha256(REPO / name) != digest]
    if mismatches:
        raise AssertionError(f"hash mismatch: {mismatches}")


def run_checks() -> None:
    check_required_files()
    check_rows_and_metrics()
    check_statuses_and_dft()
    check_reconstruction()
    check_seed_manifests_and_configs()
    test_reference_equivalence()
    test_coordinate_and_force_helpers()
    check_figures()
    check_hashes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-hashes", action="store_true", help="create/update the frozen artifact hash manifest")
    args = parser.parse_args()
    if args.write_hashes:
        write_hash_manifest()
        print(f"HASH_MANIFEST_WRITTEN={HASH_MANIFEST.relative_to(REPO)}")
    run_checks()
    print("RELEASE_VALIDATION=PASS")


if __name__ == "__main__":
    main()
