"""Fail-closed audit of the completed A diagnostic and paired P0 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
METHODS = ("B0", "F0")
SEEDS = tuple(range(710032, 710048))
MODEL_SHA = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
MATTERSIM_SHA = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = (
        "reference_review.md", "diagnostic_config.yaml", "physics_diagnostic.csv",
        "diagnostic_summary.json", "guidance_config.yaml", "guidance_trace.csv",
        "heldout_metrics.csv", "quality_metrics.csv", "efficiency_metrics.csv",
        "paired_results.csv", "decision_summary.json", "final_report.md",
        "p0_paired_per_seed.csv", "p0_bootstrap.csv", "p0_summary.csv",
        "p0_summary.json", "guidance_manifest.json", "guidance_geometry_test.json",
    )
    missing = [name for name in required if not (ROOT / name).is_file()]
    assert not missing, missing
    diagnostic = json.loads((ROOT / "diagnostic_summary.json").read_text())
    decision = json.loads((ROOT / "decision_summary.json").read_text())
    assert diagnostic["A_DIAGNOSTIC"] == "GO"
    assert diagnostic["LATE_PHYSICS_SIGNAL_RELIABLE"] is True
    assert decision["LATE_PHYSICS_SIGNAL_RELIABLE"] == "YES"
    assert decision["MATTERSIM_FORCE_GUIDANCE_P0"] == "GO"
    assert decision["NEXT"] == "FORMAL32"
    assert decision["formal32_run"] is False and decision["formal256_run"] is False

    heldout = rows(ROOT / "heldout_metrics.csv")
    paired = rows(ROOT / "paired_results.csv")
    trace = rows(ROOT / "guidance_trace.csv")
    assert len(heldout) == 32 and len(paired) == 16 and len(trace) == 320
    for method in METHODS:
        assert tuple(int(row["seed"]) for row in heldout if row["method"] == method) == SEEDS
    assert max(float(row["max_cart_correction_a"]) for row in trace) <= 0.01 + 1e-12
    assert max(float(row["max_cart_correction_a"]) for row in trace) <= 0.02 + 1e-12
    assert all(float(row["t_norm"]) <= 0.02 + 1e-8 for row in trace)
    assert {row["phase"] for row in trace} == {"predictor"}
    assert all(row["accepted"] == "True" and row["fallback"] == "False" for row in trace)
    assert min(float(row["min_distance_after_a"]) for row in trace) >= 0.5

    for method in METHODS:
        for seed in SEEDS:
            summary = json.loads(
                (ROOT / "p0_generation" / method / str(seed) / "run_summary.json").read_text()
            )
            assert summary["success"] is True and int(summary["seed"]) == seed
            assert summary["adaptive_cfg_used"] is False
            assert summary["stress_guidance"] is False
            assert summary["cell_guidance"] is False
            assert summary["atomic_guidance"] is False
            assert float(summary["guidance_scale"]) == 2.0
            assert int(summary["sampling_steps"]) == 1000
            assert int(summary["corrector_steps_per_timestep"]) == 1
            assert int(summary["mattergen_score_calls"]) == 2000
            assert summary["checkpoint_sha256"] == MODEL_SHA
            if method == "F0":
                assert summary["position_force_guidance"] is True
                assert summary["mattersim_checkpoint_sha256"] == MATTERSIM_SHA
            else:
                assert summary["position_force_guidance"] is False

    manifest = json.loads((ROOT / "guidance_manifest.json").read_text())
    assert manifest["mattersim_package_version"] == "1.2.3"
    assert manifest["chgnet_package_version"] == "0.3.0"
    assert len(manifest["stash_list_preserved"]) == 5
    assert sha256(Path(manifest["mattergen_checkpoint"])) == MODEL_SHA
    assert sha256(Path(manifest["mattersim_checkpoint"])) == MATTERSIM_SHA
    subprocess.run(["git", "diff", "--check"], cwd=PROJECT_ROOT, check=True)
    payload = {
        "AUDIT": "PASS", "required_files_present": len(required),
        "diagnostic_verdict": diagnostic["A_DIAGNOSTIC"],
        "p0_verdict": decision["MATTERSIM_FORCE_GUIDANCE_P0"],
        "n_paired": len(paired), "n_heldout_rows": len(heldout),
        "guidance_trace_rows": len(trace),
        "guidance_fallbacks": sum(row["fallback"] == "True" for row in trace),
        "maximum_cartesian_correction_a": max(float(row["max_cart_correction_a"]) for row in trace),
        "minimum_accepted_candidate_distance_a": min(float(row["min_distance_after_a"]) for row in trace),
        "stash_count_preserved": len(manifest["stash_list_preserved"]),
        "formal32_run": False, "formal256_run": False,
    }
    (ROOT / "audit_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
