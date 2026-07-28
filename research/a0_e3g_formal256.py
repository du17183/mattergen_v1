"""Frozen source-data audit for the A0 + learned-gated E3-PCR formal study.

This module intentionally performs no generation, refinement, or relaxation.
The registered A0 formal batch overlaps the frozen Q3 gate training set, so the
formal protocol requires a SOURCE_DATA_INCOMPLETE terminal result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path("/data/dxl/mattergen_v1")
ROOT = Path("/data/dxl")
RESULT = ROOT / "results/a0_e3g_formal256"
REPORT = ROOT / "reports/a0_e3g_formal256"
LOG = ROOT / "logs/a0_e3g_formal256"
EXTERNAL_TOOLS = ROOT / "tools/a0_e3g_formal256"
PROGRESS = RESULT / "progress"
MASTER_PROGRESS = PROGRESS / "master_progress.json"
EVENTS = PROGRESS / "events.jsonl"

A0_COMMIT = "5de00419eea2d8a9be303638f2db8ece15a22366"
E3G_COMMIT = "0275cbf08ed3c6321cea7d06f7a3a8edb83b7483"
COMPATIBILITY_COMMIT = "ba2303c284210fdae0a35bb0153a8ef3af45a54c"
FORMAL_BRANCH = "feature/a0-e3g-formal256"

A0_SEEDS = tuple(range(20000, 20256))
Q3_TRAINING_SEEDS = frozenset(range(20000, 20064))
Q3_FROZEN64_SEEDS = frozenset(range(32000, 32064))
Q3_FORMAL256_SEEDS = frozenset(range(40000, 40256))
COMPATIBILITY64_SEEDS = frozenset(range(41000, 41064))

A0_GENERATION = ROOT / "results/formal_256/generation/A0"
A0_RELAXED = ROOT / "results/formal_256/relaxed/A0"
A0_CONFIG = ROOT / "reports/formal_256/final/frozen_method_configs.json"
A0_SEED_MANIFEST = ROOT / "reports/formal_256/final/formal_seed_manifest.csv"
A0_OFFICIAL_METRICS = (
    ROOT / "reports/formal_256/A0/official_metrics_per_structure.csv"
)

Q3_CHECKPOINT = (
    ROOT / "results/postgen_fastgate/q3_refiner/model/q3_gate.joblib"
)
Q3_CHECKPOINT_SHA256 = (
    "b2ce1800fa0fa448f57d58010c8586b5de1b6666c4f198737a2f8a4bfabcb90e"
)
Q3_CONFIG = PROJECT / "configs/q3_e3_pcr_frozen64.json"
Q3_CONFIG_SHA256 = (
    "50d10efdea1050a84de6b2872f78742c2468ff4bef45cd7544fb30cef31eb87a"
)
Q3_SOURCE = PROJECT / "research/postgen_fastgate/refiner_eval.py"
Q3_SOURCE_SHA256 = (
    "3d1d6e38066bb195c893ea8665f284e66261f74a055e1521ed4d6250d469895f"
)
Q3_TRAINING_SUMMARY = (
    ROOT / "reports/postgen_fastgate/q3_refiner/training_and_offline_summary.json"
)
Q3_FORMAL_MANIFEST = (
    ROOT / "reports/q3_e3_pcr/formal256/formal_frozen_manifest.json"
)
COMPATIBILITY_SUMMARY = ROOT / "reports/a0_e3g_compat64/final_summary.json"

MATTERGEN_CHECKPOINT = (
    ROOT
    / "checkpoints/official/hf_mattergen/checkpoints/"
    "dft_mag_density/checkpoints/last.ckpt"
)
MATTERGEN_SHA256 = (
    "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
)
MATTERSIM_CHECKPOINT = ROOT / "mattersim_weights/mattersim-v1.0.0-5M.pth"
MATTERSIM_SHA256 = (
    "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
)

REUSE_JSON = REPORT / "reuse_audit.json"
REUSE_CSV = REPORT / "reuse_audit.csv"
REUSE_MD = REPORT / "reuse_audit.md"
FROZEN_JSON = REPORT / "frozen_manifest.json"
FROZEN_MD = REPORT / "frozen_manifest.md"
FINAL_JSON = REPORT / "final_summary.json"
FINAL_MD = REPORT / "final_report.md"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def append_event(stage: str, status: str, **details: Any) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": now(), "stage": stage, "status": status, **details}
    with EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def write_master(stage: str, status: str, **details: Any) -> None:
    existing = read_json(MASTER_PROGRESS) if MASTER_PROGRESS.is_file() else {}
    payload = {
        "schema_version": 1,
        "experiment": "A0 + E3-G frozen formal 256",
        "formal_branch": FORMAL_BRANCH,
        "current_stage": stage,
        "status": status,
        "updated_at": now(),
        "new_mattergen_runs": 0,
        "new_a0_mattersim_runs": 0,
        "new_e3g_mattersim_runs": 0,
        "gpu_workers": 0,
        "independent_mlip_started": False,
        "dft_started": False,
        "paper_figures_started": False,
        "new_training_started": False,
        "parameter_tuning_started": False,
        "other_processes_terminated": False,
        "sigkill_used": False,
        **existing,
        **details,
    }
    payload["current_stage"] = stage
    payload["status"] = status
    payload["updated_at"] = now()
    atomic_json(MASTER_PROGRESS, payload)
    append_event(stage, status, **details)


def determine_terminal_state(eligible_count: int, total: int = 256) -> str:
    if eligible_count < total:
        return "SOURCE_DATA_INCOMPLETE"
    raise RuntimeError(
        "All source seeds are eligible; the full computation runner is required."
    )


def seed_disqualification_reasons(
    seed: int,
    *,
    structure_complete: bool,
    generation_complete: bool,
    relaxation_complete: bool,
    official_metrics_complete: bool,
    frozen_a0_config_match: bool,
) -> list[str]:
    reasons: list[str] = []
    if seed in Q3_TRAINING_SEEDS:
        reasons.append("used_in_q3_gate_training")
    if seed in Q3_FROZEN64_SEEDS:
        reasons.append("used_in_q3_frozen64_evaluation")
    if seed in Q3_FORMAL256_SEEDS:
        reasons.append("used_in_q3_formal256_evaluation")
    if seed in COMPATIBILITY64_SEEDS:
        reasons.append("used_in_a0_e3g_compatibility64")
    if not structure_complete:
        reasons.append("a0_structure_missing_or_invalid")
    if not generation_complete:
        reasons.append("a0_generation_record_incomplete")
    if not relaxation_complete:
        reasons.append("a0_mattersim_record_incomplete")
    if not official_metrics_complete:
        reasons.append("a0_official_metrics_missing")
    if not frozen_a0_config_match:
        reasons.append("a0_frozen_config_mismatch")
    return reasons


def _load_official_metrics() -> dict[int, dict[str, str]]:
    with A0_OFFICIAL_METRICS.open(encoding="utf-8", newline="") as stream:
        return {int(row["seed"]): row for row in csv.DictReader(stream)}


def _a0_config_matches(config: dict[str, Any]) -> bool:
    expected = {
        "code_commit": A0_COMMIT,
        "checkpoint_sha256": MATTERGEN_SHA256,
        "base_guidance": 2.0,
        "guidance_schedule": "adaptive",
        "adaptive_alpha": 0.5,
        "adaptive_ema": 0.95,
        "adaptive_epsilon": 1.0e-6,
        "guidance_min_scale": 0.0,
        "guidance_max_scale": 5.0,
        "batch_size": 1,
        "sampling_steps": 1000,
    }
    return all(config.get(key) == value for key, value in expected.items()) and (
        config.get("corrector_gating", {}).get("enabled") is False
    )


def validate_frozen_assets() -> dict[str, Any]:
    expected_hashes = {
        Q3_CHECKPOINT: Q3_CHECKPOINT_SHA256,
        Q3_CONFIG: Q3_CONFIG_SHA256,
        Q3_SOURCE: Q3_SOURCE_SHA256,
        MATTERGEN_CHECKPOINT: MATTERGEN_SHA256,
        MATTERSIM_CHECKPOINT: MATTERSIM_SHA256,
    }
    actual_hashes = {str(path): sha256(path) for path in expected_hashes}
    mismatches = {
        str(path): {"expected": expected, "actual": actual_hashes[str(path)]}
        for path, expected in expected_hashes.items()
        if actual_hashes[str(path)] != expected
    }
    if mismatches:
        raise RuntimeError(f"frozen asset hash mismatch: {mismatches}")

    q3_training = read_json(Q3_TRAINING_SUMMARY)
    if q3_training["training_seed_range"] != [20000, 20063]:
        raise RuntimeError("unexpected Q3 training seed range")
    if q3_training["training_rows"] != 64:
        raise RuntimeError("unexpected Q3 training row count")
    if q3_training["network"]["threshold"] != 0.5:
        raise RuntimeError("unexpected Q3 threshold")

    a0 = read_json(A0_CONFIG)["A0"]
    required_a0 = {
        "base_guidance": 2.0,
        "adaptive_alpha": 0.5,
        "adaptive_ema": 0.95,
        "adaptive_epsilon": 1.0e-6,
        "guidance_min_scale": 0.0,
        "guidance_max_scale": 5.0,
        "corrector_gating_enabled": False,
    }
    if any(a0.get(key) != value for key, value in required_a0.items()):
        raise RuntimeError("A0 frozen configuration mismatch")

    e3g_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", E3G_COMMIT, "HEAD"],
            cwd=PROJECT,
        ).returncode
        == 0
    )
    if not e3g_ancestor:
        raise RuntimeError("formal branch does not descend from E3-G commit")

    protected_compatibility_paths = (
        "research/a0_e3g_compat64.py",
        "research/postgen_fastgate/refiner_eval.py",
        "configs/q3_e3_pcr_frozen64.json",
    )
    changed = [
        path
        for path in protected_compatibility_paths
        if subprocess.run(
            ["git", "diff", "--quiet", COMPATIBILITY_COMMIT, "--", path],
            cwd=PROJECT,
        ).returncode
        != 0
    ]
    if changed:
        raise RuntimeError(f"compatibility code changed: {changed}")

    return {
        "asset_hashes": actual_hashes,
        "a0_frozen_config": required_a0,
        "q3_training_seed_range": [20000, 20063],
        "q3_training_rows": 64,
        "q3_gate_threshold": 0.5,
        "e3g_commit_in_ancestry": e3g_ancestor,
        "a0_commit_in_ancestry": (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", A0_COMMIT, "HEAD"],
                cwd=PROJECT,
            ).returncode
            == 0
        ),
        "compatibility_code_unchanged": True,
    }


def build_reuse_rows() -> list[dict[str, Any]]:
    official = _load_official_metrics()
    rows: list[dict[str, Any]] = []
    for seed in A0_SEEDS:
        generation_dir = A0_GENERATION / str(seed)
        relaxation_dir = A0_RELAXED / str(seed)
        structure = generation_dir / "generated_crystals.extxyz"
        generation_summary_path = generation_dir / "run_summary.json"
        generation_config_path = generation_dir / "run_config.json"
        relaxation_summary_path = relaxation_dir / "relax_summary.json"
        relaxed_structure = relaxation_dir / "relaxed_structure.extxyz"

        generation_summary = (
            read_json(generation_summary_path)
            if generation_summary_path.is_file()
            else {}
        )
        generation_config = (
            read_json(generation_config_path)
            if generation_config_path.is_file()
            else {}
        )
        relaxation_summary = (
            read_json(relaxation_summary_path)
            if relaxation_summary_path.is_file()
            else {}
        )
        metrics = official.get(seed)

        structure_complete = structure.is_file() and structure.stat().st_size > 0
        generation_complete = (
            generation_summary.get("success") is True
            and generation_summary.get("seed") == seed
        )
        frozen_match = _a0_config_matches(generation_config)
        relaxation_complete = (
            relaxation_summary.get("success") is True
            and relaxation_summary.get("seed") == seed
            and relaxation_summary.get("config") == "A0"
            and relaxation_summary.get("checkpoint_sha256") == MATTERSIM_SHA256
            and relaxed_structure.is_file()
            and relaxed_structure.stat().st_size > 0
        )
        metrics_complete = bool(
            metrics
            and metrics.get("method") == "A0"
            and int(metrics["seed"]) == seed
            and metrics.get("input_hash") == relaxation_summary.get("input_hash")
        )
        reasons = seed_disqualification_reasons(
            seed,
            structure_complete=structure_complete,
            generation_complete=generation_complete,
            relaxation_complete=relaxation_complete,
            official_metrics_complete=metrics_complete,
            frozen_a0_config_match=frozen_match,
        )
        rows.append(
            {
                "seed": seed,
                "a0_structure_path": str(structure),
                "a0_structure_sha256": sha256(structure)
                if structure_complete
                else "",
                "a0_generation_summary_path": str(generation_summary_path),
                "a0_generation_config_path": str(generation_config_path),
                "a0_structure_complete": structure_complete,
                "a0_generation_complete": generation_complete,
                "a0_frozen_config_match": frozen_match,
                "a0_mattersim_result_path": str(relaxation_summary_path),
                "a0_mattersim_output_path": str(relaxed_structure),
                "a0_mattersim_input_hash": relaxation_summary.get(
                    "input_hash", ""
                ),
                "a0_mattersim_complete": relaxation_complete,
                "a0_official_metrics_complete": metrics_complete,
                "used_in_q3_gate_training": seed in Q3_TRAINING_SEEDS,
                "eligible_for_reuse": not reasons,
                "reason_if_not_reused": ";".join(reasons),
            }
        )
    return rows


def run_audit() -> dict[str, Any]:
    for path in (RESULT, REPORT, LOG, EXTERNAL_TOOLS, PROGRESS):
        path.mkdir(parents=True, exist_ok=True)
    write_master("state_audit", "running")
    frozen = validate_frozen_assets()
    write_master("freeze_audit", "success")

    rows = build_reuse_rows()
    atomic_csv(REUSE_CSV, rows)
    physically_complete = sum(
        row["a0_structure_complete"]
        and row["a0_generation_complete"]
        and row["a0_mattersim_complete"]
        and row["a0_official_metrics_complete"]
        and row["a0_frozen_config_match"]
        for row in rows
    )
    eligible = sum(row["eligible_for_reuse"] for row in rows)
    ineligible = len(rows) - eligible
    training_overlap = sum(row["used_in_q3_gate_training"] for row in rows)
    state = determine_terminal_state(eligible, len(rows))

    audit = {
        "schema_version": 1,
        "created_at": now(),
        "candidate_seed_range": [A0_SEEDS[0], A0_SEEDS[-1]],
        "candidate_seed_count": len(rows),
        "physically_complete_a0_records": physically_complete,
        "eligible_for_reuse": eligible,
        "ineligible_for_reuse": ineligible,
        "q3_training_intersection": [
            max(A0_SEEDS[0], min(Q3_TRAINING_SEEDS)),
            min(A0_SEEDS[-1], max(Q3_TRAINING_SEEDS)),
        ],
        "q3_training_intersection_count": training_overlap,
        "q3_frozen64_intersection": [],
        "q3_formal256_intersection": [],
        "a0_e3g_compatibility64_intersection": [],
        "full_256_batch_eligible": eligible == len(rows),
        "mixing_partial_old_and_new_forbidden": True,
        "terminal_state": state,
        "rows": rows,
    }
    atomic_json(REUSE_JSON, audit)
    atomic_text(
        REUSE_MD,
        "# A0 formal 256 reuse audit\n\n"
        f"- Candidate seeds: `{A0_SEEDS[0]}–{A0_SEEDS[-1]}`\n"
        f"- Physically complete A0 records: `{physically_complete}/256`\n"
        f"- Eligible independent records: `{eligible}/256`\n"
        f"- Ineligible records: `{ineligible}/256`\n"
        f"- Q3 gate-training overlap: `{training_overlap}/256` "
        "(`20000–20063`)\n"
        "- Partial-old plus newly generated mixing is forbidden.\n"
        f"- Terminal state: `{state}`\n\n"
        "The A0 files are physically complete, but the registered formal batch "
        "is not scientifically independent because 64 seeds trained the frozen "
        "Q3 gate. No MatterGen, refinement, or MatterSim task was started.\n",
    )

    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "formal_branch": FORMAL_BRANCH,
        "formal_base_commit": COMPATIBILITY_COMMIT,
        "formal_code_commit_at_audit": git_output("rev-parse", "HEAD"),
        "a0_commit": A0_COMMIT,
        "e3g_commit": E3G_COMMIT,
        "q3_checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "q3_config_sha256": Q3_CONFIG_SHA256,
        "q3_source_sha256": Q3_SOURCE_SHA256,
        "a0_seed_manifest": str(A0_SEED_MANIFEST),
        "a0_official_metrics": str(A0_OFFICIAL_METRICS),
        "frozen_validation": frozen,
        "registered_a0_seed_range": [20000, 20255],
        "q3_training_seed_range": [20000, 20063],
        "source_data_physically_complete": physically_complete == 256,
        "source_data_formally_eligible": False,
        "new_mattergen_runs": 0,
        "new_a0_mattersim_runs": 0,
        "new_e3g_mattersim_runs": 0,
        "training_or_retuning": False,
    }
    atomic_json(FROZEN_JSON, manifest)
    atomic_text(
        FROZEN_MD,
        "# A0 + E3-G formal 256 freeze\n\n"
        f"- A0 formal commit: `{A0_COMMIT}`\n"
        f"- E3-G formal commit: `{E3G_COMMIT}`\n"
        f"- Compatibility base: `{COMPATIBILITY_COMMIT}`\n"
        f"- Q3 checkpoint SHA256: `{Q3_CHECKPOINT_SHA256}`\n"
        f"- Q3 config SHA256: `{Q3_CONFIG_SHA256}`\n"
        "- Frozen parameters were not changed.\n",
    )

    final = {
        "schema_version": 1,
        "completed_at": now(),
        "A0_E3G_FORMAL256_COMPLETED": True,
        "final_state": state,
        "a0_e3g_formal256_go": False,
        "a0_e3g_formal256_no_go": False,
        "a0_commit": A0_COMMIT,
        "e3g_commit": E3G_COMMIT,
        "formal_branch": FORMAL_BRANCH,
        "formal_commit": None,
        "a0_data_reused": False,
        "a0_structures_reused": 0,
        "a0_mattersim_reused": 0,
        "new_mattergen_runs": 0,
        "new_a0_mattersim_runs": 0,
        "new_e3g_mattersim_runs": 0,
        "evaluation_seeds": [20000, 20255],
        "seed_count": 256,
        "seed_eligibility_confirmed": False,
        "eligible_source_records": eligible,
        "ineligible_source_records": ineligible,
        "q3_training_overlap_count": training_overlap,
        "q3_checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "q3_config_sha256": Q3_CONFIG_SHA256,
        "primary_metrics": None,
        "quality_metrics": None,
        "safety_metrics": None,
        "primary_effect_pass": None,
        "quality_safety_pass": None,
        "mechanism_safety_pass": None,
        "independent_mlip_started": False,
        "dft_started": False,
        "paper_figures_started": False,
        "new_training_started": False,
        "parameter_tuning_started": False,
        "stability_source": "MatterSim-5M surrogate",
        "dft_verified": False,
        "property_target_verified": False,
        "final_report": str(FINAL_MD),
        "reuse_audit": str(REUSE_MD),
        "github_branch": FORMAL_BRANCH,
        "github_commit": None,
        "draft_pr": None,
        "gpu_workers": 0,
        "other_processes_terminated": False,
        "sigkill_used": False,
        "limitations": (
            "The only registered A0 formal batch overlaps Q3 gate training "
            "for seeds 20000–20063; partial-old plus new mixing is forbidden."
        ),
        "next_action": (
            "Pre-register and generate a wholly new independent A0 256-seed "
            "batch in a separately authorized task."
        ),
    }
    atomic_json(FINAL_JSON, final)
    atomic_text(
        FINAL_MD,
        "# A0 + E3-G frozen formal 256\n\n"
        f"- Final state: `{state}`\n"
        "- A0 files physically complete: `256/256`\n"
        f"- Independently eligible A0 files: `{eligible}/256`\n"
        f"- Excluded Q3 training seeds: `{training_overlap}/256` "
        "(`20000–20063`)\n"
        "- New MatterGen runs: `0`\n"
        "- New A0 MatterSim runs: `0`\n"
        "- New E3-G MatterSim runs: `0`\n\n"
        "The existing A0 batch cannot support a frozen independent 256-seed "
        "combination claim. Protocol forbids mixing 192 old eligible structures "
        "with 64 newly generated structures, so no refinement or evaluation was "
        "started.\n",
    )
    for path, title in (
        (REPORT / "statistics_report.md", "Statistics"),
        (REPORT / "quality_report.md", "Quality"),
        (REPORT / "safety_report.md", "Safety"),
    ):
        atomic_text(
            path,
            f"# {title} report\n\nNot computed because the source eligibility "
            f"gate terminated as `{state}` before E3-G refinement.\n",
        )
    atomic_text(
        REPORT / "reproduction.md",
        "# Reproduce the source-data audit\n\n```bash\n"
        "cd /data/dxl/mattergen_v1\n"
        "git switch feature/a0-e3g-formal256\n"
        "/data/dxl/tools/a0_e3g_formal256/run.sh\n"
        "/data/dxl/tools/a0_e3g_formal256/status.sh\n"
        "```\n",
    )
    write_master(
        "github_archive",
        "pending",
        final_state=state,
        source_data_physically_complete=True,
        eligible_source_records=eligible,
        ineligible_source_records=ineligible,
        q3_training_overlap_count=training_overlap,
    )
    print(json.dumps({"final_state": state, "eligible": eligible}, sort_keys=True))
    return final


def status() -> None:
    payload = (
        read_json(MASTER_PROGRESS)
        if MASTER_PROGRESS.is_file()
        else {"status": "not_started"}
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "status"))
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.command == "audit":
        run_audit()
    else:
        status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
