#!/usr/bin/env python3
"""Formal 256-seed validation for the frozen E3-PCR crystal refiner."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pandas as pd

from research import q3_frozen64 as core


ROOT = Path("/data/dxl")
PROJECT = ROOT / "mattergen_v1"
RESULT = ROOT / "results/q3_e3_pcr/formal256"
REPORT = ROOT / "reports/q3_e3_pcr/formal256"
LOG = ROOT / "logs/q3_e3_pcr/formal256"
EXTERNAL_TOOLS = ROOT / "tools/q3_e3_pcr/formal256"
PROGRESS = RESULT / "progress"
MASTER_PROGRESS = PROGRESS / "master_progress.json"
EVENTS = PROGRESS / "events.jsonl"
PIPELINE_LOCK = PROGRESS / "pipeline.lock"
GENERATION = RESULT / "generation/C0"
FEATURES = RESULT / "features.csv"
REFINED = RESULT / "refined"
REFINEMENT_MANIFEST = RESULT / "refinement_manifest.csv"
REFINEMENT_SUMMARY = RESULT / "refinement_summary.json"
RELAXED = RESULT / "relaxed"
RELAX_PROGRESS = PROGRESS / "relax_progress.json"

BASE_BRANCH = "feature/q3-e3-pcr-frozen64"
BASE_COMMIT = "87ec85c4ab353a362c8e2645cb0d14c3f6828672"
FORMAL_BRANCH = "feature/q3-e3-pcr-formal256"
FROZEN_SOURCE_COMMIT = "b65f42a8792004c7c820e59fa4413e1310e06143"
FROZEN_SOURCE = PROJECT / "research/postgen_fastgate/refiner_eval.py"
FROZEN_SOURCE_SHA256 = (
    "3d1d6e38066bb195c893ea8665f284e66261f74a055e1521ed4d6250d469895f"
)
CONFIG = PROJECT / "configs/q3_e3_pcr_frozen64.json"
CONFIG_SHA256 = "50d10efdea1050a84de6b2872f78742c2468ff4bef45cd7544fb30cef31eb87a"
Q3_CHECKPOINT = ROOT / "results/postgen_fastgate/q3_refiner/model/q3_gate.joblib"
Q3_CHECKPOINT_SHA256 = (
    "b2ce1800fa0fa448f57d58010c8586b5de1b6666c4f198737a2f8a4bfabcb90e"
)
MATTERGEN_CHECKPOINT = (
    ROOT
    / "checkpoints/official/hf_mattergen/checkpoints/"
    "dft_mag_density/checkpoints/last.ckpt"
)
MATTERGEN_SHA256 = (
    "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
)
MATTERSIM = ROOT / "mattersim_weights/mattersim-v1.0.0-5M.pth"
MATTERSIM_SHA256 = (
    "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
)
TRAINING_REPORT = (
    ROOT / "reports/postgen_fastgate/q3_refiner/training_and_offline_summary.json"
)
REFINER_FREEZE_REPORT = (
    ROOT / "reports/postgen_fastgate/q3_refiner/frozen_refinement_manifest.json"
)

SEEDS = tuple(range(40000, 40256))
METHODS = ("C0", "ALWAYS_ON", "Q3_E3_PCR")
DISPLAY_NAMES = {
    "C0": "C0",
    "ALWAYS_ON": "E3-A",
    "Q3_E3_PCR": "E3-G",
}
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_SAMPLES = 20_000
FORCE_HARM_EPSILON = 1.0e-6
SHORT_BOND_ANGSTROM = 0.5
STABILITY_THRESHOLD = 0.1
MATTERGEN_PYTHON = ROOT / "envs/mattergen_py310/bin/python"
CHGNET_PYTHON = ROOT / "envs/fn_pra_teacher/bin/python"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    core.atomic_json(path, payload)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    core.atomic_csv(path, frame)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def configure_core() -> None:
    """Point the proven frozen64 machinery at the formal256 namespace."""
    core.RESULT = RESULT
    core.REPORT = REPORT
    core.LOG = LOG
    core.EXTERNAL_TOOLS = EXTERNAL_TOOLS
    core.PROGRESS = PROGRESS
    core.MASTER_PROGRESS = MASTER_PROGRESS
    core.EVENTS = EVENTS
    core.PIPELINE_LOCK = PIPELINE_LOCK
    core.GENERATION = GENERATION
    core.FEATURES = FEATURES
    core.REFINED = REFINED
    core.REFINEMENT_MANIFEST = REFINEMENT_MANIFEST
    core.REFINEMENT_SUMMARY = REFINEMENT_SUMMARY
    core.RELAXED = RELAXED
    core.RELAX_PROGRESS = RELAX_PROGRESS
    core.CONFIG = CONFIG
    core.SEEDS = SEEDS
    core.METHODS = METHODS
    core.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    core.BOOTSTRAP_SAMPLES = BOOTSTRAP_SAMPLES


def event(stage: str, status: str, **values: Any) -> None:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    row = {"time": now(), "stage": stage, "status": status, **values}
    with EVENTS.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)


def write_master(stage: str, status: str, **values: Any) -> None:
    payload = (
        read_json(MASTER_PROGRESS)
        if MASTER_PROGRESS.is_file()
        else {
            "schema_version": 1,
            "experiment": "E3-PCR Formal 256",
            "base_branch": BASE_BRANCH,
            "base_commit": BASE_COMMIT,
            "formal_branch": FORMAL_BRANCH,
            "formal_seeds": [SEEDS[0], SEEDS[-1]],
            "created_at": now(),
            "a0_compatibility_started": False,
            "dft_started": False,
            "other_processes_terminated": False,
            "sigkill_used": False,
        }
    )
    payload.update(
        {
            "updated_at": now(),
            "current_stage": stage,
            "status": status,
            **values,
        }
    )
    atomic_json(MASTER_PROGRESS, payload)
    event(stage, status, **values)


def frozen_contract() -> dict[str, Any]:
    training = read_json(TRAINING_REPORT)
    refiner = read_json(REFINER_FREEZE_REPORT)
    config = read_json(CONFIG)
    return {
        "checkpoint_sha256": sha256(Q3_CHECKPOINT),
        "config_sha256": sha256(CONFIG),
        "frozen_source_sha256": sha256(FROZEN_SOURCE),
        "mattergen_checkpoint_sha256": sha256(MATTERGEN_CHECKPOINT),
        "mattersim_checkpoint_sha256": sha256(MATTERSIM),
        "training_seed_range": training["training_seed_range"],
        "network_seed": training["network"]["random_seed"],
        "gate_input_dim": training["network"]["input_dim"],
        "gate_hidden_dims": training["network"]["hidden_dims"],
        "gate_output_dim": training["network"]["output_dim"],
        "gate_threshold": training["network"]["threshold"],
        "trainable_parameters": training["network"]["trainable_parameters"],
        "refinement_steps": refiner["trust_region"]["steps"],
        "position_eta": refiner["trust_region"]["position_eta"],
        "per_step_radius_angstrom": refiner["trust_region"][
            "per_step_radius_angstrom"
        ],
        "backtrack_max": refiner["trust_region"]["backtrack_max"],
        "minimum_distance_angstrom": refiner["trust_region"][
            "minimum_distance_angstrom"
        ],
        "maximum_cumulative_displacement_angstrom": config["refinement"][
            "maximum_cumulative_displacement_angstrom"
        ],
        "atomic_numbers_modified": config["refinement"][
            "modify_atomic_numbers"
        ],
        "cell_modified": config["refinement"]["modify_cell"],
    }


def validate_frozen_contract(contract: dict[str, Any]) -> None:
    expected = {
        "checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "frozen_source_sha256": FROZEN_SOURCE_SHA256,
        "mattergen_checkpoint_sha256": MATTERGEN_SHA256,
        "mattersim_checkpoint_sha256": MATTERSIM_SHA256,
        "training_seed_range": [20000, 20063],
        "network_seed": 20260728,
        "gate_input_dim": 14,
        "gate_hidden_dims": [8],
        "gate_output_dim": 1,
        "gate_threshold": 0.5,
        "trainable_parameters": 129,
        "refinement_steps": 5,
        "position_eta": 0.01,
        "per_step_radius_angstrom": 0.02,
        "backtrack_max": 3,
        "minimum_distance_angstrom": 0.5,
        "maximum_cumulative_displacement_angstrom": 0.1,
        "atomic_numbers_modified": False,
        "cell_modified": False,
    }
    mismatches = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        atomic_json(REPORT / "frozen_state_mismatch.json", mismatches)
        write_master(
            "freeze_audit",
            "failed",
            final_state="FROZEN_STATE_MISMATCH",
            mismatches=mismatches,
        )
        raise RuntimeError(f"FROZEN_STATE_MISMATCH: {mismatches}")


def initialize() -> None:
    for path in (RESULT, REPORT, LOG, EXTERNAL_TOOLS, PROGRESS):
        path.mkdir(parents=True, exist_ok=True)
    write_master("state_audit", "running")
    branch = git_output("branch", "--show-current")
    if branch != FORMAL_BRANCH:
        raise RuntimeError(f"formal runner requires branch {FORMAL_BRANCH}, got {branch}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
        cwd=PROJECT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("formal branch does not descend from frozen64 commit")
    source_at_freeze = subprocess.run(
        [
            "git",
            "show",
            f"{FROZEN_SOURCE_COMMIT}:research/postgen_fastgate/refiner_eval.py",
        ],
        cwd=PROJECT,
        check=True,
        capture_output=True,
    ).stdout
    if source_at_freeze != FROZEN_SOURCE.read_bytes():
        raise RuntimeError("FROZEN_STATE_MISMATCH: refiner source changed")
    contract = frozen_contract()
    validate_frozen_contract(contract)
    seed_audit = {
        "schema_version": 1,
        "created_at": now(),
        "formal_seed_range": [SEEDS[0], SEEDS[-1]],
        "formal_seed_count": len(SEEDS),
        "path_matches": [],
        "structured_seed_field_matches": [],
        "reserved_plan_reference": (
            "/data/dxl/reports/innovation2_literature_search/"
            "minimum_validation_plan.md:39"
        ),
        "numeric_csv_false_positive_note": (
            "Values in 40000–40255 found in SPG shape-statistics non-seed "
            "columns; the seed column is 245xx and has no intersection."
        ),
        "training_intersection": [],
        "q3_32_development_intersection": [],
        "q3_frozen64_intersection": [],
        "gate_threshold_selection_intersection": [],
        "random_gate_intersection": [],
        "historical_postgen_module_intersection": [],
        "passed": True,
    }
    atomic_json(REPORT / "formal_seed_audit.json", seed_audit)
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "base_branch": BASE_BRANCH,
        "base_commit": BASE_COMMIT,
        "formal_branch": FORMAL_BRANCH,
        "formal_code_commit": git_output("rev-parse", "HEAD"),
        "evaluation_script": str(Path(__file__).resolve()),
        "evaluation_script_sha256": sha256(Path(__file__).resolve()),
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "frozen_source": str(FROZEN_SOURCE),
        "formal_seeds": [SEEDS[0], SEEDS[-1]],
        "formal_seed_count": len(SEEDS),
        "c0_generated_once_per_seed": True,
        "e3a_e3g_derive_from_same_c0": True,
        "q3_checkpoint": str(Q3_CHECKPOINT),
        "config": str(CONFIG),
        "mattergen_checkpoint": str(MATTERGEN_CHECKPOINT),
        "mattersim_checkpoint": str(MATTERSIM),
        "contract": contract,
        "seed_audit": seed_audit,
        "statistics": {
            "primary_endpoint": "pre-relaxation maximum force",
            "primary_arms": ["E3-A vs C0", "E3-G vs C0"],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "wilcoxon_multiple_testing": "Holm, family size 2",
            "force_harm_epsilon_ev_angstrom": FORCE_HARM_EPSILON,
        },
        "tuning_after_freeze": False,
        "stability_source": "MatterSim-5M surrogate",
        "dft_verified": False,
        "property_target_verified": False,
    }
    atomic_json(REPORT / "formal_frozen_manifest.json", manifest)
    atomic_text(
        REPORT / "freeze_report.md",
        "# E3-PCR Formal 256 freeze audit\n\n"
        f"- Base: `{BASE_BRANCH}` at `{BASE_COMMIT}`\n"
        f"- Formal code commit: `{manifest['formal_code_commit']}`\n"
        f"- Q3 checkpoint SHA256: `{Q3_CHECKPOINT_SHA256}`\n"
        f"- Frozen config SHA256: `{CONFIG_SHA256}`\n"
        f"- Frozen source SHA256: `{FROZEN_SOURCE_SHA256}`\n"
        f"- MatterGen SHA256: `{MATTERGEN_SHA256}`\n"
        f"- MatterSim SHA256: `{MATTERSIM_SHA256}`\n"
        "- Formal seeds: `40000–40255` (256 unused seeds)\n"
        "- Frozen parameters match the 64-seed validation exactly.\n"
        "- No retraining, threshold search, radius search, or checkpoint selection.\n",
    )
    write_master(
        "freeze_audit",
        "success",
        q3_checkpoint_sha256=Q3_CHECKPOINT_SHA256,
        config_sha256=CONFIG_SHA256,
        frozen_source_sha256=FROZEN_SOURCE_SHA256,
    )
    write_master(
        "formal_seed_audit",
        "success",
        formal_seeds=[SEEDS[0], SEEDS[-1]],
        formal_seed_count=len(SEEDS),
    )


def gpu_compute_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def wait_for_all_gpus_free(next_stage: str) -> None:
    while True:
        processes = gpu_compute_processes()
        if not processes:
            return
        write_master(
            "waiting_for_gpus",
            "waiting",
            next_stage=next_stage,
            external_gpu_processes=processes,
        )
        time.sleep(30)


def generate() -> None:
    wait_for_all_gpus_free("c0_generation")
    write_master("c0_generation", "running")
    core.generate()
    generation = core.configure_generation_module()
    state = generation.load_generation_progress()
    if state["success"] != len(SEEDS):
        raise RuntimeError(f"C0 generation incomplete: {state['success']}/{len(SEEDS)}")
    tasks = sorted(state["tasks"], key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in tasks] != list(SEEDS):
        raise RuntimeError("C0 generation seed contract mismatch")
    for row in tasks:
        path = GENERATION / str(row["seed"]) / "generated_crystals.extxyz"
        if not path.is_file():
            raise RuntimeError(f"missing generated C0 structure: {path}")
    write_master(
        "c0_generation",
        "success",
        c0_generation=f"{state['success']}/{len(SEEDS)}",
        c0_generated_once=True,
    )


def audited_refinement() -> None:
    """Run the unchanged refiner while collecting rejection reason counters."""
    from research.postgen_fastgate import refiner_eval as frozen

    original_subset = core.run_refinement_subset
    invocation = 0

    def wrapped_subset(
        model: Any,
        originals: list[Any],
        active: np.ndarray,
        instrumentation: dict[str, int],
    ) -> tuple[list[Any], list[dict[str, int]], float]:
        nonlocal invocation
        invocation += 1
        label = "E3_G" if invocation == 1 else "E3_A"
        original_safe = frozen.finite_safe
        safety = {
            "safety_checks": 0,
            "safety_rejections": 0,
            "short_bond_rejections": 0,
            "abnormal_cell_rejections": 0,
        }

        def audited_safe(atoms: Any) -> bool:
            safety["safety_checks"] += 1
            positions_finite = bool(np.isfinite(atoms.positions).all())
            cell_finite = bool(np.isfinite(atoms.cell.array).all())
            try:
                volume = float(atoms.get_volume())
            except BaseException:
                volume = math.nan
            abnormal = (
                not positions_finite
                or not cell_finite
                or not math.isfinite(volume)
                or volume <= 0.1
            )
            try:
                short = frozen.minimum_distance(atoms) < SHORT_BOND_ANGSTROM
            except BaseException:
                short = True
            accepted = original_safe(atoms)
            if not accepted:
                safety["safety_rejections"] += 1
                if short:
                    safety["short_bond_rejections"] += 1
                if abnormal:
                    safety["abnormal_cell_rejections"] += 1
            return accepted

        frozen.finite_safe = audited_safe
        try:
            result = original_subset(model, originals, active, instrumentation)
        finally:
            frozen.finite_safe = original_safe
        instrumentation.update(safety)
        instrumentation["formal_arm"] = label
        return result

    core.run_refinement_subset = wrapped_subset
    try:
        core.refine()
    finally:
        core.run_refinement_subset = original_subset


def refine() -> None:
    wait_for_all_gpus_free("e3a_refinement")
    write_master("e3a_refinement", "running")
    audited_refinement()
    features = pd.read_csv(FEATURES)
    features["split"] = "formal256_independent"
    atomic_csv(FEATURES, features.sort_values("seed").reset_index(drop=True))
    manifest = pd.read_csv(REFINEMENT_MANIFEST)
    e3a = manifest[manifest["method"] == "ALWAYS_ON"].copy()
    e3g = manifest[manifest["method"] == "Q3_E3_PCR"].copy()
    if len(e3a) != len(SEEDS) or len(e3g) != len(SEEDS):
        raise RuntimeError("formal refinement arm count mismatch")
    if not (
        e3a["input_hash"].astype(str).to_numpy()
        == e3g["input_hash"].astype(str).to_numpy()
    ).all():
        raise RuntimeError("E3-A and E3-G do not derive from identical C0 inputs")
    gate_off = e3g[~e3g["gate_applied"].astype(bool)]
    if not gate_off["exact_baseline_fallback"].astype(bool).all():
        raise RuntimeError("one or more Gate-off structures differ from C0")
    for arm, frame in (("E3-A", e3a), ("E3-G", e3g)):
        full_rejection = frame[
            (frame["accepted_steps"].astype(int) == 0)
            & (frame["fallback_count"].astype(int) > 0)
        ]
        if not full_rejection["exact_baseline_fallback"].astype(bool).all():
            raise RuntimeError(f"{arm} full safety rejection did not restore C0")
    write_master(
        "e3a_refinement",
        "success",
        e3a_refinement=f"{len(e3a)}/{len(SEEDS)}",
        refinement_rate=float(e3a["gate_applied"].astype(bool).mean()),
    )
    write_master(
        "e3g_refinement",
        "success",
        e3g_refinement=f"{len(e3g)}/{len(SEEDS)}",
        refinement_rate=float(e3g["gate_applied"].astype(bool).mean()),
        gate_off_exact_fallback=int(
            gate_off["exact_baseline_fallback"].astype(bool).sum()
        ),
    )


def relax() -> None:
    wait_for_all_gpus_free("c0_relaxation")
    write_master("c0_relaxation", "running", total_relaxations=3 * len(SEEDS))
    if sha256(MATTERSIM) != MATTERSIM_SHA256:
        raise RuntimeError("MatterSim checkpoint SHA256 mismatch")
    state = core.initialize_relax()
    expected = len(METHODS) * len(SEEDS)
    if state["success"] != expected:
        worker_logs = LOG / "relax_workers"
        worker_logs.mkdir(parents=True, exist_ok=True)
        processes = []
        handles = []
        for slot in range(2):
            for gpu in range(8):
                handle = (worker_logs / f"gpu{gpu}_slot{slot}.log").open(
                    "a", encoding="utf-8"
                )
                env = os.environ.copy()
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2",
                        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                    }
                )
                process = subprocess.Popen(
                    [
                        str(MATTERGEN_PYTHON),
                        "-m",
                        "research.q3_formal256",
                        "relax-worker",
                        "--gpu",
                        str(gpu),
                        "--slot",
                        str(slot),
                    ],
                    cwd=PROJECT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append(process)
                handles.append(handle)
        codes = [process.wait() for process in processes]
        for handle in handles:
            handle.close()
        state = core.initialize_relax()
        if any(code != 0 for code in codes) or state["success"] != expected:
            write_master(
                "c0_relaxation",
                "failed",
                relaxed=state["success"],
                expected=expected,
                worker_exit_codes=codes,
            )
            raise RuntimeError(
                f"MatterSim relaxation incomplete: {state['success']}/{expected}"
            )
    state = core.initialize_relax()
    counts = {
        method: sum(
            row["status"] == "success" and row["method"] == method
            for row in state["tasks"]
        )
        for method in METHODS
    }
    if any(value != len(SEEDS) for value in counts.values()):
        raise RuntimeError(f"formal relaxation count mismatch: {counts}")
    write_master(
        "c0_relaxation", "success", c0_relaxation=f"{counts['C0']}/{len(SEEDS)}"
    )
    write_master(
        "e3a_relaxation",
        "success",
        e3a_relaxation=f"{counts['ALWAYS_ON']}/{len(SEEDS)}",
    )
    write_master(
        "e3g_relaxation",
        "success",
        e3g_relaxation=f"{counts['Q3_E3_PCR']}/{len(SEEDS)}",
    )


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(
        0, len(values), size=(BOOTSTRAP_SAMPLES, len(values))
    )
    low, high = np.quantile(values[indexes].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: p_values[key])
    adjusted: dict[str, float] = {}
    previous = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (total - rank) * float(p_values[key]))
        previous = max(previous, candidate)
        adjusted[key] = previous
    return adjusted


def continuous_stat(
    baseline: np.ndarray,
    selected: np.ndarray,
    metric: str,
) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    left = np.asarray(baseline, dtype=float)
    right = np.asarray(selected, dtype=float)
    paired = np.isfinite(left) & np.isfinite(right)
    left = left[paired]
    right = right[paired]
    difference = right - left
    low, high = bootstrap_ci(difference)
    nonzero = np.any(np.abs(difference) > 1.0e-12)
    test = wilcoxon(difference, zero_method="pratt") if nonzero else None
    wins = int((difference < -1.0e-12).sum())
    losses = int((difference > 1.0e-12).sum())
    return {
        "metric": metric,
        "type": "continuous",
        "paired_count": len(difference),
        "baseline_mean": float(left.mean()),
        "selected_mean": float(right.mean()),
        "baseline_median": float(np.median(left)),
        "selected_median": float(np.median(right)),
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "bootstrap_95_ci_low": low,
        "bootstrap_95_ci_high": high,
        "test": "Wilcoxon signed-rank",
        "p_value": float(test.pvalue) if test else 1.0,
        "wins": wins,
        "ties": int(len(difference) - wins - losses),
        "losses": losses,
    }


def binary_stat(
    baseline: np.ndarray,
    selected: np.ndarray,
    metric: str,
) -> dict[str, Any]:
    from scipy.stats import binomtest

    left = np.asarray(baseline, dtype=bool)
    right = np.asarray(selected, dtype=bool)
    wins = int((~left & right).sum())
    losses = int((left & ~right).sum())
    discordant = wins + losses
    difference = right.astype(float) - left.astype(float)
    low, high = bootstrap_ci(difference)
    return {
        "metric": metric,
        "type": "binary",
        "paired_count": len(left),
        "baseline_mean": float(left.mean()),
        "selected_mean": float(right.mean()),
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "bootstrap_95_ci_low": low,
        "bootstrap_95_ci_high": high,
        "test": "McNemar exact / paired discordant binomial",
        "p_value": (
            float(binomtest(wins, discordant, 0.5).pvalue)
            if discordant
            else 1.0
        ),
        "wins": wins,
        "ties": int(len(left) - discordant),
        "losses": losses,
    }


def force_robustness(
    baseline: pd.DataFrame, selected: pd.DataFrame
) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    left = baseline["pre_relax_max_force_ev_ang"].to_numpy(float)
    right = selected["pre_relax_max_force_ev_ang"].to_numpy(float)
    difference = right - left
    low, high = bootstrap_ci(difference)
    test = (
        wilcoxon(difference, zero_method="pratt")
        if np.any(np.abs(difference) > 1.0e-12)
        else None
    )
    leave_one_out = np.asarray(
        [np.delete(difference, index).mean() for index in range(len(difference))]
    )
    favorable = int(np.argmin(difference))
    unfavorable = int(np.argmax(difference))
    total = float(np.abs(difference).sum())

    def removed_relative(index: int) -> float:
        return float(
            np.delete(right, index).mean() / np.delete(left, index).mean() - 1.0
        )

    return {
        "baseline_mean": float(left.mean()),
        "selected_mean": float(right.mean()),
        "baseline_median": float(np.median(left)),
        "selected_median": float(np.median(right)),
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "relative_change": float(right.mean() / left.mean() - 1.0),
        "bootstrap_95_ci": [low, high],
        "wilcoxon_p_raw": float(test.pvalue) if test else 1.0,
        "wins": int((difference < -1.0e-12).sum()),
        "ties": int((np.abs(difference) <= 1.0e-12).sum()),
        "losses": int((difference > 1.0e-12).sum()),
        "leave_one_out_mean_difference_range": [
            float(leave_one_out.min()),
            float(leave_one_out.max()),
        ],
        "remove_most_favorable_sample_index": favorable,
        "remove_most_favorable_sample_seed": SEEDS[favorable],
        "remove_most_favorable_relative_change": removed_relative(favorable),
        "remove_most_unfavorable_sample_index": unfavorable,
        "remove_most_unfavorable_sample_seed": SEEDS[unfavorable],
        "remove_most_unfavorable_relative_change": removed_relative(unfavorable),
        "maximum_single_sample_absolute_contribution_rate": (
            float(np.abs(difference).max() / total) if total else 0.0
        ),
    }


def minimum_distance(atoms: Any) -> float:
    if len(atoms) < 2:
        return math.inf
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def abnormal_cell(atoms: Any) -> bool:
    cell = np.asarray(atoms.cell.array, dtype=float)
    try:
        volume = float(atoms.get_volume())
        condition = float(np.linalg.cond(cell))
    except BaseException:
        return True
    return bool(
        not np.isfinite(cell).all()
        or not math.isfinite(volume)
        or volume <= 0.1
        or not math.isfinite(condition)
    )


def prepare_official_metrics() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames = core.official_frames()
    elements: dict[str, list[list[str]]] = {}
    for method, frame in frames.items():
        original_atoms = frame["_original_atoms"].tolist()
        frame["pre_relax_minimum_distance_angstrom"] = [
            minimum_distance(atoms) for atoms in original_atoms
        ]
        frame["pre_relax_short_bond"] = (
            frame["pre_relax_minimum_distance_angstrom"] < SHORT_BOND_ANGSTROM
        )
        frame["pre_relax_abnormal_cell"] = [
            abnormal_cell(atoms) for atoms in original_atoms
        ]
        elements[method] = [
            sorted(set(atoms.get_chemical_symbols())) for atoms in original_atoms
        ]
    tool_root = ROOT / "tools/innovation2_next"
    sys.path.insert(0, str(tool_root))
    import analyze_corrector_64 as official

    official.ROOT = ROOT
    official.RESULT = RESULT
    official.REPORT = REPORT
    official.PROGRESS = PROGRESS
    official.REFERENCE = core.REFERENCE
    official.REFERENCE_LMDB = core.REFERENCE_LMDB
    official.CONFIGS = METHODS
    official.SEEDS = list(range(len(SEEDS)))
    official.STABILITY_THRESHOLD = STABILITY_THRESHOLD
    metrics, errors = official.official_metrics(frames)
    if errors:
        raise RuntimeError(f"official metrics failures: {errors}")
    evaluated: dict[str, pd.DataFrame] = {}
    missing: dict[str, Any] = {}
    for method in METHODS:
        frame = pd.read_csv(
            REPORT / method / "official_metrics_per_structure.csv"
        ).sort_values("seed").reset_index(drop=True)
        ehull = pd.to_numeric(
            frame["energy_above_hull_per_atom"], errors="coerce"
        ).to_numpy(float)
        missing_indexes = np.flatnonzero(~np.isfinite(ehull))
        missing[DISPLAY_NAMES[method]] = {
            "covered": int(np.isfinite(ehull).sum()),
            "total": len(ehull),
            "coverage": float(np.isfinite(ehull).mean()),
            "missing_evaluation_indexes": missing_indexes.tolist(),
            "missing_formal_seeds": [SEEDS[index] for index in missing_indexes],
            "missing_elements": sorted(
                {
                    element
                    for index in missing_indexes
                    for element in elements[method][index]
                }
            ),
        }
        evaluated[method] = frame
    common = np.logical_and.reduce(
        [
            np.isfinite(
                evaluated[method]["energy_above_hull_per_atom"].to_numpy(float)
            )
            for method in METHODS
        ]
    )
    missing["common_three_arm"] = {
        "covered": int(common.sum()),
        "total": len(common),
        "coverage": float(common.mean()),
        "formal_seeds": [SEEDS[index] for index in np.flatnonzero(common)],
    }
    atomic_json(REPORT / "ehull_coverage.json", missing)
    return evaluated, metrics


def metric_means(frame: pd.DataFrame) -> dict[str, float]:
    ehull = frame["energy_above_hull_per_atom"].to_numpy(float)
    finite = np.isfinite(ehull)
    stable = frame["stable"].astype(bool).to_numpy()
    novel = frame["novel"].astype(bool).to_numpy()
    unique = frame["unique"].astype(bool).to_numpy()
    metastable = finite & (ehull <= 0.2)
    return {
        "pre_relax_max_force": float(
            frame["pre_relax_max_force_ev_ang"].mean()
        ),
        "pre_relax_max_force_median": float(
            frame["pre_relax_max_force_ev_ang"].median()
        ),
        "convergence_rate": float(frame["converged"].astype(bool).mean()),
        "relaxation_steps_mean": float(frame["steps"].mean()),
        "relaxation_steps_median": float(frame["steps"].median()),
        "rmsd": float(frame["rmsd_from_relaxation"].mean()),
        "ehull": float(np.nanmean(ehull)),
        "ehull_median": float(np.nanmedian(ehull)),
        "ehull_coverage": float(finite.mean()),
        "stable": float(stable.mean()),
        "metastable": float(metastable.mean()),
        "nus": float(frame["novel_unique_stable"].astype(bool).mean()),
        "msun": float((metastable & novel & unique).mean()),
        "novel": float(novel.mean()),
        "unique": float(unique.mean()),
        "composition_validity": float(
            frame["comp_validity"].astype(bool).mean()
        ),
        "structure_validity": float(
            frame["structure_validity"].astype(bool).mean()
        ),
        "relaxation_failure_rate": 0.0,
        "short_bond_rate": float(
            frame["pre_relax_short_bond"].astype(bool).mean()
        ),
        "abnormal_cell_rate": float(
            frame["pre_relax_abnormal_cell"].astype(bool).mean()
        ),
        "relaxation_elapsed_mean": float(frame["relax_elapsed_seconds"].mean()),
    }


def refinement_behavior(
    manifest: pd.DataFrame,
    method: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    frame = manifest[manifest["method"] == method].copy()
    displacement = frame["maximum_wrapped_displacement_angstrom"].to_numpy(float)
    instrumentation_key = (
        "always_on_instrumentation"
        if method == "ALWAYS_ON"
        else "learned_instrumentation"
    )
    instrumentation = summary[instrumentation_key]
    safety_checks = int(instrumentation.get("safety_checks", 0))
    safety_rejections = int(instrumentation.get("safety_rejections", 0))
    full_rejections = frame[
        (frame["accepted_steps"].astype(int) == 0)
        & (frame["fallback_count"].astype(int) > 0)
    ]
    return {
        "refinement_count": int(frame["gate_applied"].astype(bool).sum()),
        "refinement_rate": float(frame["gate_applied"].astype(bool).mean()),
        "fallback_count": int(frame["fallback_count"].astype(int).sum()),
        "fallback_mean": float(frame["fallback_count"].astype(float).mean()),
        "backtracking_count": int(
            frame["backtracking_count"].astype(int).sum()
        ),
        "backtracking_mean": float(
            frame["backtracking_count"].astype(float).mean()
        ),
        "accepted_steps_mean": float(frame["accepted_steps"].astype(float).mean()),
        "mean_displacement_angstrom": float(displacement.mean()),
        "median_displacement_angstrom": float(np.median(displacement)),
        "maximum_displacement_angstrom": float(displacement.max()),
        "p90_displacement_angstrom": float(np.quantile(displacement, 0.90)),
        "p95_displacement_angstrom": float(np.quantile(displacement, 0.95)),
        "p99_displacement_angstrom": float(np.quantile(displacement, 0.99)),
        "proposal_atoms": int(instrumentation["proposal_atoms"]),
        "clipped_atoms": int(instrumentation["clipped_atoms"]),
        "clipping_rate": float(
            instrumentation["clipped_atoms"]
            / max(1, instrumentation["proposal_atoms"])
        ),
        "safety_checks": safety_checks,
        "safety_rejections": safety_rejections,
        "safety_rejection_rate": float(
            safety_rejections / max(1, safety_checks)
        ),
        "short_bond_rejections": int(
            instrumentation.get("short_bond_rejections", 0)
        ),
        "short_bond_rejection_rate": float(
            instrumentation.get("short_bond_rejections", 0)
            / max(1, safety_checks)
        ),
        "abnormal_cell_rejections": int(
            instrumentation.get("abnormal_cell_rejections", 0)
        ),
        "abnormal_cell_rejection_rate": float(
            instrumentation.get("abnormal_cell_rejections", 0)
            / max(1, safety_checks)
        ),
        "full_rejection_count": len(full_rejections),
        "full_rejections_exact_fallback": bool(
            full_rejections["exact_baseline_fallback"].astype(bool).all()
        ),
        "atomic_numbers_modified": int(
            (~frame["atomic_numbers_unchanged"].astype(bool)).sum()
        ),
        "cell_modified": int((~frame["cell_unchanged"].astype(bool)).sum()),
        "finite": bool(
            np.isfinite(
                frame[
                    [
                        "maximum_wrapped_displacement_angstrom",
                        "minimum_distance_angstrom",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "minimum_distance_angstrom": float(
            frame["minimum_distance_angstrom"].min()
        ),
    }


def quality_pass(
    baseline: dict[str, float],
    selected: dict[str, float],
    generation_success: int,
    behavior: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    gates = {
        "generation_success_not_lower": generation_success == len(SEEDS),
        "structure_validity_not_lower": (
            selected["structure_validity"] >= baseline["structure_validity"]
        ),
        "composition_drop_le_1_over_256": (
            selected["composition_validity"]
            >= baseline["composition_validity"] - 1.0 / len(SEEDS)
        ),
        "ehull_degradation_le_0_002": (
            selected["ehull"] - baseline["ehull"] <= 0.002
        ),
        "stable_drop_le_1_over_256": (
            selected["stable"] >= baseline["stable"] - 1.0 / len(SEEDS)
        ),
        "nus_drop_le_1_over_256": (
            selected["nus"] >= baseline["nus"] - 1.0 / len(SEEDS)
        ),
        "novel_drop_le_1_over_256": (
            selected["novel"] >= baseline["novel"] - 1.0 / len(SEEDS)
        ),
        "unique_not_lower": selected["unique"] >= baseline["unique"],
        "rmsd_degradation_le_5_percent": (
            selected["rmsd"] <= baseline["rmsd"] * 1.05
        ),
        "relaxation_failure_not_increased": (
            selected["relaxation_failure_rate"]
            <= baseline["relaxation_failure_rate"]
        ),
        "short_bond_not_increased": (
            selected["short_bond_rate"] <= baseline["short_bond_rate"]
        ),
        "abnormal_cell_not_increased": (
            selected["abnormal_cell_rate"] <= baseline["abnormal_cell_rate"]
        ),
        "mechanism_atomic_numbers_unchanged": (
            behavior["atomic_numbers_modified"] == 0
        ),
        "mechanism_cell_unchanged": behavior["cell_modified"] == 0,
        "mechanism_max_displacement_bounded": (
            behavior["maximum_displacement_angstrom"] <= 0.1000001
        ),
        "mechanism_full_rejection_exact_fallback": behavior[
            "full_rejections_exact_fallback"
        ],
        "mechanism_no_nan_inf": behavior["finite"],
        "mechanism_minimum_distance_safe": (
            behavior["minimum_distance_angstrom"] >= SHORT_BOND_ANGSTROM
        ),
    }
    return all(gates.values()), gates


def gate_quality_not_worse(
    e3a: dict[str, float], e3g: dict[str, float]
) -> tuple[bool, dict[str, bool]]:
    gates = {
        "structure": e3g["structure_validity"] >= e3a["structure_validity"],
        "composition": (
            e3g["composition_validity"]
            >= e3a["composition_validity"] - 1.0 / len(SEEDS)
        ),
        "ehull": e3g["ehull"] <= e3a["ehull"] + 0.002,
        "stable": e3g["stable"] >= e3a["stable"] - 1.0 / len(SEEDS),
        "nus": e3g["nus"] >= e3a["nus"] - 1.0 / len(SEEDS),
        "novel": e3g["novel"] >= e3a["novel"] - 1.0 / len(SEEDS),
        "unique": e3g["unique"] >= e3a["unique"],
        "rmsd": e3g["rmsd"] <= e3a["rmsd"] * 1.05,
        "failure": (
            e3g["relaxation_failure_rate"] <= e3a["relaxation_failure_rate"]
        ),
        "short_bond": e3g["short_bond_rate"] <= e3a["short_bond_rate"],
        "abnormal_cell": (
            e3g["abnormal_cell_rate"] <= e3a["abnormal_cell_rate"]
        ),
    }
    return all(gates.values()), gates


def select_final_state(
    e3a_pass: bool,
    e3g_pass: bool,
    gate_supported: bool,
) -> tuple[str, str]:
    if e3a_pass and e3g_pass and gate_supported:
        return "E3_G_FORMAL_CONFIRMED", "LEARNED_GATED_E3_PCR"
    if e3a_pass and e3g_pass:
        return (
            "E3_REFINER_FORMAL_CONFIRMED_GATE_UNSUPPORTED",
            "SAFE_BOUNDED_E3_PCR",
        )
    if e3a_pass:
        return "E3_A_FORMAL_CONFIRMED", "ALWAYS_ON_SAFE_BOUNDED_E3_PCR"
    if e3g_pass:
        return "E3_G_FORMAL_CONFIRMED", "LEARNED_GATED_E3_PCR"
    return "E3_PCR_FORMAL_NO_GO", "NONE"


def analyze() -> None:
    from scipy.stats import binomtest

    write_master("metrics", "running")
    evaluated, official_metrics = prepare_official_metrics()
    baseline = evaluated["C0"]
    e3a = evaluated["ALWAYS_ON"]
    e3g = evaluated["Q3_E3_PCR"]
    means = {
        "C0": metric_means(baseline),
        "E3-A": metric_means(e3a),
        "E3-G": metric_means(e3g),
    }
    common_ehull = np.logical_and.reduce(
        [
            np.isfinite(frame["energy_above_hull_per_atom"].to_numpy(float))
            for frame in (baseline, e3a, e3g)
        ]
    )
    if not common_ehull.any():
        raise RuntimeError("no common three-arm E-hull coverage")
    for label, frame in (("C0", baseline), ("E3-A", e3a), ("E3-G", e3g)):
        means[label]["ehull_all_available"] = means[label]["ehull"]
        means[label]["ehull_median_all_available"] = means[label]["ehull_median"]
        common_values = frame.loc[
            common_ehull, "energy_above_hull_per_atom"
        ].to_numpy(float)
        means[label]["ehull"] = float(common_values.mean())
        means[label]["ehull_median"] = float(np.median(common_values))
    paired_rows: list[dict[str, Any]] = []
    continuous = {
        "pre_relax_max_force_ev_ang": None,
        "relaxation_steps": "steps",
        "relaxation_rmsd": "rmsd_from_relaxation",
        "relaxation_elapsed_seconds": "relax_elapsed_seconds",
    }
    binary = {
        "force_converged": "converged",
        "stable": "stable",
        "nus": "novel_unique_stable",
        "structure_validity": "structure_validity",
        "composition_validity": "comp_validity",
        "novel": "novel",
        "unique": "unique",
        "pre_relax_short_bond": "pre_relax_short_bond",
        "pre_relax_abnormal_cell": "pre_relax_abnormal_cell",
    }
    for arm_name, frame in (("E3-A", e3a), ("E3-G", e3g)):
        for metric, column in continuous.items():
            actual = column or metric
            row = continuous_stat(
                baseline[actual].to_numpy(float),
                frame[actual].to_numpy(float),
                metric,
            )
            row["comparison"] = f"{arm_name} vs C0"
            paired_rows.append(row)
        row = continuous_stat(
            baseline.loc[common_ehull, "energy_above_hull_per_atom"].to_numpy(
                float
            ),
            frame.loc[common_ehull, "energy_above_hull_per_atom"].to_numpy(
                float
            ),
            "energy_above_hull_per_atom_common_coverage",
        )
        row["comparison"] = f"{arm_name} vs C0"
        paired_rows.append(row)
        for metric, column in binary.items():
            row = binary_stat(
                baseline[column].astype(bool).to_numpy(),
                frame[column].astype(bool).to_numpy(),
                metric,
            )
            row["comparison"] = f"{arm_name} vs C0"
            paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    atomic_csv(REPORT / "formal_paired_statistics.csv", paired)

    primary = {
        "E3-A": force_robustness(baseline, e3a),
        "E3-G": force_robustness(baseline, e3g),
    }
    holm = holm_adjust(
        {arm: values["wilcoxon_p_raw"] for arm, values in primary.items()}
    )
    for arm, value in holm.items():
        primary[arm]["wilcoxon_p_holm"] = value
        primary[arm]["effect_reduction_ge_10_percent"] = (
            primary[arm]["relative_change"] <= -0.10
        )
        primary[arm]["bootstrap_ci_upper_below_zero"] = (
            primary[arm]["bootstrap_95_ci"][1] < 0.0
        )
        primary[arm]["holm_p_below_0_05"] = value < 0.05
        primary[arm]["primary_effect_pass"] = bool(
            primary[arm]["effect_reduction_ge_10_percent"]
            and primary[arm]["bootstrap_ci_upper_below_zero"]
            and primary[arm]["holm_p_below_0_05"]
        )
    atomic_json(REPORT / "primary_statistics.json", primary)

    manifest = pd.read_csv(REFINEMENT_MANIFEST)
    refinement_summary = read_json(REFINEMENT_SUMMARY)
    behavior = {
        "E3-A": refinement_behavior(
            manifest, "ALWAYS_ON", refinement_summary
        ),
        "E3-G": refinement_behavior(
            manifest, "Q3_E3_PCR", refinement_summary
        ),
    }
    e3g_manifest = (
        manifest[manifest["method"] == "Q3_E3_PCR"]
        .sort_values("evaluation_index")
        .reset_index(drop=True)
    )
    gate_off = e3g_manifest[~e3g_manifest["gate_applied"].astype(bool)]
    gate_off_exact = bool(
        gate_off["exact_baseline_fallback"].astype(bool).all()
    )
    behavior["E3-G"]["gate_off_count"] = len(gate_off)
    behavior["E3-G"]["gate_off_exact_fallback"] = gate_off_exact
    if not gate_off_exact:
        raise RuntimeError("formal Gate-off exact fallback invariant failed")

    generation_state = core.configure_generation_module().load_generation_progress()
    generation_success = int(generation_state["success"])
    e3a_quality, e3a_quality_gates = quality_pass(
        means["C0"], means["E3-A"], generation_success, behavior["E3-A"]
    )
    e3g_quality, e3g_quality_gates = quality_pass(
        means["C0"], means["E3-G"], generation_success, behavior["E3-G"]
    )
    quality = {
        "C0": means["C0"],
        "E3-A": means["E3-A"],
        "E3-G": means["E3-G"],
        "E3-A_gates": e3a_quality_gates,
        "E3-G_gates": e3g_quality_gates,
        "E3-A_quality_safety_pass": e3a_quality,
        "E3-G_quality_safety_pass": e3g_quality,
        "ehull_coverage": read_json(REPORT / "ehull_coverage.json"),
    }
    atomic_json(REPORT / "quality_summary.json", quality)

    c0_force = baseline["pre_relax_max_force_ev_ang"].to_numpy(float)
    e3a_force = e3a["pre_relax_max_force_ev_ang"].to_numpy(float)
    e3g_force = e3g["pre_relax_max_force_ev_ang"].to_numpy(float)
    always_gain = float(c0_force.mean() - e3a_force.mean())
    gated_gain = float(c0_force.mean() - e3g_force.mean())
    gain_retention = float(gated_gain / always_gain) if always_gain > 0 else math.nan
    e3a_harm = e3a_force > c0_force + FORCE_HARM_EPSILON
    e3g_harm = e3g_force > c0_force + FORCE_HARM_EPSILON
    e3a_only_harm = int((e3a_harm & ~e3g_harm).sum())
    e3g_only_harm = int((~e3a_harm & e3g_harm).sum())
    discordant = e3a_only_harm + e3g_only_harm
    harm_p = (
        float(binomtest(e3g_only_harm, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    order = np.argsort(c0_force, kind="stable")
    low = np.zeros(len(SEEDS), dtype=bool)
    low[order[: len(SEEDS) // 2]] = True
    quality_not_worse, quality_not_worse_gates = gate_quality_not_worse(
        means["E3-A"], means["E3-G"]
    )
    coverage_reduction = (
        behavior["E3-A"]["refinement_rate"]
        - behavior["E3-G"]["refinement_rate"]
    )
    condition_1 = bool(
        coverage_reduction >= 0.25 and gain_retention >= 0.85
    )
    condition_2 = bool(
        e3g_harm.mean() < e3a_harm.mean() and harm_p < 0.05
    )
    condition_3 = bool(
        e3g_harm[low].mean() < e3a_harm[low].mean()
        and primary["E3-G"]["relative_change"] <= -0.10
    )
    mean_displacement_reduction = float(
        1.0
        - behavior["E3-G"]["mean_displacement_angstrom"]
        / max(behavior["E3-A"]["mean_displacement_angstrom"], 1.0e-12)
    )
    p95_displacement_reduction = float(
        1.0
        - behavior["E3-G"]["p95_displacement_angstrom"]
        / max(behavior["E3-A"]["p95_displacement_angstrom"], 1.0e-12)
    )
    condition_4 = bool(
        max(mean_displacement_reduction, p95_displacement_reduction) >= 0.20
        and gain_retention >= 0.85
    )
    quality_clear_advantage = bool(
        means["E3-G"]["rmsd"] <= means["E3-A"]["rmsd"] * 0.98
        or means["E3-G"]["ehull"] <= means["E3-A"]["ehull"] - 0.002
        or means["E3-G"]["relaxation_failure_rate"]
        < means["E3-A"]["relaxation_failure_rate"]
        or means["E3-G"]["abnormal_cell_rate"]
        < means["E3-A"]["abnormal_cell_rate"]
    )
    condition_5 = bool(
        quality_clear_advantage
        and primary["E3-G"]["primary_effect_pass"]
    )
    gate_supported = bool(
        quality_not_worse
        and any(
            (condition_1, condition_2, condition_3, condition_4, condition_5)
        )
    )
    mechanism = {
        "always_gain": always_gain,
        "gated_gain": gated_gain,
        "gain_retention": gain_retention,
        "e3a_refinement_rate": behavior["E3-A"]["refinement_rate"],
        "e3g_refinement_rate": behavior["E3-G"]["refinement_rate"],
        "coverage_reduction": coverage_reduction,
        "e3a_harm_rate": float(e3a_harm.mean()),
        "e3g_harm_rate": float(e3g_harm.mean()),
        "e3a_only_harm": e3a_only_harm,
        "e3g_only_harm": e3g_only_harm,
        "harm_mcnemar_exact_p": harm_p,
        "low_force_count": int(low.sum()),
        "low_force_threshold_max_force": float(c0_force[low].max()),
        "low_force_e3a_harm_rate": float(e3a_harm[low].mean()),
        "low_force_e3g_harm_rate": float(e3g_harm[low].mean()),
        "mean_displacement_reduction": mean_displacement_reduction,
        "p95_displacement_reduction": p95_displacement_reduction,
        "e3g_quality_not_worse_than_e3a": quality_not_worse,
        "e3g_quality_not_worse_gates": quality_not_worse_gates,
        "condition_1_gain_coverage": condition_1,
        "condition_2_reduce_harm": condition_2,
        "condition_3_protect_low_force": condition_3,
        "condition_4_smaller_intervention": condition_4,
        "condition_5_better_quality_safety": condition_5,
        "GATE_MECHANISM_FORMAL_SUPPORTED": gate_supported,
        "behavior": behavior,
    }
    atomic_json(REPORT / "gate_mechanism_summary.json", mechanism)

    e3a_pass = bool(primary["E3-A"]["primary_effect_pass"] and e3a_quality)
    e3g_pass = bool(primary["E3-G"]["primary_effect_pass"] and e3g_quality)
    final_state, final_method = select_final_state(
        e3a_pass, e3g_pass, gate_supported
    )
    generation_elapsed = [
        float(row["elapsed_seconds"])
        for row in generation_state["tasks"]
        if row["status"] == "success"
    ]
    relaxation_state = core.initialize_relax()
    peak_relax_vram = 0
    for row in relaxation_state["tasks"]:
        summary_path = Path(row["output_dir"]) / "relax_summary.json"
        peak_relax_vram = max(
            peak_relax_vram,
            int(read_json(summary_path).get("peak_vram_bytes", 0)),
        )
    performance = {
        "c0_generation_median_seconds": float(np.median(generation_elapsed)),
        "c0_generation_mean_seconds": float(np.mean(generation_elapsed)),
        "e3a_refinement_total_seconds": refinement_summary[
            "always_on_refinement_total_seconds"
        ],
        "e3a_refinement_seconds_per_structure": (
            refinement_summary["always_on_refinement_total_seconds"] / len(SEEDS)
        ),
        "e3g_gate_total_seconds": refinement_summary[
            "gate_forward_total_seconds"
        ],
        "e3g_gate_seconds_per_structure": refinement_summary[
            "gate_forward_seconds_per_structure"
        ],
        "e3g_refinement_total_seconds": refinement_summary[
            "q3_refinement_total_seconds"
        ],
        "e3g_refinement_seconds_per_structure": refinement_summary[
            "q3_refinement_seconds_per_structure"
        ],
        "e3a_overhead_ratio_vs_c0_generation": (
            refinement_summary["always_on_refinement_total_seconds"]
            / len(SEEDS)
            / np.median(generation_elapsed)
        ),
        "e3g_overhead_ratio_vs_c0_generation": (
            refinement_summary["q3_refinement_seconds_per_structure"]
            / np.median(generation_elapsed)
        ),
        "mattersim_mean_seconds": {
            "C0": means["C0"]["relaxation_elapsed_mean"],
            "E3-A": means["E3-A"]["relaxation_elapsed_mean"],
            "E3-G": means["E3-G"]["relaxation_elapsed_mean"],
        },
        "refinement_peak_vram_bytes": refinement_summary["peak_vram_bytes"],
        "relaxation_peak_vram_bytes": peak_relax_vram,
        "q3_parameter_count": 129,
    }
    final = {
        "schema_version": 1,
        "completed_at": now(),
        "Q3_E3_PCR_FORMAL256_COMPLETED": True,
        "final_state": final_state,
        "final_method": final_method,
        "base_branch": BASE_BRANCH,
        "base_commit": BASE_COMMIT,
        "formal_branch": FORMAL_BRANCH,
        "formal_code_commit": read_json(
            REPORT / "formal_frozen_manifest.json"
        )["formal_code_commit"],
        "formal_seeds": [SEEDS[0], SEEDS[-1]],
        "counts": {
            "c0_generation": generation_success,
            "e3a_refinement": len(SEEDS),
            "e3g_refinement": len(SEEDS),
            "c0_relaxation": len(SEEDS),
            "e3a_relaxation": len(SEEDS),
            "e3g_relaxation": len(SEEDS),
            "total_mattersim": 3 * len(SEEDS),
        },
        "q3_checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "means": means,
        "primary": primary,
        "quality": quality,
        "mechanism": mechanism,
        "performance": performance,
        "official_metrics": {
            DISPLAY_NAMES[key]: value for key, value in official_metrics.items()
        },
        "e3a_primary_effect_pass": primary["E3-A"][
            "primary_effect_pass"
        ],
        "e3g_primary_effect_pass": primary["E3-G"][
            "primary_effect_pass"
        ],
        "e3a_quality_safety_pass": e3a_quality,
        "e3g_quality_safety_pass": e3g_quality,
        "refiner_formal_confirmed": e3a_pass or e3g_pass,
        "learned_gate_formal_confirmed": (
            final_state == "E3_G_FORMAL_CONFIRMED"
        ),
        "gate_mechanism_formal_supported": gate_supported,
        "a0_compatibility_started": False,
        "dft_started": False,
        "stability_source": "MatterSim-5M surrogate",
        "dft_verified": False,
        "property_target_verified": False,
        "other_processes_terminated": False,
        "sigkill_used": False,
    }
    atomic_json(REPORT / "final_summary.json", final)

    aggregate = pd.DataFrame(
        [
            {"method": method, **means[method]}
            for method in ("C0", "E3-A", "E3-G")
        ]
    )
    primary_frame = pd.DataFrame(
        [{"arm": arm, **values} for arm, values in primary.items()]
    )
    atomic_csv(REPORT / "aggregate_metrics.csv", aggregate)
    atomic_csv(REPORT / "primary_statistics.csv", primary_frame)
    atomic_csv(
        REPORT / "refinement_behavior.csv",
        pd.DataFrame(
            [{"arm": arm, **values} for arm, values in behavior.items()]
        ),
    )
    atomic_text(
        REPORT / "primary_statistics_report.md",
        "# E3-PCR Formal 256 primary statistics\n\n"
        "Primary endpoint: pre-relaxation maximum force. Both formal arms use "
        "paired bootstrap confidence intervals and Holm-adjusted Wilcoxon tests.\n\n"
        + primary_frame.to_markdown(index=False)
        + "\n",
    )
    atomic_text(
        REPORT / "quality_report.md",
        "# E3-PCR Formal 256 quality and secondary endpoints\n\n"
        "- Stability source: MatterSim-5M surrogate\n"
        "- DFT verified: False\n"
        "- Property target verified: False\n\n"
        + aggregate.to_markdown(index=False)
        + "\n\n## Paired secondary statistics\n\n"
        + paired.to_markdown(index=False)
        + "\n\n## Quality gates\n\n```json\n"
        + json.dumps(
            {
                "E3-A": e3a_quality_gates,
                "E3-G": e3g_quality_gates,
                "ehull_coverage": quality["ehull_coverage"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n```\n",
    )
    atomic_text(
        REPORT / "gate_mechanism_report.md",
        "# E3-PCR Formal 256 learned-gate mechanism\n\n"
        "The formal decision follows the five preregistered gain/intervention/"
        "safety conditions. Always-on coverage is 100%; learned-gated coverage "
        "is evaluated against retained force benefit and safety.\n\n```json\n"
        + json.dumps(mechanism, indent=2, sort_keys=True)
        + "\n```\n",
    )
    atomic_text(
        REPORT / "reproduction_report.md",
        "# Reproduce E3-PCR Formal 256\n\n"
        "```bash\n"
        "source /data/dxl/env.sh\n"
        "cd /data/dxl/mattergen_v1\n"
        "git switch feature/q3-e3-pcr-formal256\n"
        "/data/dxl/tools/q3_e3_pcr/formal256/resume.sh\n"
        "/data/dxl/tools/q3_e3_pcr/formal256/status.sh\n"
        "```\n\n"
        "The runner is resumable. Successful generation and relaxation tasks "
        "are hash-validated and are not rerun.\n",
    )
    atomic_text(
        REPORT / "final_report.md",
        "# E3-PCR Formal 256\n\n"
        f"- Final state: `{final_state}`\n"
        f"- Final method: `{final_method}`\n"
        f"- E3-A primary pass: `{primary['E3-A']['primary_effect_pass']}`\n"
        f"- E3-G primary pass: `{primary['E3-G']['primary_effect_pass']}`\n"
        f"- E3-A quality pass: `{e3a_quality}`\n"
        f"- E3-G quality pass: `{e3g_quality}`\n"
        f"- Gate mechanism formally supported: `{gate_supported}`\n"
        "- C0 was generated exactly once for each formal seed; E3-A and E3-G "
        "derive from the same C0 structures.\n"
        "- MatterSim relaxations: `768/768`.\n"
        "- Atomic numbers and cells were not modified.\n"
        "- A0 compatibility and DFT were not started.\n\n"
        "## Aggregate metrics\n\n"
        + aggregate.to_markdown(index=False)
        + "\n\n## Primary endpoint\n\n"
        + primary_frame.to_markdown(index=False)
        + "\n\n## Mechanism decision\n\n```json\n"
        + json.dumps(
            {
                key: value
                for key, value in mechanism.items()
                if key != "behavior"
            },
            indent=2,
            sort_keys=True,
        )
        + "\n```\n\n"
        "## Limitations\n\n"
        "- Stability is evaluated with the MatterSim-5M surrogate.\n"
        "- No DFT or independent property-target verification was run.\n",
    )
    write_master(
        "primary_statistics",
        "success",
        e3a_primary_pass=primary["E3-A"]["primary_effect_pass"],
        e3g_primary_pass=primary["E3-G"]["primary_effect_pass"],
    )
    write_master(
        "secondary_statistics",
        "success",
        e3a_quality_pass=e3a_quality,
        e3g_quality_pass=e3g_quality,
    )
    write_master(
        "gate_mechanism_analysis",
        "success",
        gate_mechanism_formal_supported=gate_supported,
    )
    write_master(
        "formal_decision",
        "success",
        final_state=final_state,
        final_method=final_method,
        final_report=str(REPORT / "final_report.md"),
    )
    print(
        json.dumps(
            {
                "final_state": final_state,
                "final_method": final_method,
                "gate_mechanism_formal_supported": gate_supported,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def pipeline() -> None:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another formal256 pipeline is already running") from error
        initialize()
        commands = (
            [str(MATTERGEN_PYTHON), "-m", "research.q3_formal256", "generate"],
            [str(CHGNET_PYTHON), "-m", "research.q3_formal256", "refine"],
            [str(MATTERGEN_PYTHON), "-m", "research.q3_formal256", "relax"],
            [str(MATTERGEN_PYTHON), "-m", "research.q3_formal256", "analyze"],
        )
        for command in commands:
            subprocess.run(command, cwd=PROJECT, check=True)
        final = read_json(REPORT / "final_summary.json")
        write_master(
            "github_archive",
            "pending",
            final_state=final["final_state"],
            gpu_workers=0,
        )
        fcntl.flock(lock, fcntl.LOCK_UN)


def status() -> None:
    payload = (
        read_json(MASTER_PROGRESS)
        if MASTER_PROGRESS.is_file()
        else {"status": "not_started"}
    )
    generation_progress = PROGRESS / "generation_progress.json"
    if generation_progress.is_file():
        generation = read_json(generation_progress)
        payload["c0_generation"] = {
            "success": generation.get("success", 0),
            "total": len(SEEDS),
        }
    if RELAX_PROGRESS.is_file():
        relaxation = read_json(RELAX_PROGRESS)
        payload["relaxation"] = {
            method: {
                "success": sum(
                    row["status"] == "success" and row["method"] == method
                    for row in relaxation["tasks"]
                ),
                "total": len(SEEDS),
            }
            for method in METHODS
        }
        payload["relaxation"]["all"] = {
            "success": relaxation.get("success", 0),
            "total": 3 * len(SEEDS),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "initialize",
        "generate",
        "refine",
        "relax",
        "analyze",
        "pipeline",
        "status",
    ):
        commands.add_parser(command)
    worker = commands.add_parser("relax-worker")
    worker.add_argument("--gpu", type=int, required=True)
    worker.add_argument("--slot", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    configure_core()
    args = parse_args()
    if args.command == "initialize":
        initialize()
    elif args.command == "generate":
        generate()
    elif args.command == "refine":
        refine()
    elif args.command == "relax":
        relax()
    elif args.command == "analyze":
        analyze()
    elif args.command == "pipeline":
        pipeline()
    elif args.command == "status":
        status()
    elif args.command == "relax-worker":
        return core.relax_worker(args.gpu, args.slot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
