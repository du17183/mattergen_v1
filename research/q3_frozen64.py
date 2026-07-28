#!/usr/bin/env python3
"""Frozen 64-seed validation for the Q3 E3-PCR post-generation refiner."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ase.io
import joblib
import numpy as np
import pandas as pd


ROOT = Path("/data/dxl")
PROJECT = ROOT / "mattergen_v1"
sys.path.insert(0, str(PROJECT))
RESULT = ROOT / "results/q3_e3_pcr/frozen64"
REPORT = ROOT / "reports/q3_e3_pcr/frozen64"
LOG = ROOT / "logs/q3_e3_pcr/frozen64"
EXTERNAL_TOOLS = ROOT / "tools/q3_e3_pcr/frozen64"
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
CONFIG = PROJECT / "configs/q3_e3_pcr_frozen64.json"
FROZEN_SOURCE_COMMIT = "b65f42a8792004c7c820e59fa4413e1310e06143"
FROZEN_SOURCE = PROJECT / "research/postgen_fastgate/refiner_eval.py"
Q3_CHECKPOINT = (
    ROOT / "results/postgen_fastgate/q3_refiner/model/q3_gate.joblib"
)
Q3_CHECKPOINT_SHA256 = (
    "b2ce1800fa0fa448f57d58010c8586b5de1b6666c4f198737a2f8a4bfabcb90e"
)
Q3_TRAINING_REPORT = (
    ROOT / "reports/postgen_fastgate/q3_refiner/training_and_offline_summary.json"
)
Q3_FREEZE_REPORT = (
    ROOT / "reports/postgen_fastgate/q3_refiner/frozen_refinement_manifest.json"
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
MATTERGEN_PYTHON = ROOT / "envs/mattergen_py310/bin/python"
CHGNET_PYTHON = ROOT / "envs/fn_pra_teacher/bin/python"
TASK_RUNNER = ROOT / "tools/guidance_stage7/run_sample.py"
RELAX_COMMON = ROOT / "tools/guidance_stage7_eval"
REFERENCE = ROOT / "reference_assets/reference_TRI2024correction.gz"
REFERENCE_LMDB = ROOT / "reference_assets/reference_TRI2024correction.lmdb"
SEEDS = tuple(range(32000, 32064))
RANDOM_GATE_SEEDS = (
    2026072901,
    2026072902,
    2026072903,
    2026072904,
    2026072905,
)
METHODS = ("C0", "Q3_E3_PCR", "ALWAYS_ON")
FEATURE_COLUMNS = (
    "num_atoms",
    "volume_per_atom",
    "mass_density_amu_ang3",
    "minimum_distance_angstrom",
    "atomic_number_mean",
    "atomic_number_std",
    "cell_condition",
    "chgnet_energy_per_atom_ev",
    "chgnet_force_rms_ev_ang",
    "chgnet_max_force_ev_ang",
    "chgnet_force_mean_ev_ang",
    "chgnet_stress_rms_gpa",
    "chgnet_stress_max_abs_gpa",
    "chgnet_mag_density",
)
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_SAMPLES = 20_000
STATE_LOCK = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def structure_hash(atoms: Any) -> str:
    digest = hashlib.sha256()
    for value in (atoms.numbers, atoms.positions, atoms.cell.array):
        array = np.asarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def cell_hash(atoms: Any) -> str:
    value = np.ascontiguousarray(np.asarray(atoms.cell.array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def event(stage: str, status: str, **values: Any) -> None:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    row = {
        "time": now(),
        "stage": stage,
        "status": status,
        **values,
    }
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
            "experiment": "Q3 E3-PCR frozen 64-seed validation",
            "base_commit": FROZEN_SOURCE_COMMIT,
            "evaluation_seeds": [SEEDS[0], SEEDS[-1]],
            "created_at": now(),
            "formal_256_started": False,
            "a0_compatibility_started": False,
            "dft_started": False,
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


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def initialize() -> None:
    for path in (RESULT, REPORT, LOG, EXTERNAL_TOOLS, PROGRESS):
        path.mkdir(parents=True, exist_ok=True)
    write_master("state_audit", "running")
    if git_output("rev-parse", f"{FROZEN_SOURCE_COMMIT}^{{commit}}") != (
        FROZEN_SOURCE_COMMIT
    ):
        raise RuntimeError("frozen source commit is unavailable")
    source_at_commit = subprocess.run(
        ["git", "show", f"{FROZEN_SOURCE_COMMIT}:research/postgen_fastgate/refiner_eval.py"],
        cwd=PROJECT,
        check=True,
        capture_output=True,
    ).stdout
    current_source = FROZEN_SOURCE.read_bytes()
    if source_at_commit != current_source:
        raise RuntimeError("frozen Q3 source differs from b65f42a")
    required_hashes = {
        Q3_CHECKPOINT: Q3_CHECKPOINT_SHA256,
        MATTERGEN_CHECKPOINT: MATTERGEN_SHA256,
        MATTERSIM: MATTERSIM_SHA256,
    }
    actual_hashes = {}
    for path, expected in required_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        actual_hashes[str(path)] = actual
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}")
    training = read_json(Q3_TRAINING_REPORT)
    frozen = read_json(Q3_FREEZE_REPORT)
    if (
        training["model_sha256"] != Q3_CHECKPOINT_SHA256
        or training["training_seed_range"] != [20000, 20063]
        or training["network"]["random_seed"] != 20260728
        or training["network"]["threshold"] != 0.5
        or training["network"]["trainable_parameters"] != 129
        or frozen["trust_region"]["steps"] != 5
        or frozen["trust_region"]["position_eta"] != 0.01
        or frozen["trust_region"]["per_step_radius_angstrom"] != 0.02
        or frozen["trust_region"]["backtrack_max"] != 3
        or frozen["trust_region"]["minimum_distance_angstrom"] != 0.5
    ):
        raise RuntimeError("frozen Q3 training/refinement contract mismatch")
    config_hash = sha256(CONFIG)
    source_hash = hashlib.sha256(current_source).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "base_branch": "feature/postgen-quality-modules-fastgate",
        "base_commit": FROZEN_SOURCE_COMMIT,
        "frozen_branch": "feature/q3-e3-pcr-frozen64",
        "frozen_source_path": str(FROZEN_SOURCE),
        "frozen_source_sha256": source_hash,
        "evaluation_script_path": str(Path(__file__).resolve()),
        "evaluation_base_commit": FROZEN_SOURCE_COMMIT,
        "config_path": str(CONFIG),
        "config_sha256": config_hash,
        "q3_checkpoint": str(Q3_CHECKPOINT),
        "q3_checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "mattergen_checkpoint": str(MATTERGEN_CHECKPOINT),
        "mattergen_checkpoint_sha256": MATTERGEN_SHA256,
        "mattersim_checkpoint": str(MATTERSIM),
        "mattersim_checkpoint_sha256": MATTERSIM_SHA256,
        "training_seed_range": [20000, 20063],
        "training_data_split": "frozen 64 A0 historical outputs",
        "network_seed": 20260728,
        "gate_threshold": 0.5,
        "trainable_parameters": 129,
        "input_features": list(FEATURE_COLUMNS),
        "refinement": {
            "steps": 5,
            "position_eta": 0.01,
            "per_step_radius_angstrom": 0.02,
            "maximum_cumulative_displacement_angstrom": 0.1,
            "backtrack_max": 3,
            "minimum_distance_angstrom": 0.5,
            "atomic_numbers_modified": False,
            "cell_modified": False,
        },
        "fallback_conditions": [
            "learned gate probability below 0.5",
            "non-finite proposal or CHGNet energy",
            "minimum distance below 0.5 angstrom",
            "new CHGNet energy above prior energy",
            "all three backtracking scales rejected",
        ],
        "evaluation_seeds": [SEEDS[0], SEEDS[-1]],
        "seed_count": len(SEEDS),
        "seed_audit": {
            "training_intersection": [],
            "development_33000_33127_intersection": [],
            "known_cg_tdr_intersection": [],
            "manual_repository_audit": "no seed-field or path conflicts",
        },
        "hashes": actual_hashes,
        "frozen_files_match_base_commit": True,
        "tuning_after_freeze": False,
    }
    atomic_json(REPORT / "frozen_manifest.json", manifest)
    lines = [
        "# Q3 E3-PCR frozen64 manifest",
        "",
        f"- Frozen source commit: `{FROZEN_SOURCE_COMMIT}`",
        f"- Q3 checkpoint: `{Q3_CHECKPOINT}`",
        f"- Q3 checkpoint SHA256: `{Q3_CHECKPOINT_SHA256}`",
        f"- Config SHA256: `{config_hash}`",
        "- Training seeds: `20000–20063`",
        "- Evaluation seeds: `32000–32063`",
        "- Gate: 14 → 8 → 1 tanh MLP, threshold 0.5, 129 parameters",
        "- Refiner: 5 steps, eta 0.01, 0.02 Å per-step radius, 3 backtracks",
        "- Atomic numbers modified: `False`",
        "- Cell modified: `False`",
        "- Tuning after freeze: `False`",
        "",
    ]
    (REPORT / "frozen_manifest.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    write_master(
        "freeze_manifest",
        "success",
        frozen_manifest=str(REPORT / "frozen_manifest.json"),
        q3_checkpoint_sha256=Q3_CHECKPOINT_SHA256,
        config_sha256=config_hash,
    )


def configure_generation_module() -> Any:
    from research.postgen_fastgate import new_eval as generation

    generation.RESULT = RESULT
    generation.REPORT = REPORT
    generation.LOG = LOG
    generation.GENERATION = GENERATION
    generation.PROGRESS = PROGRESS
    generation.GEN_PROGRESS = PROGRESS / "generation_progress.json"
    generation.MASTER_PROGRESS = MASTER_PROGRESS
    generation.SEEDS = SEEDS
    generation.POOL_SIZE = 1
    generation.POOL_COUNT = len(SEEDS)
    return generation


def generate() -> None:
    write_master("c0_generation", "running")
    generation = configure_generation_module()
    generation.run_generation()
    state = generation.load_generation_progress()
    if state["success"] != len(SEEDS):
        raise RuntimeError("C0 generation incomplete")
    elapsed = [
        float(row["elapsed_seconds"])
        for row in state["tasks"]
        if row["elapsed_seconds"] is not None
    ]
    write_master(
        "c0_generation",
        "success",
        c0_generation=f"{state['success']}/{len(SEEDS)}",
        median_elapsed_seconds=float(np.median(elapsed)),
    )


def extract_features(model: Any) -> pd.DataFrame:
    from pymatgen.io.ase import AseAtomsAdaptor
    from research.postgen_fastgate.features import structure_features

    existing = pd.read_csv(FEATURES) if FEATURES.is_file() else pd.DataFrame()
    done = set(existing["seed"].astype(int)) if len(existing) else set()
    rows = existing.to_dict(orient="records")
    pending = [seed for seed in SEEDS if seed not in done]
    for start in range(0, len(pending), 32):
        chunk = pending[start : start + 32]
        atoms = [
            ase.io.read(GENERATION / str(seed) / "generated_crystals.extxyz")
            for seed in chunk
        ]
        structures = [AseAtomsAdaptor.get_structure(value) for value in atoms]
        predictions = model.predict_structure(
            structures,
            task="efsm",
            return_site_energies=False,
            batch_size=32,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        if len(predictions) != len(chunk):
            raise RuntimeError("CHGNet feature count mismatch")
        for seed, structure, prediction in zip(
            chunk, atoms, predictions, strict=True
        ):
            index = seed - SEEDS[0]
            path = GENERATION / str(seed) / "generated_crystals.extxyz"
            rows.append(
                {
                    "method": "C0",
                    "seed": seed,
                    "evaluation_index": index,
                    "split": "frozen64_independent",
                    "input_path": str(path),
                    "input_hash": structure_hash(structure),
                    **structure_features(structure, prediction),
                }
            )
        atomic_csv(FEATURES, pd.DataFrame(rows).sort_values("seed"))
        print(
            json.dumps(
                {
                    "stage": "feature_extraction",
                    "completed": len(rows),
                    "total": len(SEEDS),
                }
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    if (
        len(frame) != len(SEEDS)
        or frame["seed"].astype(int).tolist() != list(SEEDS)
        or frame.loc[:, FEATURE_COLUMNS].isna().any().any()
    ):
        raise RuntimeError("frozen64 feature manifest mismatch")
    return frame


def run_refinement_subset(
    model: Any,
    originals: list[Any],
    active: np.ndarray,
    instrumentation: dict[str, int],
) -> tuple[list[Any], list[dict[str, int]], float]:
    from research.postgen_fastgate import refiner_eval as frozen

    if (
        frozen.REFINEMENT_STEPS != 5
        or frozen.POSITION_ETA != 0.01
        or frozen.POSITION_RADIUS_ANGSTROM != 0.02
        or frozen.BACKTRACK_MAX != 3
        or frozen.MINIMUM_DISTANCE_ANGSTROM != 0.5
    ):
        raise RuntimeError("runtime refiner constants differ from frozen contract")
    outputs = [atoms.copy() for atoms in originals]
    counters = [
        {"accepted_steps": 0, "fallback_count": 0, "backtracking_count": 0}
        for _ in originals
    ]
    if not len(active):
        return outputs, counters, 0.0
    active_atoms = [outputs[index] for index in active]
    active_counters = [counters[index] for index in active]
    original_proposal = frozen.position_proposal

    def instrumented_proposal(
        atoms: Any, forces: np.ndarray, scale: float
    ) -> Any:
        raw = frozen.POSITION_ETA * scale * np.asarray(forces, dtype=float)
        norms = np.linalg.norm(raw, axis=1)
        cap = frozen.POSITION_RADIUS_ANGSTROM * scale
        instrumentation["proposal_atoms"] += len(norms)
        instrumentation["clipped_atoms"] += int((norms > cap).sum())
        instrumentation["proposal_calls"] += 1
        return original_proposal(atoms, forces, scale)

    frozen.position_proposal = instrumented_proposal
    started = time.monotonic()
    try:
        for _step in range(frozen.REFINEMENT_STEPS):
            active_atoms, _predictions = frozen.advance(
                model, active_atoms, active_counters
            )
    finally:
        frozen.position_proposal = original_proposal
    elapsed = time.monotonic() - started
    for local, index in enumerate(active):
        outputs[index] = active_atoms[local]
        counters[index] = active_counters[local]
    return outputs, counters, elapsed


def refine() -> None:
    import torch
    from chgnet.model.model import CHGNet
    from research.postgen_fastgate import refiner_eval as frozen

    write_master("q3_refinement", "running")
    if sha256(Q3_CHECKPOINT) != Q3_CHECKPOINT_SHA256:
        raise RuntimeError("Q3 checkpoint changed after freeze")
    torch.cuda.reset_peak_memory_stats()
    model = CHGNet.load(model_name="0.3.0", verbose=False, use_device="cuda")
    features = extract_features(model)
    network = joblib.load(Q3_CHECKPOINT)
    values = features.loc[:, FEATURE_COLUMNS].to_numpy(float)
    gate_started = time.perf_counter()
    probabilities = network.predict_proba(values)[:, 1]
    gate_forward_seconds = time.perf_counter() - gate_started
    apply_gate = probabilities >= 0.5
    originals = [ase.io.read(path) for path in features["input_path"]]
    learned_instrumentation = {
        "proposal_atoms": 0,
        "clipped_atoms": 0,
        "proposal_calls": 0,
    }
    always_instrumentation = {
        "proposal_atoms": 0,
        "clipped_atoms": 0,
        "proposal_calls": 0,
    }
    learned, learned_counters, learned_elapsed = run_refinement_subset(
        model,
        originals,
        np.flatnonzero(apply_gate),
        learned_instrumentation,
    )
    always, always_counters, always_elapsed = run_refinement_subset(
        model,
        originals,
        np.arange(len(originals)),
        always_instrumentation,
    )
    rows = []
    for index, (
        seed,
        original,
        learned_atoms,
        always_atoms,
        probability,
        apply,
    ) in enumerate(
        zip(
            SEEDS,
            originals,
            learned,
            always,
            probabilities,
            apply_gate,
            strict=True,
        )
    ):
        for method, output_atoms, counters in (
            ("Q3_E3_PCR", learned_atoms, learned_counters[index]),
            ("ALWAYS_ON", always_atoms, always_counters[index]),
        ):
            if not np.array_equal(output_atoms.numbers, original.numbers):
                raise RuntimeError(f"{method} changed atomic numbers")
            if not np.array_equal(output_atoms.cell.array, original.cell.array):
                raise RuntimeError(f"{method} changed cell")
            if not frozen.finite_safe(output_atoms):
                raise RuntimeError(f"{method} produced unsafe structure")
            output_dir = REFINED / method / f"{index:02d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "refined_structure.extxyz"
            ase.io.write(output_path, output_atoms, format="extxyz")
            checked = ase.io.read(output_path)
            displacement = frozen.wrapped_displacement_max(original, checked)
            if displacement > 0.1000001:
                raise RuntimeError("maximum cumulative displacement exceeded")
            input_hash = structure_hash(original)
            output_hash = structure_hash(checked)
            gate_applied = bool(apply) if method == "Q3_E3_PCR" else True
            if method == "Q3_E3_PCR" and not gate_applied:
                if input_hash != output_hash:
                    raise RuntimeError("gate-rejected Q3 sample is not exact fallback")
            rows.append(
                {
                    "method": method,
                    "evaluation_index": index,
                    "seed": seed,
                    "input_path": features.iloc[index]["input_path"],
                    "output_path": str(output_path),
                    "gate_probability": float(probability),
                    "gate_applied": gate_applied,
                    **counters,
                    "maximum_wrapped_displacement_angstrom": displacement,
                    "minimum_distance_angstrom": frozen.minimum_distance(checked),
                    "input_hash": input_hash,
                    "output_hash": output_hash,
                    "atomic_numbers_unchanged": bool(
                        np.array_equal(checked.numbers, original.numbers)
                    ),
                    "cell_unchanged": cell_hash(checked) == cell_hash(original),
                    "exact_baseline_fallback": input_hash == output_hash,
                }
            )
    manifest = pd.DataFrame(rows).sort_values(
        ["method", "evaluation_index"]
    ).reset_index(drop=True)
    atomic_csv(REFINEMENT_MANIFEST, manifest)
    q3 = manifest[manifest["method"] == "Q3_E3_PCR"]
    summary = {
        "schema_version": 1,
        "created_at": now(),
        "q3_checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "q3_parameter_count": 129,
        "gate_threshold": 0.5,
        "gate_on_count": int(q3["gate_applied"].sum()),
        "gate_on_rate": float(q3["gate_applied"].mean()),
        "gate_forward_total_seconds": gate_forward_seconds,
        "gate_forward_seconds_per_structure": gate_forward_seconds / len(SEEDS),
        "q3_refinement_total_seconds": learned_elapsed,
        "q3_refinement_seconds_per_structure": learned_elapsed / len(SEEDS),
        "always_on_refinement_total_seconds": always_elapsed,
        "learned_instrumentation": learned_instrumentation,
        "always_on_instrumentation": always_instrumentation,
        "learned_clipping_rate": (
            learned_instrumentation["clipped_atoms"]
            / max(1, learned_instrumentation["proposal_atoms"])
        ),
        "gate_rejected_exact_fallback_count": int(
            ((~q3["gate_applied"].astype(bool)) & q3["exact_baseline_fallback"]).sum()
        ),
        "maximum_displacement_angstrom": float(
            q3["maximum_wrapped_displacement_angstrom"].max()
        ),
        "mean_displacement_angstrom": float(
            q3["maximum_wrapped_displacement_angstrom"].mean()
        ),
        "median_displacement_angstrom": float(
            q3["maximum_wrapped_displacement_angstrom"].median()
        ),
        "backtracking_mean": float(q3["backtracking_count"].mean()),
        "fallback_mean": float(q3["fallback_count"].mean()),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "atomic_numbers_modified": 0,
        "cell_modified": 0,
    }
    atomic_json(REFINEMENT_SUMMARY, summary)
    write_master(
        "q3_refinement",
        "success",
        q3_refinement=f"{len(q3)}/{len(SEEDS)}",
        gate_on_count=summary["gate_on_count"],
        gate_on_rate=summary["gate_on_rate"],
        maximum_displacement_angstrom=summary[
            "maximum_displacement_angstrom"
        ],
    )


def relax_rows() -> list[dict[str, Any]]:
    if not REFINEMENT_MANIFEST.is_file():
        raise RuntimeError("refinement manifest does not exist")
    manifest = pd.read_csv(REFINEMENT_MANIFEST)
    by_key = {
        (str(row.method), int(row.seed)): row
        for row in manifest.itertuples(index=False)
    }
    rows = []
    for index, seed in enumerate(SEEDS):
        inputs = {
            "C0": GENERATION / str(seed) / "generated_crystals.extxyz",
            "Q3_E3_PCR": Path(by_key[("Q3_E3_PCR", seed)].output_path),
            "ALWAYS_ON": Path(by_key[("ALWAYS_ON", seed)].output_path),
        }
        for method in METHODS:
            atoms = ase.io.read(inputs[method])
            rows.append(
                {
                    "task_id": f"{method}_{seed}",
                    "method": method,
                    "evaluation_index": index,
                    "seed": seed,
                    "input_path": str(inputs[method]),
                    "input_hash": structure_hash(atoms),
                    "output_dir": str(RELAXED / method / f"{index:02d}"),
                    "status": "pending",
                    "attempt": 0,
                    "gpu": None,
                    "slot": None,
                    "elapsed_seconds": None,
                    "pre_relax_max_force_ev_ang": None,
                    "error": "",
                }
            )
    return rows


def validate_relax(row: dict[str, Any]) -> bool:
    try:
        output = Path(row["output_dir"])
        summary = read_json(output / "relax_summary.json")
        atoms = ase.io.read(output / "relaxed_structure.extxyz")
        return (
            summary.get("success") is True
            and summary["task_id"] == row["task_id"]
            and summary["input_hash"] == row["input_hash"]
            and sha256(MATTERSIM) == summary["checkpoint_sha256"]
            and np.isfinite(atoms.positions).all()
            and np.isfinite(atoms.cell.array).all()
        )
    except BaseException:
        return False


def save_relax_progress(state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    state["success"] = sum(row["status"] == "success" for row in state["tasks"])
    state["total"] = len(state["tasks"])
    atomic_json(RELAX_PROGRESS, state)
    atomic_csv(PROGRESS / "relax_progress.csv", pd.DataFrame(state["tasks"]))


def locked_relax(operation: Any) -> Any:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    with (PROGRESS / "relax_progress.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = (
            read_json(RELAX_PROGRESS)
            if RELAX_PROGRESS.is_file()
            else {"schema_version": 1, "created_at": now(), "tasks": relax_rows()}
        )
        result = operation(state)
        save_relax_progress(state)
        fcntl.flock(lock, fcntl.LOCK_UN)
        return result


def initialize_relax() -> dict[str, Any]:
    def operation(state: dict[str, Any]) -> dict[str, Any]:
        expected = relax_rows()
        if [row["task_id"] for row in state["tasks"]] != [
            row["task_id"] for row in expected
        ]:
            raise RuntimeError("frozen64 relaxation contract mismatch")
        for row, contract in zip(state["tasks"], expected, strict=True):
            row["input_path"] = contract["input_path"]
            row["input_hash"] = contract["input_hash"]
            row["output_dir"] = contract["output_dir"]
            if validate_relax(row):
                row["status"] = "success"
                summary = read_json(Path(row["output_dir"]) / "relax_summary.json")
                row["pre_relax_max_force_ev_ang"] = summary[
                    "pre_relax_max_force_ev_ang"
                ]
            elif row["status"] in {"running", "success"}:
                row["status"] = "interrupted"
        return state

    return locked_relax(operation)


def claim_relax(gpu: int, slot: int) -> dict[str, Any] | None:
    def operation(state: dict[str, Any]) -> dict[str, Any] | None:
        for row in state["tasks"]:
            if row["status"] in {"pending", "interrupted", "failed"} and int(
                row["attempt"]
            ) < 2:
                row.update(
                    status="running",
                    attempt=int(row["attempt"]) + 1,
                    gpu=gpu,
                    slot=slot,
                    error="",
                )
                return dict(row)
        return None

    return locked_relax(operation)


def finish_relax(task_id: str, success: bool, **values: Any) -> None:
    def operation(state: dict[str, Any]) -> None:
        row = next(item for item in state["tasks"] if item["task_id"] == task_id)
        row.update(status="success" if success else "failed", **values)

    locked_relax(operation)


def quarantine(path: Path) -> None:
    if path.exists():
        destination = path.with_name(
            f"{path.name}.incomplete."
            f"{datetime.now().strftime('%Y%m%dT%H%M%S')}.{os.getpid()}"
        )
        os.replace(path, destination)


def relax_worker(gpu: int, slot: int) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    import torch
    from mattersim.forcefield import MatterSimCalculator

    sys.path.insert(0, str(RELAX_COMMON))
    from relax_common import load_potential, relax_group

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    potential = load_potential("cuda")
    stop = False

    def handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    while not stop:
        row = claim_relax(gpu, slot)
        if row is None:
            break
        output = Path(row["output_dir"])
        try:
            quarantine(output)
            output.mkdir(parents=True)
            atoms = ase.io.read(row["input_path"])
            if structure_hash(atoms) != row["input_hash"]:
                raise RuntimeError("relax input hash mismatch")
            probe = atoms.copy()
            probe.calc = MatterSimCalculator.from_potential(
                potential=potential, device="cuda"
            )
            initial_energy = float(probe.get_potential_energy())
            initial_forces = np.asarray(probe.get_forces(), dtype=float)
            initial_max_force = float(
                np.linalg.norm(initial_forces, axis=1).max()
            )
            started = time.monotonic()
            result = relax_group(potential, [atoms])[0]
            elapsed = time.monotonic() - started
            output_path = output / "relaxed_structure.extxyz"
            ase.io.write(output_path, result["atoms"], format="extxyz")
            checked = ase.io.read(output_path)
            summary = {
                "success": True,
                "task_id": row["task_id"],
                "method": row["method"],
                "evaluation_index": row["evaluation_index"],
                "seed": row["seed"],
                "gpu": gpu,
                "slot": slot,
                "input_path": row["input_path"],
                "input_hash": row["input_hash"],
                "output_hash": structure_hash(checked),
                "checkpoint": str(MATTERSIM),
                "checkpoint_sha256": MATTERSIM_SHA256,
                "elapsed_seconds": elapsed,
                "initial_energy_ev": initial_energy,
                "initial_energy_per_atom_ev": initial_energy / len(atoms),
                "pre_relax_max_force_ev_ang": initial_max_force,
                "energy_ev": result["energy_ev"],
                "energy_per_atom_ev": result["energy_per_atom_ev"],
                "maximum_force_ev_ang": result["max_force_ev_ang"],
                "steps": result["steps"],
                "converged": result["converged"],
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            }
            atomic_json(output / "relax_summary.json", summary)
            finish_relax(
                row["task_id"],
                True,
                elapsed_seconds=elapsed,
                pre_relax_max_force_ev_ang=initial_max_force,
                error="",
            )
        except BaseException:
            error = traceback.format_exc()
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.log").write_text(error, encoding="utf-8")
            finish_relax(row["task_id"], False, error=error[-4000:])
    return 0


def relax() -> None:
    if sha256(MATTERSIM) != MATTERSIM_SHA256:
        raise RuntimeError("MatterSim checkpoint SHA256 mismatch")
    write_master("mattersim_relaxation", "running")
    state = initialize_relax()
    if state["success"] == len(METHODS) * len(SEEDS):
        write_master(
            "mattersim_relaxation", "success", relaxed=state["success"]
        )
        return
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
                    str(Path(__file__).resolve()),
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
    state = initialize_relax()
    expected = len(METHODS) * len(SEEDS)
    if any(code != 0 for code in codes) or state["success"] != expected:
        write_master(
            "mattersim_relaxation",
            "failed",
            relaxed=state["success"],
            expected=expected,
        )
        raise RuntimeError(
            f"MatterSim relaxation incomplete: {state['success']}/{expected}"
        )
    write_master("mattersim_relaxation", "success", relaxed=state["success"])


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(
        0, len(values), size=(BOOTSTRAP_SAMPLES, len(values))
    )
    low, high = np.quantile(values[indexes].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def official_frames() -> dict[str, pd.DataFrame]:
    from pymatgen.io.ase import AseAtomsAdaptor

    state = initialize_relax()
    if state["success"] != len(METHODS) * len(SEEDS):
        raise RuntimeError("official metrics require complete relaxation")
    rows_by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHODS
    }
    for row in state["tasks"]:
        output = Path(row["output_dir"])
        summary = read_json(output / "relax_summary.json")
        original = ase.io.read(row["input_path"])
        relaxed_atoms = ase.io.read(output / "relaxed_structure.extxyz")
        structure = AseAtomsAdaptor.get_structure(relaxed_atoms)
        rows_by_method[row["method"]].append(
            {
                "method": row["method"],
                "seed": int(row["evaluation_index"]),
                "pool_id": int(row["evaluation_index"]),
                "candidate_seed": int(row["seed"]),
                "energy_ev": float(summary["energy_ev"]),
                "energy_per_atom_ev": float(summary["energy_per_atom_ev"]),
                "maximum_force_ev_ang": float(summary["maximum_force_ev_ang"]),
                "pre_relax_max_force_ev_ang": float(
                    summary["pre_relax_max_force_ev_ang"]
                ),
                "converged": bool(summary["converged"]),
                "relax_elapsed_seconds": float(summary["elapsed_seconds"]),
                "steps": int(summary["steps"]),
                "formula": structure.composition.reduced_formula,
                "chemical_system": structure.composition.chemical_system,
                "_relaxed_atoms": relaxed_atoms,
                "_original_atoms": original,
            }
        )
    return {
        method: pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
        for method, rows in rows_by_method.items()
    }


def paired_statistics(
    baseline: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    from scipy.stats import binomtest, wilcoxon

    rows = []
    continuous = (
        "energy_above_hull_per_atom",
        "rmsd_from_relaxation",
        "pre_relax_max_force_ev_ang",
        "steps",
    )
    for column in continuous:
        difference = (
            selected[column].to_numpy(float)
            - baseline[column].to_numpy(float)
        )
        low, high = bootstrap_ci(difference)
        test = (
            wilcoxon(difference, zero_method="pratt")
            if np.any(np.abs(difference) > 1.0e-12)
            else None
        )
        wins = int((difference < -1.0e-12).sum())
        losses = int((difference > 1.0e-12).sum())
        rows.append(
            {
                "metric": column,
                "type": "continuous",
                "baseline_mean": float(baseline[column].mean()),
                "selected_mean": float(selected[column].mean()),
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
        )
    for column in (
        "stable",
        "novel_unique_stable",
        "comp_validity",
        "structure_validity",
        "novel",
        "unique",
        "converged",
    ):
        left = baseline[column].astype(bool).to_numpy()
        right = selected[column].astype(bool).to_numpy()
        wins = int((~left & right).sum())
        losses = int((left & ~right).sum())
        discordant = wins + losses
        difference = right.astype(float) - left.astype(float)
        low, high = bootstrap_ci(difference)
        rows.append(
            {
                "metric": column,
                "type": "binary",
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
        )
    return pd.DataFrame(rows)


def metric_means(frame: pd.DataFrame) -> dict[str, float]:
    ehull = frame["energy_above_hull_per_atom"].to_numpy(float)
    stable = frame["stable"].astype(bool).to_numpy()
    novel = frame["novel"].astype(bool).to_numpy()
    unique = frame["unique"].astype(bool).to_numpy()
    metastable = ehull <= 0.2
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
        "ehull": float(ehull.mean()),
        "ehull_median": float(np.median(ehull)),
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
    }


def force_robustness(
    baseline: pd.DataFrame, selected: pd.DataFrame
) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    left = baseline["pre_relax_max_force_ev_ang"].to_numpy(float)
    right = selected["pre_relax_max_force_ev_ang"].to_numpy(float)
    difference = right - left
    low, high = bootstrap_ci(difference)
    test = wilcoxon(difference, zero_method="pratt")
    leave_one_out = np.asarray(
        [
            np.delete(difference, index).mean()
            for index in range(len(difference))
        ]
    )
    most_favorable = int(np.argmin(difference))
    without_most_favorable_left = np.delete(left, most_favorable)
    without_most_favorable_right = np.delete(right, most_favorable)
    total = float(np.abs(difference).sum())
    return {
        "baseline_mean": float(left.mean()),
        "selected_mean": float(right.mean()),
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "relative_change": float(right.mean() / left.mean() - 1.0),
        "bootstrap_95_ci": [low, high],
        "wilcoxon_p": float(test.pvalue),
        "wins": int((difference < -1.0e-12).sum()),
        "ties": int((np.abs(difference) <= 1.0e-12).sum()),
        "losses": int((difference > 1.0e-12).sum()),
        "leave_one_out_mean_difference_range": [
            float(leave_one_out.min()),
            float(leave_one_out.max()),
        ],
        "remove_most_favorable_sample_index": most_favorable,
        "remove_most_favorable_relative_change": float(
            without_most_favorable_right.mean()
            / without_most_favorable_left.mean()
            - 1.0
        ),
        "maximum_single_sample_absolute_contribution_rate": (
            float(np.abs(difference).max() / total) if total else 0.0
        ),
    }


def random_gate_ablation(
    baseline: pd.DataFrame,
    always: pd.DataFrame,
    learned_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline_force = baseline["pre_relax_max_force_ev_ang"].to_numpy(float)
    always_force = always["pre_relax_max_force_ev_ang"].to_numpy(float)
    rows = []
    for random_seed in RANDOM_GATE_SEEDS:
        rng = np.random.default_rng(random_seed)
        selected = np.sort(
            rng.choice(len(SEEDS), size=learned_count, replace=False)
        )
        mask = np.zeros(len(SEEDS), dtype=bool)
        mask[selected] = True
        random_force = np.where(mask, always_force, baseline_force)
        rows.append(
            {
                "random_seed": random_seed,
                "gate_on_count": learned_count,
                "gate_on_rate": learned_count / len(SEEDS),
                "mean_force": float(random_force.mean()),
                "force_absolute_change": float(
                    random_force.mean() - baseline_force.mean()
                ),
                "force_relative_change": float(
                    random_force.mean() / baseline_force.mean() - 1.0
                ),
                "wins": int((random_force < baseline_force - 1.0e-12).sum()),
                "ties": int(
                    (np.abs(random_force - baseline_force) <= 1.0e-12).sum()
                ),
                "losses": int((random_force > baseline_force + 1.0e-12).sum()),
                "selected_indexes": " ".join(str(value) for value in selected),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "random_gate_runs": len(frame),
        "mean_force_relative_change": float(
            frame["force_relative_change"].mean()
        ),
        "force_relative_change_range": [
            float(frame["force_relative_change"].min()),
            float(frame["force_relative_change"].max()),
        ],
        "mean_wins": float(frame["wins"].mean()),
        "mean_losses": float(frame["losses"].mean()),
    }
    return frame, summary


def mechanism_analysis(
    baseline: pd.DataFrame,
    q3: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from scipy.stats import spearmanr

    q3_manifest = (
        manifest[manifest["method"] == "Q3_E3_PCR"]
        .sort_values("evaluation_index")
        .reset_index(drop=True)
    )
    features = pd.read_csv(FEATURES).sort_values("evaluation_index").reset_index(
        drop=True
    )
    left_force = baseline["pre_relax_max_force_ev_ang"].to_numpy(float)
    right_force = q3["pre_relax_max_force_ev_ang"].to_numpy(float)
    improvement = left_force - right_force
    table = features[
        [
            "seed",
            "num_atoms",
            "volume_ang3",
            "minimum_distance_angstrom",
            "chgnet_energy_per_atom_ev",
            "chgnet_max_force_ev_ang",
            "chgnet_mag_density",
        ]
    ].copy()
    table["gate_probability"] = q3_manifest["gate_probability"].to_numpy(float)
    table["gate_applied"] = q3_manifest["gate_applied"].astype(bool).to_numpy()
    table["initial_mattersim_max_force"] = left_force
    table["q3_mattersim_max_force"] = right_force
    table["force_improvement"] = improvement
    magnetic_numbers = {
        21, 22, 23, 24, 25, 26, 27, 28, 29, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79,
    }
    magnetic_counts = []
    for seed in SEEDS:
        atoms = ase.io.read(
            GENERATION / str(seed) / "generated_crystals.extxyz"
        )
        magnetic_counts.append(
            sum(int(number) in magnetic_numbers for number in atoms.numbers)
        )
    table["magnetic_element_count"] = magnetic_counts
    force_median = float(np.median(left_force))
    high = left_force >= force_median
    low = ~high
    correlation = spearmanr(
        table["gate_probability"], table["force_improvement"]
    )
    applied = table["gate_applied"].to_numpy(bool)
    summary = {
        "initial_force_median": force_median,
        "high_initial_force_count": int(high.sum()),
        "high_initial_force_improvement_rate": float(
            (improvement[high] > 1.0e-12).mean()
        ),
        "high_initial_force_gate_on_rate": float(applied[high].mean()),
        "low_initial_force_count": int(low.sum()),
        "low_initial_force_exact_fallback_rate": float((~applied[low]).mean()),
        "gate_confidence_force_improvement_spearman": float(
            correlation.statistic
        ),
        "gate_confidence_force_improvement_spearman_p": float(
            correlation.pvalue
        ),
        "gate_on_force_improvement_rate": float(
            (improvement[applied] > 1.0e-12).mean()
        )
        if applied.any()
        else 0.0,
        "gate_on_force_worsening_rate": float(
            (improvement[applied] < -1.0e-12).mean()
        )
        if applied.any()
        else 0.0,
    }
    return table, summary


def analyze() -> None:
    write_master("metrics", "running")
    frames = official_frames()
    tool_root = ROOT / "tools/innovation2_next"
    sys.path.insert(0, str(tool_root))
    import analyze_corrector_64 as official

    official.ROOT = ROOT
    official.RESULT = RESULT
    official.REPORT = REPORT
    official.PROGRESS = PROGRESS
    official.REFERENCE = REFERENCE
    official.REFERENCE_LMDB = REFERENCE_LMDB
    official.CONFIGS = METHODS
    official.SEEDS = list(range(len(SEEDS)))
    official.STABILITY_THRESHOLD = 0.1
    metrics, errors = official.official_metrics(frames)
    if errors:
        raise RuntimeError(f"official metrics failures: {errors}")
    evaluated = {}
    for method in METHODS:
        frame = pd.read_csv(
            REPORT / method / "official_metrics_per_structure.csv"
        ).sort_values("seed").reset_index(drop=True)
        raw = frames[method].sort_values("seed").reset_index(drop=True)
        frame["pre_relax_max_force_ev_ang"] = raw[
            "pre_relax_max_force_ev_ang"
        ]
        frame["steps"] = raw["steps"]
        frame["converged"] = raw["converged"]
        evaluated[method] = frame
    baseline = evaluated["C0"]
    q3 = evaluated["Q3_E3_PCR"]
    always = evaluated["ALWAYS_ON"]
    stats = paired_statistics(baseline, q3)
    atomic_csv(REPORT / "paired_statistics.csv", stats)
    robustness = force_robustness(baseline, q3)
    atomic_json(REPORT / "force_robustness.json", robustness)
    baseline_mean = metric_means(baseline)
    q3_mean = metric_means(q3)
    always_mean = metric_means(always)
    changes = {
        key: q3_mean[key] - baseline_mean[key]
        for key in baseline_mean
    }
    changes["pre_relax_max_force_relative"] = (
        q3_mean["pre_relax_max_force"]
        / baseline_mean["pre_relax_max_force"]
        - 1.0
    )
    changes["rmsd_relative"] = q3_mean["rmsd"] / baseline_mean["rmsd"] - 1.0
    q3_manifest = pd.read_csv(REFINEMENT_MANIFEST)
    q3_rows = q3_manifest[q3_manifest["method"] == "Q3_E3_PCR"]
    learned_count = int(q3_rows["gate_applied"].sum())
    random_frame, random_summary = random_gate_ablation(
        baseline, always, learned_count
    )
    atomic_csv(REPORT / "random_gate_ablation.csv", random_frame)
    learned_force_change = changes["pre_relax_max_force_relative"]
    always_force_change = (
        always_mean["pre_relax_max_force"]
        / baseline_mean["pre_relax_max_force"]
        - 1.0
    )
    learned_difference = (
        q3["pre_relax_max_force_ev_ang"].to_numpy(float)
        - baseline["pre_relax_max_force_ev_ang"].to_numpy(float)
    )
    always_difference = (
        always["pre_relax_max_force_ev_ang"].to_numpy(float)
        - baseline["pre_relax_max_force_ev_ang"].to_numpy(float)
    )
    ablation = {
        "baseline_mean_force": baseline_mean["pre_relax_max_force"],
        "always_on_mean_force": always_mean["pre_relax_max_force"],
        "learned_gate_mean_force": q3_mean["pre_relax_max_force"],
        "always_on_force_relative_change": always_force_change,
        "random_gate": random_summary,
        "learned_gate_force_relative_change": learned_force_change,
        "always_on_worsening_rate": float(
            (always_difference > 1.0e-12).mean()
        ),
        "learned_gate_worsening_rate": float(
            (learned_difference > 1.0e-12).mean()
        ),
        "learned_vs_always_on": float(
            q3_mean["pre_relax_max_force"]
            - always_mean["pre_relax_max_force"]
        ),
        "learned_vs_random_gate_mean_relative_change": float(
            learned_force_change
            - random_summary["mean_force_relative_change"]
        ),
    }
    mechanism_table, mechanism = mechanism_analysis(
        baseline, q3, q3_manifest
    )
    atomic_csv(REPORT / "mechanism_per_structure.csv", mechanism_table)
    gate_mechanism_supported = bool(
        learned_force_change
        < random_summary["mean_force_relative_change"]
        and ablation["learned_gate_worsening_rate"]
        <= ablation["always_on_worsening_rate"]
    )
    ablation["gate_mechanism_supported"] = gate_mechanism_supported
    atomic_json(REPORT / "ablation_summary.json", ablation)
    atomic_json(REPORT / "mechanism_summary.json", mechanism)
    refinement = read_json(REFINEMENT_SUMMARY)
    generation_state = configure_generation_module().load_generation_progress()
    generation_success = int(generation_state["success"])
    relaxation_success = initialize_relax()["success"]
    force_stat = stats[
        stats["metric"] == "pre_relax_max_force_ev_ang"
    ].iloc[0]
    effect_gate = bool(
        changes["pre_relax_max_force_relative"] <= -0.10
        and (
            float(force_stat["bootstrap_95_ci_high"]) < 0.0
            or float(force_stat["p_value"]) < 0.05
        )
    )
    quality_gate = bool(
        generation_success == len(SEEDS)
        and changes["structure_validity"] >= 0.0
        and changes["composition_validity"] >= -1.0 / len(SEEDS)
        and changes["ehull"] <= 0.002
        and changes["stable"] >= -1.0 / len(SEEDS)
        and changes["nus"] >= -1.0 / len(SEEDS)
        and changes["novel"] >= -1.0 / len(SEEDS)
        and changes["unique"] >= 0.0
        and changes["rmsd_relative"] <= 0.05
        and relaxation_success == len(METHODS) * len(SEEDS)
    )
    gate_off = q3_rows[~q3_rows["gate_applied"].astype(bool)]
    mechanism_safety = bool(
        q3_rows["atomic_numbers_unchanged"].astype(bool).all()
        and q3_rows["cell_unchanged"].astype(bool).all()
        and q3_rows["maximum_wrapped_displacement_angstrom"].max()
        <= 0.1000001
        and gate_off["exact_baseline_fallback"].astype(bool).all()
        and np.isfinite(
            q3_rows[
                [
                    "maximum_wrapped_displacement_angstrom",
                    "minimum_distance_angstrom",
                ]
            ].to_numpy(float)
        ).all()
        and q3_rows["minimum_distance_angstrom"].min() >= 0.5
    )
    final_go = bool(effect_gate and quality_gate and mechanism_safety)
    final_state = (
        "Q3_FROZEN_64_GO" if final_go else "Q3_FROZEN_64_NO_GO"
    )
    summary = {
        "schema_version": 1,
        "completed_at": now(),
        "final_state": final_state,
        "candidate": "Q3_E3_PCR",
        "base_commit": FROZEN_SOURCE_COMMIT,
        "evaluation_seeds": [SEEDS[0], SEEDS[-1]],
        "q3_checkpoint": str(Q3_CHECKPOINT),
        "q3_checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "q3_parameter_count": 129,
        "training_seed": 20260728,
        "counts": {
            "c0_generation": generation_success,
            "q3_refinement": len(q3_rows),
            "c0_relaxation": len(SEEDS),
            "q3_relaxation": len(SEEDS),
            "always_on_relaxation": len(SEEDS),
        },
        "baseline": baseline_mean,
        "q3": q3_mean,
        "always_on": always_mean,
        "changes": changes,
        "force_robustness": robustness,
        "refinement": refinement,
        "ablation": ablation,
        "mechanism": mechanism,
        "gates": {
            "effect_gate": effect_gate,
            "quality_gate": quality_gate,
            "mechanism_safety": mechanism_safety,
            "gate_mechanism_supported": gate_mechanism_supported,
            "Q3_FROZEN_64_GO": final_go,
            "Q3_FROZEN_64_NO_GO": not final_go,
        },
        "official_metrics": metrics,
        "formal_256_started": False,
        "a0_compatibility_started": False,
        "dft_started": False,
        "other_processes_terminated": False,
        "sigkill_used": False,
    }
    atomic_json(REPORT / "final_summary.json", summary)
    aggregate = pd.DataFrame(
        [
            {"method": "C0", **baseline_mean},
            {"method": "Q3_E3_PCR", **q3_mean},
            {"method": "ALWAYS_ON", **always_mean},
        ]
    )
    final_lines = [
        "# Q3 E3-PCR frozen 64-seed validation",
        "",
        f"- Final state: `{final_state}`",
        f"- Effect gate: `{effect_gate}`",
        f"- Quality gate: `{quality_gate}`",
        f"- Mechanism safety: `{mechanism_safety}`",
        f"- Gate mechanism supported: `{gate_mechanism_supported}`",
        "- Evaluation seeds: `32000–32063`",
        "- MatterGen sampling was run once per seed and shared by every method.",
        "- Atomic numbers and cells were unchanged.",
        "- Formal 256, A0 compatibility, and DFT were not started.",
        "",
        "## Aggregate metrics",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## Q3 changes versus C0",
        "",
        pd.DataFrame([changes]).to_markdown(index=False),
        "",
        "## Force robustness",
        "",
        "```json",
        json.dumps(robustness, indent=2, sort_keys=True),
        "```",
        "",
        "## Ablation",
        "",
        "```json",
        json.dumps(ablation, indent=2, sort_keys=True),
        "```",
        "",
        "## Limits",
        "",
        "- MatterSim-5M is the independent evaluator; no DFT was run.",
        "- This is an independent frozen 64-seed validation, not a 256-seed "
        "formal confirmation.",
        "",
    ]
    (REPORT / "final_report.md").write_text(
        "\n".join(final_lines), encoding="utf-8"
    )
    (REPORT / "statistics_report.md").write_text(
        "# Frozen64 paired statistics\n\n"
        + stats.to_markdown(index=False)
        + "\n\n"
        + "## Maximum-force robustness\n\n```json\n"
        + json.dumps(robustness, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    (REPORT / "ablation_report.md").write_text(
        "# Frozen64 gate ablation\n\n```json\n"
        + json.dumps(ablation, indent=2, sort_keys=True)
        + "\n```\n\n"
        + random_frame.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    (REPORT / "mechanism_report.md").write_text(
        "# Frozen64 mechanism analysis\n\n```json\n"
        + json.dumps(mechanism, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    write_master(
        "frozen64_go_no_go",
        "success",
        final_state=final_state,
        q3_frozen_64_go=final_go,
        final_report=str(REPORT / "final_report.md"),
    )
    print(json.dumps(summary["gates"], sort_keys=True), flush=True)


def pipeline() -> None:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another frozen64 pipeline is already running") from error
        initialize()
        commands = (
            [str(MATTERGEN_PYTHON), str(Path(__file__).resolve()), "generate"],
            [str(CHGNET_PYTHON), str(Path(__file__).resolve()), "refine"],
            [str(MATTERGEN_PYTHON), str(Path(__file__).resolve()), "relax"],
            [str(MATTERGEN_PYTHON), str(Path(__file__).resolve()), "analyze"],
        )
        for command in commands:
            subprocess.run(command, cwd=PROJECT, check=True)
        write_master(
            "stop_for_review",
            "success",
            final_state=read_json(REPORT / "final_summary.json")["final_state"],
            gpu_workers=0,
        )
        fcntl.flock(lock, fcntl.LOCK_UN)


def status() -> None:
    payload = (
        read_json(MASTER_PROGRESS)
        if MASTER_PROGRESS.is_file()
        else {"status": "not_started"}
    )
    if (PROGRESS / "generation_progress.json").is_file():
        payload["generation"] = read_json(
            PROGRESS / "generation_progress.json"
        ).get("success", 0)
    if RELAX_PROGRESS.is_file():
        payload["relaxation"] = read_json(RELAX_PROGRESS).get("success", 0)
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
        return relax_worker(args.gpu, args.slot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
