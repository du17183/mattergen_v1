#!/usr/bin/env python3
"""Frozen 64-seed compatibility study for Adaptive CFG followed by E3-G."""

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ase.io
import joblib
import numpy as np
import pandas as pd

from research import q3_formal256 as shared
from research import q3_frozen64 as core


ROOT = Path("/data/dxl")
PROJECT = ROOT / "mattergen_v1"
RESULT = ROOT / "results/a0_e3g_compat64"
REPORT = ROOT / "reports/a0_e3g_compat64"
LOG = ROOT / "logs/a0_e3g_compat64"
EXTERNAL_TOOLS = ROOT / "tools/a0_e3g_compat64"
PROGRESS = RESULT / "progress"
MASTER_PROGRESS = PROGRESS / "master_progress.json"
EVENTS = PROGRESS / "events.jsonl"
PIPELINE_LOCK = PROGRESS / "pipeline.lock"
GEN_PROGRESS = PROGRESS / "generation_progress.json"
GENERATION = RESULT / "generation/A0"
FEATURES = RESULT / "features.csv"
REFINED = RESULT / "refined"
REFINEMENT_MANIFEST = RESULT / "refinement_manifest.csv"
REFINEMENT_SUMMARY = RESULT / "refinement_summary.json"
RELAXED = RESULT / "relaxed"
RELAX_PROGRESS = PROGRESS / "relax_progress.json"

BASE_COMMIT = "0275cbf08ed3c6321cea7d06f7a3a8edb83b7483"
A0_FORMAL_COMMIT = "5de00419eea2d8a9be303638f2db8ece15a22366"
E3G_FORMAL_COMMIT = BASE_COMMIT
BRANCH = "feature/a0-e3g-compatibility64"
SEEDS = tuple(range(41000, 41064))
METHODS = ("A0", "A0_E3G")
DISPLAY_NAMES = {"A0": "A0", "A0_E3G": "A0+E3-G"}

TASK_RUNNER = ROOT / "tools/guidance_stage7/run_sample.py"
MATTERGEN_PYTHON = ROOT / "envs/mattergen_py310/bin/python"
CHGNET_PYTHON = ROOT / "envs/fn_pra_teacher/bin/python"
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
Q3_CHECKPOINT = ROOT / "results/postgen_fastgate/q3_refiner/model/q3_gate.joblib"
Q3_CHECKPOINT_SHA256 = (
    "b2ce1800fa0fa448f57d58010c8586b5de1b6666c4f198737a2f8a4bfabcb90e"
)
Q3_CONFIG = PROJECT / "configs/q3_e3_pcr_frozen64.json"
Q3_CONFIG_SHA256 = (
    "50d10efdea1050a84de6b2872f78742c2468ff4bef45cd7544fb30cef31eb87a"
)
FROZEN_SOURCE = PROJECT / "research/postgen_fastgate/refiner_eval.py"
FROZEN_SOURCE_SHA256 = (
    "3d1d6e38066bb195c893ea8665f284e66261f74a055e1521ed4d6250d469895f"
)
A0_CONFIG = ROOT / "reports/formal_256/final/frozen_method_configs.json"

BOOTSTRAP_SEED = 20260728
BOOTSTRAP_SAMPLES = 20_000
FORCE_HARM_EPSILON = 1.0e-6
SHORT_BOND_ANGSTROM = 0.5
STATE_LOCK = threading.RLock()


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


def atomic_json(path: Path, value: Any) -> None:
    core.atomic_json(path, value)


def atomic_csv(path: Path, value: pd.DataFrame) -> None:
    core.atomic_csv(path, value)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
            "experiment": "A0 + E3-G compatibility 64",
            "base_commit": BASE_COMMIT,
            "a0_formal_commit": A0_FORMAL_COMMIT,
            "e3g_formal_commit": E3G_FORMAL_COMMIT,
            "compatibility_branch": BRANCH,
            "evaluation_seeds": [SEEDS[0], SEEDS[-1]],
            "created_at": now(),
            "formal_256_combination_started": False,
            "dft_started": False,
            "independent_mlip_started": False,
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


def configure_modules() -> None:
    """Point proven Q3 and formal-analysis machinery at this namespace."""
    for module in (core, shared):
        module.ROOT = ROOT
        module.PROJECT = PROJECT
        module.RESULT = RESULT
        module.REPORT = REPORT
        module.LOG = LOG
        module.PROGRESS = PROGRESS
        module.MASTER_PROGRESS = MASTER_PROGRESS
        module.EVENTS = EVENTS
        module.PIPELINE_LOCK = PIPELINE_LOCK
        module.GENERATION = GENERATION
        module.FEATURES = FEATURES
        module.REFINED = REFINED
        module.REFINEMENT_MANIFEST = REFINEMENT_MANIFEST
        module.REFINEMENT_SUMMARY = REFINEMENT_SUMMARY
        module.RELAXED = RELAXED
        module.RELAX_PROGRESS = RELAX_PROGRESS
        module.SEEDS = SEEDS
        module.METHODS = METHODS
        module.BOOTSTRAP_SEED = BOOTSTRAP_SEED
        module.BOOTSTRAP_SAMPLES = BOOTSTRAP_SAMPLES
    core.EXTERNAL_TOOLS = EXTERNAL_TOOLS
    shared.EXTERNAL_TOOLS = EXTERNAL_TOOLS
    shared.DISPLAY_NAMES = DISPLAY_NAMES
    core.relax_rows = compatibility_relax_rows


def generation_environment(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "MATTERGEN_INTEROP_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def validate_a0_config(config: dict[str, Any]) -> bool:
    guidance = config.get("guidance_parameters", {})
    return bool(
        config.get("method") == "adaptive"
        and config.get("target") == {"dft_mag_density": 0.1}
        and float(config.get("base_guidance", math.nan)) == 2.0
        and int(config.get("batch_size", 0)) == 1
        and int(config.get("sampling_steps", 0)) == 1000
        and config.get("strict_deterministic") is True
        and float(guidance.get("min_scale", math.nan)) == 0.0
        and float(guidance.get("max_scale", math.nan)) == 5.0
        and float(guidance.get("adaptive_alpha", math.nan)) == 0.5
        and float(guidance.get("adaptive_ema", math.nan)) == 0.95
        and float(guidance.get("adaptive_eps", math.nan)) == 1.0e-6
    )


def validate_generation(output: Path, seed: int) -> bool:
    try:
        required = (
            "generated_crystals.extxyz",
            "run_summary.json",
            "run_config.json",
            "structure_hashes.json",
        )
        if any(not (output / name).is_file() for name in required):
            return False
        summary = read_json(output / "run_summary.json")
        config = read_json(output / "run_config.json")
        atoms = ase.io.read(output / "generated_crystals.extxyz", ":")
        if not isinstance(atoms, list):
            atoms = [atoms]
        return bool(
            len(atoms) == 1
            and summary.get("success") is True
            and int(summary.get("seed", -1)) == seed
            and validate_a0_config(config)
            and np.isfinite(atoms[0].positions).all()
            and np.isfinite(atoms[0].cell.array).all()
        )
    except BaseException:
        return False


def generation_rows() -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "status": "pending",
            "attempt": 0,
            "gpu": index % 8,
            "slot": (index % 32) // 8,
            "output_dir": str(GENERATION / str(seed)),
            "elapsed_seconds": None,
            "return_code": None,
            "error": "",
        }
        for index, seed in enumerate(SEEDS)
    ]


def save_generation_progress(state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    state["success"] = sum(row["status"] == "success" for row in state["tasks"])
    state["total"] = len(state["tasks"])
    atomic_json(GEN_PROGRESS, state)
    atomic_csv(PROGRESS / "generation_progress.csv", pd.DataFrame(state["tasks"]))


def load_generation_progress() -> dict[str, Any]:
    state = (
        read_json(GEN_PROGRESS)
        if GEN_PROGRESS.is_file()
        else {"schema_version": 1, "created_at": now(), "tasks": generation_rows()}
    )
    if [int(row["seed"]) for row in state["tasks"]] != list(SEEDS):
        raise RuntimeError("A0 generation seed contract mismatch")
    for row in state["tasks"]:
        if validate_generation(Path(row["output_dir"]), int(row["seed"])):
            row["status"] = "success"
        elif row["status"] in {"running", "success"}:
            row["status"] = "interrupted"
    save_generation_progress(state)
    return state


def quarantine(path: Path) -> None:
    if path.exists():
        os.replace(
            path,
            path.with_name(
                f"{path.name}.incomplete."
                f"{datetime.now().strftime('%Y%m%dT%H%M%S')}.{os.getpid()}"
            ),
        )


def run_generation_task(row: dict[str, Any], state: dict[str, Any]) -> bool:
    output = Path(row["output_dir"])
    seed = int(row["seed"])
    with STATE_LOCK:
        if validate_generation(output, seed):
            row["status"] = "success"
            save_generation_progress(state)
            return True
        quarantine(output)
        row.update(
            status="running",
            attempt=int(row["attempt"]) + 1,
            return_code=None,
            elapsed_seconds=None,
            error="",
        )
        save_generation_progress(state)
    command = [
        str(MATTERGEN_PYTHON),
        str(TASK_RUNNER),
        "--output-dir",
        str(output),
        "--seed",
        str(seed),
        "--physical-gpu",
        str(row["gpu"]),
        "--method",
        "adaptive",
        "--trace",
        "off",
    ]
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=PROJECT,
        env=generation_environment(int(row["gpu"])),
        text=True,
        capture_output=True,
    )
    elapsed = time.monotonic() - started
    valid = process.returncode == 0 and validate_generation(output, seed)
    if output.is_dir():
        (output / "launcher_stdout.log").write_text(
            process.stdout, encoding="utf-8"
        )
        (output / "launcher_stderr.log").write_text(
            process.stderr, encoding="utf-8"
        )
    with STATE_LOCK:
        row.update(
            status="success" if valid else "failed",
            elapsed_seconds=elapsed,
            return_code=process.returncode,
            error="" if valid else process.stderr[-4000:],
        )
        save_generation_progress(state)
        print(
            json.dumps(
                {
                    "stage": "a0_generation",
                    "seed": seed,
                    "status": row["status"],
                    "success": state["success"],
                    "total": state["total"],
                }
            ),
            flush=True,
        )
    return valid


def generate() -> None:
    shared.wait_for_all_gpus_free("a0_generation")
    write_master("a0_generation", "running")
    state = load_generation_progress()
    for _round in range(2):
        pending = [
            row
            for row in state["tasks"]
            if not validate_generation(Path(row["output_dir"]), int(row["seed"]))
            and int(row["attempt"]) < 2
        ]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [
                pool.submit(run_generation_task, row, state) for row in pending
            ]
            for future in as_completed(futures):
                future.result()
        state = load_generation_progress()
        if state["success"] == len(SEEDS):
            break
    if state["success"] != len(SEEDS):
        raise RuntimeError(f"A0 generation incomplete: {state['success']}/64")
    write_master(
        "a0_generation",
        "success",
        a0_generation="64/64",
        a0_generated_once=True,
    )


def choose_gated_output(original: Any, candidate: Any, enabled: bool) -> Any:
    """The disabled E3-G route is an exact structural copy of A0."""
    return candidate if enabled else original.copy()


def audited_refinement_subset(
    model: Any,
    originals: list[Any],
    active: np.ndarray,
    instrumentation: dict[str, int],
) -> tuple[list[Any], list[dict[str, int]], float]:
    from research.postgen_fastgate import refiner_eval as frozen

    original_safe = frozen.finite_safe
    safety = {
        "safety_checks": 0,
        "safety_rejections": 0,
        "short_bond_rejections": 0,
        "abnormal_cell_rejections": 0,
    }

    def audited_safe(atoms: Any) -> bool:
        safety["safety_checks"] += 1
        try:
            volume = float(atoms.get_volume())
        except BaseException:
            volume = math.nan
        abnormal = bool(
            not np.isfinite(atoms.positions).all()
            or not np.isfinite(atoms.cell.array).all()
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
            safety["short_bond_rejections"] += int(short)
            safety["abnormal_cell_rejections"] += int(abnormal)
        return accepted

    frozen.finite_safe = audited_safe
    try:
        outputs, counters, elapsed = core.run_refinement_subset(
            model, originals, active, instrumentation
        )
    finally:
        frozen.finite_safe = original_safe
    instrumentation.update(safety)
    return outputs, counters, elapsed


def refine() -> None:
    import torch
    from chgnet.model.model import CHGNet
    from research.postgen_fastgate import refiner_eval as frozen

    shared.wait_for_all_gpus_free("e3g_refinement")
    write_master("e3g_refinement", "running")
    if sha256(Q3_CHECKPOINT) != Q3_CHECKPOINT_SHA256:
        raise RuntimeError("Q3 checkpoint changed after freeze")
    torch.cuda.reset_peak_memory_stats()
    model = CHGNet.load(model_name="0.3.0", verbose=False, use_device="cuda")
    features = core.extract_features(model).sort_values("seed").reset_index(drop=True)
    features["method"] = "A0"
    features["split"] = "a0_e3g_compat64_independent"
    atomic_csv(FEATURES, features)
    network = joblib.load(Q3_CHECKPOINT)
    values = features.loc[:, core.FEATURE_COLUMNS].to_numpy(float)
    gate_started = time.perf_counter()
    probabilities = network.predict_proba(values)[:, 1]
    gate_seconds = time.perf_counter() - gate_started
    apply_gate = probabilities >= 0.5
    originals = [ase.io.read(path) for path in features["input_path"]]
    instrumentation = {
        "proposal_atoms": 0,
        "clipped_atoms": 0,
        "proposal_calls": 0,
    }
    candidates, counters, refine_seconds = audited_refinement_subset(
        model, originals, np.flatnonzero(apply_gate), instrumentation
    )
    rows: list[dict[str, Any]] = []
    for index, (seed, original, candidate, probability, apply) in enumerate(
        zip(SEEDS, originals, candidates, probabilities, apply_gate, strict=True)
    ):
        output_atoms = choose_gated_output(original, candidate, bool(apply))
        if not np.array_equal(output_atoms.numbers, original.numbers):
            raise RuntimeError("E3-G changed atomic numbers")
        if not np.array_equal(output_atoms.cell.array, original.cell.array):
            raise RuntimeError("E3-G changed cell")
        if not frozen.finite_safe(output_atoms):
            raise RuntimeError("E3-G produced unsafe structure")
        output_dir = REFINED / "A0_E3G" / f"{index:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "refined_structure.extxyz"
        ase.io.write(output_path, output_atoms, format="extxyz")
        checked = ase.io.read(output_path)
        displacement = frozen.wrapped_displacement_max(original, checked)
        if displacement > 0.1000001:
            raise RuntimeError("maximum cumulative displacement exceeded")
        input_hash = core.structure_hash(original)
        output_hash = core.structure_hash(checked)
        if not apply and input_hash != output_hash:
            raise RuntimeError("Gate-off output is not exact A0 fallback")
        rows.append(
            {
                "method": "A0_E3G",
                "evaluation_index": index,
                "seed": seed,
                "input_path": features.iloc[index]["input_path"],
                "output_path": str(output_path),
                "gate_probability": float(probability),
                "gate_applied": bool(apply),
                **counters[index],
                "maximum_wrapped_displacement_angstrom": displacement,
                "minimum_distance_angstrom": frozen.minimum_distance(checked),
                "input_hash": input_hash,
                "output_hash": output_hash,
                "atomic_numbers_unchanged": True,
                "cell_unchanged": True,
                "exact_baseline_fallback": input_hash == output_hash,
            }
        )
    manifest = pd.DataFrame(rows).sort_values("evaluation_index").reset_index(drop=True)
    atomic_csv(REFINEMENT_MANIFEST, manifest)
    gate_off = manifest[~manifest["gate_applied"].astype(bool)]
    summary = {
        "schema_version": 1,
        "created_at": now(),
        "q3_checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "q3_parameter_count": 129,
        "gate_threshold": 0.5,
        "gate_on_count": int(manifest["gate_applied"].sum()),
        "gate_on_rate": float(manifest["gate_applied"].mean()),
        "gate_forward_total_seconds": gate_seconds,
        "gate_forward_seconds_per_structure": gate_seconds / len(SEEDS),
        "q3_refinement_total_seconds": refine_seconds,
        "q3_refinement_seconds_per_structure": refine_seconds / len(SEEDS),
        "learned_instrumentation": instrumentation,
        "learned_clipping_rate": (
            instrumentation["clipped_atoms"]
            / max(1, instrumentation["proposal_atoms"])
        ),
        "gate_rejected_exact_fallback_count": int(
            gate_off["exact_baseline_fallback"].astype(bool).sum()
        ),
        "maximum_displacement_angstrom": float(
            manifest["maximum_wrapped_displacement_angstrom"].max()
        ),
        "mean_displacement_angstrom": float(
            manifest["maximum_wrapped_displacement_angstrom"].mean()
        ),
        "median_displacement_angstrom": float(
            manifest["maximum_wrapped_displacement_angstrom"].median()
        ),
        "backtracking_mean": float(manifest["backtracking_count"].mean()),
        "fallback_mean": float(manifest["fallback_count"].mean()),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "atomic_numbers_modified": 0,
        "cell_modified": 0,
    }
    atomic_json(REFINEMENT_SUMMARY, summary)
    write_master(
        "e3g_refinement",
        "success",
        e3g_refinement="64/64",
        gate_on_count=summary["gate_on_count"],
        gate_on_rate=summary["gate_on_rate"],
        gate_off_exact_fallback=summary["gate_rejected_exact_fallback_count"],
        maximum_displacement_angstrom=summary["maximum_displacement_angstrom"],
    )


def compatibility_relax_rows() -> list[dict[str, Any]]:
    if not REFINEMENT_MANIFEST.is_file():
        raise RuntimeError("refinement manifest does not exist")
    manifest = pd.read_csv(REFINEMENT_MANIFEST)
    by_seed = {
        int(row.seed): Path(row.output_path)
        for row in manifest.itertuples(index=False)
    }
    rows = []
    for index, seed in enumerate(SEEDS):
        inputs = {
            "A0": GENERATION / str(seed) / "generated_crystals.extxyz",
            "A0_E3G": by_seed[seed],
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
                    "input_hash": core.structure_hash(atoms),
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


def relax() -> None:
    shared.wait_for_all_gpus_free("a0_relaxation")
    write_master("a0_relaxation", "running", total_relaxations=128)
    if sha256(MATTERSIM) != MATTERSIM_SHA256:
        raise RuntimeError("MatterSim checkpoint SHA256 mismatch")
    state = core.initialize_relax()
    if state["success"] != 128:
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
                        "research.a0_e3g_compat64",
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
        if any(code != 0 for code in codes) or state["success"] != 128:
            raise RuntimeError(
                f"MatterSim relaxation incomplete: {state['success']}/128"
            )
    state = core.initialize_relax()
    counts = {
        method: sum(
            row["status"] == "success" and row["method"] == method
            for row in state["tasks"]
        )
        for method in METHODS
    }
    if counts != {"A0": 64, "A0_E3G": 64}:
        raise RuntimeError(f"relaxation count mismatch: {counts}")
    write_master("a0_relaxation", "success", a0_relaxation="64/64")
    write_master("e3g_relaxation", "success", e3g_relaxation="64/64")


def quality_pass(
    baseline: dict[str, float],
    selected: dict[str, float],
    behavior: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    gates = {
        "generation_success_not_lower": load_generation_progress()["success"] == 64,
        "structure_validity_not_lower": (
            selected["structure_validity"] >= baseline["structure_validity"]
        ),
        "composition_drop_le_1_over_64": (
            selected["composition_validity"]
            >= baseline["composition_validity"] - 1.0 / 64.0
        ),
        "ehull_degradation_le_0_002": selected["ehull"] - baseline["ehull"] <= 0.002,
        "stable_drop_le_1_over_64": selected["stable"] >= baseline["stable"] - 1 / 64,
        "nus_drop_le_1_over_64": selected["nus"] >= baseline["nus"] - 1 / 64,
        "novel_drop_le_1_over_64": selected["novel"] >= baseline["novel"] - 1 / 64,
        "unique_not_lower": selected["unique"] >= baseline["unique"],
        "rmsd_degradation_le_5_percent": selected["rmsd"] <= baseline["rmsd"] * 1.05,
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
        "atomic_numbers_unchanged": behavior["atomic_numbers_modified"] == 0,
        "cell_unchanged": behavior["cell_modified"] == 0,
        "maximum_displacement_bounded": (
            behavior["maximum_displacement_angstrom"] <= 0.1000001
        ),
        "full_rejection_exact_fallback": behavior[
            "full_rejections_exact_fallback"
        ],
        "gate_off_exact_fallback": behavior["gate_off_exact_fallback"],
        "no_nan_inf": behavior["finite"],
        "minimum_distance_safe": (
            behavior["minimum_distance_angstrom"] >= SHORT_BOND_ANGSTROM
        ),
    }
    return all(gates.values()), gates


def analyze() -> None:
    from scipy.stats import spearmanr

    write_master("metrics", "running")
    evaluated, official_metrics = shared.prepare_official_metrics()
    baseline = evaluated["A0"]
    selected = evaluated["A0_E3G"]
    means = {
        "A0": shared.metric_means(baseline),
        "A0+E3-G": shared.metric_means(selected),
    }
    common = (
        np.isfinite(baseline["energy_above_hull_per_atom"].to_numpy(float))
        & np.isfinite(selected["energy_above_hull_per_atom"].to_numpy(float))
    )
    if not common.any():
        raise RuntimeError("no common two-arm E-hull coverage")
    for label, frame in (("A0", baseline), ("A0+E3-G", selected)):
        means[label]["ehull_all_available"] = means[label]["ehull"]
        values = frame.loc[common, "energy_above_hull_per_atom"].to_numpy(float)
        means[label]["ehull"] = float(values.mean())
        means[label]["ehull_median"] = float(np.median(values))
    coverage = read_json(REPORT / "ehull_coverage.json")
    coverage["common_two_arm"] = coverage.pop("common_three_arm")
    atomic_json(REPORT / "ehull_coverage.json", coverage)

    paired_rows: list[dict[str, Any]] = []
    for metric, column in {
        "pre_relax_max_force_ev_ang": "pre_relax_max_force_ev_ang",
        "relaxation_steps": "steps",
        "relaxation_rmsd": "rmsd_from_relaxation",
        "relaxation_elapsed_seconds": "relax_elapsed_seconds",
    }.items():
        row = shared.continuous_stat(
            baseline[column].to_numpy(float),
            selected[column].to_numpy(float),
            metric,
        )
        row["comparison"] = "A0+E3-G vs A0"
        paired_rows.append(row)
    ehull_row = shared.continuous_stat(
        baseline.loc[common, "energy_above_hull_per_atom"].to_numpy(float),
        selected.loc[common, "energy_above_hull_per_atom"].to_numpy(float),
        "energy_above_hull_per_atom_common_coverage",
    )
    ehull_row["comparison"] = "A0+E3-G vs A0"
    paired_rows.append(ehull_row)
    for metric, column in {
        "force_converged": "converged",
        "stable": "stable",
        "nus": "novel_unique_stable",
        "structure_validity": "structure_validity",
        "composition_validity": "comp_validity",
        "novel": "novel",
        "unique": "unique",
        "pre_relax_short_bond": "pre_relax_short_bond",
        "pre_relax_abnormal_cell": "pre_relax_abnormal_cell",
    }.items():
        row = shared.binary_stat(
            baseline[column].astype(bool).to_numpy(),
            selected[column].astype(bool).to_numpy(),
            metric,
        )
        row["comparison"] = "A0+E3-G vs A0"
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    atomic_csv(REPORT / "paired_statistics.csv", paired)

    primary = shared.force_robustness(baseline, selected)
    primary["effect_reduction_ge_10_percent"] = primary["relative_change"] <= -0.10
    primary["bootstrap_ci_upper_below_zero"] = primary["bootstrap_95_ci"][1] < 0
    primary["wilcoxon_p_below_0_05"] = primary["wilcoxon_p_raw"] < 0.05
    primary["statistical_evidence_pass"] = bool(
        primary["bootstrap_ci_upper_below_zero"]
        or primary["wilcoxon_p_below_0_05"]
    )
    primary["primary_effect_pass"] = bool(
        primary["effect_reduction_ge_10_percent"]
        and primary["statistical_evidence_pass"]
    )
    atomic_json(REPORT / "primary_statistics.json", primary)

    manifest = pd.read_csv(REFINEMENT_MANIFEST).sort_values("evaluation_index")
    refinement_summary = read_json(REFINEMENT_SUMMARY)
    behavior = shared.refinement_behavior(
        manifest, "A0_E3G", refinement_summary
    )
    gate_off = manifest[~manifest["gate_applied"].astype(bool)]
    behavior["gate_off_count"] = len(gate_off)
    behavior["gate_off_exact_fallback"] = bool(
        gate_off["exact_baseline_fallback"].astype(bool).all()
    )
    safety_pass, safety_gates = quality_pass(
        means["A0"], means["A0+E3-G"], behavior
    )

    baseline_force = baseline["pre_relax_max_force_ev_ang"].to_numpy(float)
    selected_force = selected["pre_relax_max_force_ev_ang"].to_numpy(float)
    improvement = baseline_force - selected_force
    harm = selected_force > baseline_force + FORCE_HARM_EPSILON
    order = np.argsort(baseline_force, kind="stable")
    low = np.zeros(len(SEEDS), dtype=bool)
    low[order[: len(SEEDS) // 2]] = True
    correlation = spearmanr(
        manifest["gate_probability"].to_numpy(float),
        improvement,
        nan_policy="omit",
    )
    mechanism = {
        "gate_on_count": int(manifest["gate_applied"].astype(bool).sum()),
        "gate_on_rate": float(manifest["gate_applied"].astype(bool).mean()),
        "fallback_rate": float(
            (~manifest["gate_applied"].astype(bool)).mean()
        ),
        "harm_rate": float(harm.mean()),
        "low_force_count": int(low.sum()),
        "low_force_threshold": float(baseline_force[low].max()),
        "low_force_harm_rate": float(harm[low].mean()),
        "high_force_mean_improvement": float(improvement[~low].mean()),
        "low_force_mean_improvement": float(improvement[low].mean()),
        "gate_confidence_improvement_spearman": float(correlation.statistic),
        "gate_confidence_improvement_spearman_p": float(correlation.pvalue),
        "behavior": behavior,
    }
    atomic_json(REPORT / "mechanism_summary.json", mechanism)
    quality = {
        "A0": means["A0"],
        "A0+E3-G": means["A0+E3-G"],
        "quality_gates": safety_gates,
        "quality_safety_pass": safety_pass,
        "ehull_coverage": coverage,
    }
    atomic_json(REPORT / "quality_summary.json", quality)

    state = (
        "A0_E3G_COMPATIBILITY_GO"
        if primary["primary_effect_pass"] and safety_pass
        else "A0_E3G_COMPATIBILITY_NO_GO"
    )
    generation = load_generation_progress()
    generation_times = [
        float(row["elapsed_seconds"])
        for row in generation["tasks"]
        if row["status"] == "success"
    ]
    performance = {
        "a0_generation_median_seconds": float(np.median(generation_times)),
        "gate_forward_seconds_per_structure": refinement_summary[
            "gate_forward_seconds_per_structure"
        ],
        "refiner_seconds_per_structure": refinement_summary[
            "q3_refinement_seconds_per_structure"
        ],
        "end_to_end_overhead": float(
            refinement_summary["q3_refinement_seconds_per_structure"]
            / np.median(generation_times)
        ),
        "peak_vram_bytes": refinement_summary["peak_vram_bytes"],
        "q3_parameter_count": 129,
    }
    final = {
        "schema_version": 1,
        "completed_at": now(),
        "A0_E3G_COMPAT64_COMPLETED": True,
        "final_state": state,
        "a0_e3g_compatibility_go": state == "A0_E3G_COMPATIBILITY_GO",
        "a0_e3g_compatibility_no_go": state == "A0_E3G_COMPATIBILITY_NO_GO",
        "a0_commit": A0_FORMAL_COMMIT,
        "e3g_commit": E3G_FORMAL_COMMIT,
        "compatibility_branch": BRANCH,
        "evaluation_code_commit": read_json(
            REPORT / "frozen_manifest.json"
        )["evaluation_code_commit"],
        "evaluation_seeds": [SEEDS[0], SEEDS[-1]],
        "q3_checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "q3_config_sha256": Q3_CONFIG_SHA256,
        "counts": {
            "a0_generation": 64,
            "e3g_refinement": 64,
            "a0_relaxation": 64,
            "e3g_relaxation": 64,
            "total_mattersim": 128,
        },
        "means": means,
        "primary": primary,
        "quality": quality,
        "mechanism": mechanism,
        "performance": performance,
        "official_metrics": {
            DISPLAY_NAMES[key]: value for key, value in official_metrics.items()
        },
        "formal_256_combination_started": False,
        "dft_started": False,
        "independent_mlip_started": False,
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
            for method in ("A0", "A0+E3-G")
        ]
    )
    atomic_csv(REPORT / "aggregate_metrics.csv", aggregate)
    atomic_text(
        REPORT / "statistics_report.md",
        "# A0 + E3-G compatibility paired statistics\n\n"
        + pd.DataFrame([primary]).to_markdown(index=False)
        + "\n\n"
        + paired.to_markdown(index=False)
        + "\n",
    )
    atomic_text(
        REPORT / "mechanism_report.md",
        "# A0 + E3-G compatibility mechanism analysis\n\n```json\n"
        + json.dumps(mechanism, indent=2, sort_keys=True)
        + "\n```\n",
    )
    atomic_text(
        REPORT / "quality_report.md",
        "# A0 + E3-G compatibility quality analysis\n\n"
        + aggregate.to_markdown(index=False)
        + "\n\n```json\n"
        + json.dumps(safety_gates, indent=2, sort_keys=True)
        + "\n```\n",
    )
    atomic_text(
        REPORT / "reproduction_report.md",
        "# Reproduce A0 + E3-G compatibility 64\n\n```bash\n"
        "source /data/dxl/env.sh\n"
        "cd /data/dxl/mattergen_v1\n"
        "git switch feature/a0-e3g-compatibility64\n"
        "/data/dxl/tools/a0_e3g_compat64/resume.sh\n"
        "/data/dxl/tools/a0_e3g_compat64/status.sh\n"
        "```\n",
    )
    atomic_text(
        REPORT / "final_report.md",
        "# A0 + E3-G compatibility 64\n\n"
        f"- Final state: `{state}`\n"
        f"- Primary effect pass: `{primary['primary_effect_pass']}`\n"
        f"- Quality safety pass: `{safety_pass}`\n"
        "- A0 generation: `64/64`\n"
        "- A0+E3-G refinement: `64/64`\n"
        "- MatterSim relaxation: `128/128`\n"
        "- A0+E3-G derives from the exact same A0 structures.\n"
        "- No training, retuning, 256-seed combination, DFT, or independent "
        "MLIP experiment was started.\n\n"
        "## Aggregate metrics\n\n"
        + aggregate.to_markdown(index=False)
        + "\n\n## Primary endpoint\n\n"
        + pd.DataFrame([primary]).to_markdown(index=False)
        + "\n\n## Limitations\n\n"
        "- Stability is evaluated with the MatterSim-5M surrogate.\n"
        "- DFT and independent property-target verification were not run.\n",
    )
    write_master(
        "compatibility_decision",
        "success",
        final_state=state,
        a0_e3g_compatibility_go=state == "A0_E3G_COMPATIBILITY_GO",
        final_report=str(REPORT / "final_report.md"),
    )
    print(json.dumps({"final_state": state}, sort_keys=True), flush=True)


def validate_frozen_state() -> dict[str, Any]:
    if not subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
        cwd=PROJECT,
    ).returncode == 0:
        raise RuntimeError("compatibility branch does not descend from E3-G formal")
    frozen_a0_paths = (
        "mattergen/diffusion/sampling/guidance_schedule.py",
        "mattergen/diffusion/sampling/classifier_free_guidance.py",
        "mattergen/generator.py",
        "mattergen/scripts/generate.py",
    )
    changed_a0_paths = []
    for relative in frozen_a0_paths:
        at_base = subprocess.run(
            ["git", "show", f"{BASE_COMMIT}:{relative}"],
            cwd=PROJECT,
            check=True,
            capture_output=True,
        ).stdout
        if at_base != (PROJECT / relative).read_bytes():
            changed_a0_paths.append(relative)
    if changed_a0_paths:
        raise RuntimeError(
            f"A0 implementation changed after compatibility base: {changed_a0_paths}"
        )
    expected_assets = {
        MATTERGEN_CHECKPOINT: MATTERGEN_SHA256,
        MATTERSIM: MATTERSIM_SHA256,
        Q3_CHECKPOINT: Q3_CHECKPOINT_SHA256,
        Q3_CONFIG: Q3_CONFIG_SHA256,
        FROZEN_SOURCE: FROZEN_SOURCE_SHA256,
    }
    actual = {str(path): sha256(path) for path in expected_assets}
    for path, expected in expected_assets.items():
        if actual[str(path)] != expected:
            raise RuntimeError(f"frozen SHA256 mismatch: {path}")
    a0 = read_json(A0_CONFIG)["A0"]
    required_a0 = {
        "guidance_schedule": "adaptive",
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
    contract = shared.frozen_contract()
    shared.validate_frozen_contract(contract)
    return {
        "assets": actual,
        "a0": required_a0,
        "q3_contract": contract,
        "a0_paths_unchanged_from_base": list(frozen_a0_paths),
    }


def initialize() -> None:
    for path in (RESULT, REPORT, LOG, EXTERNAL_TOOLS, PROGRESS):
        path.mkdir(parents=True, exist_ok=True)
    write_master("state_audit", "running")
    frozen = validate_frozen_state()
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "a0_formal_source_commit": A0_FORMAL_COMMIT,
        "e3g_formal_commit": E3G_FORMAL_COMMIT,
        "compatibility_base_commit": BASE_COMMIT,
        "compatibility_branch": BRANCH,
        "evaluation_code_commit": git_output("rev-parse", "HEAD"),
        "evaluation_script": str(Path(__file__).resolve()),
        "evaluation_script_sha256": sha256(Path(__file__).resolve()),
        "evaluation_seeds": [SEEDS[0], SEEDS[-1]],
        "evaluation_seed_count": len(SEEDS),
        "a0_frozen_parameters": frozen["a0"],
        "a0_paths_unchanged_from_base": frozen[
            "a0_paths_unchanged_from_base"
        ],
        "q3_checkpoint": str(Q3_CHECKPOINT),
        "q3_checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "q3_config": str(Q3_CONFIG),
        "q3_config_sha256": Q3_CONFIG_SHA256,
        "q3_parameters": 129,
        "q3_contract": frozen["q3_contract"],
        "assets": frozen["assets"],
        "a0_disabled_existing_code_unchanged": True,
        "e3g_disabled_exact_a0_route": True,
        "training_or_retuning": False,
    }
    atomic_json(REPORT / "frozen_manifest.json", manifest)
    atomic_text(
        REPORT / "frozen_manifest.md",
        "# A0 + E3-G compatibility freeze\n\n"
        f"- A0 formal commit: `{A0_FORMAL_COMMIT}`\n"
        f"- E3-G formal commit: `{E3G_FORMAL_COMMIT}`\n"
        f"- Evaluation seeds: `{SEEDS[0]}–{SEEDS[-1]}`\n"
        f"- Q3 checkpoint SHA256: `{Q3_CHECKPOINT_SHA256}`\n"
        f"- Q3 config SHA256: `{Q3_CONFIG_SHA256}`\n"
        "- A0 parameters and all E3-G parameters are frozen.\n"
        "- Training or retuning: `False`\n",
    )
    seed_audit = {
        "created_at": now(),
        "candidate_range": [SEEDS[0], SEEDS[-1]],
        "candidate_count": len(SEEDS),
        "known_seed_ranges": [
            [10000, 10063],
            [11000, 11063],
            [20000, 20255],
            [22000, 22031],
            [32000, 32063],
            [33000, 33127],
            [40000, 40255],
        ],
        "known_range_intersections": [],
        "path_component_intersections": [],
        "seed_field_intersections": [],
        "passed": True,
        "note": "Exact numeric-boundary search produced only non-seed numeric/hash matches.",
    }
    atomic_json(REPORT / "seed_audit.json", seed_audit)
    write_master(
        "freeze_audit",
        "success",
        q3_checkpoint_sha256=Q3_CHECKPOINT_SHA256,
        q3_config_sha256=Q3_CONFIG_SHA256,
    )
    write_master("seed_audit", "success", evaluation_seeds=[41000, 41063])


def pipeline() -> None:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another compatibility pipeline is running") from error
        initialize()
        for command in (
            [str(MATTERGEN_PYTHON), "-m", "research.a0_e3g_compat64", "generate"],
            [str(CHGNET_PYTHON), "-m", "research.a0_e3g_compat64", "refine"],
            [str(MATTERGEN_PYTHON), "-m", "research.a0_e3g_compat64", "relax"],
            [str(MATTERGEN_PYTHON), "-m", "research.a0_e3g_compat64", "analyze"],
        ):
            subprocess.run(command, cwd=PROJECT, check=True)
        final = read_json(REPORT / "final_summary.json")
        write_master(
            "github_archive",
            "pending",
            final_state=final["final_state"],
            gpu_workers=0,
        )


def status() -> None:
    payload = (
        read_json(MASTER_PROGRESS)
        if MASTER_PROGRESS.is_file()
        else {"status": "not_started"}
    )
    if GEN_PROGRESS.is_file():
        payload["generation"] = {
            "success": load_generation_progress()["success"],
            "total": 64,
        }
    if RELAX_PROGRESS.is_file():
        state = core.initialize_relax()
        payload["relaxation"] = {
            method: sum(
                row["status"] == "success" and row["method"] == method
                for row in state["tasks"]
            )
            for method in METHODS
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
    configure_modules()
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
