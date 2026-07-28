#!/usr/bin/env python3
"""Contaminated 256-seed A0/E3-G diagnostic, split by training overlap.

This experiment is intentionally *not* an independent validation.  Seeds
20000--20063 were used to train the frozen Q3 gate.  The remaining 192 seeds
are reported separately and the mixed 256-seed result must never be presented
as a formal result.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pandas as pd

from research import a0_e3g_compat64 as compat
from research import q3_formal256 as shared
from research import q3_frozen64 as core


ROOT = Path("/data/dxl")
PROJECT = ROOT / "mattergen_v1"
RESULT = ROOT / "results/a0_e3g_leakage256"
REPORT = ROOT / "reports/a0_e3g_leakage256"
LOG = ROOT / "logs/a0_e3g_leakage256"
EXTERNAL_TOOLS = ROOT / "tools/a0_e3g_leakage256"
PROGRESS = RESULT / "progress"
MASTER_PROGRESS = PROGRESS / "master_progress.json"
EVENTS = PROGRESS / "events.jsonl"
PIPELINE_LOCK = PROGRESS / "pipeline.lock"

SOURCE_GENERATION = ROOT / "results/formal_256/generation/A0"
SOURCE_A0_RELAXED = ROOT / "results/formal_256/relaxed/A0"
SOURCE_A0_METRICS = (
    ROOT / "reports/formal_256/A0/official_metrics_per_structure.csv"
)
GENERATION = SOURCE_GENERATION
FEATURES = RESULT / "features.csv"
REFINED = RESULT / "refined"
REFINEMENT_MANIFEST = RESULT / "refinement_manifest.csv"
REFINEMENT_SUMMARY = RESULT / "refinement_summary.json"
RELAXED = RESULT / "relaxed"
RELAX_PROGRESS = PROGRESS / "e3g_relax_progress.json"
PROBE_PROGRESS = PROGRESS / "a0_probe_progress.json"
PROBE_OUTPUT = RESULT / "a0_force_probe"

SEEDS = tuple(range(20000, 20256))
TRAIN_SEEDS = tuple(range(20000, 20064))
HELDOUT_SEEDS = tuple(range(20064, 20256))
METHODS = ("A0_E3G",)
DISPLAY_NAMES = {"A0_E3G": "A0+E3-G"}
BRANCH = "experiment/a0-e3g-leakage-diagnostic256"
SESSION = "mattergen_a0_e3g_leakage256"

MATTERGEN_PYTHON = ROOT / "envs/mattergen_py310/bin/python"
CHGNET_PYTHON = ROOT / "envs/fn_pra_teacher/bin/python"
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
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_SAMPLES = 20_000


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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
            "experiment": "A0 E3-G contaminated 256 leakage diagnostic",
            "branch": BRANCH,
            "seed_range": [SEEDS[0], SEEDS[-1]],
            "training_overlap": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
            "heldout_range": [HELDOUT_SEEDS[0], HELDOUT_SEEDS[-1]],
            "independent_validation": False,
            "formal_go_no_go_allowed": False,
            "training_or_retuning": False,
            "created_at": now(),
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


def e3g_relax_rows() -> list[dict[str, Any]]:
    if not REFINEMENT_MANIFEST.is_file():
        raise RuntimeError("refinement manifest does not exist")
    manifest = pd.read_csv(REFINEMENT_MANIFEST).sort_values("evaluation_index")
    if manifest["seed"].astype(int).tolist() != list(SEEDS):
        raise RuntimeError("refinement manifest seed contract mismatch")
    rows: list[dict[str, Any]] = []
    for record in manifest.itertuples(index=False):
        atoms = ase.io.read(record.output_path)
        index = int(record.evaluation_index)
        seed = int(record.seed)
        rows.append(
            {
                "task_id": f"A0_E3G_{seed}",
                "method": "A0_E3G",
                "evaluation_index": index,
                "seed": seed,
                "input_path": str(record.output_path),
                "input_hash": core.structure_hash(atoms),
                "output_dir": str(RELAXED / "A0_E3G" / f"{index:03d}"),
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


def configure_modules() -> None:
    values = {
        "ROOT": ROOT,
        "PROJECT": PROJECT,
        "RESULT": RESULT,
        "REPORT": REPORT,
        "LOG": LOG,
        "EXTERNAL_TOOLS": EXTERNAL_TOOLS,
        "PROGRESS": PROGRESS,
        "MASTER_PROGRESS": MASTER_PROGRESS,
        "EVENTS": EVENTS,
        "PIPELINE_LOCK": PIPELINE_LOCK,
        "GENERATION": GENERATION,
        "FEATURES": FEATURES,
        "REFINED": REFINED,
        "REFINEMENT_MANIFEST": REFINEMENT_MANIFEST,
        "REFINEMENT_SUMMARY": REFINEMENT_SUMMARY,
        "RELAXED": RELAXED,
        "RELAX_PROGRESS": RELAX_PROGRESS,
        "SEEDS": SEEDS,
        "METHODS": METHODS,
        "DISPLAY_NAMES": DISPLAY_NAMES,
        "BRANCH": BRANCH,
        "MATTERGEN_PYTHON": MATTERGEN_PYTHON,
        "CHGNET_PYTHON": CHGNET_PYTHON,
        "MATTERSIM": MATTERSIM,
        "MATTERSIM_SHA256": MATTERSIM_SHA256,
        "Q3_CHECKPOINT": Q3_CHECKPOINT,
        "Q3_CHECKPOINT_SHA256": Q3_CHECKPOINT_SHA256,
        "Q3_CONFIG": Q3_CONFIG,
        "Q3_CONFIG_SHA256": Q3_CONFIG_SHA256,
        "FROZEN_SOURCE": FROZEN_SOURCE,
        "FROZEN_SOURCE_SHA256": FROZEN_SOURCE_SHA256,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "BOOTSTRAP_SAMPLES": BOOTSTRAP_SAMPLES,
    }
    for name, value in values.items():
        setattr(compat, name, value)
    compat.write_master = write_master
    compat.event = event
    compat.configure_modules()
    core.relax_rows = e3g_relax_rows
    core.write_master = write_master
    shared.write_master = write_master


def validate_source() -> dict[str, Any]:
    if core.sha256(MATTERSIM) != MATTERSIM_SHA256:
        raise RuntimeError("MatterSim checkpoint SHA256 mismatch")
    if core.sha256(Q3_CHECKPOINT) != Q3_CHECKPOINT_SHA256:
        raise RuntimeError("Q3 checkpoint SHA256 mismatch")
    if core.sha256(Q3_CONFIG) != Q3_CONFIG_SHA256:
        raise RuntimeError("Q3 config SHA256 mismatch")
    if core.sha256(FROZEN_SOURCE) != FROZEN_SOURCE_SHA256:
        raise RuntimeError("frozen E3-PCR source SHA256 mismatch")
    source_rows = pd.read_csv(SOURCE_A0_METRICS)
    if sorted(source_rows["seed"].astype(int)) != list(SEEDS):
        raise RuntimeError("A0 official metrics do not cover exact 256 seeds")
    missing_generation = [
        seed
        for seed in SEEDS
        if not (
            SOURCE_GENERATION / str(seed) / "generated_crystals.extxyz"
        ).is_file()
    ]
    missing_relaxation = [
        seed
        for seed in SEEDS
        if not (
            SOURCE_A0_RELAXED / str(seed) / "relax_summary.json"
        ).is_file()
        or not (
            SOURCE_A0_RELAXED / str(seed) / "relaxed_structure.extxyz"
        ).is_file()
    ]
    if missing_generation or missing_relaxation:
        raise RuntimeError(
            f"A0 source incomplete generation={missing_generation} "
            f"relaxation={missing_relaxation}"
        )
    return {
        "a0_generation_reused": 256,
        "a0_relaxation_reused": 256,
        "a0_official_metrics_reused": 256,
        "q3_training_overlap_count": len(TRAIN_SEEDS),
        "heldout_count": len(HELDOUT_SEEDS),
    }


def initialize() -> None:
    for path in (RESULT, REPORT, LOG, EXTERNAL_TOOLS, PROGRESS):
        path.mkdir(parents=True, exist_ok=True)
    write_master("source_audit", "running")
    source = validate_source()
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "purpose": "training-test leakage effect diagnostic",
        "independent_validation": False,
        "formal_claim_allowed": False,
        "mixed_256_contaminated": True,
        "training_overlap_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
        "heldout_seeds": [HELDOUT_SEEDS[0], HELDOUT_SEEDS[-1]],
        "a0_generation_source": str(SOURCE_GENERATION),
        "a0_relaxation_source": str(SOURCE_A0_RELAXED),
        "a0_metrics_source": str(SOURCE_A0_METRICS),
        "q3_checkpoint": str(Q3_CHECKPOINT),
        "q3_checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "q3_config": str(Q3_CONFIG),
        "q3_config_sha256": Q3_CONFIG_SHA256,
        "frozen_source_sha256": FROZEN_SOURCE_SHA256,
        "gate_threshold": 0.5,
        "refinement_steps": 5,
        "position_eta": 0.01,
        "per_step_radius_angstrom": 0.02,
        "maximum_cumulative_displacement_angstrom": 0.10,
        "backtrack_max": 3,
        "training_or_retuning": False,
        "a0_generation_rerun": False,
        "a0_relaxation_rerun": False,
        "matterSim_checkpoint_sha256": MATTERSIM_SHA256,
        "stability_source": "MatterSim-5M surrogate",
        "dft_verified": False,
        "property_target_verified": False,
        **source,
    }
    atomic_json(REPORT / "diagnostic_manifest.json", manifest)
    atomic_text(
        REPORT / "protocol.md",
        "# A0 + E3-G training-test leakage diagnostic\n\n"
        "- Seeds `20000–20063`: overlap the frozen Q3 gate training set.\n"
        "- Seeds `20064–20255`: not used for Q3 gate training.\n"
        "- The overall 256 result is intentionally contaminated and is not a "
        "formal validation.\n"
        "- A0 generation and completed A0 relaxation are reused exactly.\n"
        "- No training, retuning, threshold change, or refinement change is "
        "permitted.\n",
    )
    write_master("source_audit", "success", **source)


def refinement_complete() -> bool:
    if not REFINEMENT_MANIFEST.is_file() or not REFINEMENT_SUMMARY.is_file():
        return False
    try:
        frame = pd.read_csv(REFINEMENT_MANIFEST)
        return bool(
            len(frame) == len(SEEDS)
            and frame["seed"].astype(int).tolist() == list(SEEDS)
            and all(Path(path).is_file() for path in frame["output_path"])
        )
    except BaseException:
        return False


def refine() -> None:
    if refinement_complete():
        write_master("e3g_refinement", "success", e3g_refinement="256/256")
        return
    compat.refine()
    frame = pd.read_csv(FEATURES)
    frame["split"] = np.where(
        frame["seed"].astype(int).isin(TRAIN_SEEDS),
        "q3_train_overlap",
        "q3_train_heldout",
    )
    atomic_csv(FEATURES, frame)
    if not refinement_complete():
        raise RuntimeError("E3-G refinement did not produce 256 valid outputs")
    write_master("e3g_refinement", "success", e3g_refinement="256/256")


def launch_workers(command: str, worker_count_per_gpu: int = 2) -> list[int]:
    gpus = compat.wait_for_free_gpus(command)
    worker_logs = LOG / f"{command}_workers"
    worker_logs.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[Any]] = []
    handles = []
    for slot in range(worker_count_per_gpu):
        for gpu in gpus:
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
                    "research.a0_e3g_leakage256",
                    f"{command}-worker",
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
    return codes


def relax() -> None:
    write_master("e3g_mattersim_relaxation", "running")
    state = core.initialize_relax()
    if state["success"] != len(SEEDS):
        codes = launch_workers("relax")
        state = core.initialize_relax()
        if any(codes) or state["success"] != len(SEEDS):
            raise RuntimeError(
                f"E3-G MatterSim relaxation incomplete: "
                f"{state['success']}/{len(SEEDS)}, codes={codes}"
            )
    write_master(
        "e3g_mattersim_relaxation",
        "success",
        e3g_relaxation=f"{state['success']}/{len(SEEDS)}",
        a0_relaxation_reused="256/256",
    )


def probe_rows() -> list[dict[str, Any]]:
    manifest = pd.read_csv(REFINEMENT_MANIFEST).sort_values("evaluation_index")
    active = manifest[manifest["gate_applied"].astype(bool)]
    rows: list[dict[str, Any]] = []
    for record in active.itertuples(index=False):
        index = int(record.evaluation_index)
        seed = int(record.seed)
        input_path = SOURCE_GENERATION / str(seed) / "generated_crystals.extxyz"
        atoms = ase.io.read(input_path)
        rows.append(
            {
                "task_id": f"A0_FORCE_{seed}",
                "evaluation_index": index,
                "seed": seed,
                "input_path": str(input_path),
                "input_hash": core.structure_hash(atoms),
                "output_dir": str(PROBE_OUTPUT / f"{index:03d}"),
                "status": "pending",
                "attempt": 0,
                "gpu": None,
                "slot": None,
                "pre_relax_max_force_ev_ang": None,
                "error": "",
            }
        )
    return rows


def validate_probe(row: dict[str, Any]) -> bool:
    try:
        summary = read_json(Path(row["output_dir"]) / "probe_summary.json")
        return bool(
            summary["success"]
            and summary["task_id"] == row["task_id"]
            and summary["input_hash"] == row["input_hash"]
            and summary["checkpoint_sha256"] == MATTERSIM_SHA256
            and np.isfinite(summary["pre_relax_max_force_ev_ang"])
        )
    except BaseException:
        return False


def save_probe(state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    state["success"] = sum(row["status"] == "success" for row in state["tasks"])
    state["total"] = len(state["tasks"])
    atomic_json(PROBE_PROGRESS, state)
    atomic_csv(PROGRESS / "a0_probe_progress.csv", pd.DataFrame(state["tasks"]))


def locked_probe(operation: Any) -> Any:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    with (PROGRESS / "a0_probe_progress.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = (
            read_json(PROBE_PROGRESS)
            if PROBE_PROGRESS.is_file()
            else {"schema_version": 1, "created_at": now(), "tasks": probe_rows()}
        )
        expected = probe_rows()
        if [row["task_id"] for row in state["tasks"]] != [
            row["task_id"] for row in expected
        ]:
            raise RuntimeError("A0 force-probe contract mismatch")
        result = operation(state)
        save_probe(state)
        fcntl.flock(lock, fcntl.LOCK_UN)
        return result


def initialize_probe() -> dict[str, Any]:
    def operation(state: dict[str, Any]) -> dict[str, Any]:
        expected = probe_rows()
        for row, contract in zip(state["tasks"], expected, strict=True):
            row["input_path"] = contract["input_path"]
            row["input_hash"] = contract["input_hash"]
            row["output_dir"] = contract["output_dir"]
            if validate_probe(row):
                row["status"] = "success"
                summary = read_json(
                    Path(row["output_dir"]) / "probe_summary.json"
                )
                row["pre_relax_max_force_ev_ang"] = summary[
                    "pre_relax_max_force_ev_ang"
                ]
            elif row["status"] in {"running", "success"}:
                row["status"] = "interrupted"
        return state

    return locked_probe(operation)


def claim_probe(gpu: int, slot: int) -> dict[str, Any] | None:
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

    return locked_probe(operation)


def finish_probe(task_id: str, success: bool, **values: Any) -> None:
    def operation(state: dict[str, Any]) -> None:
        row = next(item for item in state["tasks"] if item["task_id"] == task_id)
        row.update(status="success" if success else "failed", **values)

    locked_probe(operation)


def probe_worker(gpu: int, slot: int) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from mattersim.forcefield import MatterSimCalculator

    sys.path.insert(0, str(core.RELAX_COMMON))
    from relax_common import load_potential

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    potential = load_potential("cuda")
    stop = False

    def handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    while not stop:
        row = claim_probe(gpu, slot)
        if row is None:
            break
        output = Path(row["output_dir"])
        try:
            output.mkdir(parents=True, exist_ok=True)
            atoms = ase.io.read(row["input_path"])
            if core.structure_hash(atoms) != row["input_hash"]:
                raise RuntimeError("A0 force-probe input hash mismatch")
            probe = atoms.copy()
            probe.calc = MatterSimCalculator.from_potential(
                potential=potential, device="cuda"
            )
            started = time.monotonic()
            forces = np.asarray(probe.get_forces(), dtype=float)
            elapsed = time.monotonic() - started
            maximum_force = float(np.linalg.norm(forces, axis=1).max())
            summary = {
                "success": True,
                "task_id": row["task_id"],
                "evaluation_index": row["evaluation_index"],
                "seed": row["seed"],
                "gpu": gpu,
                "slot": slot,
                "input_path": row["input_path"],
                "input_hash": row["input_hash"],
                "checkpoint": str(MATTERSIM),
                "checkpoint_sha256": MATTERSIM_SHA256,
                "elapsed_seconds": elapsed,
                "pre_relax_max_force_ev_ang": maximum_force,
            }
            atomic_json(output / "probe_summary.json", summary)
            finish_probe(
                row["task_id"],
                True,
                pre_relax_max_force_ev_ang=maximum_force,
                elapsed_seconds=elapsed,
                error="",
            )
        except BaseException:
            error = traceback.format_exc()
            output.mkdir(parents=True, exist_ok=True)
            atomic_text(output / "error.log", error)
            finish_probe(row["task_id"], False, error=error[-4000:])
    return 0


def probe() -> None:
    state = initialize_probe()
    write_master(
        "a0_missing_force_probe",
        "running",
        required=state["total"],
        reused_gate_off=len(SEEDS) - state["total"],
    )
    if state["success"] != state["total"]:
        codes = launch_workers("probe")
        state = initialize_probe()
        if any(codes) or state["success"] != state["total"]:
            raise RuntimeError(
                f"A0 force probe incomplete: {state['success']}/"
                f"{state['total']}, codes={codes}"
            )
    write_master(
        "a0_missing_force_probe",
        "success",
        probed=state["success"],
        reused_gate_off=len(SEEDS) - state["total"],
    )


def add_a0_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("seed").reset_index(drop=True).copy()
    minimum: list[float] = []
    abnormal: list[bool] = []
    for seed in SEEDS:
        atoms = ase.io.read(
            SOURCE_GENERATION / str(seed) / "generated_crystals.extxyz"
        )
        minimum.append(shared.minimum_distance(atoms))
        abnormal.append(shared.abnormal_cell(atoms))
    result["pre_relax_minimum_distance_angstrom"] = minimum
    result["pre_relax_short_bond"] = np.asarray(minimum) < 0.5
    result["pre_relax_abnormal_cell"] = abnormal
    return result


def attach_force_values(
    baseline: pd.DataFrame, selected: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(REFINEMENT_MANIFEST).sort_values("evaluation_index")
    state = initialize_probe()
    probes = {
        int(row["seed"]): float(row["pre_relax_max_force_ev_ang"])
        for row in state["tasks"]
        if row["status"] == "success"
    }
    selected = selected.sort_values("seed").reset_index(drop=True).copy()
    selected["formal_seed"] = list(SEEDS)
    selected_force = selected["pre_relax_max_force_ev_ang"].to_numpy(float)
    baseline_force = selected_force.copy()
    for index, record in enumerate(manifest.itertuples(index=False)):
        if bool(record.gate_applied):
            baseline_force[index] = probes[int(record.seed)]
        elif not bool(record.exact_baseline_fallback):
            raise RuntimeError("gate-off row is not exact A0 fallback")
    baseline = baseline.sort_values("seed").reset_index(drop=True).copy()
    baseline["pre_relax_max_force_ev_ang"] = baseline_force
    baseline["formal_seed"] = list(SEEDS)
    atomic_csv(
        REPORT / "a0_force_reconstruction.csv",
        pd.DataFrame(
            {
                "seed": SEEDS,
                "gate_applied": manifest["gate_applied"].astype(bool),
                "a0_force": baseline_force,
                "e3g_force": selected_force,
                "a0_force_source": np.where(
                    manifest["gate_applied"].astype(bool),
                    "new_mattersim_single_point_probe",
                    "exact_gate_off_e3g_reuse",
                ),
            }
        ),
    )
    return baseline, selected


def cohort_statistics(
    name: str,
    indexes: np.ndarray,
    baseline: pd.DataFrame,
    selected: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    left = baseline.iloc[indexes].reset_index(drop=True)
    right = selected.iloc[indexes].reset_index(drop=True)
    seeds = tuple(SEEDS[index] for index in indexes)
    old_seeds = shared.SEEDS
    shared.SEEDS = seeds
    try:
        primary = shared.force_robustness(left, right)
    finally:
        shared.SEEDS = old_seeds
    paired: list[dict[str, Any]] = []
    continuous = {
        "pre_relax_max_force_ev_ang": "pre_relax_max_force_ev_ang",
        "relaxation_steps": "steps",
        "relaxation_rmsd": "rmsd_from_relaxation",
    }
    for metric, column in continuous.items():
        row = shared.continuous_stat(
            left[column].to_numpy(float),
            right[column].to_numpy(float),
            metric,
        )
        paired.append(row)
    common = (
        np.isfinite(left["energy_above_hull_per_atom"].to_numpy(float))
        & np.isfinite(right["energy_above_hull_per_atom"].to_numpy(float))
    )
    if common.any():
        paired.append(
            shared.continuous_stat(
                left.loc[common, "energy_above_hull_per_atom"].to_numpy(float),
                right.loc[common, "energy_above_hull_per_atom"].to_numpy(float),
                "energy_above_hull_per_atom_common_coverage",
            )
        )
    for metric, column in {
        "force_converged": "converged",
        "stable": "stable",
        "nus": "novel_unique_stable",
        "composition_validity": "comp_validity",
        "structure_validity": "structure_validity",
        "novel": "novel",
        "unique": "unique",
    }.items():
        paired.append(
            shared.binary_stat(
                left[column].astype(bool).to_numpy(),
                right[column].astype(bool).to_numpy(),
                metric,
            )
        )
    cohort_dir = REPORT / "cohorts" / name
    atomic_csv(cohort_dir / "paired_statistics.csv", pd.DataFrame(paired))
    left_means = shared.metric_means(left)
    right_means = shared.metric_means(right)
    subset = manifest.iloc[indexes]
    force_difference = (
        right["pre_relax_max_force_ev_ang"].to_numpy(float)
        - left["pre_relax_max_force_ev_ang"].to_numpy(float)
    )
    result = {
        "cohort": name,
        "seed_count": len(indexes),
        "seed_range": [seeds[0], seeds[-1]],
        "overlaps_q3_training": name != "heldout_192",
        "a0": left_means,
        "a0_e3g": right_means,
        "primary_force": primary,
        "gate_on_count": int(subset["gate_applied"].astype(bool).sum()),
        "gate_on_rate": float(subset["gate_applied"].astype(bool).mean()),
        "force_harm_rate": float((force_difference > 1.0e-6).mean()),
        "force_improvement_rate": float((force_difference < -1.0e-6).mean()),
        "ehull_common_coverage": int(common.sum()),
    }
    atomic_json(cohort_dir / "summary.json", result)
    return result


def bootstrap_gap(
    train_difference: np.ndarray, heldout_difference: np.ndarray
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    train_index = rng.integers(
        0,
        len(train_difference),
        size=(BOOTSTRAP_SAMPLES, len(train_difference)),
    )
    heldout_index = rng.integers(
        0,
        len(heldout_difference),
        size=(BOOTSTRAP_SAMPLES, len(heldout_difference)),
    )
    gap = (
        train_difference[train_index].mean(axis=1)
        - heldout_difference[heldout_index].mean(axis=1)
    )
    low, high = np.quantile(gap, [0.025, 0.975])
    return float(low), float(high)


def analyze() -> None:
    write_master("official_metrics", "running")
    evaluated, official = shared.prepare_official_metrics()
    selected = evaluated["A0_E3G"]
    baseline = add_a0_geometry(pd.read_csv(SOURCE_A0_METRICS))
    baseline, selected = attach_force_values(baseline, selected)
    manifest = pd.read_csv(REFINEMENT_MANIFEST).sort_values("evaluation_index")
    if len(selected) != 256 or len(baseline) != 256 or len(manifest) != 256:
        raise RuntimeError("analysis input count mismatch")
    atomic_csv(REPORT / "A0_reused_metrics_augmented.csv", baseline)
    atomic_csv(REPORT / "A0_E3G_metrics_augmented.csv", selected)

    cohorts = {
        "train_overlap_64": np.arange(0, 64),
        "heldout_192": np.arange(64, 256),
        "mixed_256": np.arange(0, 256),
    }
    summaries = {
        name: cohort_statistics(
            name, indexes, baseline, selected, manifest
        )
        for name, indexes in cohorts.items()
    }
    force_difference = (
        selected["pre_relax_max_force_ev_ang"].to_numpy(float)
        - baseline["pre_relax_max_force_ev_ang"].to_numpy(float)
    )
    train_difference = force_difference[:64]
    heldout_difference = force_difference[64:]
    gap = float(train_difference.mean() - heldout_difference.mean())
    gap_ci = bootstrap_gap(train_difference, heldout_difference)
    train_relative = summaries["train_overlap_64"]["primary_force"][
        "relative_change"
    ]
    heldout_relative = summaries["heldout_192"]["primary_force"][
        "relative_change"
    ]
    relative_gap = float(train_relative - heldout_relative)
    if gap_ci[1] < 0:
        leakage_state = "LEAKAGE_INFLATION_DETECTED"
    elif gap_ci[0] > 0:
        leakage_state = "HELDOUT_EFFECT_STRONGER"
    else:
        leakage_state = "NO_CLEAR_LEAKAGE_INFLATION"
    leakage = {
        "state": leakage_state,
        "train_mean_force_difference": float(train_difference.mean()),
        "heldout_mean_force_difference": float(heldout_difference.mean()),
        "train_minus_heldout_effect_gap": gap,
        "bootstrap_95_ci": list(gap_ci),
        "train_relative_force_change": train_relative,
        "heldout_relative_force_change": heldout_relative,
        "train_minus_heldout_relative_change_gap": relative_gap,
        "interpretation": (
            "Negative effect gap means the apparent force reduction is stronger "
            "on gate-training seeds."
        ),
    }
    atomic_json(REPORT / "leakage_effect.json", leakage)
    summary = {
        "schema_version": 1,
        "completed_at": now(),
        "LEAKAGE_DIAGNOSTIC_COMPLETED": True,
        "final_state": leakage_state,
        "independent_validation": False,
        "formal_go_no_go_allowed": False,
        "mixed_256_contaminated": True,
        "training_overlap_count": 64,
        "heldout_count": 192,
        "a0_generation_reused": 256,
        "a0_relaxation_reused": 256,
        "a0_new_full_relaxations": 0,
        "e3g_new_refinements": 256,
        "e3g_new_mattersim_relaxations": 256,
        "a0_new_single_point_force_probes": initialize_probe()["success"],
        "q3_checkpoint_sha256": Q3_CHECKPOINT_SHA256,
        "cohorts": summaries,
        "leakage_effect": leakage,
        "official_e3g_metrics": official,
        "stability_source": "MatterSim-5M surrogate",
        "dft_verified": False,
        "property_target_verified": False,
    }
    atomic_json(REPORT / "final_summary.json", summary)
    train = summaries["train_overlap_64"]
    heldout = summaries["heldout_192"]
    mixed = summaries["mixed_256"]
    report = f"""# A0 + E3-G training-test leakage diagnostic

## Status

`{leakage_state}`

This is an intentionally contaminated diagnostic. Seeds `20000–20063` were
used to train the frozen Q3 gate. The mixed 256 result is not an independent
validation and must not be used as a formal thesis result.

## Primary pre-relaxation force effect

| Cohort | N | Gate rate | A0 mean | A0+E3-G mean | Relative change | Wins/Ties/Losses |
|---|---:|---:|---:|---:|---:|---:|
| Train overlap | 64 | {train['gate_on_rate']:.3%} | {train['primary_force']['baseline_mean']:.6f} | {train['primary_force']['selected_mean']:.6f} | {train['primary_force']['relative_change']:.3%} | {train['primary_force']['wins']}/{train['primary_force']['ties']}/{train['primary_force']['losses']} |
| Held out | 192 | {heldout['gate_on_rate']:.3%} | {heldout['primary_force']['baseline_mean']:.6f} | {heldout['primary_force']['selected_mean']:.6f} | {heldout['primary_force']['relative_change']:.3%} | {heldout['primary_force']['wins']}/{heldout['primary_force']['ties']}/{heldout['primary_force']['losses']} |
| Mixed (contaminated) | 256 | {mixed['gate_on_rate']:.3%} | {mixed['primary_force']['baseline_mean']:.6f} | {mixed['primary_force']['selected_mean']:.6f} | {mixed['primary_force']['relative_change']:.3%} | {mixed['primary_force']['wins']}/{mixed['primary_force']['ties']}/{mixed['primary_force']['losses']} |

Train-minus-heldout mean force-effect gap: `{gap:.6f}` eV/Å,
bootstrap 95% CI `{gap_ci[0]:.6f}, {gap_ci[1]:.6f}`.

## Interpretation

- A negative gap means the measured force improvement is stronger on samples
  seen by the gate during training.
- The held-out 192 cohort is the only scientifically informative cohort in
  this run, although it still reuses an already selected seed pool and is not
  a replacement for a prospectively frozen independent validation.
- A0 generation and all 256 A0 relaxations were reused. Only gate-on A0
  structures received new MatterSim single-point force probes.
- Stability is a MatterSim-5M surrogate; DFT and target-property verification
  were not performed.
"""
    atomic_text(REPORT / "final_report.md", report)
    write_master(
        "stop_for_review",
        "success",
        final_state=leakage_state,
        leakage_diagnostic_completed=True,
        gpu_workers=0,
    )


def pipeline() -> None:
    PROGRESS.mkdir(parents=True, exist_ok=True)
    with PIPELINE_LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another leakage diagnostic is running") from error
        initialize()
        commands = (
            [str(CHGNET_PYTHON), "-m", __name__, "refine"],
            [str(MATTERGEN_PYTHON), "-m", __name__, "relax"],
            [str(MATTERGEN_PYTHON), "-m", __name__, "probe"],
            [str(MATTERGEN_PYTHON), "-m", __name__, "analyze"],
        )
        for command in commands:
            subprocess.run(command, cwd=PROJECT, check=True)


def status() -> None:
    payload = (
        read_json(MASTER_PROGRESS)
        if MASTER_PROGRESS.is_file()
        else {"status": "not_started"}
    )
    if REFINEMENT_MANIFEST.is_file():
        payload["refinement"] = {
            "success": len(pd.read_csv(REFINEMENT_MANIFEST)),
            "total": len(SEEDS),
        }
    if RELAX_PROGRESS.is_file():
        state = read_json(RELAX_PROGRESS)
        payload["e3g_relaxation"] = {
            "success": state.get("success", 0),
            "total": state.get("total", len(SEEDS)),
        }
    if PROBE_PROGRESS.is_file():
        state = read_json(PROBE_PROGRESS)
        payload["a0_force_probe"] = {
            "success": state.get("success", 0),
            "total": state.get("total", 0),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "initialize",
        "refine",
        "relax",
        "probe",
        "analyze",
        "pipeline",
        "status",
    ):
        commands.add_parser(command)
    for command in ("relax-worker", "probe-worker"):
        worker = commands.add_parser(command)
        worker.add_argument("--gpu", type=int, required=True)
        worker.add_argument("--slot", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    configure_modules()
    args = parse_args()
    if args.command == "initialize":
        initialize()
    elif args.command == "refine":
        refine()
    elif args.command == "relax":
        relax()
    elif args.command == "probe":
        probe()
    elif args.command == "analyze":
        analyze()
    elif args.command == "pipeline":
        pipeline()
    elif args.command == "status":
        status()
    elif args.command == "relax-worker":
        return core.relax_worker(args.gpu, args.slot)
    elif args.command == "probe-worker":
        return probe_worker(args.gpu, args.slot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
