#!/usr/bin/env python3
"""Blind 32x4 evaluation for the frozen post-generation SetRank selector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
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
import numpy as np
import pandas as pd


ROOT = Path("/data/dxl")
PROJECT = ROOT / "mattergen_v1"
sys.path.insert(0, str(PROJECT))
RESULT = ROOT / "results/postgen_fastgate/new_eval"
REPORT = ROOT / "reports/postgen_fastgate/new_eval"
LOG = ROOT / "logs/postgen_fastgate/new_eval"
GENERATION = RESULT / "generation/C0"
RELAXED = RESULT / "relaxed"
PROGRESS = RESULT / "progress"
GEN_PROGRESS = PROGRESS / "generation_progress.json"
RELAX_PROGRESS = PROGRESS / "relax_progress.json"
MASTER_PROGRESS = PROGRESS / "master_progress.json"
FEATURES = RESULT / "candidate_features.csv"
SCORES = RESULT / "pool_scores.csv"
SELECTION = RESULT / "selection.csv"
MATTERGEN_PYTHON = ROOT / "envs/mattergen_py310/bin/python"
CHGNET_PYTHON = ROOT / "envs/fn_pra_teacher/bin/python"
TASK_RUNNER = ROOT / "tools/guidance_stage7/run_sample.py"
QUALITY_ROOT = ROOT / "results/postgen_fastgate/quality_model/checkpoints"
SETRANK_ROOT = ROOT / "results/postgen_fastgate/setrank/checkpoints"
RELAX_COMMON = ROOT / "tools/guidance_stage7_eval"
REFERENCE = ROOT / "reference_assets/reference_TRI2024correction.gz"
REFERENCE_LMDB = ROOT / "reference_assets/reference_TRI2024correction.lmdb"
MATTERSIM = ROOT / "mattersim_weights/mattersim-v1.0.0-5M.pth"
MATTERSIM_SHA256 = (
    "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
)
SEEDS = tuple(range(33000, 33128))
POOL_SIZE = 4
POOL_COUNT = 32
METHODS = ("C0_FIRST", "Q6_NS_SETRANK")
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


def pool_for_seed(seed: int) -> tuple[int, int]:
    if seed not in SEEDS:
        raise ValueError(f"seed outside frozen blind range: {seed}")
    offset = seed - SEEDS[0]
    return offset // POOL_SIZE, offset % POOL_SIZE


def gate_decision(changes: dict[str, float], failures: dict[str, int]) -> dict[str, bool]:
    safety = (
        changes["structure_validity"] >= 0.0
        and changes["composition_validity"] >= -1.0 / POOL_COUNT
        and changes["stable"] >= -1.0 / POOL_COUNT
        and changes["nus"] >= -1.0 / POOL_COUNT
        and changes["novel"] >= -0.02
        and changes["unique"] >= -0.02
        and failures["selected"] <= failures["baseline"]
    )
    positive = (
        changes["ehull"] <= -0.005
        or changes["stable"] >= 1.0 / POOL_COUNT
        or changes["nus"] >= 1.0 / POOL_COUNT
        or changes["rmsd_relative"] <= -0.10
        or changes["pre_relax_max_force_relative"] <= -0.10
    )
    return {
        "safety_gate": bool(safety),
        "positive_gate": bool(positive),
        "Q6_NS_SETRANK_FINAL_GO": bool(safety and positive),
    }


def write_master(stage: str, status: str, **values: Any) -> None:
    payload = read_json(MASTER_PROGRESS) if MASTER_PROGRESS.is_file() else {
        "schema_version": 1,
        "experiment": "Frozen Q6 SetRank 32x4 blind evaluation",
        "created_at": now(),
        "seed_start": SEEDS[0],
        "seed_end": SEEDS[-1],
        "pool_size": POOL_SIZE,
        "pool_count": POOL_COUNT,
        "generator": "C0 original MatterGen",
        "sampling_trajectory_modified": False,
        "mattergen_backbone_trainable": False,
    }
    payload.update(
        {
            "updated_at": now(),
            "current_stage": stage,
            "status": status,
            **values,
        }
    )
    atomic_json(MASTER_PROGRESS, payload)


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
        if (
            len(atoms) != 1
            or not summary.get("success")
            or int(summary.get("seed", -1)) != seed
            or config.get("method") != "constant"
        ):
            return False
        arrays = (atoms[0].numbers, atoms[0].positions, atoms[0].cell.array)
        return all(np.isfinite(np.asarray(value)).all() for value in arrays)
    except BaseException:
        return False


def generation_rows() -> list[dict[str, Any]]:
    rows = []
    for index, seed in enumerate(SEEDS):
        pool_id, candidate_index = pool_for_seed(seed)
        rows.append(
            {
                "seed": seed,
                "pool_id": pool_id,
                "candidate_index": candidate_index,
                "status": "pending",
                "attempt": 0,
                "gpu": index % 8,
                "slot": (index % 32) // 8,
                "output_dir": str(GENERATION / str(seed)),
                "elapsed_seconds": None,
                "return_code": None,
                "error": "",
            }
        )
    return rows


def load_generation_progress() -> dict[str, Any]:
    state = (
        read_json(GEN_PROGRESS)
        if GEN_PROGRESS.is_file()
        else {"schema_version": 1, "created_at": now(), "tasks": generation_rows()}
    )
    if [int(row["seed"]) for row in state["tasks"]] != list(SEEDS):
        raise RuntimeError("generation progress seed contract mismatch")
    for row in state["tasks"]:
        if validate_generation(Path(row["output_dir"]), int(row["seed"])):
            row["status"] = "success"
        elif row["status"] in {"running", "success"}:
            row["status"] = "interrupted"
    save_generation_progress(state)
    return state


def save_generation_progress(state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    state["success"] = sum(row["status"] == "success" for row in state["tasks"])
    state["total"] = len(state["tasks"])
    atomic_json(GEN_PROGRESS, state)
    atomic_csv(PROGRESS / "generation_progress.csv", pd.DataFrame(state["tasks"]))


def quarantine(path: Path) -> None:
    if path.exists():
        destination = path.with_name(
            f"{path.name}.incomplete.{datetime.now().strftime('%Y%m%dT%H%M%S')}.{os.getpid()}"
        )
        os.replace(path, destination)


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
            "MATTERGEN_POSTGEN_FASTGATE": "1",
        }
    )
    return env


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
        "constant",
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
                    "stage": "generation",
                    "seed": seed,
                    "status": row["status"],
                    "success": state["success"],
                    "total": state["total"],
                }
            ),
            flush=True,
        )
    return valid


def run_generation() -> None:
    write_master("generation", "running")
    state = load_generation_progress()
    for attempt in range(2):
        pending = [
            row
            for row in state["tasks"]
            if not validate_generation(Path(row["output_dir"]), int(row["seed"]))
            and int(row["attempt"]) < 2
        ]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = {
                pool.submit(run_generation_task, row, state): row
                for row in pending
            }
            for future in as_completed(futures):
                future.result()
        state = load_generation_progress()
        if state["success"] == len(SEEDS):
            break
        print(json.dumps({"generation_retry_round": attempt + 1}), flush=True)
    state = load_generation_progress()
    if state["success"] != len(SEEDS):
        write_master("generation", "failed", generated=state["success"])
        raise RuntimeError(f"generation incomplete: {state['success']}/{len(SEEDS)}")
    write_master("generation", "success", generated=len(SEEDS))


def extract_and_select(device: str) -> None:
    from chgnet.model.model import CHGNet
    import torch

    from research.postgen_fastgate.features import structure_features
    from research.postgen_fastgate.setrank import SetRankNetwork

    write_master("frozen_feature_scoring", "running")
    if load_generation_progress()["success"] != len(SEEDS):
        raise RuntimeError("all 128 C0 candidates must exist before scoring")
    existing = pd.read_csv(FEATURES) if FEATURES.is_file() else pd.DataFrame()
    done = set(existing["seed"].astype(int)) if len(existing) else set()
    model = CHGNet.load(model_name="0.3.0", verbose=False, use_device=device)
    rows = existing.to_dict(orient="records")
    pending = [seed for seed in SEEDS if seed not in done]
    for start in range(0, len(pending), 32):
        chunk = pending[start : start + 32]
        atoms = [
            ase.io.read(GENERATION / str(seed) / "generated_crystals.extxyz")
            for seed in chunk
        ]
        from pymatgen.io.ase import AseAtomsAdaptor

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
            raise RuntimeError("CHGNet candidate feature count mismatch")
        for seed, atom, prediction in zip(chunk, atoms, predictions, strict=True):
            pool_id, candidate_index = pool_for_seed(seed)
            rows.append(
                {
                    "method": "C0",
                    "seed": seed,
                    "split": "blind_new",
                    "pool_id": pool_id,
                    "candidate_index": candidate_index,
                    "input_path": str(
                        GENERATION / str(seed) / "generated_crystals.extxyz"
                    ),
                    **structure_features(atom, prediction),
                }
            )
        atomic_csv(FEATURES, pd.DataFrame(rows).sort_values("seed"))
        print(
            json.dumps(
                {"stage": "feature_extraction", "completed": len(rows), "total": 128}
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    if len(frame) != len(SEEDS) or frame["seed"].astype(int).tolist() != list(SEEDS):
        raise RuntimeError("candidate feature manifest mismatch")

    quality_checkpoint = sorted(QUALITY_ROOT.glob("quality_member_*.pt"))[0]
    quality_payload = torch.load(
        quality_checkpoint, map_location="cpu", weights_only=True
    )
    metadata = quality_payload["metadata"]
    columns = metadata["feature_columns"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing frozen features: {missing}")
    feature = frame.loc[:, columns].to_numpy(float)
    feature = (
        feature - np.asarray(metadata["feature_mean"], dtype=float)
    ) / np.asarray(metadata["feature_std"], dtype=float)
    feature_tensor = torch.tensor(feature, dtype=torch.float32, device=device)

    checkpoint_paths = sorted(SETRANK_ROOT.glob("setrank_member_*.pt"))
    if len(checkpoint_paths) != 3:
        raise RuntimeError("frozen SetRank ensemble must have three members")
    pools = torch.tensor(
        np.arange(len(SEEDS)).reshape(POOL_COUNT, POOL_SIZE),
        dtype=torch.long,
        device=device,
    )
    predictions = []
    for checkpoint in checkpoint_paths:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        ranker = SetRankNetwork(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload["dropout"]),
        ).to(device)
        ranker.load_state_dict(payload["state_dict"])
        ranker.eval()
        with torch.no_grad():
            predictions.append(ranker(feature_tensor[pools]).cpu().numpy())
    stack = np.stack(predictions)
    conservative = stack.mean(axis=0) - 0.25 * stack.std(axis=0)
    score_rows = []
    selections = []
    for pool_id in range(POOL_COUNT):
        best = int(np.argmax(conservative[pool_id]))
        for candidate_index in range(POOL_SIZE):
            seed = SEEDS[pool_id * POOL_SIZE + candidate_index]
            score_rows.append(
                {
                    "pool_id": pool_id,
                    "candidate_index": candidate_index,
                    "seed": seed,
                    "score_mean": float(stack[:, pool_id, candidate_index].mean()),
                    "score_std": float(stack[:, pool_id, candidate_index].std()),
                    "conservative_score": float(
                        conservative[pool_id, candidate_index]
                    ),
                    "selected": candidate_index == best,
                    "baseline": candidate_index == 0,
                }
            )
        selections.append(
            {
                "pool_id": pool_id,
                "baseline_seed": SEEDS[pool_id * POOL_SIZE],
                "selected_seed": SEEDS[pool_id * POOL_SIZE + best],
                "selected_candidate_index": best,
            }
        )
    atomic_csv(SCORES, pd.DataFrame(score_rows))
    atomic_csv(SELECTION, pd.DataFrame(selections))
    freeze = {
        "schema_version": 1,
        "created_at": now(),
        "selector": "Q6_NS_SETRANK_v2",
        "selection_rule": "ensemble mean minus 0.25 standard deviations",
        "candidate_pool_size": POOL_SIZE,
        "candidate_pool_count": POOL_COUNT,
        "selected_before_mattersim": True,
        "quality_checkpoint_metadata_source": str(quality_checkpoint),
        "quality_checkpoint_sha256": sha256(quality_checkpoint),
        "setrank_checkpoints": [
            {"path": str(path), "sha256": sha256(path)}
            for path in checkpoint_paths
        ],
        "selection_path": str(SELECTION),
        "selection_sha256": sha256(SELECTION),
        "mattergen_backbone_trainable": False,
        "chgnet_trainable": False,
        "sampling_trajectory_modified": False,
    }
    atomic_json(REPORT / "frozen_selection_manifest.json", freeze)
    write_master(
        "frozen_feature_scoring",
        "success",
        feature_rows=len(frame),
        selected_pools=len(selections),
    )


def structure_hash(atoms: Any) -> str:
    digest = hashlib.sha256()
    for value in (atoms.numbers, atoms.positions, atoms.cell.array):
        array = np.asarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def relax_rows() -> list[dict[str, Any]]:
    selection = pd.read_csv(SELECTION)
    rows = []
    for item in selection.itertuples(index=False):
        for method, seed in (
            ("C0_FIRST", int(item.baseline_seed)),
            ("Q6_NS_SETRANK", int(item.selected_seed)),
        ):
            input_path = GENERATION / str(seed) / "generated_crystals.extxyz"
            atoms = ase.io.read(input_path)
            rows.append(
                {
                    "task_id": f"{method}_pool_{int(item.pool_id):02d}",
                    "method": method,
                    "pool_id": int(item.pool_id),
                    "candidate_seed": seed,
                    "status": "pending",
                    "attempt": 0,
                    "input_path": str(input_path),
                    "input_hash": structure_hash(atoms),
                    "output_dir": str(RELAXED / method / f"{int(item.pool_id):02d}"),
                    "gpu": None,
                    "slot": None,
                    "elapsed_seconds": None,
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
    import fcntl

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
        expected = [row["task_id"] for row in relax_rows()]
        if [row["task_id"] for row in state["tasks"]] != expected:
            raise RuntimeError("relax progress task contract mismatch")
        for row in state["tasks"]:
            if validate_relax(row):
                row["status"] = "success"
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


def relax_worker(gpu: int, slot: int) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
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
            path = output / "relaxed_structure.extxyz"
            ase.io.write(path, result["atoms"], format="extxyz")
            checked = ase.io.read(path)
            summary = {
                "success": True,
                "task_id": row["task_id"],
                "method": row["method"],
                "pool_id": row["pool_id"],
                "candidate_seed": row["candidate_seed"],
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


def run_relax() -> None:
    if sha256(MATTERSIM) != MATTERSIM_SHA256:
        raise RuntimeError("MatterSim checkpoint SHA256 mismatch")
    write_master("mattersim_relax", "running")
    state = initialize_relax()
    if state["success"] == 2 * POOL_COUNT:
        write_master("mattersim_relax", "success", relaxed=state["success"])
        return
    logs = LOG / "relax_workers"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    for slot in range(2):
        for gpu in range(8):
            handle = (logs / f"gpu{gpu}_slot{slot}.log").open(
                "a", encoding="utf-8"
            )
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
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
    if any(code != 0 for code in codes) or state["success"] != 2 * POOL_COUNT:
        write_master("mattersim_relax", "failed", relaxed=state["success"])
        raise RuntimeError(
            f"MatterSim relaxation incomplete: {state['success']}/{2 * POOL_COUNT}"
        )
    write_master("mattersim_relax", "success", relaxed=state["success"])


def bootstrap_ci(values: np.ndarray, samples: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(20260728)
    indexes = rng.integers(0, len(values), size=(samples, len(values)))
    return tuple(
        float(value)
        for value in np.quantile(values[indexes].mean(axis=1), [0.025, 0.975])
    )


def paired_statistics(
    baseline: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    from scipy.stats import binomtest, wilcoxon

    rows = []
    continuous = (
        ("energy_above_hull_per_atom", True),
        ("rmsd_from_relaxation", True),
        ("pre_relax_max_force_ev_ang", True),
    )
    for column, lower_is_better in continuous:
        difference = selected[column].to_numpy(float) - baseline[column].to_numpy(
            float
        )
        low, high = bootstrap_ci(difference)
        test = (
            wilcoxon(difference, zero_method="pratt")
            if np.any(np.abs(difference) > 1e-12)
            else None
        )
        wins = int((difference < -1e-12).sum()) if lower_is_better else int(
            (difference > 1e-12).sum()
        )
        losses = int((difference > 1e-12).sum()) if lower_is_better else int(
            (difference < -1e-12).sum()
        )
        rows.append(
            {
                "metric": column,
                "type": "continuous",
                "baseline_mean": float(baseline[column].mean()),
                "selected_mean": float(selected[column].mean()),
                "mean_difference_selected_minus_baseline": float(difference.mean()),
                "median_difference_selected_minus_baseline": float(
                    np.median(difference)
                ),
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
                "mean_difference_selected_minus_baseline": float(difference.mean()),
                "median_difference_selected_minus_baseline": float(
                    np.median(difference)
                ),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "test": "exact paired discordant-binomial",
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


def analyze() -> None:
    import sys
    from pymatgen.io.ase import AseAtomsAdaptor

    write_master("official_metrics", "running")
    state = initialize_relax()
    if state["success"] != 2 * POOL_COUNT:
        raise RuntimeError("official metrics require 64 successful relax tasks")
    rows_by_method: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS}
    for row in state["tasks"]:
        summary = read_json(Path(row["output_dir"]) / "relax_summary.json")
        original = ase.io.read(row["input_path"])
        relaxed = ase.io.read(Path(row["output_dir"]) / "relaxed_structure.extxyz")
        structure = AseAtomsAdaptor.get_structure(relaxed)
        rows_by_method[row["method"]].append(
            {
                "method": row["method"],
                "seed": int(row["pool_id"]),
                "pool_id": int(row["pool_id"]),
                "candidate_seed": int(row["candidate_seed"]),
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
                "_relaxed_atoms": relaxed,
                "_original_atoms": original,
            }
        )
    frames = {
        method: pd.DataFrame(rows).sort_values("pool_id").reset_index(drop=True)
        for method, rows in rows_by_method.items()
    }
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
    official.SEEDS = list(range(POOL_COUNT))
    official.STABILITY_THRESHOLD = 0.1
    metrics, errors = official.official_metrics(frames)
    if errors:
        raise RuntimeError(f"official metrics failures: {errors}")
    baseline = frames["C0_FIRST"].sort_values("pool_id").reset_index(drop=True)
    selected = frames["Q6_NS_SETRANK"].sort_values("pool_id").reset_index(
        drop=True
    )
    stats = paired_statistics(baseline, selected)
    atomic_csv(REPORT / "paired_statistics.csv", stats)

    def means(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "ehull": float(frame["energy_above_hull_per_atom"].mean()),
            "rmsd": float(frame["rmsd_from_relaxation"].mean()),
            "stable": float(frame["stable"].astype(bool).mean()),
            "nus": float(frame["novel_unique_stable"].astype(bool).mean()),
            "composition_validity": float(frame["comp_validity"].astype(bool).mean()),
            "structure_validity": float(
                frame["structure_validity"].astype(bool).mean()
            ),
            "novel": float(frame["novel"].astype(bool).mean()),
            "unique": float(frame["unique"].astype(bool).mean()),
            "converged": float(frame["converged"].astype(bool).mean()),
            "pre_relax_max_force_ev_ang": float(
                frame["pre_relax_max_force_ev_ang"].mean()
            ),
        }

    baseline_mean = means(baseline)
    selected_mean = means(selected)
    changes = {
        "ehull": selected_mean["ehull"] - baseline_mean["ehull"],
        "rmsd_relative": selected_mean["rmsd"] / baseline_mean["rmsd"] - 1.0,
        "pre_relax_max_force_relative": (
            selected_mean["pre_relax_max_force_ev_ang"]
            / baseline_mean["pre_relax_max_force_ev_ang"]
            - 1.0
        ),
        "stable": selected_mean["stable"] - baseline_mean["stable"],
        "nus": selected_mean["nus"] - baseline_mean["nus"],
        "composition_validity": (
            selected_mean["composition_validity"]
            - baseline_mean["composition_validity"]
        ),
        "structure_validity": (
            selected_mean["structure_validity"]
            - baseline_mean["structure_validity"]
        ),
        "novel": selected_mean["novel"] - baseline_mean["novel"],
        "unique": selected_mean["unique"] - baseline_mean["unique"],
        "converged": selected_mean["converged"] - baseline_mean["converged"],
    }
    failures = {"baseline": 0, "selected": 0}
    gates = gate_decision(changes, failures)
    summary = {
        "schema_version": 1,
        "completed_at": now(),
        "candidate": "Q6_NS_SETRANK_v2",
        "comparison": "Frozen selector versus first C0 trajectory in each pool",
        "pool_count": POOL_COUNT,
        "pool_size": POOL_SIZE,
        "generated_trajectories": len(SEEDS),
        "evaluated_baseline": POOL_COUNT,
        "evaluated_selected": POOL_COUNT,
        "baseline": baseline_mean,
        "selected": selected_mean,
        "changes": changes,
        "relaxation_failures": failures,
        "gates": gates,
        "official_metrics": metrics,
        "selection_manifest": str(REPORT / "frozen_selection_manifest.json"),
        "paired_statistics": str(REPORT / "paired_statistics.csv"),
        "sampling_trajectory_modified": False,
        "mattergen_backbone_trainable": False,
        "quality_network_trainable_during_evaluation": False,
        "mattersim_used_for_selection": False,
        "dft_verified": False,
    }
    atomic_json(REPORT / "final_summary.json", summary)
    lines = [
        "# Q6 NS-SetRank 32-pool blind evaluation",
        "",
        f"- Final GO: `{gates['Q6_NS_SETRANK_FINAL_GO']}`",
        f"- Safety gate: `{gates['safety_gate']}`",
        f"- Positive gate: `{gates['positive_gate']}`",
        "- Generator: original C0 MatterGen; sampling trajectory unchanged",
        "- Pool: 32 independent pools x 4 new C0 trajectories",
        "- Selection: frozen CHGNet features + frozen three-member SetRank ensemble",
        "- MatterSim was used only after candidate selection",
        "- DFT verified: False",
        "",
        "## Aggregate metrics",
        "",
        pd.DataFrame(
            [
                {"method": "C0_FIRST", **baseline_mean},
                {"method": "Q6_NS_SETRANK", **selected_mean},
            ]
        ).to_markdown(index=False),
        "",
        "## Changes",
        "",
        pd.DataFrame([changes]).to_markdown(index=False),
        "",
    ]
    (REPORT / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_master(
        "go_no_go",
        "success",
        final_go=gates["Q6_NS_SETRANK_FINAL_GO"],
        final_summary=str(REPORT / "final_summary.json"),
    )
    print(json.dumps(gates, sort_keys=True), flush=True)


def pipeline() -> None:
    write_master("pipeline", "running")
    commands = (
        [str(MATTERGEN_PYTHON), str(Path(__file__).resolve()), "generate"],
        [
            str(CHGNET_PYTHON),
            str(Path(__file__).resolve()),
            "score",
            "--device",
            "cuda",
        ],
        [str(MATTERGEN_PYTHON), str(Path(__file__).resolve()), "relax"],
        [str(MATTERGEN_PYTHON), str(Path(__file__).resolve()), "analyze"],
    )
    for command in commands:
        subprocess.run(command, cwd=PROJECT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate")
    score = commands.add_parser("score")
    score.add_argument("--device", default="cuda")
    commands.add_parser("relax")
    commands.add_parser("analyze")
    commands.add_parser("pipeline")
    worker = commands.add_parser("relax-worker")
    worker.add_argument("--gpu", type=int, required=True)
    worker.add_argument("--slot", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "generate":
        run_generation()
    elif args.command == "score":
        extract_and_select(args.device)
    elif args.command == "relax":
        run_relax()
    elif args.command == "analyze":
        analyze()
    elif args.command == "pipeline":
        pipeline()
    elif args.command == "relax-worker":
        return relax_worker(args.gpu, args.slot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
