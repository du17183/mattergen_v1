#!/usr/bin/env python3
"""Q3 equivariant trust-region post-generation crystal refiner."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_IMPORT_ROOT))

import ase.io
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.postgen_fastgate.new_eval import (
    GENERATION,
    POOL_COUNT,
    PROJECT,
    REFERENCE,
    REFERENCE_LMDB,
    RELAXED,
    RESULT,
    REPORT,
    atomic_csv,
    atomic_json,
    now,
    paired_statistics,
    read_json,
    sha256,
    structure_hash,
)


Q3_RESULT = Path("/data/dxl/results/postgen_fastgate/q3_refiner")
Q3_REPORT = Path("/data/dxl/reports/postgen_fastgate/q3_refiner")
Q3_LOG = Path("/data/dxl/logs/postgen_fastgate/q3_refiner")
Q3_MODEL = Q3_RESULT / "model/q3_gate.joblib"
Q3_REFINED = Q3_RESULT / "refined"
Q3_RELAXED = Q3_RESULT / "relaxed"
Q3_MANIFEST = Q3_RESULT / "refinement_manifest.csv"
Q3_PROGRESS = Q3_RESULT / "relax_progress.json"
FEATURES = RESULT / "candidate_features.csv"
HISTORICAL_FEATURES = Path(
    "/data/dxl/results/postgen_fastgate/features/historical_features.csv"
)
RP_DIRECTION = Path(
    "/data/dxl/reports/rp_qtfg/phase0/offline_direction"
)
MATTERGEN_PYTHON = Path("/data/dxl/envs/mattergen_py310/bin/python")
CHGNET_PYTHON = Path("/data/dxl/envs/fn_pra_teacher/bin/python")
RELAX_COMMON = Path("/data/dxl/tools/guidance_stage7_eval")

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
TRAIN_SEEDS = tuple(range(20000, 20064))
NETWORK_HIDDEN = (8,)
NETWORK_ALPHA = 0.1
NETWORK_THRESHOLD = 0.5
NETWORK_SEED = 20260728
REFINEMENT_STEPS = 5
POSITION_ETA = 0.01
POSITION_RADIUS_ANGSTROM = 0.02
BACKTRACK_MAX = 3
MINIMUM_DISTANCE_ANGSTROM = 0.5


def build_network() -> Any:
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=NETWORK_HIDDEN,
            activation="tanh",
            alpha=NETWORK_ALPHA,
            max_iter=2000,
            random_state=NETWORK_SEED,
        ),
    )


def historical_training_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(HISTORICAL_FEATURES)
    features = features[
        (features["method"] == "A0")
        & features["seed"].astype(int).isin(TRAIN_SEEDS)
    ].sort_values("seed").reset_index(drop=True)
    baseline = pd.read_csv(
        RP_DIRECTION / "baseline_per_structure.csv"
    ).sort_values("seed").reset_index(drop=True)
    refined = pd.read_csv(
        RP_DIRECTION / "pos_5_per_structure.csv"
    ).sort_values("seed").reset_index(drop=True)
    if (
        len(features) != 64
        or features["seed"].astype(int).tolist() != list(TRAIN_SEEDS)
        or baseline["seed"].astype(int).tolist() != list(TRAIN_SEEDS)
        or refined["seed"].astype(int).tolist() != list(TRAIN_SEEDS)
    ):
        raise RuntimeError("Q3 historical training contract mismatch")
    labels = (
        refined["initial_max_force_ev_ang"].to_numpy(float)
        < baseline["initial_max_force_ev_ang"].to_numpy(float)
    ).astype(int)
    return features, labels, baseline, refined


def train_gate() -> dict[str, Any]:
    features, labels, baseline, refined = historical_training_data()
    values = features.loc[:, FEATURE_COLUMNS].to_numpy(float)
    cross_validation = StratifiedKFold(
        n_splits=8, shuffle=True, random_state=NETWORK_SEED
    )
    probabilities = cross_val_predict(
        build_network(),
        values,
        labels,
        cv=cross_validation,
        method="predict_proba",
    )[:, 1]
    applied = probabilities >= NETWORK_THRESHOLD
    baseline_force = baseline["initial_max_force_ev_ang"].to_numpy(float)
    refined_force = refined["initial_max_force_ev_ang"].to_numpy(float)
    gated_force = np.where(applied, refined_force, baseline_force)
    force_relative = float(gated_force.mean() / baseline_force.mean() - 1.0)

    def gated_change(column: str) -> float:
        left = baseline[column].to_numpy(float)
        right = refined[column].to_numpy(float)
        return float(np.where(applied, right, left).mean() - left.mean())

    stable_change = gated_change("stable")
    composition_change = gated_change("comp_validity")
    structure_change = gated_change("structure_validity")
    novel_change = gated_change("novel")
    unique_change = gated_change("unique")
    baseline_nus = (
        baseline["stable"].astype(bool)
        & baseline["novel"].astype(bool)
        & baseline["unique"].astype(bool)
    ).to_numpy(float)
    refined_nus = (
        refined["stable"].astype(bool)
        & refined["novel"].astype(bool)
        & refined["unique"].astype(bool)
    ).to_numpy(float)
    nus_change = float(
        np.where(applied, refined_nus, baseline_nus).mean()
        - baseline_nus.mean()
    )
    safety = (
        stable_change >= -1.0 / 32.0
        and composition_change >= -1.0 / 32.0
        and structure_change >= 0.0
        and novel_change >= -0.02
        and unique_change >= -0.02
        and nus_change >= -1.0 / 32.0
    )
    positive = force_relative <= -0.10
    network = build_network()
    network.fit(values, labels)
    Q3_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(network, Q3_MODEL)
    report = {
        "schema_version": 1,
        "created_at": now(),
        "training_rows": len(features),
        "training_generator": "A0 historical outputs",
        "training_seed_range": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
        "feature_columns": list(FEATURE_COLUMNS),
        "network": {
            "type": "StandardScaler + MLPClassifier",
            "input_dim": len(FEATURE_COLUMNS),
            "hidden_dims": list(NETWORK_HIDDEN),
            "output_dim": 1,
            "trainable_parameters": (
                len(FEATURE_COLUMNS) * NETWORK_HIDDEN[0]
                + NETWORK_HIDDEN[0]
                + NETWORK_HIDDEN[0]
                + 1
            ),
            "alpha": NETWORK_ALPHA,
            "threshold": NETWORK_THRESHOLD,
            "random_seed": NETWORK_SEED,
        },
        "eight_fold_out_of_fold": {
            "auroc": float(roc_auc_score(labels, probabilities)),
            "accuracy": float(
                accuracy_score(labels, probabilities >= NETWORK_THRESHOLD)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    labels, probabilities >= NETWORK_THRESHOLD
                )
            ),
            "apply_rate": float(applied.mean()),
            "force_relative_change": force_relative,
            "force_improvement_rate": float((gated_force < baseline_force).mean()),
            "force_worsening_rate": float((gated_force > baseline_force).mean()),
            "stable_change": stable_change,
            "nus_change": nus_change,
            "composition_change": composition_change,
            "structure_change": structure_change,
            "novel_change": novel_change,
            "unique_change": unique_change,
        },
        "safety_gate": bool(safety),
        "positive_gate": bool(positive),
        "Q3_OFFLINE_GO": bool(safety and positive),
        "model_path": str(Q3_MODEL),
        "model_sha256": sha256(Q3_MODEL),
        "mattergen_backbone_trainable": False,
        "chgnet_trainable": False,
    }
    atomic_json(Q3_REPORT / "training_and_offline_summary.json", report)
    atomic_csv(
        Q3_REPORT / "oof_predictions.csv",
        pd.DataFrame(
            {
                "seed": TRAIN_SEEDS,
                "label_force_improves": labels,
                "oof_probability": probabilities,
                "apply": applied,
                "baseline_force": baseline_force,
                "pos5_force": refined_force,
                "gated_force": gated_force,
            }
        ),
    )
    if not report["Q3_OFFLINE_GO"]:
        raise RuntimeError("Q3 failed frozen historical offline gate")
    return report


def minimum_distance(atoms: Any) -> float:
    if len(atoms) < 2:
        return math.inf
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def finite_safe(atoms: Any) -> bool:
    volume = float(atoms.get_volume())
    return (
        np.isfinite(atoms.positions).all()
        and np.isfinite(atoms.cell.array).all()
        and math.isfinite(volume)
        and volume > 0.1
        and minimum_distance(atoms) >= MINIMUM_DISTANCE_ANGSTROM
    )


def position_proposal(atoms: Any, forces: np.ndarray, scale: float) -> Any:
    candidate = atoms.copy()
    displacement = POSITION_ETA * scale * np.asarray(forces, dtype=float)
    norms = np.linalg.norm(displacement, axis=1)
    cap = POSITION_RADIUS_ANGSTROM * scale
    displacement *= np.minimum(
        1.0, cap / np.maximum(norms, 1.0e-12)
    )[:, None]
    candidate.positions[:] = candidate.positions + displacement
    candidate.wrap()
    return candidate


def predict_chgnet(model: Any, atoms_list: list[Any]) -> list[dict[str, Any]]:
    from pymatgen.io.ase import AseAtomsAdaptor

    structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]
    prediction = model.predict_structure(
        structures,
        task="efs",
        batch_size=max(1, len(structures)),
    )
    return prediction if isinstance(prediction, list) else [prediction]


def advance(
    model: Any,
    atoms_list: list[Any],
    counters: list[dict[str, int]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    old_predictions = predict_chgnet(model, atoms_list)
    unresolved = list(range(len(atoms_list)))
    accepted: dict[int, tuple[Any, dict[str, Any], int]] = {}
    for backtrack in range(BACKTRACK_MAX):
        if not unresolved:
            break
        scale = 0.5**backtrack
        candidates = [
            position_proposal(
                atoms_list[index],
                np.asarray(old_predictions[index]["f"], dtype=float),
                scale,
            )
            for index in unresolved
        ]
        safe_local = [
            index
            for index, candidate in enumerate(candidates)
            if finite_safe(candidate)
        ]
        predictions = (
            predict_chgnet(
                model, [candidates[index] for index in safe_local]
            )
            if safe_local
            else []
        )
        by_local = dict(zip(safe_local, predictions, strict=True))
        remaining = []
        for local_index, global_index in enumerate(unresolved):
            prediction = by_local.get(local_index)
            old_energy = float(
                np.asarray(old_predictions[global_index]["e"]).reshape(-1)[0]
            )
            new_energy = (
                float(np.asarray(prediction["e"]).reshape(-1)[0])
                if prediction is not None
                else math.inf
            )
            if math.isfinite(new_energy) and new_energy <= old_energy + 1.0e-7:
                accepted[global_index] = (
                    candidates[local_index],
                    prediction,
                    backtrack,
                )
            else:
                remaining.append(global_index)
        unresolved = remaining
    outputs = []
    final_predictions = []
    for index, atoms in enumerate(atoms_list):
        if index in accepted:
            candidate, prediction, backtrack = accepted[index]
            outputs.append(candidate)
            final_predictions.append(prediction)
            counters[index]["accepted_steps"] += 1
            counters[index]["backtracking_count"] += backtrack
        else:
            outputs.append(atoms.copy())
            final_predictions.append(old_predictions[index])
            counters[index]["fallback_count"] += 1
    return outputs, final_predictions


def wrapped_displacement_max(original: Any, refined: Any) -> float:
    fractional_delta = refined.get_scaled_positions() - original.get_scaled_positions()
    fractional_delta -= np.round(fractional_delta)
    cartesian_delta = fractional_delta @ np.asarray(original.cell.array)
    return float(np.linalg.norm(cartesian_delta, axis=1).max())


def refine_new() -> None:
    from chgnet.model.model import CHGNet

    report = train_gate()
    network = joblib.load(Q3_MODEL)
    raw = pd.read_csv(FEATURES)
    raw = raw[
        raw["candidate_index"].astype(int) == 0
    ].sort_values("pool_id").reset_index(drop=True)
    if len(raw) != POOL_COUNT:
        raise RuntimeError("Q3 requires 32 first-candidate C0 baselines")
    probabilities = network.predict_proba(
        raw.loc[:, FEATURE_COLUMNS].to_numpy(float)
    )[:, 1]
    apply_gate = probabilities >= NETWORK_THRESHOLD
    originals = [ase.io.read(path) for path in raw["input_path"]]
    numbers = [atoms.numbers.copy() for atoms in originals]
    outputs = [atoms.copy() for atoms in originals]
    counters = [
        {"accepted_steps": 0, "fallback_count": 0, "backtracking_count": 0}
        for _ in originals
    ]
    active = np.flatnonzero(apply_gate)
    if len(active):
        model = CHGNet.load(
            model_name="0.3.0", verbose=False, use_device="cuda"
        )
        active_atoms = [outputs[index] for index in active]
        active_counters = [counters[index] for index in active]
        for _step in range(REFINEMENT_STEPS):
            active_atoms, _predictions = advance(
                model, active_atoms, active_counters
            )
        for local, index in enumerate(active):
            outputs[index] = active_atoms[local]
            counters[index] = active_counters[local]

    rows = []
    for index, (metadata, original, refined, probability, apply) in enumerate(
        zip(
            raw.to_dict(orient="records"),
            originals,
            outputs,
            probabilities,
            apply_gate,
            strict=True,
        )
    ):
        pool_id = int(metadata["pool_id"])
        seed = int(metadata["seed"])
        if not np.array_equal(refined.numbers, numbers[index]):
            raise RuntimeError("Q3 changed atomic numbers")
        if not finite_safe(refined):
            raise RuntimeError("Q3 produced unsafe refined structure")
        output_dir = Q3_REFINED / f"{pool_id:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "refined_structure.extxyz"
        ase.io.write(path, refined, format="extxyz")
        checked = ase.io.read(path)
        rows.append(
            {
                "pool_id": pool_id,
                "seed": seed,
                "input_path": metadata["input_path"],
                "output_path": str(path),
                "gate_probability": float(probability),
                "gate_applied": bool(apply),
                **counters[index],
                "maximum_wrapped_displacement_angstrom": (
                    wrapped_displacement_max(original, checked)
                ),
                "minimum_distance_angstrom": minimum_distance(checked),
                "input_hash": structure_hash(original),
                "output_hash": structure_hash(checked),
                "atomic_numbers_unchanged": bool(
                    np.array_equal(checked.numbers, numbers[index])
                ),
            }
        )
    manifest = pd.DataFrame(rows)
    atomic_csv(Q3_MANIFEST, manifest)
    freeze = {
        "schema_version": 1,
        "created_at": now(),
        "candidate": "Q3_E3_PCR",
        "network": report["network"],
        "model_sha256": sha256(Q3_MODEL),
        "gate_apply_count": int(manifest["gate_applied"].sum()),
        "accepted_step_mean": float(manifest["accepted_steps"].mean()),
        "fallback_mean": float(manifest["fallback_count"].mean()),
        "backtracking_mean": float(
            manifest["backtracking_count"].mean()
        ),
        "maximum_displacement_angstrom": float(
            manifest["maximum_wrapped_displacement_angstrom"].max()
        ),
        "trust_region": {
            "steps": REFINEMENT_STEPS,
            "position_eta": POSITION_ETA,
            "per_step_radius_angstrom": POSITION_RADIUS_ANGSTROM,
            "backtrack_max": BACKTRACK_MAX,
            "minimum_distance_angstrom": MINIMUM_DISTANCE_ANGSTROM,
        },
        "manifest_sha256": sha256(Q3_MANIFEST),
        "refinement_frozen_before_mattersim": True,
        "new_blind_labels_used": False,
        "mattergen_backbone_trainable": False,
        "chgnet_trainable": False,
        "sampling_trajectory_modified": False,
    }
    atomic_json(Q3_REPORT / "frozen_refinement_manifest.json", freeze)
    print(json.dumps(freeze, sort_keys=True), flush=True)


def relax_rows() -> list[dict[str, Any]]:
    manifest = pd.read_csv(Q3_MANIFEST)
    rows = []
    for item in manifest.itertuples(index=False):
        reuse = (not bool(item.gate_applied)) or item.input_hash == item.output_hash
        rows.append(
            {
                "task_id": f"Q3_E3_PCR_pool_{int(item.pool_id):02d}",
                "pool_id": int(item.pool_id),
                "seed": int(item.seed),
                "input_path": str(item.output_path),
                "input_hash": str(item.output_hash),
                "output_dir": str(Q3_RELAXED / f"{int(item.pool_id):02d}"),
                "reused_output_dir": (
                    str(RELAXED / "C0_FIRST" / f"{int(item.pool_id):02d}")
                    if reuse
                    else ""
                ),
                "status": "reused" if reuse else "pending",
                "attempt": 0,
                "gpu": None,
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
            summary["success"] is True
            and int(summary["pool_id"]) == int(row["pool_id"])
            and int(summary["seed"]) == int(row["seed"])
            and summary["input_hash"] == row["input_hash"]
            and np.isfinite(atoms.positions).all()
            and np.isfinite(atoms.cell.array).all()
        )
    except BaseException:
        return False


def save_progress(state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    state["reused"] = sum(row["status"] == "reused" for row in state["tasks"])
    state["success"] = sum(row["status"] == "success" for row in state["tasks"])
    state["complete"] = state["reused"] + state["success"]
    atomic_json(Q3_PROGRESS, state)
    atomic_csv(Q3_RESULT / "relax_progress.csv", pd.DataFrame(state["tasks"]))


def load_progress() -> dict[str, Any]:
    expected = relax_rows()
    state = (
        read_json(Q3_PROGRESS)
        if Q3_PROGRESS.is_file()
        else {"schema_version": 1, "created_at": now(), "tasks": expected}
    )
    if [row["task_id"] for row in state["tasks"]] != [
        row["task_id"] for row in expected
    ]:
        raise RuntimeError("Q3 relax task contract mismatch")
    for row, contract in zip(state["tasks"], expected, strict=True):
        row["reused_output_dir"] = contract["reused_output_dir"]
        if contract["reused_output_dir"]:
            row["status"] = "reused"
        elif validate_relax(row):
            row["status"] = "success"
        elif row["status"] == "running":
            row["status"] = "interrupted"
    save_progress(state)
    return state


def run_relax_one(row: dict[str, Any], gpu: int) -> dict[str, Any]:
    command = [
        str(MATTERGEN_PYTHON),
        str(Path(__file__).resolve()),
        "relax-task",
        "--pool-id",
        str(row["pool_id"]),
        "--seed",
        str(row["seed"]),
        "--input-path",
        str(row["input_path"]),
        "--input-hash",
        str(row["input_hash"]),
        "--gpu",
        str(gpu),
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        }
    )
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=PROJECT,
        env=env,
        text=True,
        capture_output=True,
    )
    return {
        "return_code": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stderr": process.stderr,
    }


def run_relax() -> None:
    state = load_progress()
    for _attempt in range(2):
        pending = [
            row
            for row in state["tasks"]
            if row["status"] not in {"success", "reused"}
            and int(row["attempt"]) < 2
        ]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for index, row in enumerate(pending):
                row.update(
                    status="running",
                    attempt=int(row["attempt"]) + 1,
                    gpu=index % 8,
                    error="",
                )
                futures[
                    pool.submit(run_relax_one, dict(row), index % 8)
                ] = row
            save_progress(state)
            for future in as_completed(futures):
                row = futures[future]
                result = future.result()
                valid = result["return_code"] == 0 and validate_relax(row)
                row.update(
                    status="success" if valid else "failed",
                    elapsed_seconds=result["elapsed_seconds"],
                    error="" if valid else result["stderr"][-4000:],
                )
                save_progress(state)
                print(
                    json.dumps(
                        {
                            "pool_id": row["pool_id"],
                            "status": row["status"],
                            "complete": state["complete"],
                            "total": POOL_COUNT,
                        }
                    ),
                    flush=True,
                )
        state = load_progress()
    if state["complete"] != POOL_COUNT:
        raise RuntimeError(
            f"Q3 relax incomplete: {state['complete']}/{POOL_COUNT}"
        )


def relax_task(
    pool_id: int,
    seed: int,
    input_path: Path,
    input_hash: str,
    gpu: int,
) -> None:
    from mattersim.forcefield import MatterSimCalculator

    sys.path.insert(0, str(RELAX_COMMON))
    from relax_common import load_potential, relax_group

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = Q3_RELAXED / f"{pool_id:02d}"
    row = {
        "output_dir": str(output),
        "pool_id": pool_id,
        "seed": seed,
        "input_hash": input_hash,
    }
    if output.exists() and validate_relax(row):
        return
    if output.exists():
        os.replace(output, output.with_name(f"{output.name}.incomplete.{os.getpid()}"))
    output.mkdir(parents=True)
    atoms = ase.io.read(input_path)
    if structure_hash(atoms) != input_hash:
        raise RuntimeError("Q3 refined input hash mismatch")
    potential = load_potential("cuda")
    probe = atoms.copy()
    probe.calc = MatterSimCalculator.from_potential(
        potential=potential, device="cuda"
    )
    initial_energy = float(probe.get_potential_energy())
    initial_force = np.asarray(probe.get_forces(), dtype=float)
    initial_max_force = float(np.linalg.norm(initial_force, axis=1).max())
    started = time.monotonic()
    result = relax_group(potential, [atoms])[0]
    elapsed = time.monotonic() - started
    path = output / "relaxed_structure.extxyz"
    ase.io.write(path, result["atoms"], format="extxyz")
    checked = ase.io.read(path)
    atomic_json(
        output / "relax_summary.json",
        {
            "success": True,
            "method": "Q3_E3_PCR",
            "pool_id": pool_id,
            "seed": seed,
            "gpu": gpu,
            "input_path": str(input_path),
            "input_hash": input_hash,
            "output_hash": structure_hash(checked),
            "initial_energy_ev": initial_energy,
            "initial_energy_per_atom_ev": initial_energy / len(atoms),
            "pre_relax_max_force_ev_ang": initial_max_force,
            "energy_ev": result["energy_ev"],
            "energy_per_atom_ev": result["energy_per_atom_ev"],
            "maximum_force_ev_ang": result["max_force_ev_ang"],
            "elapsed_seconds": elapsed,
            "steps": result["steps"],
            "converged": result["converged"],
        },
    )


def output_for(row: dict[str, Any]) -> Path:
    return Path(row["reused_output_dir"] or row["output_dir"])


def analyze() -> None:
    from pymatgen.io.ase import AseAtomsAdaptor

    state = load_progress()
    if state["complete"] != POOL_COUNT:
        raise RuntimeError("Q3 analysis requires 32 complete tasks")
    rows = []
    for row in sorted(state["tasks"], key=lambda item: int(item["pool_id"])):
        output = output_for(row)
        summary = read_json(output / "relax_summary.json")
        original = ase.io.read(row["input_path"])
        relaxed = ase.io.read(output / "relaxed_structure.extxyz")
        structure = AseAtomsAdaptor.get_structure(relaxed)
        rows.append(
            {
                "method": "Q3_E3_PCR",
                "seed": int(row["pool_id"]),
                "pool_id": int(row["pool_id"]),
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
                "_relaxed_atoms": relaxed,
                "_original_atoms": original,
            }
        )
    q3 = pd.DataFrame(rows).sort_values("pool_id").reset_index(drop=True)
    tool_root = Path("/data/dxl/tools/innovation2_next")
    sys.path.insert(0, str(tool_root))
    import analyze_corrector_64 as official

    official.ROOT = Path("/data/dxl")
    official.RESULT = Q3_RESULT
    official.REPORT = Q3_REPORT
    official.PROGRESS = Q3_RESULT
    official.REFERENCE = REFERENCE
    official.REFERENCE_LMDB = REFERENCE_LMDB
    official.CONFIGS = ("Q3_E3_PCR",)
    official.SEEDS = list(range(POOL_COUNT))
    official.STABILITY_THRESHOLD = 0.1
    metrics, errors = official.official_metrics({"Q3_E3_PCR": q3})
    if errors:
        raise RuntimeError(f"Q3 official metrics failure: {errors}")
    q3 = pd.read_csv(
        Q3_REPORT / "Q3_E3_PCR/official_metrics_per_structure.csv"
    ).sort_values("pool_id").reset_index(drop=True)
    baseline = pd.read_csv(
        REPORT / "C0_FIRST/official_metrics_per_structure.csv"
    ).sort_values("pool_id").reset_index(drop=True)
    baseline_force = []
    for pool_id in range(POOL_COUNT):
        summary = read_json(
            RELAXED
            / "C0_FIRST"
            / f"{pool_id:02d}"
            / "relax_summary.json"
        )
        baseline_force.append(float(summary["pre_relax_max_force_ev_ang"]))
    baseline["pre_relax_max_force_ev_ang"] = baseline_force
    stats = paired_statistics(baseline, q3)
    atomic_csv(Q3_REPORT / "paired_statistics.csv", stats)

    def means(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "ehull": float(frame["energy_above_hull_per_atom"].mean()),
            "rmsd": float(frame["rmsd_from_relaxation"].mean()),
            "stable": float(frame["stable"].astype(bool).mean()),
            "nus": float(frame["novel_unique_stable"].astype(bool).mean()),
            "composition_validity": float(
                frame["comp_validity"].astype(bool).mean()
            ),
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

    left = means(baseline)
    right = means(q3)
    changes = {
        "ehull": right["ehull"] - left["ehull"],
        "rmsd_relative": right["rmsd"] / left["rmsd"] - 1.0,
        "pre_relax_max_force_relative": (
            right["pre_relax_max_force_ev_ang"]
            / left["pre_relax_max_force_ev_ang"]
            - 1.0
        ),
        "stable": right["stable"] - left["stable"],
        "nus": right["nus"] - left["nus"],
        "composition_validity": (
            right["composition_validity"] - left["composition_validity"]
        ),
        "structure_validity": (
            right["structure_validity"] - left["structure_validity"]
        ),
        "novel": right["novel"] - left["novel"],
        "unique": right["unique"] - left["unique"],
        "converged": right["converged"] - left["converged"],
    }
    safety = (
        changes["structure_validity"] >= 0.0
        and changes["composition_validity"] >= -1.0 / POOL_COUNT
        and changes["stable"] >= -1.0 / POOL_COUNT
        and changes["nus"] >= -1.0 / POOL_COUNT
        and changes["novel"] >= -0.02
        and changes["unique"] >= -0.02
    )
    positive = (
        changes["ehull"] <= -0.005
        or changes["stable"] >= 1.0 / POOL_COUNT
        or changes["nus"] >= 1.0 / POOL_COUNT
        or changes["rmsd_relative"] <= -0.10
        or changes["pre_relax_max_force_relative"] <= -0.10
    )
    summary = {
        "schema_version": 1,
        "completed_at": now(),
        "candidate": "Q3_E3_PCR",
        "baseline": left,
        "refined": right,
        "changes": changes,
        "gates": {
            "safety_gate": bool(safety),
            "positive_gate": bool(positive),
            "Q3_E3_PCR_FINAL_GO": bool(safety and positive),
        },
        "official_metrics": metrics,
        "frozen_refinement_manifest": str(
            Q3_REPORT / "frozen_refinement_manifest.json"
        ),
        "training_summary": str(
            Q3_REPORT / "training_and_offline_summary.json"
        ),
        "paired_statistics": str(Q3_REPORT / "paired_statistics.csv"),
        "new_blind_labels_used_for_refinement": False,
        "sampling_trajectory_modified": False,
        "mattergen_backbone_trainable": False,
        "chgnet_trainable": False,
        "dft_verified": False,
    }
    atomic_json(Q3_REPORT / "final_summary.json", summary)
    table = pd.DataFrame(
        [{"method": "C0_FIRST", **left}, {"method": "Q3_E3_PCR", **right}]
    )
    (Q3_REPORT / "final_report.md").write_text(
        "# Q3 equivariant post-generation crystal refiner\n\n"
        f"- Final GO: `{summary['gates']['Q3_E3_PCR_FINAL_GO']}`\n"
        f"- Safety gate: `{summary['gates']['safety_gate']}`\n"
        f"- Positive gate: `{summary['gates']['positive_gate']}`\n"
        "- Original MatterGen sampling trajectory and backbone are unchanged.\n"
        "- Atomic species and lattice are unchanged; position updates are "
        "equivariant force-vector steps under a learned invariant scalar gate.\n\n"
        + table.to_markdown(index=False)
        + "\n\n"
        + pd.DataFrame([changes]).to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["gates"], sort_keys=True), flush=True)


def pipeline() -> None:
    commands = (
        [str(CHGNET_PYTHON), str(Path(__file__).resolve()), "refine"],
        [str(MATTERGEN_PYTHON), str(Path(__file__).resolve()), "relax"],
        [str(MATTERGEN_PYTHON), str(Path(__file__).resolve()), "analyze"],
    )
    for command in commands:
        subprocess.run(command, cwd=PROJECT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train")
    commands.add_parser("refine")
    commands.add_parser("relax")
    commands.add_parser("analyze")
    commands.add_parser("pipeline")
    task = commands.add_parser("relax-task")
    task.add_argument("--pool-id", type=int, required=True)
    task.add_argument("--seed", type=int, required=True)
    task.add_argument("--input-path", type=Path, required=True)
    task.add_argument("--input-hash", required=True)
    task.add_argument("--gpu", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "train":
        print(json.dumps(train_gate(), sort_keys=True))
    elif args.command == "refine":
        refine_new()
    elif args.command == "relax":
        run_relax()
    elif args.command == "analyze":
        analyze()
    elif args.command == "pipeline":
        pipeline()
    elif args.command == "relax-task":
        try:
            relax_task(
                args.pool_id,
                args.seed,
                args.input_path,
                args.input_hash,
                args.gpu,
            )
        except BaseException:
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
