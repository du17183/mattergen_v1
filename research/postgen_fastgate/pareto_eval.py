#!/usr/bin/env python3
"""Frozen Q5 condition-quality Pareto selector and blind evaluation."""

from __future__ import annotations

import argparse
import json
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
import numpy as np
import pandas as pd
import torch

from research.postgen_fastgate.model import QualityNetwork
from research.postgen_fastgate.new_eval import (
    GENERATION,
    METHODS,
    POOL_COUNT,
    POOL_SIZE,
    PROJECT,
    REFERENCE,
    REFERENCE_LMDB,
    RELAXED,
    RESULT,
    REPORT,
    SEEDS,
    atomic_csv,
    atomic_json,
    now,
    paired_statistics,
    read_json,
    sha256,
    structure_hash,
)
from research.postgen_fastgate.train_quality import (
    BINARY_TARGETS,
    CONTINUOUS_TARGETS,
    inverse_continuous,
)


Q5_RESULT = Path("/data/dxl/results/postgen_fastgate/q5_new_eval")
Q5_REPORT = Path("/data/dxl/reports/postgen_fastgate/q5_new_eval")
Q5_LOG = Path("/data/dxl/logs/postgen_fastgate/q5_new_eval")
Q5_RELAXED = Q5_RESULT / "relaxed"
Q5_SELECTION = Q5_RESULT / "selection.csv"
Q5_SCORES = Q5_RESULT / "pool_scores.csv"
Q5_PROGRESS = Q5_RESULT / "relax_progress.json"
FEATURES = RESULT / "candidate_features.csv"
HISTORICAL_FEATURES = Path(
    "/data/dxl/results/postgen_fastgate/features/historical_features.csv"
)
HISTORICAL_PREDICTIONS = Path(
    "/data/dxl/results/postgen_fastgate/quality_model/predictions.csv"
)
QUALITY_ROOT = Path(
    "/data/dxl/results/postgen_fastgate/quality_model/checkpoints"
)
MATTERGEN_PYTHON = Path("/data/dxl/envs/mattergen_py310/bin/python")
MATTERSIM = Path("/data/dxl/mattersim_weights/mattersim-v1.0.0-5M.pth")
RELAX_COMMON = Path("/data/dxl/tools/guidance_stage7_eval")

# Frozen solely from historical C0 validation; the blind MatterSim labels are not
# inputs to this selector.
NOVEL_UNIQUE_MARGIN = 0.10
COMPOSITION_MARGIN = 0.05
EHULL_TOLERANCE = 0.05
ENSEMBLE_STD_MULTIPLIER = 1.0
TARGET_MAG_DENSITY = 0.10
TARGET_HIT_RADIUS = 0.02


def add_quality_predictions(frame: pd.DataFrame, device: str) -> pd.DataFrame:
    checkpoints = sorted(QUALITY_ROOT.glob("quality_member_*.pt"))
    if len(checkpoints) != 5:
        raise RuntimeError("frozen quality ensemble must contain five members")
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    metadata = first["metadata"]
    columns = metadata["feature_columns"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing frozen feature columns: {missing}")
    features = frame.loc[:, columns].to_numpy(float)
    features = (
        features - np.asarray(metadata["feature_mean"], dtype=float)
    ) / np.asarray(metadata["feature_std"], dtype=float)
    tensor = torch.tensor(features, dtype=torch.float32, device=device)
    continuous = []
    probabilities = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        model = QualityNetwork(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload["dropout"]),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            output = model(tensor)
        standardized = output["continuous"].cpu().numpy()
        transformed = (
            standardized * np.asarray(metadata["continuous_std"], dtype=float)
            + np.asarray(metadata["continuous_mean"], dtype=float)
        )
        continuous.append(inverse_continuous(transformed))
        probabilities.append(
            torch.sigmoid(output["binary_logits"]).cpu().numpy()
        )
    continuous_stack = np.stack(continuous)
    probability_stack = np.stack(probabilities)
    result = frame.copy()
    for index, target in enumerate(CONTINUOUS_TARGETS):
        result[f"pred_{target}"] = continuous_stack[:, :, index].mean(axis=0)
        result[f"std_{target}"] = continuous_stack[:, :, index].std(axis=0)
    for index, target in enumerate(BINARY_TARGETS):
        result[f"prob_{target}"] = probability_stack[:, :, index].mean(axis=0)
        result[f"std_prob_{target}"] = probability_stack[:, :, index].std(
            axis=0
        )
    return result


def pareto_scores(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    novelty_lcb = (
        frame["prob_novel"].to_numpy(float)
        - ENSEMBLE_STD_MULTIPLIER
        * frame["std_prob_novel"].to_numpy(float)
    )
    unique_lcb = (
        frame["prob_unique"].to_numpy(float)
        - ENSEMBLE_STD_MULTIPLIER
        * frame["std_prob_unique"].to_numpy(float)
    )
    composition_lcb = (
        frame["prob_comp_validity"].to_numpy(float)
        - ENSEMBLE_STD_MULTIPLIER
        * frame["std_prob_comp_validity"].to_numpy(float)
    )
    ehull_ucb = (
        frame["pred_energy_above_hull_per_atom"].to_numpy(float)
        + ENSEMBLE_STD_MULTIPLIER
        * frame["std_energy_above_hull_per_atom"].to_numpy(float)
    )
    target_error = np.abs(
        frame["chgnet_mag_density"].to_numpy(float) - TARGET_MAG_DENSITY
    )
    nus_lcb = (
        frame["prob_novel_unique_stable"].to_numpy(float)
        - ENSEMBLE_STD_MULTIPLIER
        * frame["std_prob_novel_unique_stable"].to_numpy(float)
    )
    stable_lcb = (
        frame["prob_stable"].to_numpy(float)
        - ENSEMBLE_STD_MULTIPLIER
        * frame["std_prob_stable"].to_numpy(float)
    )
    utility = (
        -target_error / TARGET_HIT_RADIUS
        + 0.75 * nus_lcb
        + 0.25 * stable_lcb
        - 0.20 * ehull_ucb / 0.10
    )
    auxiliary = np.stack(
        [
            novelty_lcb,
            unique_lcb,
            composition_lcb,
            ehull_ucb,
            target_error,
        ],
        axis=1,
    )
    return utility, auxiliary


def select_pools(frame: pd.DataFrame, pools: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    utility, auxiliary = pareto_scores(frame)
    novelty, unique, composition, ehull, _target = auxiliary.T
    baseline = pools[:, 0]
    eligible = (
        (novelty[pools] >= novelty[baseline, None] - NOVEL_UNIQUE_MARGIN)
        & (unique[pools] >= unique[baseline, None] - NOVEL_UNIQUE_MARGIN)
        & (
            composition[pools]
            >= composition[baseline, None] - COMPOSITION_MARGIN
        )
        & (ehull[pools] <= ehull[baseline, None] + EHULL_TOLERANCE)
    )
    if not np.all(eligible[:, 0]):
        raise RuntimeError("baseline candidate must always be Pareto eligible")
    masked = np.where(eligible, utility[pools], -1.0e9)
    selected_position = masked.argmax(axis=1)
    selected = np.take_along_axis(
        pools, selected_position[:, None], axis=1
    )[:, 0]
    return selected, eligible


def historical_summary(frame: pd.DataFrame, split: str) -> dict[str, Any]:
    selected_frame = frame[
        (frame["method"] == "C0") & (frame["split"] == split)
    ].reset_index(drop=True)
    rng = np.random.default_rng(20260728)
    pools = np.stack(
        [
            rng.permutation(len(selected_frame)).reshape(-1, POOL_SIZE)
            for _ in range(1000)
        ]
    )
    flat_pools = pools.reshape(-1, POOL_SIZE)
    selected, _eligible = select_pools(selected_frame, flat_pools)
    selected = selected.reshape(pools.shape[:2])
    baseline = pools[:, :, 0]

    def array(column: str) -> np.ndarray:
        return selected_frame[column].to_numpy(float)

    def change(column: str) -> np.ndarray:
        values = array(column)
        return values[selected].mean(axis=1) - values[baseline].mean(axis=1)

    target_error = np.abs(
        array("chgnet_mag_density") - TARGET_MAG_DENSITY
    )
    hit_change = (
        (target_error[selected] <= TARGET_HIT_RADIUS).mean(axis=1)
        - (target_error[baseline] <= TARGET_HIT_RADIUS).mean(axis=1)
    )
    changes = {
        "ehull": change("energy_above_hull_per_atom"),
        "stable": change("stable"),
        "nus": change("novel_unique_stable"),
        "novel": change("novel"),
        "unique": change("unique"),
        "composition_validity": change("comp_validity"),
        "target_hit_0_02": hit_change,
        "nonbaseline_selection_rate": (selected != baseline).mean(axis=1),
    }
    summary = {
        name: {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
        }
        for name, values in changes.items()
    }
    safety = (
        summary["stable"]["mean"] >= -1.0 / POOL_COUNT
        and summary["nus"]["mean"] >= -1.0 / POOL_COUNT
        and summary["novel"]["mean"] >= -0.02
        and summary["unique"]["mean"] >= -0.02
        and summary["composition_validity"]["mean"] >= -1.0 / POOL_COUNT
    )
    positive = (
        summary["ehull"]["mean"] <= -0.005
        or summary["stable"]["mean"] >= 1.0 / POOL_COUNT
        or summary["nus"]["mean"] >= 1.0 / POOL_COUNT
        or summary["target_hit_0_02"]["mean"] >= 0.10
    )
    return {
        "split": split,
        "rows": len(selected_frame),
        "trials": len(pools),
        "summary": summary,
        "safety_gate": bool(safety),
        "positive_gate": bool(positive),
        "go": bool(safety and positive),
    }


def select_new(device: str) -> None:
    historical = pd.read_csv(HISTORICAL_FEATURES).merge(
        pd.read_csv(HISTORICAL_PREDICTIONS),
        on=["method", "seed", "split"],
        validate="one_to_one",
    )
    offline = {
        "schema_version": 1,
        "frozen_parameters": frozen_parameters(),
        "validation": historical_summary(historical, "validation"),
        "test": historical_summary(historical, "test"),
        "new_blind_labels_used": False,
    }
    if not offline["validation"]["go"] or not offline["test"]["go"]:
        raise RuntimeError("Q5 failed historical validation/test gate")
    atomic_json(Q5_REPORT / "historical_offline_summary.json", offline)

    raw = pd.read_csv(FEATURES).sort_values("seed").reset_index(drop=True)
    if raw["seed"].astype(int).tolist() != list(SEEDS):
        raise RuntimeError("new candidate feature order mismatch")
    frame = add_quality_predictions(raw, device)
    pools = np.arange(len(frame)).reshape(POOL_COUNT, POOL_SIZE)
    selected, eligible = select_pools(frame, pools)
    utility, auxiliary = pareto_scores(frame)
    target_error = auxiliary[:, 4]
    rows = []
    selections = []
    for pool_id, indexes in enumerate(pools):
        chosen = int(selected[pool_id])
        for position, index in enumerate(indexes):
            rows.append(
                {
                    "pool_id": pool_id,
                    "candidate_index": position,
                    "seed": int(frame.loc[index, "seed"]),
                    "pareto_eligible": bool(eligible[pool_id, position]),
                    "utility": float(utility[index]),
                    "target_error": float(target_error[index]),
                    "target_hit_0_02": bool(
                        target_error[index] <= TARGET_HIT_RADIUS
                    ),
                    "selected": index == chosen,
                    "baseline": position == 0,
                }
            )
        selections.append(
            {
                "pool_id": pool_id,
                "baseline_seed": int(frame.loc[indexes[0], "seed"]),
                "selected_seed": int(frame.loc[chosen, "seed"]),
                "selected_candidate_index": int(
                    np.flatnonzero(indexes == chosen)[0]
                ),
                "baseline_target_hit_0_02": bool(
                    target_error[indexes[0]] <= TARGET_HIT_RADIUS
                ),
                "selected_target_hit_0_02": bool(
                    target_error[chosen] <= TARGET_HIT_RADIUS
                ),
            }
        )
    selection = pd.DataFrame(selections)
    atomic_csv(Q5_SCORES, pd.DataFrame(rows))
    atomic_csv(Q5_SELECTION, selection)
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "selector": "Q5_CQPS",
        "parameters": frozen_parameters(),
        "selected_before_q5_mattersim": True,
        "new_blind_labels_used": False,
        "historical_offline_summary": str(
            Q5_REPORT / "historical_offline_summary.json"
        ),
        "selection_sha256": sha256(Q5_SELECTION),
        "nonbaseline_selected": int(
            (selection["selected_candidate_index"] != 0).sum()
        ),
        "baseline_hit_rate": float(
            selection["baseline_target_hit_0_02"].mean()
        ),
        "selected_hit_rate": float(
            selection["selected_target_hit_0_02"].mean()
        ),
        "hit_rate_change": float(
            selection["selected_target_hit_0_02"].mean()
            - selection["baseline_target_hit_0_02"].mean()
        ),
        "mattergen_backbone_trainable": False,
        "sampling_trajectory_modified": False,
    }
    atomic_json(Q5_REPORT / "frozen_selection_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)


def frozen_parameters() -> dict[str, float]:
    return {
        "novel_unique_margin": NOVEL_UNIQUE_MARGIN,
        "composition_margin": COMPOSITION_MARGIN,
        "ehull_tolerance": EHULL_TOLERANCE,
        "ensemble_std_multiplier": ENSEMBLE_STD_MULTIPLIER,
        "target_mag_density": TARGET_MAG_DENSITY,
        "target_hit_radius": TARGET_HIT_RADIUS,
        "target_weight": 1.0 / TARGET_HIT_RADIUS,
        "nus_weight": 0.75,
        "stable_weight": 0.25,
        "ehull_weight": 0.20 / 0.10,
    }


def reuse_source(pool_id: int, selected_seed: int) -> Path | None:
    q6 = pd.read_csv(RESULT / "selection.csv")
    row = q6[q6["pool_id"] == pool_id].iloc[0]
    if selected_seed == int(row["baseline_seed"]):
        return RELAXED / "C0_FIRST" / f"{pool_id:02d}"
    if selected_seed == int(row["selected_seed"]):
        return RELAXED / "Q6_NS_SETRANK" / f"{pool_id:02d}"
    return None


def relax_task_rows() -> list[dict[str, Any]]:
    selection = pd.read_csv(Q5_SELECTION)
    rows = []
    for item in selection.itertuples(index=False):
        pool_id = int(item.pool_id)
        selected_seed = int(item.selected_seed)
        reused = reuse_source(pool_id, selected_seed)
        rows.append(
            {
                "task_id": f"Q5_CQPS_pool_{pool_id:02d}",
                "pool_id": pool_id,
                "selected_seed": selected_seed,
                "input_path": str(
                    GENERATION / str(selected_seed) / "generated_crystals.extxyz"
                ),
                "output_dir": str(Q5_RELAXED / f"{pool_id:02d}"),
                "reused_output_dir": str(reused) if reused else "",
                "status": "reused" if reused else "pending",
                "attempt": 0,
                "gpu": None,
                "elapsed_seconds": None,
                "error": "",
            }
        )
    return rows


def validate_new_relax(row: dict[str, Any]) -> bool:
    try:
        output = Path(row["output_dir"])
        summary = read_json(output / "relax_summary.json")
        atoms = ase.io.read(output / "relaxed_structure.extxyz")
        return (
            summary["success"] is True
            and int(summary["pool_id"]) == int(row["pool_id"])
            and int(summary["selected_seed"]) == int(row["selected_seed"])
            and np.isfinite(atoms.positions).all()
            and np.isfinite(atoms.cell.array).all()
        )
    except BaseException:
        return False


def load_relax_progress() -> dict[str, Any]:
    expected = relax_task_rows()
    state = (
        read_json(Q5_PROGRESS)
        if Q5_PROGRESS.is_file()
        else {"schema_version": 1, "created_at": now(), "tasks": expected}
    )
    if [row["task_id"] for row in state["tasks"]] != [
        row["task_id"] for row in expected
    ]:
        raise RuntimeError("Q5 relax task contract mismatch")
    for row, contract in zip(state["tasks"], expected, strict=True):
        row["reused_output_dir"] = contract["reused_output_dir"]
        if contract["reused_output_dir"]:
            row["status"] = "reused"
        elif validate_new_relax(row):
            row["status"] = "success"
        elif row["status"] == "running":
            row["status"] = "interrupted"
    save_relax_progress(state)
    return state


def save_relax_progress(state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    state["reused"] = sum(row["status"] == "reused" for row in state["tasks"])
    state["success"] = sum(row["status"] == "success" for row in state["tasks"])
    state["complete"] = state["reused"] + state["success"]
    atomic_json(Q5_PROGRESS, state)
    atomic_csv(Q5_RESULT / "relax_progress.csv", pd.DataFrame(state["tasks"]))


def run_relax_one(row: dict[str, Any], gpu: int) -> dict[str, Any]:
    command = [
        str(MATTERGEN_PYTHON),
        str(Path(__file__).resolve()),
        "relax-task",
        "--pool-id",
        str(row["pool_id"]),
        "--selected-seed",
        str(row["selected_seed"]),
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
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def run_relax() -> None:
    state = load_relax_progress()
    for attempt in range(2):
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
            save_relax_progress(state)
            for future in as_completed(futures):
                row = futures[future]
                result = future.result()
                valid = (
                    result["return_code"] == 0 and validate_new_relax(row)
                )
                row.update(
                    status="success" if valid else "failed",
                    elapsed_seconds=result["elapsed_seconds"],
                    error="" if valid else result["stderr"][-4000:],
                )
                save_relax_progress(state)
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
        state = load_relax_progress()
    if state["complete"] != POOL_COUNT:
        raise RuntimeError(
            f"Q5 relaxation incomplete: {state['complete']}/{POOL_COUNT}"
        )


def relax_task(pool_id: int, selected_seed: int, gpu: int) -> None:
    from mattersim.forcefield import MatterSimCalculator

    sys.path.insert(0, str(RELAX_COMMON))
    from relax_common import load_potential, relax_group

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    input_path = GENERATION / str(selected_seed) / "generated_crystals.extxyz"
    output = Q5_RELAXED / f"{pool_id:02d}"
    if output.exists() and validate_new_relax(
        {
            "output_dir": str(output),
            "pool_id": pool_id,
            "selected_seed": selected_seed,
        }
    ):
        return
    if output.exists():
        os.replace(output, output.with_name(f"{output.name}.incomplete.{os.getpid()}"))
    output.mkdir(parents=True)
    atoms = ase.io.read(input_path)
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
            "method": "Q5_CQPS",
            "pool_id": pool_id,
            "selected_seed": selected_seed,
            "gpu": gpu,
            "input_path": str(input_path),
            "input_hash": structure_hash(atoms),
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


def q5_output(row: dict[str, Any]) -> Path:
    return Path(row["reused_output_dir"] or row["output_dir"])


def analyze() -> None:
    from pymatgen.io.ase import AseAtomsAdaptor

    state = load_relax_progress()
    if state["complete"] != POOL_COUNT:
        raise RuntimeError("Q5 analysis requires 32 complete selections")
    rows = []
    for row in sorted(state["tasks"], key=lambda item: int(item["pool_id"])):
        output = q5_output(row)
        summary = read_json(output / "relax_summary.json")
        original = ase.io.read(row["input_path"])
        relaxed = ase.io.read(output / "relaxed_structure.extxyz")
        structure = AseAtomsAdaptor.get_structure(relaxed)
        rows.append(
            {
                "method": "Q5_CQPS",
                "seed": int(row["pool_id"]),
                "pool_id": int(row["pool_id"]),
                "candidate_seed": int(row["selected_seed"]),
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
    q5 = pd.DataFrame(rows).sort_values("pool_id").reset_index(drop=True)
    tool_root = Path("/data/dxl/tools/innovation2_next")
    sys.path.insert(0, str(tool_root))
    import analyze_corrector_64 as official

    official.ROOT = Path("/data/dxl")
    official.RESULT = Q5_RESULT
    official.REPORT = Q5_REPORT
    official.PROGRESS = Q5_RESULT
    official.REFERENCE = REFERENCE
    official.REFERENCE_LMDB = REFERENCE_LMDB
    official.CONFIGS = ("Q5_CQPS",)
    official.SEEDS = list(range(POOL_COUNT))
    official.STABILITY_THRESHOLD = 0.1
    metrics, errors = official.official_metrics({"Q5_CQPS": q5})
    if errors:
        raise RuntimeError(f"Q5 official metrics failure: {errors}")
    q5 = pd.read_csv(Q5_REPORT / "Q5_CQPS/official_metrics_per_structure.csv")

    baseline = pd.read_csv(
        REPORT / "C0_FIRST/official_metrics_per_structure.csv"
    ).sort_values("pool_id").reset_index(drop=True)
    force = []
    for pool_id in range(POOL_COUNT):
        summary = read_json(
            RELAXED
            / "C0_FIRST"
            / f"{pool_id:02d}"
            / "relax_summary.json"
        )
        force.append(float(summary["pre_relax_max_force_ev_ang"]))
    baseline["pre_relax_max_force_ev_ang"] = force
    q5 = q5.sort_values("pool_id").reset_index(drop=True)
    stats = paired_statistics(baseline, q5)
    atomic_csv(Q5_REPORT / "paired_statistics.csv", stats)

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
    right = means(q5)
    selection = pd.read_csv(Q5_SELECTION)
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
        "target_hit_0_02": float(
            selection["selected_target_hit_0_02"].mean()
            - selection["baseline_target_hit_0_02"].mean()
        ),
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
        or changes["target_hit_0_02"] >= 0.10
    )
    summary = {
        "schema_version": 1,
        "completed_at": now(),
        "candidate": "Q5_CQPS",
        "baseline": left,
        "selected": right,
        "changes": changes,
        "gates": {
            "safety_gate": bool(safety),
            "positive_gate": bool(positive),
            "Q5_CQPS_FINAL_GO": bool(safety and positive),
        },
        "official_metrics": metrics,
        "frozen_selection_manifest": str(
            Q5_REPORT / "frozen_selection_manifest.json"
        ),
        "paired_statistics": str(Q5_REPORT / "paired_statistics.csv"),
        "chgnet_mag_oracle_previously_verified": True,
        "target_hit_is_chgnet_surrogate_not_dft": True,
        "new_blind_labels_used_for_selection": False,
        "sampling_trajectory_modified": False,
        "mattergen_backbone_trainable": False,
        "dft_verified": False,
    }
    atomic_json(Q5_REPORT / "final_summary.json", summary)
    report = pd.DataFrame(
        [{"method": "C0_FIRST", **left}, {"method": "Q5_CQPS", **right}]
    )
    (Q5_REPORT / "final_report.md").write_text(
        "# Q5 CQPS 32-pool blind evaluation\n\n"
        f"- Final GO: `{summary['gates']['Q5_CQPS_FINAL_GO']}`\n"
        f"- Safety gate: `{summary['gates']['safety_gate']}`\n"
        f"- Positive gate: `{summary['gates']['positive_gate']}`\n"
        "- CHGNet target hit is a validated surrogate, not DFT proof.\n"
        "- New MatterSim labels were not used for selection.\n\n"
        + report.to_markdown(index=False)
        + "\n\n"
        + pd.DataFrame([changes]).to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["gates"], sort_keys=True), flush=True)


def pipeline(device: str) -> None:
    select_new(device)
    run_relax()
    analyze()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--device", default="cuda")
    commands.add_parser("relax")
    commands.add_parser("analyze")
    pipeline_parser = commands.add_parser("pipeline")
    pipeline_parser.add_argument("--device", default="cuda")
    task = commands.add_parser("relax-task")
    task.add_argument("--pool-id", type=int, required=True)
    task.add_argument("--selected-seed", type=int, required=True)
    task.add_argument("--gpu", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "select":
        select_new(args.device)
    elif args.command == "relax":
        run_relax()
    elif args.command == "analyze":
        analyze()
    elif args.command == "pipeline":
        pipeline(args.device)
    elif args.command == "relax-task":
        try:
            relax_task(args.pool_id, args.selected_seed, args.gpu)
        except BaseException:
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
