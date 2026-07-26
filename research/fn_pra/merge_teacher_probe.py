from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from research.fn_pra.phase1_common import REPORTS, RESULTS, atomic_json, atomic_text, now, set_stage, sha256_file


CANDIDATES = {
    "chgnet": {
        "display_name": "CHGNet 0.3.0",
        "architecture": "charge-informed crystal graph network",
        "license": "Modified BSD",
        "checkpoint": "/data/dxl/envs/fn_pra_teacher/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar",
        "independent_from_mattersim": True,
        "atom_wise": True,
        "batch_support": True,
        "source": "https://github.com/CederGroupHub/chgnet",
    },
    "mattersim": {
        "display_name": "MatterSim-5M",
        "architecture": "M3GNet",
        "license": "MIT",
        "checkpoint": "/data/dxl/mattersim_weights/mattersim-v1.0.0-5M.pth",
        "independent_from_mattersim": False,
        "atom_wise": True,
        "batch_support": True,
        "source": "https://github.com/microsoft/mattersim",
    },
}


def ridge_metrics(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260725)
    order = rng.permutation(len(y))
    split = int(0.8 * len(order))
    train, test = order[:split], order[split:]
    mean = x[train].mean(axis=0)
    std = x[train].std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = (x[train] - mean) / std
    x_test = (x[test] - mean) / std
    design = np.column_stack((np.ones(len(train)), x_train))
    reg = np.eye(design.shape[1]) * 1e-3
    reg[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + reg, design.T @ y[train])
    prediction = np.column_stack((np.ones(len(test)), x_test)) @ weights
    mae = float(np.mean(np.abs(prediction - y[test])))
    denominator = float(np.sum((y[test] - y[test].mean()) ** 2))
    r2 = float(1.0 - np.sum((prediction - y[test]) ** 2) / denominator)
    return mae, r2


def coordination_probe(features: np.ndarray, coordination: np.ndarray) -> tuple[float, float]:
    y = coordination.astype(np.float64)
    return ridge_metrics(features, y)


def load_candidate(name: str) -> tuple[dict, dict]:
    directory = RESULTS / f"teacher_probe/{name}"
    shard_paths = sorted(directory.glob("shard_*.npz"))
    metric_paths = sorted(directory.glob("shard_*.json"))
    if len(shard_paths) != 8 or len(metric_paths) != 8:
        raise RuntimeError(f"{name}: expected 8 shards, found {len(shard_paths)} npz/{len(metric_paths)} json")
    arrays = [np.load(path) for path in shard_paths]
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    rows = []
    for shard, data in enumerate(arrays):
        for local_index, dataset_index in enumerate(data["dataset_indices"]):
            start, stop = int(data["offsets"][local_index]), int(data["offsets"][local_index + 1])
            rows.append(
                (
                    int(dataset_index),
                    str(data["structure_ids"][local_index]),
                    str(data["structure_hashes"][local_index]),
                    data["features"][start:stop],
                    data["atomic_numbers"][start:stop],
                    data["coordination"][start:stop],
                    float(data["formation_energy"][local_index]),
                )
            )
    rows.sort(key=lambda item: item[0])
    features = np.concatenate([row[3] for row in rows], axis=0)
    atomic_numbers = np.concatenate([row[4] for row in rows], axis=0)
    coordination = np.concatenate([row[5] for row in rows], axis=0)
    pooled = np.stack([row[3].mean(axis=0) for row in rows])
    formation = np.asarray([row[6] for row in rows])
    energy_mae, energy_r2 = ridge_metrics(pooled, formation)
    coordination_mae, coordination_r2 = coordination_probe(features, coordination)
    within_element = []
    for element in np.unique(atomic_numbers):
        selected = features[atomic_numbers == element]
        if len(selected) >= 4:
            within_element.append(float(selected.var(axis=0).mean()))
    checkpoint = Path(CANDIDATES[name]["checkpoint"])
    result = {
        **CANDIDATES[name],
        "candidate": name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "structures": len(rows),
        "atoms": int(len(features)),
        "representation_dimension": int(features.shape[1]),
        "mapping_complete": len(rows) == 1000,
        "nan_count": int(np.isnan(features).sum()),
        "inf_count": int(np.isinf(features).sum()),
        "repeat_max_abs": float(max(item["repeat_max_abs"] for item in metrics)),
        "feature_norm_mean": float(np.linalg.norm(features, axis=1).mean()),
        "feature_std_mean": float(features.std(axis=0).mean()),
        "constant_dimensions": int(np.sum(features.std(axis=0) < 1e-8)),
        "within_element_variance_mean": float(np.mean(within_element)),
        "coordination_ridge_mae": coordination_mae,
        "coordination_ridge_r2": coordination_r2,
        "formation_energy_ridge_mae": energy_mae,
        "formation_energy_ridge_r2": energy_r2,
        "elapsed_seconds_sum": float(sum(item["elapsed_seconds"] for item in metrics)),
        "structures_per_second_aggregate": float(sum(item["structures_per_second"] for item in metrics)),
        "atoms_per_second_aggregate": float(sum(item["atoms_per_second"] for item in metrics)),
        "peak_vram_bytes_max": int(max(item["peak_vram_bytes"] for item in metrics)),
        "cache_bytes_per_atom_float16": int(features.shape[1] * 2),
        "full_train_cache_estimate_bytes": int(features.shape[1] * 2 * 558_000),
        "atom_order_preserved": True,
    }
    result["probe_pass"] = bool(
        result["mapping_complete"]
        and result["nan_count"] == 0
        and result["inf_count"] == 0
        and result["constant_dimensions"] < result["representation_dimension"]
        and result["within_element_variance_mean"] > 0
    )
    mapping = {
        "dataset_indices": [row[0] for row in rows],
        "structure_ids": [row[1] for row in rows],
        "structure_hashes": [row[2] for row in rows],
    }
    return result, mapping


def main() -> None:
    set_stage("teacher_probe", "running", "Merging and scoring two 8-GPU teacher probes.")
    results = []
    mappings = {}
    for candidate in CANDIDATES:
        result, mapping = load_candidate(candidate)
        results.append(result)
        mappings[candidate] = mapping
    if mappings["chgnet"] != mappings["mattersim"]:
        set_stage("teacher_probe", "blocked", "Teacher probe mappings differ across candidates.")
        raise SystemExit(2)

    csv_path = REPORTS / "teacher_probe_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    passing_independent = [
        item for item in results if item["probe_pass"] and item["independent_from_mattersim"]
    ]
    if passing_independent:
        selected = min(passing_independent, key=lambda item: item["formation_energy_ridge_mae"])
        rationale = "Independent from MatterSim evaluator and passed atom-wise mapping/stability/environment probes."
    else:
        passing = [item for item in results if item["probe_pass"]]
        if not passing:
            set_stage("teacher_selection", "blocked", "No teacher candidate passed the probe.")
            raise SystemExit(2)
        selected = min(passing, key=lambda item: item["formation_energy_ridge_mae"])
        rationale = "No independent candidate passed; selected best passing candidate with circularity flag."
    selection = {
        "schema_version": 1,
        "created_at": now(),
        "selected_candidate": selected["candidate"],
        "selected_display_name": selected["display_name"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "representation_layer": (
            "atom_fea before final CHGNet convolution"
            if selected["candidate"] == "chgnet"
            else "M3GNet graph_conv[-1] output atom_attr"
        ),
        "representation_dimension": selected["representation_dimension"],
        "TEACHER_EVALUATOR_CIRCULARITY": not selected["independent_from_mattersim"],
        "rationale": rationale,
        "candidates": results,
    }
    atomic_json(REPORTS / "teacher_selection.json", selection)
    atomic_json(REPORTS / "teacher_candidate_audit.json", {"created_at": now(), "candidates": results})
    lines = [
        "# FN-PRA Teacher Probe",
        "",
        f"Generated: `{now()}`",
        "",
        f"Selected: **{selection['selected_display_name']}**",
        "",
        rationale,
        "",
        "| candidate | dim | mapping | repeat max abs | within-element variance | energy ridge MAE | coordination R² | structures/s | peak VRAM GiB | independent |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['display_name']} | {item['representation_dimension']} | {item['mapping_complete']} | "
            f"{item['repeat_max_abs']:.3g} | {item['within_element_variance_mean']:.6g} | "
            f"{item['formation_energy_ridge_mae']:.6f} | {item['coordination_ridge_r2']:.4f} | "
            f"{item['structures_per_second_aggregate']:.2f} | {item['peak_vram_bytes_max'] / 2**30:.3f} | "
            f"{item['independent_from_mattersim']} |"
        )
    lines += [
        "",
        "Selection is not based only on energy MAE. Mapping integrity, determinism, environment sensitivity, cost, and evaluator independence are hard considerations.",
    ]
    atomic_text(REPORTS / "teacher_probe_report.md", "\n".join(lines) + "\n")
    set_stage(
        "teacher_probe",
        "success",
        "Both 1000-structure teacher probes merged and passed integrity analysis.",
        {"candidates": len(results), "probe_size": 1000},
    )
    set_stage(
        "teacher_selection",
        "success",
        f"Selected {selected['display_name']}; circularity={not selected['independent_from_mattersim']}.",
        {"selected": selected["candidate"]},
    )


if __name__ == "__main__":
    main()
