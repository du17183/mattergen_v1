"""Evaluate independent held-out property, validity, trace behavior, and efficiency."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import ase.io
import ase.constraints
from ase.filters import ExpCellFilter
import numpy as np
import torch
import torch_geometric  # load the PyTorch-2.9-compatible build before adding legacy CHGNet
import torch_scatter

ase.constraints.ExpCellFilter = ExpCellFilter
sys.path.append("/mnt/datasets-livsyn/dxl/mattergen_v1/.venv/lib/python3.10/site-packages")
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/self_correcting_search_p0"
CHGNET_PATH = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar")
METHODS = ("B0", "S0")
SEEDS = tuple(range(98200, 98208))
TARGET = 0.2
TAU = 0.01


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def structural_validity(structure) -> tuple[bool, float]:
    if not np.isfinite(structure.volume) or structure.volume < 0.1 or len(structure) < 1:
        return False, float("nan")
    if len(structure) == 1:
        return True, float("inf")
    distances = np.asarray(structure.distance_matrix, dtype=float)
    distances[np.diag_indices_from(distances)] = np.inf
    minimum = float(np.min(distances))
    return bool(np.isfinite(minimum) and minimum >= 0.5), minimum


def main() -> None:
    structures = []
    metadata = []
    summaries = {}
    for method in METHODS:
        for seed in SEEDS:
            run_dir = ROOT / "search_generation" / method / str(seed)
            summary = json.loads((run_dir / "run_summary.json").read_text())
            atoms = ase.io.read(run_dir / "generated_crystals.extxyz", index=":")
            if len(atoms) != 1 or summary["method"] != method or int(summary["seed"]) != seed:
                raise RuntimeError(f"heldout artifact mismatch: {method} {seed}")
            structure = AseAtomsAdaptor.get_structure(atoms[0])
            structures.append(structure)
            metadata.append((method, seed, structure))
            summaries[(method, seed)] = summary
    model = CHGNet.from_file(CHGNET_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = model.predict_structure(structures, task="efsm", batch_size=16)
    if isinstance(predictions, dict):
        predictions = [predictions]

    property_rows = []
    quality_rows = []
    for (method, seed, structure), prediction in zip(metadata, predictions):
        value = float(np.abs(np.asarray(prediction["m"], dtype=float)).sum() / structure.volume)
        error = abs(value - TARGET)
        valid, minimum = structural_validity(structure)
        property_rows.append({
            "method": method,
            "seed": seed,
            "formula": structure.composition.reduced_formula,
            "chgnet_mag_density": value,
            "target_mag_density": TARGET,
            "mag_absolute_error": error,
            "mag_hit": error <= TAU,
            "tau": TAU,
            "surrogate_property_eval": True,
            "surrogate_calibration_mae": 0.001449,
            "surrogate_calibration_pearson": 0.9499,
            "dft_verified": False,
        })
        quality_rows.append({
            "method": method,
            "seed": seed,
            "formula": structure.composition.reduced_formula,
            "num_atoms": len(structure),
            "volume_angstrom3": float(structure.volume),
            "minimum_periodic_distance_angstrom": minimum,
            "structure_valid": valid,
            "validity_definition": "volume>=0.1A3_and_min_periodic_distance>=0.5A",
            "mattersim_run": False,
            "dft_verified": False,
        })
    write_csv(ROOT / "heldout_property_metrics.csv", property_rows)
    write_csv(ROOT / "quality_metrics.csv", quality_rows)

    aggregate = {}
    for method in METHODS:
        selected = [row for row in property_rows if row["method"] == method]
        quality = [row for row in quality_rows if row["method"] == method]
        aggregate[method] = {
            "n": len(selected),
            "mean_mag_density": float(np.mean([row["chgnet_mag_density"] for row in selected])),
            "mag_mae": float(np.mean([row["mag_absolute_error"] for row in selected])),
            "mag_median_error": float(np.median([row["mag_absolute_error"] for row in selected])),
            "mag_hits": int(sum(row["mag_hit"] for row in selected)),
            "valid_count": int(sum(row["structure_valid"] for row in quality)),
            "valid_fraction": float(np.mean([row["structure_valid"] for row in quality])),
        }
    b_by_seed = {row["seed"]: row for row in property_rows if row["method"] == "B0"}
    s_rows = [row for row in property_rows if row["method"] == "S0"]
    paired_deltas = [
        row["mag_absolute_error"] - b_by_seed[row["seed"]]["mag_absolute_error"]
        for row in s_rows
    ]
    improved = sum(delta < -1e-12 for delta in paired_deltas)
    same = sum(abs(delta) <= 1e-12 for delta in paired_deltas)
    worse = sum(delta > 1e-12 for delta in paired_deltas)
    if improved + same + worse != len(SEEDS):
        raise RuntimeError("paired comparison count mismatch")
    reduction = 1.0 - aggregate["S0"]["mag_mae"] / aggregate["B0"]["mag_mae"]
    extra_hits = aggregate["S0"]["mag_hits"] - aggregate["B0"]["mag_hits"]
    validity_drop = aggregate["B0"]["valid_fraction"] - aggregate["S0"]["valid_fraction"]
    validity_ok = validity_drop <= 0.125

    with (ROOT / "search_trace.csv").open(newline="", encoding="utf-8") as stream:
        trace = list(csv.DictReader(stream))
    accepted_deltas = [float(row["score_delta"]) for row in trace if row["accepted"].lower() == "true"]
    accepted_count = sum(row["accepted"].lower() == "true" for row in trace)
    acceptance_rate = accepted_count / len(trace) if trace else 0.0
    probe_local_improves = bool(accepted_deltas and np.mean(accepted_deltas) < 0)
    evaluator_mismatch = bool(probe_local_improves and reduction <= 0)
    property_go = bool(reduction >= 0.15 and improved >= 6 and extra_hits >= 1 and validity_ok)
    strong_go = bool(property_go and reduction >= 0.25 and extra_hits >= 2)
    definite_fail = bool(reduction < 0.08 or improved <= 4 or evaluator_mismatch)
    status = "GO" if property_go else ("FAIL" if definite_fail else "BORDERLINE")

    efficiency_rows = []
    method_efficiency = {}
    for method in METHODS:
        selected = [summaries[(method, seed)] for seed in SEEDS]
        values = {
            "method": method,
            "n": len(selected),
            "mean_runtime_seconds": float(np.mean([row["elapsed_seconds"] for row in selected])),
            "total_runtime_seconds": float(np.sum([row["elapsed_seconds"] for row in selected])),
            "peak_vram_bytes_max": int(max(row["peak_allocated_bytes"] for row in selected)),
            "mattergen_score_calls_mean": float(np.mean([row["mattergen_score_calls"] for row in selected])),
            "probe_extra_forward_calls_mean": float(np.mean([row.get("probe_extra_conditional_forward_calls", 0) for row in selected])),
            "probe_verification_seconds_mean": float(np.mean([row.get("probe_verification_seconds", 0) for row in selected])),
            "probe_scored_proposals_mean": float(np.mean([row.get("probe_scored_proposals", 0) for row in selected])),
            "acceptance_rate_mean": float(np.mean([row.get("acceptance_rate", 0) for row in selected])),
            "rejection_rate_mean": float(np.mean([1.0 - row.get("acceptance_rate", 0) for row in selected])),
        }
        method_efficiency[method] = values
        efficiency_rows.append(values)
    slowdown = method_efficiency["S0"]["mean_runtime_seconds"] / method_efficiency["B0"]["mean_runtime_seconds"]
    for row in efficiency_rows:
        row["slowdown_vs_B0"] = 1.0 if row["method"] == "B0" else slowdown
    write_csv(ROOT / "efficiency_metrics.csv", efficiency_rows)

    decision = json.loads((ROOT / "decision_summary.json").read_text())
    decision.update({
        "SELF_CORRECTING_SEARCH_P0": status,
        "NEXT": "RELIABILITY_AWARE_SC" if status == "GO" else "STOP",
        "property_go": property_go,
        "strong_go": strong_go,
        "mean_mag_mae_reduction_fraction": reduction,
        "improved_seed_count": int(improved),
        "same_seed_count": int(same),
        "worse_seed_count": int(worse),
        "extra_hit_count": int(extra_hits),
        "validity_drop_fraction": validity_drop,
        "validity_guardrail_pass": validity_ok,
        "probe_local_accepted_mean_score_delta": float(np.mean(accepted_deltas)) if accepted_deltas else None,
        "probe_exploitation_or_evaluator_mismatch": evaluator_mismatch,
        "property_aggregate": aggregate,
        "search_trace_summary": {
            "scored_proposals": len(trace),
            "accepted_proposals": int(accepted_count),
            "rejected_proposals": int(len(trace) - accepted_count),
            "acceptance_rate": acceptance_rate,
            "rejection_rate": 1.0 - acceptance_rate if trace else 0.0,
        },
        "efficiency": method_efficiency,
        "slowdown_vs_B0": slowdown,
        "mattersim_authorized_by_property_go": property_go,
        "mattersim_run": False,
    })
    (ROOT / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
