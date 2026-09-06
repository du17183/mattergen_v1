"""Summarize frozen 8-seed TCL/DML P0 quality and paired sensitivity."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
METHODS = ("C0", "FT0", "TCL", "DML")
SEEDS = tuple(range(80000, 80008))
COMPARISONS = (("TCL", "FT0"), ("TCL", "C0"), ("DML", "FT0"), ("DML", "C0"))


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def load_method(method: str, generation: pd.DataFrame) -> dict[str, np.ndarray]:
    generated = generation[generation["method"] == method].sort_values("seed")
    if tuple(generated["seed"].astype(int)) != SEEDS:
        raise RuntimeError(f"{method}: generation seed order mismatch")
    if not generated["success"].astype(bool).all():
        raise RuntimeError(f"{method}: generation failure")
    with gzip.open(ROOT / "quality" / method / "official_detailed.json.gz", "rt") as stream:
        detail = json.load(stream)
    relaxation = json.loads(
        (ROOT / "relaxation" / method / "relaxation_summary.json").read_text()
    )
    if tuple(map(int, relaxation["sample_seeds"])) != SEEDS:
        raise RuntimeError(f"{method}: relaxation seed order mismatch")
    if relaxation["success"] is not True:
        raise RuntimeError(f"{method}: MatterSim relaxation failure")
    initial = read(
        ROOT / "relaxation" / method / "initial_with_properties.extxyz", index=":"
    )
    force_norms = [np.linalg.norm(item.get_forces(), axis=1) for item in initial]
    data = {
        "seed": np.asarray(SEEDS, dtype=int),
        "formula": generated["formula"].to_numpy(),
        "e_hull": np.asarray(detail["energy_above_hull_per_atom"], dtype=float),
        "stable": np.asarray(detail["stable"], dtype=float),
        "nus": np.asarray(detail["novel_unique_stable"], dtype=float),
        "novel": np.asarray(detail["novel"], dtype=float),
        "unique": np.asarray(detail["unique"], dtype=float),
        "rmsd": np.asarray(detail["rmsd_from_relaxation"], dtype=float),
        "atomic_force_mean": np.asarray([values.mean() for values in force_norms]),
        "structure_max_force": np.asarray([values.max() for values in force_norms]),
        "relaxation_steps": np.asarray(relaxation["relaxation_steps"], dtype=float),
        "generation_seconds": generated["elapsed_seconds"].to_numpy(dtype=float),
    }
    for key, values in data.items():
        if len(values) != len(SEEDS):
            raise RuntimeError(f"{method}: {key} count mismatch")
    return data


def summary_row(method: str, values: dict[str, np.ndarray]) -> dict[str, object]:
    e_hull = values["e_hull"]
    rmsd = values["rmsd"]
    max_force = values["structure_max_force"]
    steps = values["relaxation_steps"]
    return {
        "method": method,
        "n": len(SEEDS),
        "generation_success_rate": 1.0,
        "mattersim_success_rate": 1.0,
        "dft_verified": False,
        "e_hull_mean_ev_per_atom": float(e_hull.mean()),
        "e_hull_median_ev_per_atom": float(np.median(e_hull)),
        "e_hull_p95_ev_per_atom": quantile(e_hull, 0.95),
        "e_hull_max_ev_per_atom": float(e_hull.max()),
        "stable": float(values["stable"].mean()),
        "nus": float(values["nus"].mean()),
        "novel": float(values["novel"].mean()),
        "unique": float(values["unique"].mean()),
        "rmsd_mean_a": float(rmsd.mean()),
        "rmsd_median_a": float(np.median(rmsd)),
        "rmsd_p95_a": quantile(rmsd, 0.95),
        "rmsd_max_a": float(rmsd.max()),
        "atomic_force_mean_ev_per_a": float(values["atomic_force_mean"].mean()),
        "structure_max_force_mean_ev_per_a": float(max_force.mean()),
        "structure_max_force_median_ev_per_a": float(np.median(max_force)),
        "structure_max_force_p95_ev_per_a": quantile(max_force, 0.95),
        "structure_max_force_max_ev_per_a": float(max_force.max()),
        "relaxation_steps_mean": float(steps.mean()),
        "relaxation_steps_median": float(np.median(steps)),
        "relaxation_steps_p95": quantile(steps, 0.95),
        "relaxation_steps_max": int(steps.max()),
        "relaxation_steps_gt100_count": int((steps > 100).sum()),
        "relaxation_steps_gt200_count": int((steps > 200).sum()),
        "relaxation_steps_gt400_count": int((steps > 400).sum()),
        "generation_seconds_mean": float(values["generation_seconds"].mean()),
    }


def paired_rows(data: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    metrics = {
        "e_hull": ("lower", "eV/atom"),
        "stable": ("higher", "fraction"),
        "nus": ("higher", "fraction"),
        "novel": ("higher", "fraction"),
        "unique": ("higher", "fraction"),
        "rmsd": ("lower", "angstrom"),
        "atomic_force_mean": ("lower", "eV/angstrom"),
        "structure_max_force": ("lower", "eV/angstrom"),
        "relaxation_steps": ("lower", "steps"),
    }
    rows = []
    for method, baseline in COMPARISONS:
        for metric, (direction, unit) in metrics.items():
            candidate = data[method][metric]
            reference = data[baseline][metric]
            delta = candidate - reference
            tolerance = 1e-12
            favorable = delta < -tolerance if direction == "lower" else delta > tolerance
            unfavorable = delta > tolerance if direction == "lower" else delta < -tolerance
            ties = ~(favorable | unfavorable)
            leave_one_out = (delta.sum() - delta) / (len(delta) - 1)
            observed = float(delta.mean())
            if abs(observed) <= tolerance:
                sign_robust = bool(np.all(np.abs(leave_one_out) <= tolerance))
            elif observed < 0:
                sign_robust = bool(np.all(leave_one_out < 0))
            else:
                sign_robust = bool(np.all(leave_one_out > 0))
            rows.append({
                "comparison": f"{method}-{baseline}",
                "metric": metric,
                "preferred_direction": direction,
                "unit": unit,
                "n_pairs": len(SEEDS),
                "candidate_mean": float(candidate.mean()),
                "baseline_mean": float(reference.mean()),
                "mean_delta_candidate_minus_baseline": observed,
                "median_paired_delta": float(np.median(delta)),
                "favorable_pairs": int(favorable.sum()),
                "unfavorable_pairs": int(unfavorable.sum()),
                "ties": int(ties.sum()),
                "leave_one_out_delta_min": float(leave_one_out.min()),
                "leave_one_out_delta_max": float(leave_one_out.max()),
                "leave_one_out_sign_robust": sign_robust,
            })
    return rows


def tail_rows(data: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows = []
    for method in METHODS:
        values = data[method]
        for index, seed in enumerate(SEEDS):
            flags = []
            if values["e_hull"][index] > 0.1:
                flags.append("e_hull_gt_0.1")
            if values["rmsd"][index] > 0.5:
                flags.append("rmsd_gt_0.5")
            if values["structure_max_force"][index] > 1.0:
                flags.append("max_force_gt_1")
            if values["relaxation_steps"][index] > 100:
                flags.append("steps_gt_100")
            if values["relaxation_steps"][index] > 200:
                flags.append("steps_gt_200")
            if values["relaxation_steps"][index] > 400:
                flags.append("steps_gt_400")
            if flags:
                rows.append({
                    "method": method,
                    "seed": seed,
                    "formula": values["formula"][index],
                    "e_hull_ev_per_atom": float(values["e_hull"][index]),
                    "stable": float(values["stable"][index]),
                    "rmsd_a": float(values["rmsd"][index]),
                    "atomic_force_mean_ev_per_a": float(values["atomic_force_mean"][index]),
                    "structure_max_force_ev_per_a": float(values["structure_max_force"][index]),
                    "relaxation_steps": int(values["relaxation_steps"][index]),
                    "outlier_flags": ";".join(flags),
                    "dft_verified": False,
                })
    return rows


def main() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    data = {method: load_method(method, generation) for method in METHODS}
    quality = pd.DataFrame([summary_row(method, data[method]) for method in METHODS])
    quality.to_csv(ROOT / "quality_results.csv", index=False)
    paired = pd.DataFrame(paired_rows(data))
    paired.to_csv(ROOT / "paired_comparisons.csv", index=False)
    tails = pd.DataFrame(tail_rows(data))
    if tails.empty:
        tails = pd.DataFrame(columns=(
            "method", "seed", "formula", "e_hull_ev_per_atom", "stable", "rmsd_a",
            "atomic_force_mean_ev_per_a", "structure_max_force_ev_per_a",
            "relaxation_steps", "outlier_flags", "dft_verified",
        ))
    tails.to_csv(ROOT / "tail_analysis.csv", index=False)
    print(quality.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)
    print(tails.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
