"""P1 descriptive metrics, tails, and 20,000 paired bootstraps."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_p1"
METHODS = ("C0", "FT0", "TCL", "GBSA-TCL")
SEEDS = tuple(range(85000, 85032))
COMPARISONS = (("GBSA-TCL", "TCL"), ("GBSA-TCL", "C0"), ("GBSA-TCL", "FT0"))
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260920
METRICS = {
    "e_hull": ("lower", "eV/atom", 1.0),
    "stable": ("higher", "percentage_points", 100.0),
    "nus": ("higher", "percentage_points", 100.0),
    "novel": ("higher", "percentage_points", 100.0),
    "unique": ("higher", "percentage_points", 100.0),
    "rmsd": ("lower", "angstrom", 1.0),
    "atomic_force_mean": ("lower", "eV/angstrom", 1.0),
    "structure_max_force": ("lower", "eV/angstrom", 1.0),
    "relaxation_steps": ("lower", "steps", 1.0),
}


def quantile(values, p, axis=None):
    return np.quantile(np.asarray(values, dtype=float), p, axis=axis)


def load_method(method: str, generation: pd.DataFrame) -> dict[str, np.ndarray]:
    generated = generation[generation.method == method].sort_values("seed")
    if tuple(generated.seed.astype(int)) != SEEDS or not generated.success.astype(bool).all():
        raise RuntimeError(f"{method}: generation mismatch/failure")
    with gzip.open(ROOT / "quality" / method / "official_detailed.json.gz", "rt") as stream:
        detail = json.load(stream)
    relaxation = json.loads((ROOT / "relaxation" / method / "relaxation_summary.json").read_text())
    if tuple(map(int, relaxation["sample_seeds"])) != SEEDS or relaxation["success"] is not True:
        raise RuntimeError(f"{method}: relaxation mismatch/failure")
    initial = read(ROOT / "relaxation" / method / "initial_with_properties.extxyz", index=":")
    force_norms = [np.linalg.norm(item.get_forces(), axis=1) for item in initial]
    data = {
        "seed": np.asarray(SEEDS),
        "formula": generated.formula.to_numpy(),
        "e_hull": np.asarray(detail["energy_above_hull_per_atom"], dtype=float),
        "stable": np.asarray(detail["stable"], dtype=float),
        "nus": np.asarray(detail["novel_unique_stable"], dtype=float),
        "novel": np.asarray(detail["novel"], dtype=float),
        "unique": np.asarray(detail["unique"], dtype=float),
        "rmsd": np.asarray(detail["rmsd_from_relaxation"], dtype=float),
        "atomic_force_mean": np.asarray([x.mean() for x in force_norms]),
        "structure_max_force": np.asarray([x.max() for x in force_norms]),
        "relaxation_steps": np.asarray(relaxation["relaxation_steps"], dtype=float),
        "generation_seconds": generated.elapsed_seconds.to_numpy(dtype=float),
    }
    if any(len(value) != len(SEEDS) for value in data.values()):
        raise RuntimeError(f"{method}: metric length mismatch")
    if any(not np.isfinite(value).all() for key, value in data.items() if key != "formula"):
        raise RuntimeError(f"{method}: non-finite metric")
    return data


def summary_row(method: str, values: dict[str, np.ndarray]) -> dict:
    row = {
        "method": method, "n": len(SEEDS),
        "generation_success_count": len(SEEDS), "generation_success_rate": 1.0,
        "mattersim_success_count": len(SEEDS), "mattersim_success_rate": 1.0,
        "dft_verified": False,
    }
    for metric, unit in (("e_hull", "ev_per_atom"), ("rmsd", "angstrom"),
                         ("atomic_force_mean", "ev_per_angstrom")):
        x = values[metric]
        row.update({
            f"{metric}_mean_{unit}": float(x.mean()),
            f"{metric}_median_{unit}": float(np.median(x)),
            f"{metric}_p95_{unit}": float(quantile(x, .95)),
            f"{metric}_max_{unit}": float(x.max()),
        })
    for metric in ("stable", "nus", "novel", "unique"):
        x = values[metric]
        row[f"{metric}_count"] = int(x.sum())
        row[f"{metric}_rate"] = float(x.mean())
    max_force = values["structure_max_force"]
    steps = values["relaxation_steps"]
    row.update({
        "structure_max_force_mean_ev_per_angstrom": float(max_force.mean()),
        "structure_max_force_median_ev_per_angstrom": float(np.median(max_force)),
        "structure_max_force_p95_ev_per_angstrom": float(quantile(max_force, .95)),
        "structure_max_force_p99_ev_per_angstrom": float(quantile(max_force, .99)),
        "structure_max_force_max_ev_per_angstrom": float(max_force.max()),
        "structure_max_force_gt1_count": int((max_force > 1).sum()),
        "structure_max_force_gt2_count": int((max_force > 2).sum()),
        "relaxation_steps_mean": float(steps.mean()),
        "relaxation_steps_median": float(np.median(steps)),
        "relaxation_steps_p95": float(quantile(steps, .95)),
        "relaxation_steps_max": int(steps.max()),
        "relaxation_steps_gt200_count": int((steps > 200).sum()),
        "relaxation_steps_gt400_count": int((steps > 400).sum()),
        "rmsd_gt0p5_count": int((values["rmsd"] > .5).sum()),
        "generation_seconds_mean": float(values["generation_seconds"].mean()),
    })
    return row


def paired_rows(data: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(SEEDS), size=(N_BOOTSTRAP, len(SEEDS)))
    rows = []
    for candidate, baseline in COMPARISONS:
        for metric, (direction, unit, scale) in METRICS.items():
            raw = data[candidate][metric] - data[baseline][metric]
            delta = raw * scale
            boot = delta[indices].mean(axis=1)
            tol = 1e-12
            favorable = raw < -tol if direction == "lower" else raw > tol
            unfavorable = raw > tol if direction == "lower" else raw < -tol
            ties = ~(favorable | unfavorable)
            rows.append({
                "comparison": f"{candidate}-{baseline}", "metric": metric,
                "preferred_direction": direction, "unit": unit, "n_pairs": len(SEEDS),
                "candidate_mean": float(data[candidate][metric].mean() * scale),
                "baseline_mean": float(data[baseline][metric].mean() * scale),
                "mean_delta_candidate_minus_baseline": float(delta.mean()),
                "median_paired_delta": float(np.median(delta)),
                "bootstrap_ci95_low": float(quantile(boot, .025)),
                "bootstrap_ci95_high": float(quantile(boot, .975)),
                "bootstrap_resamples": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
                "wins": int(favorable.sum()), "ties": int(ties.sum()),
                "losses": int(unfavorable.sum()),
            })
    return rows


def per_seed_rows(data: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    for method in METHODS:
        values = data[method]
        for i, seed in enumerate(SEEDS):
            flags = []
            if values["rmsd"][i] > .5: flags.append("RMSD>0.5")
            if values["structure_max_force"][i] > 1: flags.append("MaxF>1")
            if values["structure_max_force"][i] > 2: flags.append("MaxF>2")
            if values["relaxation_steps"][i] > 200: flags.append("Steps>200")
            if values["relaxation_steps"][i] > 400: flags.append("Steps>400")
            rows.append({
                "method": method, "seed": seed, "formula": values["formula"][i],
                "e_hull_ev_per_atom": values["e_hull"][i],
                "stable": bool(values["stable"][i]), "nus": bool(values["nus"][i]),
                "novel": bool(values["novel"][i]), "unique": bool(values["unique"][i]),
                "rmsd_angstrom": values["rmsd"][i],
                "atomic_force_mean_ev_per_angstrom": values["atomic_force_mean"][i],
                "structure_max_force_ev_per_angstrom": values["structure_max_force"][i],
                "relaxation_steps": int(values["relaxation_steps"][i]),
                "tail_flags": ";".join(flags), "dft_verified": False,
            })
    return rows


def main() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    data = {method: load_method(method, generation) for method in METHODS}
    quality = pd.DataFrame([summary_row(method, data[method]) for method in METHODS])
    paired = pd.DataFrame(paired_rows(data))
    per_seed = pd.DataFrame(per_seed_rows(data))
    tails = per_seed[per_seed.tail_flags.astype(bool)].copy()
    quality.to_csv(ROOT / "quality_results.csv", index=False)
    paired.to_csv(ROOT / "paired_statistics.csv", index=False)
    per_seed.to_csv(ROOT / "per_seed_results.csv", index=False)
    tails.to_csv(ROOT / "tail_analysis.csv", index=False)
    print(quality.to_string(index=False), flush=True)
    print(paired[paired.comparison == "GBSA-TCL-TCL"].to_string(index=False), flush=True)
    print("severe GBSA rows")
    print(tails[tails.method == "GBSA-TCL"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
