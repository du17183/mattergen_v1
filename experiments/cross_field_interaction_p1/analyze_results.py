"""Summarize P1 quality, tails, and 20,000 paired bootstrap resamples."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
METHODS = ("C0", "MLP", "CFI")
SEEDS = tuple(range(79000, 79032))
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260906


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def load_method(method: str, generation: pd.DataFrame) -> dict[str, np.ndarray]:
    generated = generation[generation["method"] == method].sort_values("seed")
    if generated["seed"].astype(int).tolist() != list(SEEDS):
        raise RuntimeError(f"{method}: seed order mismatch")
    if not generated["success"].astype(bool).all():
        raise RuntimeError(f"{method}: generation failure")
    with gzip.open(ROOT / "quality" / method / "official_detailed.json.gz", "rt") as stream:
        detail = json.load(stream)
    relaxation = json.loads((ROOT / "relaxation" / method / "relaxation_summary.json").read_text())
    if list(map(int, relaxation["sample_seeds"])) != list(SEEDS):
        raise RuntimeError(f"{method}: relaxation seed order mismatch")
    if relaxation["success"] is not True:
        raise RuntimeError(f"{method}: MatterSim relaxation failure")
    initial = read(ROOT / "relaxation" / method / "initial_with_properties.extxyz", index=":")
    if len(initial) != len(SEEDS):
        raise RuntimeError(f"{method}: initial structure count mismatch")
    force_norms = [np.linalg.norm(structure.get_forces(), axis=1) for structure in initial]
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
    }
    expected = len(SEEDS)
    for name, values in data.items():
        if len(values) != expected:
            raise RuntimeError(f"{method}: {name} length mismatch")
    return data


def summarize(method: str, values: dict[str, np.ndarray]) -> dict[str, object]:
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
        "relaxation_steps_gt100_rate": float((steps > 100).mean()),
        "relaxation_steps_gt200_count": int((steps > 200).sum()),
        "relaxation_steps_gt200_rate": float((steps > 200).mean()),
        "relaxation_steps_gt400_count": int((steps > 400).sum()),
        "relaxation_steps_gt400_rate": float((steps > 400).mean()),
    }


def bootstrap_rows(data: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    metrics = {
        "e_hull_mean": ("e_hull", "eV/atom", 1.0),
        "stable": ("stable", "percentage_points", 100.0),
        "nus": ("nus", "percentage_points", 100.0),
        "novel": ("novel", "percentage_points", 100.0),
        "rmsd_mean": ("rmsd", "angstrom", 1.0),
        "atomic_force_mean": ("atomic_force_mean", "eV/angstrom", 1.0),
        "structure_max_force_mean": ("structure_max_force", "eV/angstrom", 1.0),
        "relaxation_steps_mean": ("relaxation_steps", "steps", 1.0),
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(SEEDS), size=(N_BOOTSTRAP, len(SEEDS)))
    rows = []
    for baseline in ("C0", "MLP"):
        for metric, (key, unit, scale) in metrics.items():
            paired_delta = (data["CFI"][key] - data[baseline][key]) * scale
            resampled = paired_delta[indices].mean(axis=1)
            rows.append({
                "comparison": f"CFI-{baseline}",
                "metric": metric,
                "n_pairs": len(SEEDS),
                "mean_delta": float(paired_delta.mean()),
                "ci95_lower": quantile(resampled, 0.025),
                "ci95_upper": quantile(resampled, 0.975),
                "unit": unit,
                "n_bootstrap": N_BOOTSTRAP,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
    return rows


def tail_rows(data: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows = []
    for method in METHODS:
        values = data[method]
        for index, seed in enumerate(SEEDS):
            rmsd = float(values["rmsd"][index])
            steps = int(values["relaxation_steps"][index])
            max_force = float(values["structure_max_force"][index])
            flags = []
            if rmsd > 0.5:
                flags.append("rmsd_gt_0.5")
            if steps > 200:
                flags.append("steps_gt_200")
            if steps > 400:
                flags.append("steps_gt_400")
            if max_force > 1.0:
                flags.append("max_force_gt_1")
            if flags:
                rows.append({
                    "method": method,
                    "seed": seed,
                    "formula": values["formula"][index],
                    "e_hull_ev_per_atom": float(values["e_hull"][index]),
                    "rmsd_a": rmsd,
                    "atomic_force_mean_ev_per_a": float(values["atomic_force_mean"][index]),
                    "structure_max_force_ev_per_a": max_force,
                    "relaxation_steps": steps,
                    "outlier_flags": ";".join(flags),
                    "dft_verified": False,
                })
    return rows


def main() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    data = {method: load_method(method, generation) for method in METHODS}
    quality = pd.DataFrame([summarize(method, data[method]) for method in METHODS])
    quality.to_csv(ROOT / "quality_results.csv", index=False)
    paired = pd.DataFrame(bootstrap_rows(data))
    paired.to_csv(ROOT / "paired_statistics.csv", index=False)
    tails = pd.DataFrame(tail_rows(data))
    if tails.empty:
        tails = pd.DataFrame(columns=[
            "method", "seed", "formula", "e_hull_ev_per_atom", "rmsd_a",
            "atomic_force_mean_ev_per_a", "structure_max_force_ev_per_a",
            "relaxation_steps", "outlier_flags", "dft_verified",
        ])
    tails.to_csv(ROOT / "tail_analysis.csv", index=False)
    print(quality.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)
    print(tails.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
