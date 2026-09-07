"""Summarize TCL P2, paired bootstraps, tails, and P0/P1/P2 replication."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/tcl_p2"
P1_ROOT = PROJECT_ROOT / "experiments/tcl_p1"
P1_ANALYSIS = P1_ROOT / "analyze_results.py"
METHODS = ("C0", "FT0", "TCL")
SEEDS = tuple(range(82000, 82064))
COMPARISONS = (("FT0", "C0"), ("TCL", "C0"), ("TCL", "FT0"))
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260911


def load_p1_analysis():
    spec = importlib.util.spec_from_file_location("frozen_p1_analysis", P1_ANALYSIS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.SEEDS = SEEDS
    return module


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def summary_row(method: str, values: dict[str, np.ndarray]) -> dict[str, object]:
    e_hull = values["e_hull"]
    rmsd = values["rmsd"]
    max_force = values["structure_max_force"]
    steps = values["relaxation_steps"]
    return {
        "method": method,
        "n": len(SEEDS),
        "generation_success_count": len(SEEDS),
        "generation_success_rate": 1.0,
        "mattersim_success_count": len(SEEDS),
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
        "structure_max_force_p99_ev_per_a": quantile(max_force, 0.99),
        "structure_max_force_max_ev_per_a": float(max_force.max()),
        "structure_max_force_gt1_count": int((max_force > 1.0).sum()),
        "structure_max_force_gt1_rate": float((max_force > 1.0).mean()),
        "structure_max_force_gt2_count": int((max_force > 2.0).sum()),
        "structure_max_force_gt2_rate": float((max_force > 2.0).mean()),
        "relaxation_steps_mean": float(steps.mean()),
        "relaxation_steps_median": float(np.median(steps)),
        "relaxation_steps_p95": quantile(steps, 0.95),
        "relaxation_steps_max": int(steps.max()),
        "relaxation_steps_gt100_count": int((steps > 100).sum()),
        "relaxation_steps_gt100_rate": float((steps > 100).mean()),
        "relaxation_steps_gt200_count": int((steps > 200).sum()),
        "relaxation_steps_gt200_rate": float((steps > 200).mean()),
        "relaxation_steps_gt300_count": int((steps > 300).sum()),
        "relaxation_steps_gt300_rate": float((steps > 300).mean()),
        "relaxation_steps_gt400_count": int((steps > 400).sum()),
        "relaxation_steps_gt400_rate": float((steps > 400).mean()),
        "generation_seconds_mean": float(values["generation_seconds"].mean()),
    }


def paired_rows(data: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    metrics = {
        "e_hull": ("lower", "eV/atom", 1.0),
        "stable": ("higher", "percentage_points", 100.0),
        "nus": ("higher", "percentage_points", 100.0),
        "novel": ("higher", "percentage_points", 100.0),
        "unique": ("higher", "percentage_points", 100.0),
        "rmsd": ("lower", "angstrom", 1.0),
        "atomic_force_mean": ("lower", "eV/angstrom", 1.0),
        "structure_max_force": ("lower", "eV/angstrom", 1.0),
        "force_gt1": ("lower", "percentage_points", 100.0),
        "force_gt2": ("lower", "percentage_points", 100.0),
        "relaxation_steps": ("lower", "steps", 1.0),
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(SEEDS), size=(N_BOOTSTRAP, len(SEEDS)))
    rows = []
    for method, baseline in COMPARISONS:
        for metric, (preferred, unit, scale) in metrics.items():
            candidate_raw = data[method][metric]
            baseline_raw = data[baseline][metric]
            raw_delta = candidate_raw - baseline_raw
            delta = raw_delta * scale
            resampled = delta[indices].mean(axis=1)
            tolerance = 1e-12
            favorable = raw_delta < -tolerance if preferred == "lower" else raw_delta > tolerance
            unfavorable = raw_delta > tolerance if preferred == "lower" else raw_delta < -tolerance
            ties = ~(favorable | unfavorable)
            rows.append({
                "comparison": f"{method}-{baseline}",
                "metric": metric,
                "preferred_direction": preferred,
                "unit": unit,
                "n_pairs": len(SEEDS),
                "candidate_mean": float(candidate_raw.mean() * scale),
                "baseline_mean": float(baseline_raw.mean() * scale),
                "mean_delta_candidate_minus_baseline": float(delta.mean()),
                "median_paired_delta": float(np.median(delta)),
                "bootstrap_ci95_low": quantile(resampled, 0.025),
                "bootstrap_ci95_high": quantile(resampled, 0.975),
                "bootstrap_resamples": N_BOOTSTRAP,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "favorable_pairs": int(favorable.sum()),
                "unfavorable_pairs": int(unfavorable.sum()),
                "ties": int(ties.sum()),
            })
    return rows


def tail_rows(data: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows = []
    for method in METHODS:
        values = data[method]
        for index, seed in enumerate(SEEDS):
            flags = []
            if values["rmsd"][index] > 0.5:
                flags.append("rmsd_gt_0.5")
            if values["structure_max_force"][index] > 1.0:
                flags.append("max_force_gt_1")
            if values["structure_max_force"][index] > 2.0:
                flags.append("max_force_gt_2")
            if values["relaxation_steps"][index] > 200:
                flags.append("steps_gt_200")
            if values["relaxation_steps"][index] > 300:
                flags.append("steps_gt_300")
            if values["relaxation_steps"][index] > 400:
                flags.append("steps_gt_400")
            if flags:
                rows.append({
                    "method": method,
                    "seed": seed,
                    "formula": values["formula"][index],
                    "e_hull_ev_per_atom": float(values["e_hull"][index]),
                    "stable": float(values["stable"][index]),
                    "nus": float(values["nus"][index]),
                    "rmsd_a": float(values["rmsd"][index]),
                    "atomic_force_mean_ev_per_a": float(values["atomic_force_mean"][index]),
                    "structure_max_force_ev_per_a": float(values["structure_max_force"][index]),
                    "relaxation_steps": int(values["relaxation_steps"][index]),
                    "outlier_flags": ";".join(flags),
                    "dft_verified": False,
                })
    return rows


def sign(value: float, tolerance: float = 1e-12) -> str:
    if value < -tolerance:
        return "down"
    if value > tolerance:
        return "up"
    return "flat"


def replication_rows(p2: pd.DataFrame) -> list[dict[str, object]]:
    p1 = pd.read_csv(P1_ROOT / "paired_statistics.csv")
    p1 = p1[p1["comparison"] == "TCL-FT0"].set_index("metric")
    p2_index = p2[p2["comparison"] == "TCL-FT0"].set_index("metric")
    p0 = {
        "e_hull": (-0.00953533169182052, "eV/atom"),
        "stable": (0.0, "percentage_points"),
        "nus": (25.0, "percentage_points"),
        "novel": (25.0, "percentage_points"),
        "rmsd": (-0.019737251544580375, "angstrom"),
        "atomic_force_mean": (-0.083695, "eV/angstrom"),
        "structure_max_force": (-0.242409, "eV/angstrom"),
        "relaxation_steps": (-9.125, "steps"),
    }
    rows = []
    for metric, (p0_delta, unit) in p0.items():
        p1_row = p1.loc[metric]
        p2_row = p2_index.loc[metric]
        p0_direction = sign(p0_delta)
        p1_direction = sign(float(p1_row["mean_delta_candidate_minus_baseline"]))
        p2_direction = sign(float(p2_row["mean_delta_candidate_minus_baseline"]))
        if p0_direction == p1_direction == p2_direction:
            replicated = "yes_all_three"
        elif p1_direction == p2_direction:
            replicated = "yes_p1_to_p2_only"
        else:
            replicated = "no_p1_to_p2"
        rows.append({
            "metric": metric,
            "unit": unit,
            "p0_delta": p0_delta,
            "p0_direction": p0_direction,
            "p1_delta": float(p1_row["mean_delta_candidate_minus_baseline"]),
            "p1_ci95_low": float(p1_row["bootstrap_ci95_low"]),
            "p1_ci95_high": float(p1_row["bootstrap_ci95_high"]),
            "p1_direction": p1_direction,
            "p2_delta": float(p2_row["mean_delta_candidate_minus_baseline"]),
            "p2_ci95_low": float(p2_row["bootstrap_ci95_low"]),
            "p2_ci95_high": float(p2_row["bootstrap_ci95_high"]),
            "p2_direction": p2_direction,
            "replicated": replicated,
        })
    return rows


def main() -> None:
    p1_analysis = load_p1_analysis()
    generation = pd.read_csv(ROOT / "generation_results.csv")
    data = {
        method: p1_analysis.load_method(method, generation)
        for method in METHODS
    }
    for values in data.values():
        values["force_gt1"] = (values["structure_max_force"] > 1.0).astype(float)
        values["force_gt2"] = (values["structure_max_force"] > 2.0).astype(float)
    quality = pd.DataFrame([summary_row(method, data[method]) for method in METHODS])
    quality.to_csv(ROOT / "quality_results.csv", index=False)
    paired = pd.DataFrame(paired_rows(data))
    paired.to_csv(ROOT / "paired_statistics.csv", index=False)
    tails = pd.DataFrame(tail_rows(data))
    if tails.empty:
        tails = pd.DataFrame(columns=(
            "method", "seed", "formula", "e_hull_ev_per_atom", "stable", "nus",
            "rmsd_a", "atomic_force_mean_ev_per_a", "structure_max_force_ev_per_a",
            "relaxation_steps", "outlier_flags", "dft_verified",
        ))
    tails.to_csv(ROOT / "tail_analysis.csv", index=False)
    replication = pd.DataFrame(replication_rows(paired))
    replication.to_csv(ROOT / "replication_table.csv", index=False)
    print(quality.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)
    print(tails.to_string(index=False), flush=True)
    print(replication.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
