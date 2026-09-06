"""Paired bootstrap sensitivity analysis for the P1b 16-seed screen."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
ROOT = EXPERIMENT_ROOT / "screen16"
METHODS = ("C0", "M1", "M2", "M2-L1", "M2-L2")
COMPARISONS = tuple(
    (candidate, reference)
    for candidate in ("M2-L1", "M2-L2")
    for reference in ("C0", "M1", "M2")
)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260906


def force_arrays(method: str) -> tuple[np.ndarray, np.ndarray]:
    atoms = read(
        ROOT
        / "relaxation"
        / method
        / "initial_with_properties.extxyz",
        index=":",
    )
    norms = [np.linalg.norm(item.get_forces(), axis=1) for item in atoms]
    return (
        np.asarray([values.mean() for values in norms]),
        np.asarray([values.max() for values in norms]),
    )


def bootstrap(delta: np.ndarray, offset: int) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + offset)
    sampled_means = np.empty(BOOTSTRAP_REPLICATES)
    batch_size = 2_000
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_REPLICATES)
        indices = rng.integers(
            0, len(delta), size=(stop - start, len(delta))
        )
        sampled_means[start:stop] = delta[indices].mean(axis=1)
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        with gzip.open(
            ROOT / "quality" / method / "official_detailed.json.gz", "rt"
        ) as stream:
            detail = json.load(stream)
        mean_force, max_force = force_arrays(method)
        arrays[method] = {
            "e_hull_ev_per_atom": np.asarray(
                detail["energy_above_hull_per_atom"], dtype=float
            ),
            "stable": np.asarray(detail["stable"], dtype=float),
            "nus": np.asarray(
                detail["novel_unique_stable"], dtype=float
            ),
            "novel": np.asarray(detail["novel"], dtype=float),
            "unique": np.asarray(detail["unique"], dtype=float),
            "rmsd_a": np.asarray(
                detail["rmsd_from_relaxation"], dtype=float
            ),
            "structure_force_mean_ev_per_a": mean_force,
            "structure_max_force_ev_per_a": max_force,
        }
    lower_is_better = {
        "e_hull_ev_per_atom",
        "rmsd_a",
        "structure_force_mean_ev_per_a",
        "structure_max_force_ev_per_a",
    }
    rows = []
    offset = 0
    for candidate, reference in COMPARISONS:
        for metric, candidate_values in arrays[candidate].items():
            reference_values = arrays[reference][metric]
            delta = candidate_values - reference_values
            low, high = bootstrap(delta, offset)
            offset += 1
            lower = metric in lower_is_better
            favorable = delta < -1e-12 if lower else delta > 1e-12
            unfavorable = delta > 1e-12 if lower else delta < -1e-12
            ties = ~(favorable | unfavorable)
            leave_one_out = (delta.sum() - delta) / (len(delta) - 1)
            rows.append(
                {
                    "stage": "screen16_exploratory",
                    "comparison": f"{candidate}-{reference}",
                    "metric": metric,
                    "preferred_direction": "lower" if lower else "higher",
                    "n_pairs": len(delta),
                    "reference_mean": float(reference_values.mean()),
                    "candidate_mean": float(candidate_values.mean()),
                    "mean_delta_candidate_minus_reference": float(
                        delta.mean()
                    ),
                    "median_paired_delta": float(np.median(delta)),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "favorable_pairs": int(favorable.sum()),
                    "unfavorable_pairs": int(unfavorable.sum()),
                    "ties": int(ties.sum()),
                    "leave_one_out_delta_min": float(
                        leave_one_out.min()
                    ),
                    "leave_one_out_delta_max": float(
                        leave_one_out.max()
                    ),
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(EXPERIMENT_ROOT / "paired_statistics.csv", index=False)
    print(output.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
