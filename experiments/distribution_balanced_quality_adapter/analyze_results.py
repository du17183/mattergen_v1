"""Summarize screen16 quality, distribution, and paired bootstrap results."""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd
from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parent
STAGE_ROOT = ROOT / "screen16"
METHODS = ("C0", "M1", "M2", "M2-DB")
SEEDS = tuple(range(73000, 73016))
COMPARISONS = tuple(("M2-DB", reference) for reference in ("C0", "M1", "M2"))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260906


def jensen_shannon(left: Counter, right: Counter) -> float:
    keys = sorted(set(left) | set(right), key=str)
    p = np.asarray([left.get(key, 0) for key in keys], dtype=float)
    q = np.asarray([right.get(key, 0) for key in keys], dtype=float)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    return float(
        0.5 * np.sum(p[p_mask] * np.log(p[p_mask] / midpoint[p_mask]))
        + 0.5 * np.sum(q[q_mask] * np.log(q[q_mask] / midpoint[q_mask]))
    )


def bootstrap(delta: np.ndarray, offset: int) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + offset)
    means = np.empty(BOOTSTRAP_REPLICATES)
    for start in range(0, BOOTSTRAP_REPLICATES, 2_000):
        stop = min(start + 2_000, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    quality_rows = []
    distribution_rows = []
    distributions: dict[str, dict[str, Counter]] = {}
    paired_arrays: dict[str, dict[str, np.ndarray]] = {}

    for method in METHODS:
        generated = generation[generation["method"] == method].sort_values("seed")
        if generated["seed"].astype(int).tolist() != list(SEEDS):
            raise RuntimeError(f"{method}: seed order mismatch")
        if not generated["success"].astype(bool).all():
            raise RuntimeError(f"{method}: generation failure")
        summary = json.loads(
            (STAGE_ROOT / "quality" / method / "quality_summary.json").read_text()
        )
        with gzip.open(
            STAGE_ROOT / "quality" / method / "official_detailed.json.gz", "rt"
        ) as stream:
            detail = json.load(stream)
        e_hull = np.asarray(detail["energy_above_hull_per_atom"], dtype=float)
        rmsd = np.asarray(detail["rmsd_from_relaxation"], dtype=float)
        initial = read(
            STAGE_ROOT / "relaxation" / method / "initial_with_properties.extxyz",
            index=":",
        )
        atom_force_norms = [
            np.linalg.norm(structure.get_forces(), axis=1) for structure in initial
        ]
        all_force_norms = np.concatenate(atom_force_norms)
        structure_force_means = np.asarray(
            [values.mean() for values in atom_force_norms]
        )
        structure_force_max = np.asarray(
            [values.max() for values in atom_force_norms]
        )
        relaxation = json.loads(
            (STAGE_ROOT / "relaxation" / method / "relaxation_summary.json").read_text()
        )
        relaxation_steps = np.asarray(relaxation["relaxation_steps"], dtype=float)
        structures = read(
            STAGE_ROOT / "structures" / f"{method}_generated.extxyz", index=":"
        )
        element_counts: Counter = Counter()
        family_counts: Counter = Counter()
        formula_counts: Counter = Counter()
        atom_counts: Counter = Counter()
        for structure in structures:
            elements = structure.get_chemical_symbols()
            element_counts.update(elements)
            family_counts["-".join(sorted(set(elements)))] += 1
            formula_counts[Composition(Counter(elements)).reduced_formula] += 1
            atom_counts[len(structure)] += 1
        distributions[method] = {
            "element": element_counts,
            "family": family_counts,
            "formula": formula_counts,
            "atom_count": atom_counts,
        }
        for category, counter in distributions[method].items():
            total = sum(counter.values())
            for item, count in sorted(counter.items(), key=lambda pair: str(pair[0])):
                distribution_rows.append(
                    {"stage": "screen16", "method": method, "category": category,
                     "item": item, "count": count, "share": count / total}
                )
        atoms_per_structure = np.asarray([len(structure) for structure in structures])
        official = summary["official_metrics"]
        quality_rows.append(
            {
                "stage": "screen16", "method": method, "n": summary["n"],
                "generation_success_rate": float(generated["success"].astype(bool).mean()),
                "mattersim_success_rate": summary["relaxation_success_rate"], "dft_verified": False,
                "e_hull_mean_ev_per_atom": official["avg_energy_above_hull_per_atom"],
                "e_hull_median_ev_per_atom": float(np.median(e_hull)),
                "e_hull_p95_ev_per_atom": float(np.quantile(e_hull, 0.95)),
                "e_hull_max_ev_per_atom": float(e_hull.max()),
                "stable": official["frac_stable_structures"],
                "nus": official["frac_novel_unique_stable_structures"],
                "novel": official["frac_novel_structures"],
                "unique": official["frac_unique_structures"],
                "rmsd_mean_a": official["avg_rmsd_from_relaxation"],
                "rmsd_median_a": float(np.median(rmsd)),
                "rmsd_p95_a": float(np.quantile(rmsd, 0.95)),
                "rmsd_max_a": float(rmsd.max()),
                "atomic_force_mean_ev_per_a": float(all_force_norms.mean()),
                "atomic_force_p95_ev_per_a": float(np.quantile(all_force_norms, 0.95)),
                "structure_force_mean_ev_per_a": float(structure_force_means.mean()),
                "structure_max_force_mean_ev_per_a": float(structure_force_max.mean()),
                "structure_max_force_p95_ev_per_a": float(np.quantile(structure_force_max, 0.95)),
                "force_max_ev_per_a": float(all_force_norms.max()),
                "relaxation_steps_mean": float(relaxation_steps.mean()),
                "relaxation_steps_median": float(np.median(relaxation_steps)),
                "relaxation_steps_p95": float(np.quantile(relaxation_steps, 0.95)),
                "relaxation_steps_max": int(relaxation_steps.max()),
                "composition_family_count": len(family_counts),
                "unique_reduced_formula_count": len(formula_counts),
                "unique_element_count": len(element_counts),
                "num_atoms_mean": float(atoms_per_structure.mean()),
                "num_atoms_median": float(np.median(atoms_per_structure)),
                "num_atoms_min": int(atoms_per_structure.min()),
                "num_atoms_max": int(atoms_per_structure.max()),
                "num_atoms_unique_count": len(atom_counts),
            }
        )
        paired_arrays[method] = {
            "e_hull_ev_per_atom": e_hull,
            "stable": np.asarray(detail["stable"], dtype=float),
            "nus": np.asarray(detail["novel_unique_stable"], dtype=float),
            "novel": np.asarray(detail["novel"], dtype=float),
            "unique": np.asarray(detail["unique"], dtype=float),
            "rmsd_a": rmsd,
            "structure_force_mean_ev_per_a": structure_force_means,
            "structure_max_force_ev_per_a": structure_force_max,
            "relaxation_steps": relaxation_steps,
        }

    quality = pd.DataFrame(quality_rows)
    for method in METHODS:
        for reference in ("C0", "M1", "M2"):
            if method == reference:
                continue
            for category in ("element", "family", "formula", "atom_count"):
                quality.loc[
                    quality["method"] == method, f"{category}_jsd_vs_{reference}"
                ] = jensen_shannon(
                    distributions[method][category], distributions[reference][category]
                )
    quality.to_csv(ROOT / "quality_results.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(ROOT / "distribution_analysis.csv", index=False)

    lower_is_better = {
        "e_hull_ev_per_atom", "rmsd_a", "structure_force_mean_ev_per_a",
        "structure_max_force_ev_per_a", "relaxation_steps",
    }
    paired_rows = []
    offset = 0
    for candidate, reference in COMPARISONS:
        for metric, candidate_values in paired_arrays[candidate].items():
            reference_values = paired_arrays[reference][metric]
            delta = candidate_values - reference_values
            low, high = bootstrap(delta, offset)
            offset += 1
            lower = metric in lower_is_better
            favorable = delta < -1e-12 if lower else delta > 1e-12
            unfavorable = delta > 1e-12 if lower else delta < -1e-12
            ties = ~(favorable | unfavorable)
            paired_rows.append(
                {
                    "stage": "screen16_exploratory", "comparison": f"{candidate}-{reference}",
                    "metric": metric, "preferred_direction": "lower" if lower else "higher",
                    "n_pairs": len(delta), "reference_mean": float(reference_values.mean()),
                    "candidate_mean": float(candidate_values.mean()),
                    "mean_delta_candidate_minus_reference": float(delta.mean()),
                    "median_paired_delta": float(np.median(delta)),
                    "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "favorable_pairs": int(favorable.sum()),
                    "unfavorable_pairs": int(unfavorable.sum()), "ties": int(ties.sum()),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(ROOT / "paired_statistics.csv", index=False)
    print(quality.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
