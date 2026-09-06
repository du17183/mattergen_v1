"""Summarize P1 real generation and official quality evaluation results."""

from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd
from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parent
METHODS = ("C0", "M1", "M2")
EXPECTED_SEEDS = list(range(70000, 70032))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260906


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def detail_for(method: str) -> dict[str, list]:
    with gzip.open(ROOT / "quality" / method / "official_detailed.json.gz", "rt") as stream:
        return json.load(stream)


def initial_force_arrays(method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atoms = read(ROOT / "relaxation" / method / "initial_with_properties.extxyz", index=":")
    per_atom = [np.linalg.norm(item.get_forces(), axis=1) for item in atoms]
    return (
        np.concatenate(per_atom),
        np.asarray([values.mean() for values in per_atom]),
        np.asarray([values.max() for values in per_atom]),
    )


def write_quality_results() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    summaries = {
        method: json.loads((ROOT / "quality" / method / "quality_summary.json").read_text())
        for method in METHODS
    }
    details = {method: detail_for(method) for method in METHODS}
    force_arrays = {method: initial_force_arrays(method) for method in METHODS}
    rows = []
    for method in METHODS:
        summary = summaries[method]
        official = summary["official_metrics"]
        detail = details[method]
        e_hull = np.asarray(detail["energy_above_hull_per_atom"], dtype=float)
        rmsd = np.asarray(detail["rmsd_from_relaxation"], dtype=float)
        atomic_force, structure_mean_force, structure_max_force = force_arrays[method]
        steps = np.asarray(
            json.loads(
                (ROOT / "relaxation" / method / "relaxation_summary.json").read_text()
            )["relaxation_steps"],
            dtype=float,
        )
        generated = generation[generation["method"] == method].sort_values("seed")
        if generated["seed"].astype(int).tolist() != EXPECTED_SEEDS:
            raise ValueError(f"{method}: generation seeds/order mismatch")
        if len(e_hull) != len(EXPECTED_SEEDS):
            raise ValueError(f"{method}: official detail count mismatch")
        rows.append(
            {
                "method": method,
                "n": int(summary["n"]),
                "dft_verified": False,
                "generation_success_count": int(generated["success"].astype(bool).sum()),
                "generation_success_rate": float(generated["success"].astype(bool).mean()),
                "relaxation_success_rate": float(summary["relaxation_success_rate"]),
                "avg_energy_above_hull_per_atom_ev": official[
                    "avg_energy_above_hull_per_atom"
                ],
                "e_hull_median_ev_per_atom": float(np.median(e_hull)),
                "e_hull_p75_ev_per_atom": quantile(e_hull, 0.75),
                "e_hull_p90_ev_per_atom": quantile(e_hull, 0.90),
                "e_hull_p95_ev_per_atom": quantile(e_hull, 0.95),
                "e_hull_max_ev_per_atom": float(e_hull.max()),
                "frac_stable": official["frac_stable_structures"],
                "frac_novel_unique_stable": official[
                    "frac_novel_unique_stable_structures"
                ],
                "frac_novel": official["frac_novel_structures"],
                "frac_unique": official["frac_unique_structures"],
                "avg_rmsd_from_relaxation_a": official["avg_rmsd_from_relaxation"],
                "rmsd_median_a": float(np.median(rmsd)),
                "rmsd_p75_a": quantile(rmsd, 0.75),
                "rmsd_p90_a": quantile(rmsd, 0.90),
                "rmsd_p95_a": quantile(rmsd, 0.95),
                "rmsd_max_a": float(rmsd.max()),
                "pre_relaxation_atomic_force_norm_mean_ev_per_a": float(
                    atomic_force.mean()
                ),
                "pre_relaxation_atomic_force_norm_p95_ev_per_a": quantile(
                    atomic_force, 0.95
                ),
                "pre_relaxation_atomic_force_norm_p99_ev_per_a": quantile(
                    atomic_force, 0.99
                ),
                "pre_relaxation_atomic_force_norm_max_ev_per_a": float(
                    atomic_force.max()
                ),
                "pre_relaxation_structure_mean_force_mean_ev_per_a": float(
                    structure_mean_force.mean()
                ),
                "pre_relaxation_structure_mean_force_median_ev_per_a": float(
                    np.median(structure_mean_force)
                ),
                "pre_relaxation_structure_mean_force_p95_ev_per_a": quantile(
                    structure_mean_force, 0.95
                ),
                "pre_relaxation_structure_max_force_mean_ev_per_a": float(
                    structure_max_force.mean()
                ),
                "pre_relaxation_structure_max_force_median_ev_per_a": float(
                    np.median(structure_max_force)
                ),
                "pre_relaxation_structure_max_force_p95_ev_per_a": quantile(
                    structure_max_force, 0.95
                ),
                "pre_relaxation_structure_max_force_p99_ev_per_a": quantile(
                    structure_max_force, 0.99
                ),
                "pre_relaxation_structure_max_force_max_ev_per_a": float(
                    structure_max_force.max()
                ),
                "relaxation_steps_mean": float(steps.mean()),
                "relaxation_steps_median": float(np.median(steps)),
                "relaxation_steps_p75": quantile(steps, 0.75),
                "relaxation_steps_p90": quantile(steps, 0.90),
                "relaxation_steps_p95": quantile(steps, 0.95),
                "relaxation_steps_p99": quantile(steps, 0.99),
                "relaxation_steps_max": int(steps.max()),
                "generation_seconds_mean": float(generated["elapsed_seconds"].mean()),
                "generation_seconds_median": float(generated["elapsed_seconds"].median()),
                "generation_seconds_p95": float(
                    generated["elapsed_seconds"].quantile(0.95)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    baseline = frame.set_index("method").loc["C0"]
    delta_columns = {
        "avg_energy_above_hull_per_atom_ev": "delta_e_hull_vs_c0_ev_per_atom",
        "frac_stable": "delta_stable_vs_c0",
        "frac_novel_unique_stable": "delta_nus_vs_c0",
        "frac_novel": "delta_novel_vs_c0",
        "frac_unique": "delta_unique_vs_c0",
        "avg_rmsd_from_relaxation_a": "delta_rmsd_vs_c0_a",
        "pre_relaxation_atomic_force_norm_mean_ev_per_a": (
            "delta_atomic_force_mean_vs_c0_ev_per_a"
        ),
        "pre_relaxation_structure_max_force_mean_ev_per_a": (
            "delta_structure_max_force_mean_vs_c0_ev_per_a"
        ),
        "relaxation_steps_mean": "delta_relaxation_steps_mean_vs_c0",
    }
    for source, destination in delta_columns.items():
        frame[destination] = frame[source] - float(baseline[source])
    frame.to_csv(ROOT / "quality_results.csv", index=False)


def bootstrap_mean_delta(delta: np.ndarray, seed_offset: int) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    batch_size = 2_000
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        samples[start:stop] = delta[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def write_paired_statistics() -> None:
    details = {method: detail_for(method) for method in METHODS}
    forces = {method: initial_force_arrays(method) for method in METHODS}
    relaxations = {
        method: json.loads(
            (ROOT / "relaxation" / method / "relaxation_summary.json").read_text()
        )
        for method in METHODS
    }

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        arrays[method] = {
            "e_hull_ev_per_atom": np.asarray(
                details[method]["energy_above_hull_per_atom"], dtype=float
            ),
            "stable": np.asarray(details[method]["stable"], dtype=float),
            "novel_unique_stable": np.asarray(
                details[method]["novel_unique_stable"], dtype=float
            ),
            "novel": np.asarray(details[method]["novel"], dtype=float),
            "unique": np.asarray(details[method]["unique"], dtype=float),
            "rmsd_a": np.asarray(details[method]["rmsd_from_relaxation"], dtype=float),
            "atomic_force_mean_ev_per_a": forces[method][1].astype(float),
            "structure_max_force_ev_per_a": forces[method][2].astype(float),
            "relaxation_steps": np.asarray(
                relaxations[method]["relaxation_steps"], dtype=float
            ),
        }

    lower_is_better = {
        "e_hull_ev_per_atom",
        "rmsd_a",
        "atomic_force_mean_ev_per_a",
        "structure_max_force_ev_per_a",
        "relaxation_steps",
    }
    rows = []
    seed_offset = 0
    for other in ("C0", "M1"):
        for metric, m2_values in arrays["M2"].items():
            other_values = arrays[other][metric]
            delta = m2_values - other_values
            low, high = bootstrap_mean_delta(delta, seed_offset)
            seed_offset += 1
            is_lower = metric in lower_is_better
            favorable = delta < -1e-12 if is_lower else delta > 1e-12
            unfavorable = delta > 1e-12 if is_lower else delta < -1e-12
            ties = ~(favorable | unfavorable)
            if len(delta) > 1:
                leave_one_out = (delta.sum() - delta) / (len(delta) - 1)
            else:
                leave_one_out = delta.copy()
            observed = float(delta.mean())
            sign_robust = bool(
                np.all(leave_one_out <= 0.0) if is_lower and observed <= 0.0
                else np.all(leave_one_out >= 0.0) if (not is_lower and observed >= 0.0)
                else np.all(leave_one_out >= 0.0) if is_lower
                else np.all(leave_one_out <= 0.0)
            )
            rows.append(
                {
                    "comparison": f"M2-{other}",
                    "metric": metric,
                    "preferred_direction": "lower" if is_lower else "higher",
                    "n_pairs": len(delta),
                    "other_mean": float(other_values.mean()),
                    "m2_mean": float(m2_values.mean()),
                    "mean_delta_m2_minus_other": observed,
                    "median_paired_delta": float(np.median(delta)),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "favorable_pairs": int(favorable.sum()),
                    "unfavorable_pairs": int(unfavorable.sum()),
                    "ties": int(ties.sum()),
                    "leave_one_out_delta_min": float(leave_one_out.min()),
                    "leave_one_out_delta_max": float(leave_one_out.max()),
                    "leave_one_out_sign_robust": sign_robust,
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / "paired_statistics.csv", index=False)


def shannon(counter: Counter) -> float:
    probabilities = np.asarray(list(counter.values()), dtype=float)
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


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


def write_distribution_analysis() -> None:
    rows: list[dict[str, object]] = []
    counters: dict[str, dict[str, Counter]] = {}
    for method in METHODS:
        structures = read(ROOT / "structures" / f"{method}_generated.extxyz", index=":")
        element_counts: Counter = Counter()
        atom_counts: Counter = Counter()
        system_counts: Counter = Counter()
        formula_counts: Counter = Counter()
        for structure in structures:
            elements = structure.get_chemical_symbols()
            element_counts.update(elements)
            atom_counts[len(structure)] += 1
            system_counts["-".join(sorted(set(elements)))] += 1
            formula_counts[Composition(Counter(elements)).reduced_formula] += 1
        counters[method] = {
            "element_frequency": element_counts,
            "atom_count_frequency": atom_counts,
            "composition_family_frequency": system_counts,
            "reduced_formula_frequency": formula_counts,
        }
        atom_values = np.asarray([len(item) for item in structures], dtype=float)
        summaries = {
            "structure_count": len(structures),
            "total_atom_count": int(atom_values.sum()),
            "unique_element_count": len(element_counts),
            "unique_atom_count_count": len(atom_counts),
            "unique_composition_family_count": len(system_counts),
            "unique_reduced_formula_count": len(formula_counts),
            "element_entropy_nats": shannon(element_counts),
            "atom_count_entropy_nats": shannon(atom_counts),
            "composition_family_entropy_nats": shannon(system_counts),
            "atom_count_mean": float(atom_values.mean()),
            "atom_count_median": float(np.median(atom_values)),
            "atom_count_std": float(atom_values.std(ddof=0)),
            "atom_count_min": int(atom_values.min()),
            "atom_count_max": int(atom_values.max()),
            "largest_composition_family_share": max(system_counts.values()) / len(structures),
            "largest_reduced_formula_share": max(formula_counts.values()) / len(structures),
        }
        for item, value in summaries.items():
            rows.append(
                {
                    "method": method,
                    "category": "summary",
                    "item": item,
                    "count": "",
                    "share": "",
                    "value": value,
                    "reference_method": "",
                }
            )
        for category, counter in counters[method].items():
            total = sum(counter.values())
            for item, count in sorted(counter.items(), key=lambda pair: str(pair[0])):
                rows.append(
                    {
                        "method": method,
                        "category": category,
                        "item": item,
                        "count": count,
                        "share": count / total,
                        "value": "",
                        "reference_method": "",
                    }
                )
    for method, reference in (("M1", "C0"), ("M2", "C0"), ("M2", "M1")):
        for category in (
            "element_frequency",
            "atom_count_frequency",
            "composition_family_frequency",
            "reduced_formula_frequency",
        ):
            rows.append(
                {
                    "method": method,
                    "category": "distribution_comparison",
                    "item": f"{category}_jensen_shannon_nats",
                    "count": "",
                    "share": "",
                    "value": jensen_shannon(
                        counters[method][category], counters[reference][category]
                    ),
                    "reference_method": reference,
                }
            )
    with (ROOT / "distribution_analysis.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "method",
                "category",
                "item",
                "count",
                "share",
                "value",
                "reference_method",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    write_quality_results()
    write_paired_statistics()
    write_distribution_analysis()
    print((ROOT / "quality_results.csv").read_text())
    paired = pd.read_csv(ROOT / "paired_statistics.csv")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
