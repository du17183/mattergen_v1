"""Scientific diagnostics on the already generated Formal256 structures.

No model call, generation, relaxation, or external evaluator is performed.
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "experiments/tcl_formal256"
OUT = Path(__file__).resolve().parent
METHODS = ("C0", "FT0", "TCL")


def scalar(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def q(values, probabilities=(0.01, 0.05, 0.5, 0.95, 0.99)):
    array = np.asarray(values, dtype=float)
    return {f"p{int(p * 100):02d}": float(np.quantile(array, p)) for p in probabilities}


def distances(atoms):
    matrix = atoms.get_all_distances(mic=True)
    np.fill_diagonal(matrix, np.inf)
    nearest = matrix.min(axis=1)
    upper = matrix[np.triu_indices(len(atoms), k=1)]
    return nearest, upper


def descriptors(atoms, relaxed):
    nearest, pairs = distances(atoms)
    relaxed_nearest, _ = distances(relaxed)
    lengths = np.asarray(atoms.cell.lengths())
    angles = np.asarray(atoms.cell.angles())
    matrix = np.asarray(atoms.cell.array)
    volume = float(abs(np.linalg.det(matrix)))
    relaxed_volume = float(relaxed.get_volume())
    forces = np.linalg.norm(atoms.get_forces(), axis=1)
    result = {
        "num_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(mode="hill"),
        "elements": ";".join(sorted(set(atoms.get_chemical_symbols()))),
        "num_elements": len(set(atoms.get_chemical_symbols())),
        "min_distance": float(nearest.min()),
        "nearest_neighbor_p05": float(np.quantile(nearest, 0.05)),
        "nearest_neighbor_median": float(np.median(nearest)),
        "nearest_neighbor_mean": float(nearest.mean()),
        "pairs_lt_1A": int((pairs < 1.0).sum()),
        "pairs_lt_1p5A": int((pairs < 1.5).sum()),
        "pairs_lt_2A": int((pairs < 2.0).sum()),
        "mean_neighbors_lt_3A": float((pairs < 3.0).sum() * 2 / len(atoms)),
        "relaxed_min_distance": float(relaxed_nearest.min()),
        "cell_a": float(lengths[0]),
        "cell_b": float(lengths[1]),
        "cell_c": float(lengths[2]),
        "cell_alpha": float(angles[0]),
        "cell_beta": float(angles[1]),
        "cell_gamma": float(angles[2]),
        "cell_length_min": float(lengths.min()),
        "cell_length_max": float(lengths.max()),
        "cell_length_aspect_ratio": float(lengths.max() / lengths.min()),
        "cell_volume": volume,
        "cell_volume_per_atom": volume / len(atoms),
        "cell_determinant": float(np.linalg.det(matrix)),
        "cell_condition_number": float(np.linalg.cond(matrix)),
        "relaxation_volume_change_fraction": (relaxed_volume - volume) / volume,
        "initial_atomic_force_mean": float(forces.mean()),
        "initial_max_force": float(forces.max()),
    }
    return result


def load_quality(method):
    with gzip.open(FORMAL / "quality" / method / "official_detailed.json.gz", "rt") as stream:
        detailed = json.load(stream)
    per = pd.read_csv(FORMAL / "quality" / method / "per_structure.csv")
    return detailed, per


def bootstrap_delta(a, b, seed=20260914, n_resamples=20000):
    """Paired mean a-b percentile interval."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(n_resamples, len(delta)))
    sampled = delta[indices].mean(axis=1)
    return float(delta.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def correlation_rows(frame):
    rows = []
    predictors = [
        "min_distance", "nearest_neighbor_p05", "nearest_neighbor_median",
        "cell_volume_per_atom", "cell_condition_number", "cell_length_aspect_ratio",
        "pairs_lt_1p5A", "mean_neighbors_lt_3A",
    ]
    outcomes = [
        "initial_max_force", "atomic_force_mean", "rmsd", "relaxation_steps", "e_hull",
    ]
    for method in METHODS:
        sub = frame[frame.method == method]
        for x in predictors:
            for y in outcomes:
                rho, p = stats.spearmanr(sub[x], sub[y], nan_policy="omit")
                rows.append({
                    "method": method, "predictor": x, "outcome": y,
                    "spearman_rho": float(rho), "two_sided_p": float(p), "n": len(sub),
                })
    return rows


def composition_vector(atoms):
    counts = Counter(atoms.get_chemical_symbols())
    total = len(atoms)
    return {key: value / total for key, value in counts.items()}


def composition_distance(a, b):
    keys = set(a) | set(b)
    l1 = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)
    intersection = len(set(a) & set(b))
    union = len(set(a) | set(b))
    return l1, intersection / union if union else 1.0


def composition_summary(frames, structural):
    rows = []
    element_rows = []
    for method in METHODS:
        all_counts = Counter()
        bad_counts = Counter()
        normal_counts = Counter()
        bad_seeds = set(structural.loc[(structural.method == method) & structural.severe, "seed"])
        for atoms in frames[method]:
            counts = Counter(atoms.get_chemical_symbols())
            all_counts.update(counts)
            (bad_counts if int(atoms.info["sample_seed"]) in bad_seeds else normal_counts).update(counts)
        for group, counts in (("all", all_counts), ("bad", bad_counts), ("normal", normal_counts)):
            total = sum(counts.values())
            probabilities = np.asarray(list(counts.values()), dtype=float) / total
            rows.append({
                "method": method,
                "group": group,
                "atom_count": total,
                "distinct_elements": len(counts),
                "element_entropy_nats": float(-(probabilities * np.log(probabilities)).sum()),
                "Mn_fraction": counts["Mn"] / total,
                "Eu_fraction": counts["Eu"] / total,
                "O_fraction": counts["O"] / total,
                "C_fraction": counts["C"] / total,
            })
            for element, count in counts.items():
                element_rows.append({
                    "method": method, "group": group, "element": element,
                    "count": count, "fraction": count / total,
                })
    return rows, element_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    relaxed_frames = {}
    detailed = {}
    per = {}
    records = []
    for method in METHODS:
        frames[method] = read(FORMAL / "relaxation" / method / "initial_with_properties.extxyz", index=":")
        relaxed_frames[method] = read(FORMAL / "relaxation" / method / "relaxed.extxyz", index=":")
        detailed[method], per[method] = load_quality(method)
        if not (len(frames[method]) == len(relaxed_frames[method]) == len(per[method]) == 256):
            raise RuntimeError(f"incomplete existing results for {method}")
        for index, (atoms, relaxed) in enumerate(zip(frames[method], relaxed_frames[method])):
            seed = int(atoms.info["sample_seed"])
            if seed != int(per[method].iloc[index].sample_seed):
                raise RuntimeError("seed ordering mismatch")
            force = float(per[method].iloc[index].pre_relaxation_max_force)
            desc = descriptors(atoms, relaxed)
            if not math.isclose(desc["initial_max_force"], force, rel_tol=2e-5, abs_tol=2e-6):
                raise RuntimeError(f"force mismatch for {method} {seed}")
            rmsd = float(detailed[method]["rmsd_from_relaxation"][index])
            row = {
                "method": method,
                "seed": seed,
                **desc,
                "e_hull": float(detailed[method]["energy_above_hull_per_atom"][index]),
                "stable": bool(detailed[method]["stable"][index]),
                "nus": bool(detailed[method]["novel_unique_stable"][index]),
                "novel": bool(detailed[method]["novel"][index]),
                "unique": bool(detailed[method]["unique"][index]),
                "rmsd": rmsd,
                "atomic_force_mean": desc["initial_atomic_force_mean"],
                "relaxation_steps": int(per[method].iloc[index].relaxation_steps),
            }
            row["severe"] = bool(
                row["rmsd"] > 0.5
                or row["initial_max_force"] > 1.0
                or row["relaxation_steps"] > 200
            )
            records.append(row)

    structural = pd.DataFrame(records).sort_values(["method", "seed"])
    structural.to_csv(OUT / "structural_diagnostics.csv", index=False)

    correlations = correlation_rows(structural)
    pd.DataFrame(correlations).to_csv(OUT / "structure_outcome_correlations.csv", index=False)

    summaries = []
    continuous = [
        "min_distance", "nearest_neighbor_p05", "nearest_neighbor_median",
        "cell_volume_per_atom", "cell_condition_number", "cell_length_aspect_ratio",
        "initial_max_force", "rmsd", "relaxation_steps", "e_hull",
    ]
    for method in METHODS:
        for group, mask in (
            ("all", structural.method == method),
            ("bad", (structural.method == method) & structural.severe),
            ("normal", (structural.method == method) & ~structural.severe),
        ):
            sub = structural[mask]
            for metric in continuous:
                values = sub[metric].astype(float)
                summaries.append({
                    "method": method, "group": group, "metric": metric, "n": len(values),
                    "mean": float(values.mean()), **q(values), "max": float(values.max()),
                })
    pd.DataFrame(summaries).to_csv(OUT / "structural_summary.csv", index=False)

    paired = []
    by_method = {m: structural[structural.method == m].set_index("seed") for m in METHODS}
    for seed in range(83000, 83256):
        c0_atoms = frames["C0"][seed - 83000]
        tcl_atoms = frames["TCL"][seed - 83000]
        ft0_atoms = frames["FT0"][seed - 83000]
        c0_vec, tcl_vec, ft0_vec = map(composition_vector, (c0_atoms, tcl_atoms, ft0_atoms))
        tcl_l1, tcl_jaccard = composition_distance(tcl_vec, c0_vec)
        ft0_l1, ft0_jaccard = composition_distance(ft0_vec, c0_vec)
        paired.append({
            "seed": seed,
            "num_atoms": len(c0_atoms),
            "c0_formula": by_method["C0"].loc[seed, "formula"],
            "ft0_formula": by_method["FT0"].loc[seed, "formula"],
            "tcl_formula": by_method["TCL"].loc[seed, "formula"],
            "tcl_exact_composition_as_c0": Counter(tcl_atoms.get_chemical_symbols()) == Counter(c0_atoms.get_chemical_symbols()),
            "ft0_exact_composition_as_c0": Counter(ft0_atoms.get_chemical_symbols()) == Counter(c0_atoms.get_chemical_symbols()),
            "tcl_c0_composition_fraction_l1": tcl_l1,
            "ft0_c0_composition_fraction_l1": ft0_l1,
            "tcl_c0_element_set_jaccard": tcl_jaccard,
            "ft0_c0_element_set_jaccard": ft0_jaccard,
            "tcl_severe": bool(by_method["TCL"].loc[seed, "severe"]),
            "c0_severe": bool(by_method["C0"].loc[seed, "severe"]),
        })
    pd.DataFrame(paired).to_csv(OUT / "paired_composition_changes.csv", index=False)

    composition_rows, element_rows = composition_summary(frames, structural)
    pd.DataFrame(composition_rows).to_csv(OUT / "composition_summary.csv", index=False)
    pd.DataFrame(element_rows).to_csv(OUT / "element_frequencies.csv", index=False)

    tcl_bad = structural[(structural.method == "TCL") & structural.severe].copy()
    tcl_bad["severity_score"] = (
        np.maximum(tcl_bad.rmsd / 0.5, 1)
        + np.maximum(tcl_bad.initial_max_force / 1.0, 1)
        + np.maximum(tcl_bad.relaxation_steps / 200.0, 1)
    )
    tcl_bad.sort_values("severity_score", ascending=False).to_csv(OUT / "tcl_bad_seed_case_studies.csv", index=False)

    key = {}
    for method in METHODS:
        sub = by_method[method]
        key[method] = {
            "severe_count": int(sub.severe.sum()),
            "min_distance_quantiles": q(sub.min_distance),
            "volume_per_atom_quantiles": q(sub.cell_volume_per_atom),
            "cell_condition_quantiles": q(sub.cell_condition_number),
        }
    for method in ("FT0", "TCL"):
        for metric in ("min_distance", "cell_volume_per_atom", "cell_condition_number"):
            mean, low, high = bootstrap_delta(by_method[method][metric], by_method["C0"][metric])
            key[f"{method}_minus_C0_{metric}"] = {"mean": mean, "ci95": [low, high]}
    paired_frame = pd.DataFrame(paired)
    key["composition"] = {
        "tcl_exact_match_count": int(paired_frame.tcl_exact_composition_as_c0.sum()),
        "ft0_exact_match_count": int(paired_frame.ft0_exact_composition_as_c0.sum()),
        "tcl_c0_l1_mean": float(paired_frame.tcl_c0_composition_fraction_l1.mean()),
        "ft0_c0_l1_mean": float(paired_frame.ft0_c0_composition_fraction_l1.mean()),
        "tcl_c0_jaccard_mean": float(paired_frame.tcl_c0_element_set_jaccard.mean()),
        "ft0_c0_jaccard_mean": float(paired_frame.ft0_c0_element_set_jaccard.mean()),
    }
    with (OUT / "structural_key_results.json").open("w", encoding="utf-8") as stream:
        json.dump(key, stream, indent=2, sort_keys=True)
    print(json.dumps(key, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
