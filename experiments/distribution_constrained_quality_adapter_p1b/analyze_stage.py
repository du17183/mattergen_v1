"""Summarize a completed P1b stage without changing experimental artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd
from pymatgen.core import Composition


EXPERIMENT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen16", "confirm32"), required=True)
    parser.add_argument("--candidate")
    return parser.parse_args()


def jensen_shannon(left: Counter, right: Counter) -> float:
    keys = sorted(set(left) | set(right), key=str)
    p = np.asarray([left.get(key, 0) for key in keys], dtype=float)
    q = np.asarray([right.get(key, 0) for key in keys], dtype=float)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    left_mask = p > 0
    right_mask = q > 0
    return float(
        0.5
        * np.sum(p[left_mask] * np.log(p[left_mask] / midpoint[left_mask]))
        + 0.5
        * np.sum(q[right_mask] * np.log(q[right_mask] / midpoint[right_mask]))
    )


def main() -> None:
    args = parse_args()
    root = EXPERIMENT_ROOT / args.stage
    if args.stage == "screen16":
        methods = ("C0", "M1", "M2", "M2-L1", "M2-L2")
        expected_seeds = list(range(71000, 71016))
    else:
        if args.candidate not in {"M2-L1", "M2-L2"}:
            raise ValueError("confirm32 requires --candidate M2-L1 or M2-L2")
        methods = ("C0", "M1", "M2", args.candidate)
        expected_seeds = list(range(72000, 72032))

    generation = pd.read_csv(root / "generation_results.csv")
    rows = []
    distributions: dict[str, dict[str, Counter]] = {}
    for method in methods:
        method_generation = generation[
            generation["method"] == method
        ].sort_values("seed")
        if method_generation["seed"].astype(int).tolist() != expected_seeds:
            raise RuntimeError(f"{method}: seed order mismatch")
        if not method_generation["success"].astype(bool).all():
            raise RuntimeError(f"{method}: generation failure")
        summary = json.loads(
            (root / "quality" / method / "quality_summary.json").read_text()
        )
        with gzip.open(
            root / "quality" / method / "official_detailed.json.gz", "rt"
        ) as stream:
            detail = json.load(stream)
        e_hull = np.asarray(detail["energy_above_hull_per_atom"], dtype=float)
        rmsd = np.asarray(detail["rmsd_from_relaxation"], dtype=float)
        initial = read(
            root
            / "relaxation"
            / method
            / "initial_with_properties.extxyz",
            index=":",
        )
        atom_force_norms = [
            np.linalg.norm(item.get_forces(), axis=1) for item in initial
        ]
        all_force_norms = np.concatenate(atom_force_norms)
        structure_force_means = np.asarray(
            [values.mean() for values in atom_force_norms]
        )
        structure_force_max = np.asarray(
            [values.max() for values in atom_force_norms]
        )
        relaxation = json.loads(
            (
                root
                / "relaxation"
                / method
                / "relaxation_summary.json"
            ).read_text()
        )
        steps = np.asarray(relaxation["relaxation_steps"], dtype=float)
        structures = read(
            root / "structures" / f"{method}_generated.extxyz", index=":"
        )
        element_counts: Counter = Counter()
        family_counts: Counter = Counter()
        formula_counts: Counter = Counter()
        atom_counts: Counter = Counter()
        for structure in structures:
            elements = structure.get_chemical_symbols()
            element_counts.update(elements)
            family_counts["-".join(sorted(set(elements)))] += 1
            formula_counts[
                Composition(Counter(elements)).reduced_formula
            ] += 1
            atom_counts[len(structure)] += 1
        distributions[method] = {
            "element": element_counts,
            "family": family_counts,
            "formula": formula_counts,
            "atom_count": atom_counts,
        }
        official = summary["official_metrics"]
        rows.append(
            {
                "stage": args.stage,
                "method": method,
                "n": summary["n"],
                "generation_success_rate": float(
                    method_generation["success"].astype(bool).mean()
                ),
                "mattersim_success_rate": summary[
                    "relaxation_success_rate"
                ],
                "dft_verified": False,
                "e_hull_mean_ev_per_atom": official[
                    "avg_energy_above_hull_per_atom"
                ],
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
                "atomic_force_p95_ev_per_a": float(
                    np.quantile(all_force_norms, 0.95)
                ),
                "structure_force_mean_ev_per_a": float(
                    structure_force_means.mean()
                ),
                "structure_max_force_mean_ev_per_a": float(
                    structure_force_max.mean()
                ),
                "structure_max_force_p95_ev_per_a": float(
                    np.quantile(structure_force_max, 0.95)
                ),
                "force_max_ev_per_a": float(all_force_norms.max()),
                "relaxation_steps_mean": float(steps.mean()),
                "relaxation_steps_max": int(steps.max()),
                "composition_family_count": len(family_counts),
                "unique_reduced_formula_count": len(formula_counts),
                "unique_element_count": len(element_counts),
                "atom_count_unique_count": len(atom_counts),
                "atom_count_mean": float(
                    np.mean([len(item) for item in structures])
                ),
            }
        )

    frame = pd.DataFrame(rows)
    for method in methods:
        for reference in ("C0", "M1", "M2"):
            if method == reference:
                continue
            for category in ("element", "family", "formula", "atom_count"):
                frame.loc[
                    frame["method"] == method,
                    f"{category}_jsd_vs_{reference}",
                ] = jensen_shannon(
                    distributions[method][category],
                    distributions[reference][category],
                )
    frame.to_csv(root / "quality_results.csv", index=False)

    distribution_rows = []
    for method in methods:
        for category, counter in distributions[method].items():
            total = sum(counter.values())
            for item, count in sorted(
                counter.items(), key=lambda pair: str(pair[0])
            ):
                distribution_rows.append(
                    {
                        "stage": args.stage,
                        "method": method,
                        "category": category,
                        "item": item,
                        "count": count,
                        "share": count / total,
                    }
                )
    pd.DataFrame(distribution_rows).to_csv(
        root / "distribution_analysis.csv", index=False
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
