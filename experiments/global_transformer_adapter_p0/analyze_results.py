"""Create the compact P0 quality table from formal MatterSim outputs."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
METHODS = ("C0", "MLP", "Transformer")
SEEDS = tuple(range(75000, 75008))


def main() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    rows = []
    for method in METHODS:
        generated = generation[generation["method"] == method].sort_values("seed")
        if generated["seed"].astype(int).tolist() != list(SEEDS):
            raise RuntimeError(f"{method}: seed order mismatch")
        summary = json.loads((ROOT / "quality" / method / "quality_summary.json").read_text())
        with gzip.open(ROOT / "quality" / method / "official_detailed.json.gz", "rt") as stream:
            detail = json.load(stream)
        e_hull = np.asarray(detail["energy_above_hull_per_atom"], dtype=float)
        rmsd = np.asarray(detail["rmsd_from_relaxation"], dtype=float)
        initial = read(ROOT / "relaxation" / method / "initial_with_properties.extxyz", index=":")
        force_norms = [np.linalg.norm(structure.get_forces(), axis=1) for structure in initial]
        all_force_norms = np.concatenate(force_norms)
        structure_force_means = np.asarray([values.mean() for values in force_norms])
        structure_force_max = np.asarray([values.max() for values in force_norms])
        relaxation = json.loads((ROOT / "relaxation" / method / "relaxation_summary.json").read_text())
        relaxation_steps = np.asarray(relaxation["relaxation_steps"], dtype=float)
        official = summary["official_metrics"]
        rows.append({
            "method": method,
            "n": summary["n"],
            "generation_success_rate": float(generated["success"].astype(bool).mean()),
            "mattersim_success_rate": summary["relaxation_success_rate"],
            "dft_verified": False,
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
            "structure_force_mean_ev_per_a": float(structure_force_means.mean()),
            "structure_max_force_mean_ev_per_a": float(structure_force_max.mean()),
            "force_p95_ev_per_a": float(np.quantile(all_force_norms, 0.95)),
            "force_max_ev_per_a": float(all_force_norms.max()),
            "relaxation_steps_mean": float(relaxation_steps.mean()),
            "relaxation_steps_max": int(relaxation_steps.max()),
        })
        print(json.dumps({
            "method": method,
            "seeds": list(SEEDS),
            "e_hull_by_seed": dict(zip(map(str, SEEDS), e_hull.tolist())),
            "rmsd_by_seed": dict(zip(map(str, SEEDS), rmsd.tolist())),
            "structure_max_force_by_seed": dict(zip(map(str, SEEDS), structure_force_max.tolist())),
            "relaxation_steps_by_seed": dict(zip(map(str, SEEDS), relaxation_steps.tolist())),
        }, indent=2), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "quality_results.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
