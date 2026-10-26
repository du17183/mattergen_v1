"""Aggregate the frozen 16-pair P0 generated structures in seed order."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
METHODS = ("B0", "F0")
SEEDS = tuple(range(710032, 710048))


def main() -> None:
    structures_dir = ROOT / "p0_structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHODS:
        atoms = []
        for seed in SEEDS:
            run_dir = ROOT / "p0_generation" / method / str(seed)
            summary = json.loads((run_dir / "run_summary.json").read_text())
            if summary.get("success") is not True or int(summary["seed"]) != seed:
                raise RuntimeError(f"{method}/{seed}: missing successful frozen run")
            items = read(run_dir / "generated_crystals.extxyz", index=":")
            if len(items) != 1:
                raise RuntimeError(f"{method}/{seed}: expected one structure")
            item = items[0]
            item.info["sample_seed"] = seed
            item.info["sample_index_within_seed"] = 0
            item.info["p0_method"] = method
            atoms.append(item)
            rows.append({
                "method": method,
                "seed": seed,
                "success": True,
                "formula": item.get_chemical_formula(),
                "num_atoms": len(item),
                "elapsed_seconds": float(summary["elapsed_seconds"]),
                "potential_load_seconds": float(summary["potential_load_seconds"]),
                "end_to_end_seconds": float(summary["end_to_end_seconds"]),
                "physical_gpu": summary["physical_gpu"],
                "mattergen_score_calls": int(summary["mattergen_score_calls"]),
                "guidance_eligible": int(summary.get("mattersim_guidance_eligible", 0)),
                "guidance_accepted": int(summary.get("mattersim_guidance_accepted", 0)),
                "guidance_fallbacks": int(summary.get("mattersim_guidance_fallbacks", 0)),
                "guidance_seconds": float(summary.get("mattersim_guidance_seconds", 0.0)),
            })
        write(structures_dir / f"{method}_generated.extxyz", atoms)
    path = ROOT / "p0_generation_results.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "methods": list(METHODS),
        "paired_seeds": list(SEEDS),
        "n_per_method": len(SEEDS),
        "order": "method blocks; ascending seed within each file",
        "seed_namespace_audited_fresh_before_run": True,
        "pre_run_scan": "No 710000-710099 or 720000-720099 references in mattergen_v1* worktrees.",
    }
    (structures_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
