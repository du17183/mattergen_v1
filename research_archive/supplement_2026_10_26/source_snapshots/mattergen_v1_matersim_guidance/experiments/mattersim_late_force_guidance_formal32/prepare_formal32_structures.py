"""Aggregate Formal32 structures and create the required generation manifest."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
METHODS = ("C0", "F0")
SEEDS = tuple(range(730000, 730032))


def main() -> None:
    output = ROOT / "structures"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = []
    for method in METHODS:
        structures = []
        for seed in SEEDS:
            run_dir = ROOT / "generation" / method / str(seed)
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            if summary.get("success") is not True or int(summary["seed"]) != seed:
                raise RuntimeError(f"{method}/{seed}: unsuccessful frozen run")
            items = read(run_dir / "generated_crystals.extxyz", index=":")
            if len(items) != 1:
                raise RuntimeError(f"{method}/{seed}: expected exactly one structure")
            atoms = items[0]
            atoms.info.update(sample_seed=seed, sample_index_within_seed=0,
                              formal32_method=method)
            structures.append(atoms)
            rows.append({
                "method": method, "seed": seed, "success": True,
                "formula": atoms.get_chemical_formula(), "num_atoms": len(atoms),
                "elapsed_seconds": float(summary["elapsed_seconds"]),
                "potential_load_seconds": float(summary["potential_load_seconds"]),
                "end_to_end_seconds": float(summary["end_to_end_seconds"]),
                "physical_gpu": summary["physical_gpu"],
                "peak_allocated_bytes": int(summary["peak_allocated_bytes"]),
                "mattergen_score_calls": int(summary["mattergen_score_calls"]),
                "guidance_eligible": int(summary.get("mattersim_guidance_eligible", 0)),
                "guidance_attempted": int(summary.get("mattersim_guidance_eligible", 0)),
                "guidance_accepted": int(summary.get("mattersim_guidance_accepted", 0)),
                "guidance_rejected": int(summary.get("mattersim_guidance_fallbacks", 0)),
                "guidance_fallbacks": int(summary.get("mattersim_guidance_fallbacks", 0)),
                "guidance_seconds": float(summary.get("mattersim_guidance_seconds", 0.0)),
                "adaptive_cfg_used": bool(summary["adaptive_cfg_used"]),
                "surrogate_property_eval": True, "dft_verified": False,
            })
        write(output / f"{method}_generated.extxyz", structures)
    with (ROOT / "generation_manifest.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "methods": list(METHODS), "paired_seeds": list(SEEDS),
        "n_per_method": len(SEEDS), "order": "method blocks, ascending paired seed",
        "all_registered_seeds_included": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
