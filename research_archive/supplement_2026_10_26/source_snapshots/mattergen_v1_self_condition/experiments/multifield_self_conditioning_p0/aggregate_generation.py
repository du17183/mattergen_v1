"""Aggregate and materialize the 32-seed B0/S-off/S-on generation cohort."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
METHODS = ("B0", "S-off", "S-on")
SEEDS = tuple(range(95000, 95032))


def main() -> None:
    rows = []
    structures_dir = ROOT / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        atoms = []
        for seed in SEEDS:
            run_dir = ROOT / "generation" / method / str(seed)
            summary = json.loads((run_dir / "run_summary.json").read_text())
            if summary["method"] != method or int(summary["seed"]) != seed:
                raise RuntimeError(f"method/seed mismatch: {method} {seed}")
            if int(summary["mattergen_score_calls"]) != 2000:
                raise RuntimeError("sampling score-call contract mismatch")
            generated = read(run_dir / "generated_crystals.extxyz", index=":")
            if len(generated) != 1:
                raise RuntimeError(f"expected one structure: {method} {seed}")
            atom = generated[0]
            atom.info.update(
                sample_seed=seed,
                sample_index_within_seed=0,
                benchmark_method=method,
                dft_verified=False,
            )
            atoms.append(atom)
            rows.append({
                "method": method,
                "seed": seed,
                "success": summary["success"],
                "formula": summary["formula"],
                "num_atoms": summary["num_atoms"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "peak_allocated_bytes": summary["peak_allocated_bytes"],
                "mattergen_score_calls": summary["mattergen_score_calls"],
                "theoretical_baseline_score_calls": summary["theoretical_baseline_score_calls"],
                "extra_sampling_forwards": summary["extra_sampling_forwards"],
                "sc_cache_updates": summary["sc_cache_updates"],
                "sc_cache_uses": summary["sc_cache_uses"],
                "self_conditioning_enabled": summary["self_conditioning_enabled"],
                "guidance_scale": summary["guidance_scale"],
                "target_dft_mag_density": summary["target"]["dft_mag_density"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "sc_checkpoint_sha256": summary["sc_checkpoint_sha256"] or "",
                "adaptive_cfg_used": False,
                "dft_verified": False,
            })
        output = structures_dir / f"{method}_generated.extxyz"
        if output.exists():
            raise FileExistsError(output)
        write(output, atoms)
    output_csv = ROOT / "generation_results.csv"
    if output_csv.exists():
        raise FileExistsError(output_csv)
    with output_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"methods": METHODS, "seeds": [SEEDS[0], SEEDS[-1]], "rows": len(rows)}))


if __name__ == "__main__":
    main()
