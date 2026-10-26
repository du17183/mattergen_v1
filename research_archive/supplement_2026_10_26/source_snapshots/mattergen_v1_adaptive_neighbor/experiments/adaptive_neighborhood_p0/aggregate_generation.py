"""Aggregate and materialize the 32-seed B0/A0 generation cohort."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
METHODS = ("B0", "A0")
SEEDS = tuple(range(96000, 96032))


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
            atom.info.update(sample_seed=seed, sample_index_within_seed=0, benchmark_method=method, dft_verified=False)
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
                "guidance_scale": summary["guidance_scale"],
                "target_dft_mag_density": summary["target"]["dft_mag_density"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "gate_checkpoint_sha256": summary["gate_checkpoint_sha256"] or "",
                "graph_topology_modified": summary["graph_topology_modified"],
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
