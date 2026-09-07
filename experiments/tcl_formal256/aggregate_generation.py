"""Materialize method-wise generated structures for frozen TCL Formal256."""
from __future__ import annotations

import json
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
METHODS = ("C0", "FT0", "TCL")
SEEDS = tuple(range(83000, 83256))


def main() -> None:
    structures_dir = ROOT / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        structures = []
        for seed in SEEDS:
            run_dir = ROOT / "generation" / method / str(seed)
            summary = json.loads((run_dir / "run_summary.json").read_text())
            if summary["method"] != method or int(summary["seed"]) != seed:
                raise RuntimeError(f"method/seed mismatch: {method} {seed}")
            frozen = (
                int(summary["sampling_steps"]),
                int(summary["corrector_steps_per_time"]),
                float(summary["guidance_scale"]),
                float(summary["target"]["dft_mag_density"]),
            )
            if frozen != (1000, 1, 2.0, 0.1):
                raise RuntimeError(f"sampling configuration changed: {method} {seed}")
            generated = read(run_dir / "generated_crystals.extxyz", index=":")
            if len(generated) != 1:
                raise RuntimeError(f"expected one structure: {method} {seed}")
            structure = generated[0]
            structure.info.update(
                sample_seed=seed, sample_index_within_seed=0,
                benchmark_method=method, dft_verified=False,
            )
            structures.append(structure)
        output = structures_dir / f"{method}_generated.extxyz"
        if output.exists():
            raise FileExistsError(output)
        write(output, structures)
        print(json.dumps({"method": method, "count": len(structures)}), flush=True)


if __name__ == "__main__":
    main()
