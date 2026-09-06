"""Materialize seed-ordered structures for the M2-DB screen."""
from __future__ import annotations

import json
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
STAGE_ROOT = ROOT / "screen16"
METHODS = ("C0", "M1", "M2", "M2-DB")
SEEDS = tuple(range(73000, 73016))


def main() -> None:
    structures_dir = STAGE_ROOT / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        atoms = []
        for seed in SEEDS:
            run_dir = STAGE_ROOT / "generation" / method / str(seed)
            summary = json.loads((run_dir / "run_summary.json").read_text())
            if summary["method"] != method or int(summary["seed"]) != seed:
                raise RuntimeError(f"method/seed mismatch: {method} {seed}")
            if (int(summary["sampling_steps"]), int(summary["corrector_steps_per_time"])) != (1000, 1):
                raise RuntimeError("sampling configuration changed")
            generated = read(run_dir / "generated_crystals.extxyz", index=":")
            if len(generated) != 1:
                raise RuntimeError(f"expected one structure: {method} {seed}")
            atom = generated[0]
            atom.info.update(sample_seed=seed, sample_index_within_seed=0, benchmark_method=method, dft_verified=False)
            atoms.append(atom)
        output = structures_dir / f"{method}_generated.extxyz"
        if output.exists():
            raise FileExistsError(output)
        write(output, atoms)
        print(json.dumps({"method": method, "count": len(atoms)}), flush=True)


if __name__ == "__main__":
    main()
