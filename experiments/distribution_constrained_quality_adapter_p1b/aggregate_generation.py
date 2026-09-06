"""Materialize seed-ordered P1b screen structures for MatterSim evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent / "screen16"
METHODS = ("C0", "M1", "M2", "M2-L1", "M2-L2")
SEEDS = tuple(range(71000, 71016))


def main() -> None:
    structures_dir = ROOT / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        atoms = []
        for seed in SEEDS:
            run_dir = ROOT / "generation" / method / str(seed)
            summary_path = run_dir / "run_summary.json"
            structure_path = run_dir / "generated_crystals.extxyz"
            if not summary_path.is_file() or not structure_path.is_file():
                raise FileNotFoundError(
                    f"incomplete generation for {method} seed {seed}"
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("success") is not True:
                raise ValueError(f"failed summary for {method} seed {seed}")
            if summary["method"] != method or int(summary["seed"]) != seed:
                raise ValueError(f"method/seed mismatch for {method} seed {seed}")
            if (
                int(summary["sampling_steps"]) != 1000
                or int(summary["corrector_steps_per_time"]) != 1
                or float(summary["guidance_scale"]) != 2.0
                or float(summary["target"]["dft_mag_density"]) != 0.1
            ):
                raise ValueError(
                    f"sampling config mismatch for {method} seed {seed}"
                )
            generated = read(structure_path, index=":")
            if len(generated) != 1:
                raise ValueError(
                    f"expected one structure for {method} seed {seed}"
                )
            atom = generated[0]
            atom.info["sample_seed"] = seed
            atom.info["sample_index_within_seed"] = 0
            atom.info["benchmark_method"] = method
            atom.info["dft_verified"] = False
            atoms.append(atom)
        output = structures_dir / f"{method}_generated.extxyz"
        if output.exists():
            raise FileExistsError(output)
        write(output, atoms)
        print(
            json.dumps(
                {
                    "method": method,
                    "count": len(atoms),
                    "seed_min": min(SEEDS),
                    "seed_max": max(SEEDS),
                }
            )
        )


if __name__ == "__main__":
    main()
