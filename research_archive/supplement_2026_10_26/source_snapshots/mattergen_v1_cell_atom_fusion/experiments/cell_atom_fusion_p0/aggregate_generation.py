"""Materialize method-wise seed-ordered structures for evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
METHODS = ("B0", "C0")
SEEDS = tuple(range(110000, 110016))


def main() -> None:
    output_dir = ROOT / "structures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        structures = []
        for seed in SEEDS:
            run_dir = ROOT / "generation" / method / str(seed)
            summary = json.loads((run_dir / "run_summary.json").read_text())
            if summary["method"] != method or int(summary["seed"]) != seed:
                raise RuntimeError(f"method/seed mismatch: {method}/{seed}")
            if (
                int(summary["sampling_steps"]) != 1000
                or int(summary["corrector_steps_per_timestep"]) != 1
                or float(summary["guidance_scale"]) != 2.0
                or float(summary["target"]["dft_mag_density"]) != 0.2
                or bool(summary["adaptive_cfg_used"])
            ):
                raise RuntimeError("frozen sampling configuration changed")
            generated = read(run_dir / "generated_crystals.extxyz", index=":")
            if len(generated) != 1:
                raise RuntimeError(f"expected one structure: {method}/{seed}")
            atom = generated[0]
            atom.info.update(
                sample_seed=seed, benchmark_method=method,
                surrogate_property_eval=True, dft_verified=False,
            )
            structures.append(atom)
        output = output_dir / f"{method}_generated.extxyz"
        if output.exists():
            raise FileExistsError(output)
        write(output, structures)
        print(json.dumps({"method": method, "count": len(structures)}), flush=True)


if __name__ == "__main__":
    main()
