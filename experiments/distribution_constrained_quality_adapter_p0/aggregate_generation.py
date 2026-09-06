"""Materialize seed-ordered structures for the P0 MatterSim evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
METHODS = ("C0", "M1", "M2")
SEEDS = tuple(range(78000, 78008))


def main() -> None:
    structures_dir = ROOT / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {"expected_per_method": len(SEEDS), "methods": {}}
    for method in METHODS:
        atoms = []
        for seed in SEEDS:
            run_dir = ROOT / "generation" / method / str(seed)
            summary_path = run_dir / "run_summary.json"
            structure_path = run_dir / "generated_crystals.extxyz"
            if not summary_path.is_file() or not structure_path.is_file():
                raise FileNotFoundError(f"incomplete generation for {method} seed {seed}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("success") is not True or int(summary["seed"]) != seed:
                raise ValueError(f"invalid run summary for {method} seed {seed}")
            generated = read(structure_path, index=":")
            if len(generated) != 1:
                raise ValueError(f"expected one structure for {method} seed {seed}")
            atom = generated[0]
            atom.info["sample_seed"] = seed
            atom.info["sample_index_within_seed"] = 0
            atom.info["benchmark_method"] = method
            atoms.append(atom)
        output = structures_dir / f"{method}_generated.extxyz"
        if output.exists():
            raise FileExistsError(output)
        write(output, atoms)
        audit["methods"][method] = {"count": len(atoms), "seeds": list(SEEDS)}
    audit["passed"] = all(item["count"] == len(SEEDS) for item in audit["methods"].values())
    (ROOT / "generation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
