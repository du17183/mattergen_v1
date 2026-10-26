"""Assemble the 64 final structures in audited seed order."""
from __future__ import annotations

import json
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(99000, 99064))


def main() -> None:
    structures = []
    manifest = []
    for seed in SEEDS:
        run_dir = ROOT / "trajectories" / str(seed)
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        atoms = read(run_dir / "generated_crystals.extxyz", index=":")
        if len(atoms) != 1 or int(summary["seed"]) != seed or summary["success"] is not True:
            raise RuntimeError(f"bad trajectory artifact for seed {seed}")
        item = atoms[0]
        item.info["sample_seed"] = seed
        item.info["sample_index_within_seed"] = 0
        item.info["diagnostic_subset"] = summary["subset"]
        structures.append(item)
        manifest.append({
            "index": len(structures) - 1,
            "seed": seed,
            "subset": summary["subset"],
            "formula": summary["formula"],
            "num_atoms": int(summary["num_atoms"]),
        })
    output = ROOT / "generated_diagnostic.extxyz"
    write(output, structures)
    (ROOT / "structure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"success": True, "n": len(structures), "output": str(output)}))


if __name__ == "__main__":
    main()
