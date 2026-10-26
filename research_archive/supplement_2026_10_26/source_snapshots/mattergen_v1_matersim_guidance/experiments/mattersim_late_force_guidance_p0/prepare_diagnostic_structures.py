"""Aggregate frozen A0 final and clean-estimate structures in seed order."""
from __future__ import annotations

import json
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent


def main() -> None:
    finals = []
    physics = []
    manifest = []
    for seed in range(710000, 710032):
        run = ROOT / "diagnostic_trajectories" / str(seed)
        final = read(run / "generated_crystals.extxyz")
        final.info.update(seed=seed, requested_t_norm=0.0,
                          actual_model_t=0.001,
                          state_kind="final_after_eps_predictor")
        finals.append(final)
        for requested in (0.10, 0.05, 0.02, 0.01):
            atoms = read(run / "clean_estimates" / f"x0_hat_t_{requested:.2f}.extxyz")
            atoms.info.update(seed=seed, requested_t_norm=requested,
                              actual_model_t=requested,
                              state_kind="predicted_clean_x0")
            physics.append(atoms)
        physics.append(final.copy())
        manifest.append({
            "seed": seed,
            "final_index": len(finals) - 1,
            "physics_indices": list(range(len(physics) - 5, len(physics))),
            "physics_t_order": [0.10, 0.05, 0.02, 0.01, 0.00],
        })
    write(ROOT / "diagnostic_final_generated.extxyz", finals)
    write(ROOT / "diagnostic_physics_structures.extxyz", physics)
    (ROOT / "diagnostic_structure_manifest.json").write_text(
        json.dumps({"n_final": 32, "n_physics": 160, "rows": manifest},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"n_final": len(finals), "n_physics": len(physics)}))


if __name__ == "__main__":
    main()

