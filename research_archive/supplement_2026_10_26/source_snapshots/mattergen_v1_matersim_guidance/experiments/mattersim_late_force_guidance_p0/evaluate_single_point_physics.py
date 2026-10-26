"""MatterSim-5M single-point diagnostic on clean estimates and final states."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from ase.io import read
from ase.units import GPa
from mattersim.forcefield.potential import MatterSimCalculator, Potential


ROOT = Path(__file__).resolve().parent
POTENTIAL_PATH = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/"
    "mattersim/mattersim-v1.0.0-5M.pth"
)
FIELDS = (
    "seed", "subset", "t_norm", "actual_model_t", "state_kind", "num_atoms",
    "energy_t", "energy_per_atom_t", "mean_force_t", "maxF_t", "stress_norm_t",
    "min_distance_t", "volume_per_atom_t", "finite_valid", "obvious_ood",
    "failure_reason", "evaluation_seconds",
)


def _minimum_distance(atoms) -> float:
    if len(atoms) < 2:
        return float("nan")
    distance = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def _load_compat(path: Path, device: str) -> Potential:
    """Permit the trusted local 1.1.2 checkpoint under PyTorch >=2.6."""
    original = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = compatible_load
    try:
        return Potential.from_checkpoint(
            device=device, load_path=str(path), load_training_state=False
        )
    finally:
        torch.load = original


def main() -> None:
    output = ROOT / "physics_single_point.csv"
    if output.exists():
        raise FileExistsError(output)
    potential = _load_compat(POTENTIAL_PATH, "cuda")
    calculator = MatterSimCalculator(
        potential=potential, compute_stress=True, device="cuda"
    )
    structures = read(ROOT / "diagnostic_physics_structures.extxyz", index=":")
    rows = []
    expected_t = (0.10, 0.05, 0.02, 0.01, 0.0)
    for index, atoms in enumerate(structures):
        seed = int(atoms.info.get("seed", 710000 + index // 5))
        t_norm = float(atoms.info.get("requested_t_norm", expected_t[index % 5]))
        started = time.perf_counter()
        row = {
            "seed": seed,
            "subset": "calibration" if seed <= 710015 else "validation",
            "t_norm": t_norm,
            "actual_model_t": float(
                atoms.info.get("actual_model_t", 0.001 if t_norm == 0 else t_norm)
            ),
            "state_kind": atoms.info.get(
                "state_kind",
                "final_after_eps_predictor" if t_norm == 0 else "predicted_clean_x0",
            ),
            "num_atoms": len(atoms),
        }
        try:
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            stress_gpa = np.asarray(atoms.get_stress(voigt=False), dtype=float) / GPa
            force_norm = np.linalg.norm(forces, axis=1)
            min_distance = _minimum_distance(atoms)
            volume_per_atom = float(atoms.get_volume() / len(atoms))
            values = np.concatenate(
                (np.asarray([energy, min_distance, volume_per_atom]),
                 forces.ravel(), stress_gpa.ravel())
            )
            finite = bool(np.isfinite(values).all())
            max_force = float(force_norm.max())
            energy_per_atom = float(energy / len(atoms))
            obvious_ood = bool(
                not finite
                or abs(energy_per_atom) > 20.0
                or max_force > 100.0
                or min_distance < 0.5
                or not 1.0 <= volume_per_atom <= 200.0
            )
            row.update(
                energy_t=energy,
                energy_per_atom_t=energy_per_atom,
                mean_force_t=float(force_norm.mean()),
                maxF_t=max_force,
                stress_norm_t=float(np.linalg.norm(stress_gpa)),
                min_distance_t=min_distance,
                volume_per_atom_t=volume_per_atom,
                finite_valid=finite,
                obvious_ood=obvious_ood,
                failure_reason="",
            )
        except BaseException as error:
            row.update(
                energy_t=float("nan"), energy_per_atom_t=float("nan"),
                mean_force_t=float("nan"), maxF_t=float("nan"),
                stress_norm_t=float("nan"), min_distance_t=float("nan"),
                volume_per_atom_t=float("nan"), finite_valid=False,
                obvious_ood=True,
                failure_reason=f"{type(error).__name__}: {str(error)[:300]}",
            )
        row["evaluation_seconds"] = time.perf_counter() - started
        rows.append(row)
        print(
            json.dumps({"index": index, "seed": seed, "t": t_norm,
                        "finite": row["finite_valid"], "ood": row["obvious_ood"]}),
            flush=True,
        )
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
