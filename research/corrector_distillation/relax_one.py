"""Relax one structure in an isolated MatterSim process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase.io import read, write
from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--potential-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-natoms-per-batch", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atoms = read(args.input.expanduser().resolve(), index=args.index)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    potential = Potential.from_checkpoint(
        device=args.device,
        load_path=str(args.potential_path.expanduser().resolve()),
        load_training_state=False,
    )
    result = BatchRelaxer(
        potential=potential,
        filter="EXPCELLFILTER",
        fmax=args.fmax,
        max_natoms_per_batch=args.max_natoms_per_batch,
    ).relax([atoms.copy()])
    if 0 not in result or not result[0]:
        raise RuntimeError("MatterSim returned no trajectory for the structure")
    trajectory = result[0]
    trajectory_path = output_dir / "trajectory.extxyz"
    write(trajectory_path, trajectory)
    summary = {
        "success": True,
        "formula": atoms.get_chemical_formula(),
        "num_atoms": len(atoms),
        "relaxation_steps": len(trajectory),
        "trajectory": str(trajectory_path),
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
