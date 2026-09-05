"""Relax a structure file one item at a time in a minimal MatterSim process."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read, write
from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures-path", type=Path, required=True)
    parser.add_argument("--potential-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-natoms-per-batch", type=int, default=512)
    parser.add_argument("--max-structures", type=int)
    parser.add_argument(
        "--cuda-3x3-det-workaround",
        action="store_true",
        help=(
            "Use the differentiable analytic 3x3 determinant on CUDA to avoid "
            "a torch.linalg.det driver failure observed on H20/CUDA 12.1."
        ),
    )
    return parser.parse_args()


def install_cuda_3x3_det_workaround() -> None:
    original_det = torch.linalg.det

    def stable_det(value: torch.Tensor, *, out: torch.Tensor | None = None):
        if value.is_cuda and value.shape[-2:] == (3, 3):
            determinant = (
                value[..., 0, 0]
                * (
                    value[..., 1, 1] * value[..., 2, 2]
                    - value[..., 1, 2] * value[..., 2, 1]
                )
                - value[..., 0, 1]
                * (
                    value[..., 1, 0] * value[..., 2, 2]
                    - value[..., 1, 2] * value[..., 2, 0]
                )
                + value[..., 0, 2]
                * (
                    value[..., 1, 0] * value[..., 2, 1]
                    - value[..., 1, 1] * value[..., 2, 0]
                )
            )
            if out is not None:
                out.copy_(determinant)
                return out
            return determinant
        return original_det(value, out=out) if out is not None else original_det(value)

    torch.linalg.det = stable_det


def main() -> None:
    args = parse_args()
    if args.cuda_3x3_det_workaround:
        install_cuda_3x3_det_workaround()
    input_atoms = read(
        args.structures_path.expanduser().resolve(),
        index=":",
        do_not_split_by_at_sign=True,
    )
    if args.max_structures is not None:
        input_atoms = input_atoms[: args.max_structures]
    if not input_atoms:
        raise ValueError("no structures to relax")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    potential_path = args.potential_path.expanduser().resolve()
    potential = Potential.from_checkpoint(
        device=args.device,
        load_path=str(potential_path),
        load_training_state=False,
    )

    relaxed_atoms = []
    first_atoms = []
    energies = []
    max_forces = []
    steps = []
    started = time.perf_counter()
    for index, item in enumerate(input_atoms):
        result = BatchRelaxer(
            potential=potential,
            filter="EXPCELLFILTER",
            fmax=args.fmax,
            max_natoms_per_batch=args.max_natoms_per_batch,
        ).relax([item.copy()])
        if 0 not in result or not result[0]:
            raise RuntimeError(f"MatterSim returned no trajectory for index {index}")
        trajectory = result[0]
        first = trajectory[0]
        relaxed = trajectory[-1]
        first_atoms.append(first)
        relaxed_atoms.append(relaxed)
        energies.append(float(relaxed.info["total_energy"]))
        max_forces.append(
            float(np.linalg.norm(first.arrays["forces"], axis=1).max())
        )
        steps.append(len(trajectory))
        print(
            json.dumps(
                {
                    "event": "structure_relaxed",
                    "index": index,
                    "completed": index + 1,
                    "total": len(input_atoms),
                    "steps": len(trajectory),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    write(output_dir / "initial_with_properties.extxyz", first_atoms)
    write(output_dir / "relaxed.extxyz", relaxed_atoms)
    summary = {
        "schema_version": 1,
        "success": True,
        "n": len(relaxed_atoms),
        "n_input": len(input_atoms),
        "device": args.device,
        "potential_path": str(potential_path),
        "fmax": args.fmax,
        "max_natoms_per_batch": args.max_natoms_per_batch,
        "cuda_3x3_det_workaround": args.cuda_3x3_det_workaround,
        "protocol": "one structure per BatchRelaxer call; one persistent potential",
        "relaxation_seconds": elapsed,
        "energies": energies,
        "pre_relaxation_max_forces": max_forces,
        "relaxation_steps": steps,
        "sample_seeds": [
            None
            if item.info.get("sample_seed") is None
            else int(item.info["sample_seed"])
            for item in input_atoms
        ],
        "formulas": [item.get_chemical_formula() for item in input_atoms],
    }
    with (output_dir / "relaxation_summary.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"completed": True, "n": len(relaxed_atoms)}, sort_keys=True))


if __name__ == "__main__":
    main()
