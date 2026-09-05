"""Relax one benchmark arm once and compute official MatterGen metrics."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.evaluation.metrics.energy import (
    AvgEnergyAboveHullPerAtom,
    AvgRMSDFromRelaxation,
    FracNovelUniqueStableStructures,
    FracStableStructures,
)
from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.metrics.structure import FracNovelStructures, FracUniqueStructures
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.reference.reference_dataset_serializer import (
    LMDBBackedReferenceDatasetImpl,
    LMDBGZSerializer,
)
from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher


REQUIRED_METRICS = (
    AvgEnergyAboveHullPerAtom,
    FracStableStructures,
    FracNovelUniqueStableStructures,
    FracNovelStructures,
    FracUniqueStructures,
    AvgRMSDFromRelaxation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--structures-path", type=Path, required=True)
    parser.add_argument("--potential-path", type=Path, required=True)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-path", type=Path)
    reference.add_argument("--reference-lmdb-path", type=Path)
    parser.add_argument(
        "--reference-is-ordered",
        action="store_true",
        help="Use only after auditing every entry in the exact reference LMDB.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temporary-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-natoms-per-batch", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temporary_dir is not None:
        temporary_dir = args.temporary_dir.expanduser().resolve()
        temporary_dir.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(temporary_dir)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    # ASE otherwise treats ``@25`` in coverage-labelled filenames as an index
    # suffix and silently truncates the actual path.
    input_atoms = read(
        args.structures_path.expanduser().resolve(),
        index=":",
        do_not_split_by_at_sign=True,
    )
    if not input_atoms:
        raise ValueError("no generated structures")
    potential = Potential.from_checkpoint(
        device=args.device,
        load_path=str(args.potential_path.expanduser().resolve()),
        load_training_state=False,
    )
    def make_relaxer() -> BatchRelaxer:
        return BatchRelaxer(
            potential=potential,
            filter="EXPCELLFILTER",
            fmax=args.fmax,
            max_natoms_per_batch=args.max_natoms_per_batch,
        )

    started = time.perf_counter()
    relaxation_mode = "batch"
    batch_error = None
    relaxation_failures = []
    try:
        trajectories = make_relaxer().relax([item.copy() for item in input_atoms])
        if sorted(trajectories) != list(range(len(input_atoms))):
            raise RuntimeError("MatterSim did not return every input structure")
    except Exception as error:
        # A single singular/NaN structure can abort MatterSim's entire batch.
        # Retry pristine inputs independently so valid structures remain
        # measurable and every invalid index is explicitly reported.
        relaxation_mode = "individual_fallback"
        batch_error = f"{type(error).__name__}: {error}"
        trajectories = {}
        for index, item in enumerate(input_atoms):
            try:
                result = make_relaxer().relax([item.copy()])
                trajectories[index] = result[0]
            except Exception as item_error:
                relaxation_failures.append(
                    {
                        "index": index,
                        "error": f"{type(item_error).__name__}: {item_error}",
                    }
                )
    elapsed = time.perf_counter() - started
    successful_indices = sorted(trajectories)
    if not successful_indices:
        raise RuntimeError("MatterSim could not relax any input structure")

    relaxed_atoms = [trajectories[index][-1] for index in successful_indices]
    first_atoms = [trajectories[index][0] for index in successful_indices]
    energies = [float(item.info["total_energy"]) for item in relaxed_atoms]
    max_forces = [
        float(np.linalg.norm(item.arrays["forces"], axis=1).max()) for item in first_atoms
    ]
    steps = [len(trajectories[index]) for index in successful_indices]
    write(output_dir / "relaxed.extxyz", relaxed_atoms)
    np.save(output_dir / "energies.npy", np.asarray(energies))

    adaptor = AseAtomsAdaptor()
    original_structures = [adaptor.get_structure(input_atoms[index]) for index in successful_indices]
    relaxed_structures = [adaptor.get_structure(item) for item in relaxed_atoms]
    if args.reference_lmdb_path is not None:
        reference = ReferenceDataset(
            name="alex-mp MP2020 correction",
            impl=LMDBBackedReferenceDatasetImpl(
                args.reference_lmdb_path.expanduser().resolve(), cleanup_dir=False
            ),
        )
    else:
        reference = LMDBGZSerializer().deserialize(args.reference_path.expanduser().resolve())
    if args.reference_is_ordered:
        reference.__dict__["is_ordered"] = True
    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=relaxed_structures,
        energies=energies,
        original_structures=original_structures,
        reference=reference,
        structure_matcher=DefaultDisorderedStructureMatcher(),
        energy_correction_scheme=MaterialsProject2020Compatibility(),
    )
    official_metrics = evaluator.compute_metrics(
        metrics=REQUIRED_METRICS,
        save_as=output_dir / "official_metrics.json",
    )
    evaluator.as_dataframe(
        metrics=REQUIRED_METRICS,
        save_as=output_dir / "official_detailed.json.gz",
    )
    force_array = np.asarray(max_forces)
    summary = {
        "method": args.method,
        "n": len(relaxed_atoms),
        "n_input": len(input_atoms),
        "relaxation_success_rate": len(relaxed_atoms) / len(input_atoms),
        "relaxation_mode": relaxation_mode,
        "relaxation_batch_error": batch_error,
        "relaxation_failures": relaxation_failures,
        "relaxation_seconds": elapsed,
        "relaxation_seconds_per_structure": elapsed / len(relaxed_atoms),
        "pre_relaxation_max_force_mean": float(force_array.mean()),
        "pre_relaxation_max_force_median": float(np.median(force_array)),
        "pre_relaxation_max_force_p95": float(np.quantile(force_array, 0.95)),
        "pre_relaxation_max_force_max": float(force_array.max()),
        "relaxation_steps_mean": float(np.mean(steps)),
        "relaxation_steps_max": int(max(steps)),
        "official_metrics": official_metrics,
    }
    with (output_dir / "quality_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (output_dir / "per_structure.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "index",
                "sample_seed",
                "sample_index_within_seed",
                "formula",
                "num_atoms",
                "pre_relaxation_max_force",
                "relaxation_steps",
                "final_energy_ev",
            ),
        )
        writer.writeheader()
        for index, item, max_force, step_count, energy in zip(
            successful_indices, relaxed_atoms, max_forces, steps, energies
        ):
            writer.writerow(
                {
                    "index": index,
                    "sample_seed": input_atoms[index].info.get("sample_seed", ""),
                    "sample_index_within_seed": input_atoms[index].info.get(
                        "sample_index_within_seed", ""
                    ),
                    "formula": item.get_chemical_formula(),
                    "num_atoms": len(item),
                    "pre_relaxation_max_force": max_force,
                    "relaxation_steps": step_count,
                    "final_energy_ev": energy,
                }
            )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
