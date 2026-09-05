"""Relax one benchmark arm once and compute official MatterGen metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
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
    reference.add_argument("--reference-memory-fd", type=int,
                           help="Inherited read-only LMDB stored in anonymous RAM.")
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
    parser.add_argument(
        "--isolate-relaxation",
        action="store_true",
        help="Relax every structure in a fresh subprocess to isolate CUDA failures.",
    )
    parser.add_argument(
        "--individual-relaxation",
        action="store_true",
        help="Avoid multi-structure CUDA batches and relax structures one at a time.",
    )
    parser.add_argument("--isolated-retries", type=int, default=2)
    parser.add_argument(
        "--precomputed-relaxation-dir",
        type=Path,
        help="Use relaxed/initial structures from relax_many_individual.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temporary_dir is not None:
        temporary_dir = args.temporary_dir.expanduser().resolve()
        temporary_dir.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(temporary_dir)
    else:
        temporary_dir = Path(tempfile.gettempdir()).resolve()
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
    potential_path = args.potential_path.expanduser().resolve()

    def load_potential() -> Potential:
        return Potential.from_checkpoint(
            device=args.device,
            load_path=str(potential_path),
            load_training_state=False,
        )

    def make_relaxer(potential: Potential) -> BatchRelaxer:
        return BatchRelaxer(
            potential=potential,
            filter="EXPCELLFILTER",
            fmax=args.fmax,
            max_natoms_per_batch=args.max_natoms_per_batch,
        )

    def isolated_relaxation() -> tuple[dict[int, list], list[dict[str, object]]]:
        if args.isolated_retries <= 0:
            raise ValueError("isolated-retries must be positive")
        isolated_root = output_dir / "isolated_relaxation"
        isolated_root.mkdir(parents=True, exist_ok=True)
        trajectories = {}
        failures = []
        for index, item in enumerate(input_atoms):
            structure_root = isolated_root / f"{index:04d}"
            structure_root.mkdir(parents=True, exist_ok=True)
            input_path = structure_root / "input.extxyz"
            if not input_path.exists():
                write(input_path, item.copy())
            attempt_errors = []
            for attempt in range(args.isolated_retries):
                attempt_dir = structure_root / f"attempt_{attempt + 1}"
                log_path = structure_root / f"attempt_{attempt + 1}.log"
                command = [
                    sys.executable,
                    "-m",
                    "research.corrector_distillation.relax_one",
                    "--input",
                    str(input_path),
                    "--potential-path",
                    str(potential_path),
                    "--output-dir",
                    str(attempt_dir),
                    "--device",
                    args.device,
                    "--fmax",
                    str(args.fmax),
                    "--max-natoms-per-batch",
                    str(args.max_natoms_per_batch),
                ]
                environment = dict(os.environ)
                environment["TMPDIR"] = str(temporary_dir)
                with log_path.open("w", encoding="utf-8") as log_stream:
                    completed = subprocess.run(
                        command,
                        cwd=Path(__file__).resolve().parents[2],
                        env=environment,
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                trajectory_path = attempt_dir / "trajectory.extxyz"
                if completed.returncode == 0 and trajectory_path.is_file():
                    trajectories[index] = read(trajectory_path, index=":")
                    break
                attempt_errors.append(
                    {
                        "attempt": attempt + 1,
                        "return_code": completed.returncode,
                        "log_path": str(log_path),
                    }
                )
            else:
                failures.append(
                    {
                        "index": index,
                        "formula": item.get_chemical_formula(),
                        "attempts": attempt_errors,
                    }
                )
        return trajectories, failures

    precomputed_dir = (
        None
        if args.precomputed_relaxation_dir is None
        else args.precomputed_relaxation_dir.expanduser().resolve()
    )
    if precomputed_dir is not None and (
        args.isolate_relaxation or args.individual_relaxation
    ):
        raise ValueError("precomputed relaxation cannot be combined with relaxation modes")
    if precomputed_dir is not None:
        with (precomputed_dir / "relaxation_summary.json").open(
            encoding="utf-8"
        ) as stream:
            precomputed = json.load(stream)
        if precomputed.get("success") is not True:
            raise ValueError("precomputed relaxation is not marked successful")
        if Path(precomputed["potential_path"]).resolve() != potential_path:
            raise ValueError("precomputed relaxation used a different potential")
        if float(precomputed["fmax"]) != args.fmax:
            raise ValueError("precomputed relaxation used a different fmax")
        relaxed_atoms = read(precomputed_dir / "relaxed.extxyz", index=":")
        first_atoms = read(
            precomputed_dir / "initial_with_properties.extxyz", index=":"
        )
        energies = [float(value) for value in precomputed["energies"]]
        max_forces = [
            float(value) for value in precomputed["pre_relaxation_max_forces"]
        ]
        steps = [int(value) for value in precomputed["relaxation_steps"]]
        counts = {
            len(input_atoms),
            len(relaxed_atoms),
            len(first_atoms),
            len(energies),
            len(max_forces),
            len(steps),
            int(precomputed["n"]),
            int(precomputed["n_input"]),
        }
        if counts != {len(input_atoms)}:
            raise ValueError(f"precomputed relaxation count mismatch: {sorted(counts)}")
        expected_seeds = [
            None
            if item.info.get("sample_seed") is None
            else int(item.info["sample_seed"])
            for item in input_atoms
        ]
        if precomputed.get("sample_seeds") != expected_seeds:
            raise ValueError("precomputed relaxation seed order does not match inputs")
        expected_formulas = [item.get_chemical_formula() for item in input_atoms]
        if precomputed.get("formulas") != expected_formulas:
            raise ValueError("precomputed relaxation formula order does not match inputs")
        successful_indices = list(range(len(input_atoms)))
        relaxation_mode = "precomputed_individual"
        batch_error = None
        relaxation_failures = []
        elapsed = float(precomputed["relaxation_seconds"])
        relaxation_device = str(precomputed["device"])
    else:
        started = time.perf_counter()
        if args.isolate_relaxation and args.individual_relaxation:
            raise ValueError("choose at most one relaxation isolation mode")
        if args.isolate_relaxation:
            relaxation_mode = "isolated_subprocess"
            batch_error = None
            trajectories, relaxation_failures = isolated_relaxation()
        elif args.individual_relaxation:
            relaxation_mode = "individual"
            batch_error = None
            relaxation_failures = []
            trajectories = {}
            potential = load_potential()
            for index, item in enumerate(input_atoms):
                try:
                    result = make_relaxer(potential).relax([item.copy()])
                    if 0 not in result or not result[0]:
                        raise RuntimeError("MatterSim returned no trajectory")
                    trajectories[index] = result[0]
                except Exception as item_error:
                    relaxation_failures.append(
                        {
                            "index": index,
                            "formula": item.get_chemical_formula(),
                            "error": f"{type(item_error).__name__}: {item_error}",
                        }
                    )
                    break
        else:
            relaxation_mode = "batch"
            batch_error = None
            relaxation_failures = []
            potential = load_potential()
            try:
                trajectories = make_relaxer(potential).relax(
                    [item.copy() for item in input_atoms]
                )
                if sorted(trajectories) != list(range(len(input_atoms))):
                    raise RuntimeError("MatterSim did not return every input structure")
            except Exception as error:
                # A failed CUDA batch can poison its process context. Retry every
                # pristine input in a fresh child process so one invalid structure
                # cannot invalidate the rest of the method.
                relaxation_mode = "isolated_subprocess_fallback"
                batch_error = f"{type(error).__name__}: {error}"
                trajectories, relaxation_failures = isolated_relaxation()
        elapsed = time.perf_counter() - started
        successful_indices = sorted(trajectories)
        if relaxation_mode == "individual" and len(successful_indices) != len(input_atoms):
            print(
                json.dumps(
                    {
                        "relaxation_mode": relaxation_mode,
                        "successful_count": len(successful_indices),
                        "expected_count": len(input_atoms),
                        "relaxation_failures": relaxation_failures,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            raise RuntimeError("individual MatterSim relaxation was incomplete")
        if not successful_indices:
            print(
                json.dumps(
                    {
                        "batch_error": batch_error,
                        "relaxation_failure_count": len(relaxation_failures),
                        "first_relaxation_failures": relaxation_failures[:3],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            raise RuntimeError("MatterSim could not relax any input structure")
        relaxed_atoms = [trajectories[index][-1] for index in successful_indices]
        first_atoms = [trajectories[index][0] for index in successful_indices]
        energies = [float(item.info["total_energy"]) for item in relaxed_atoms]
        max_forces = [
            float(np.linalg.norm(item.arrays["forces"], axis=1).max())
            for item in first_atoms
        ]
        steps = [len(trajectories[index]) for index in successful_indices]
        relaxation_device = args.device
    write(output_dir / "relaxed.extxyz", relaxed_atoms)
    np.save(output_dir / "energies.npy", np.asarray(energies))

    adaptor = AseAtomsAdaptor()
    original_structures = [adaptor.get_structure(input_atoms[index]) for index in successful_indices]
    relaxed_structures = [adaptor.get_structure(item) for item in relaxed_atoms]
    if args.reference_memory_fd is not None:
        reference = ReferenceDataset(
            name="alex-mp MP2020 correction",
            impl=LMDBBackedReferenceDatasetImpl(
                Path(f"/proc/self/fd/{args.reference_memory_fd}"), cleanup_dir=False
            ),
        )
    elif args.reference_lmdb_path is not None:
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
        "relaxation_device": relaxation_device,
        "precomputed_relaxation_dir": (
            None if precomputed_dir is None else str(precomputed_dir)
        ),
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
