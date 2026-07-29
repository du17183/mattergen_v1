#!/usr/bin/env python3
"""Extract frozen CHGNet and geometry features for historical MatterGen outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

METHODS = ("C0", "A0", "G3")
MAX_ATOMIC_NUMBER = 94
BOOLEAN_LABELS = (
    "novel_unique_stable",
    "stable",
    "comp_validity",
    "structure_validity",
    "novel",
    "novel_unique",
    "unique",
    "converged",
)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def split_for_seed(seed: int) -> str:
    if 20000 <= seed <= 20191:
        return "train"
    if 20192 <= seed <= 20223:
        return "validation"
    if 20224 <= seed <= 20255:
        return "test"
    raise ValueError(f"seed outside frozen historical split: {seed}")


def minimum_distance(atoms: Any) -> float:
    if len(atoms) < 2:
        return math.inf
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _safe_array(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("non-finite CHGNet prediction")
    return result


def structure_features(atoms: Any, prediction: dict[str, Any]) -> dict[str, float]:
    numbers = np.asarray(atoms.numbers, dtype=int)
    if not len(numbers):
        raise ValueError("empty structure")
    if numbers.min() < 1 or numbers.max() > MAX_ATOMIC_NUMBER:
        raise ValueError("atomic number outside frozen feature vocabulary")
    volume = float(atoms.get_volume())
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("invalid cell volume")
    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    angles = np.asarray(atoms.cell.angles(), dtype=float)
    force = _safe_array(prediction["f"]).reshape(len(atoms), 3)
    stress = _safe_array(prediction["s"]).reshape(3, 3)
    magmom = _safe_array(prediction.get("m", np.zeros(len(atoms)))).reshape(-1)
    energy = float(_safe_array(prediction["e"]).reshape(-1)[0])
    force_norm = np.linalg.norm(force, axis=1)
    fractions = np.bincount(numbers, minlength=MAX_ATOMIC_NUMBER + 1)[1:]
    fractions = fractions.astype(float) / len(numbers)
    output: dict[str, float] = {
        "num_atoms": float(len(atoms)),
        "volume_ang3": volume,
        "volume_per_atom": volume / len(atoms),
        "mass_density_amu_ang3": float(np.sum(atoms.get_masses()) / volume),
        "minimum_distance_angstrom": minimum_distance(atoms),
        "atomic_number_mean": float(numbers.mean()),
        "atomic_number_std": float(numbers.std()),
        "atomic_number_min": float(numbers.min()),
        "atomic_number_max": float(numbers.max()),
        "cell_a": float(lengths[0]),
        "cell_b": float(lengths[1]),
        "cell_c": float(lengths[2]),
        "cell_alpha": float(angles[0]),
        "cell_beta": float(angles[1]),
        "cell_gamma": float(angles[2]),
        "cell_condition": float(np.linalg.cond(np.asarray(atoms.cell.array))),
        "chgnet_energy_per_atom_ev": energy,
        "chgnet_force_rms_ev_ang": float(np.sqrt(np.mean(force**2))),
        "chgnet_max_force_ev_ang": float(force_norm.max()),
        "chgnet_force_mean_ev_ang": float(force_norm.mean()),
        "chgnet_stress_rms_gpa": float(np.sqrt(np.mean(stress**2))),
        "chgnet_stress_max_abs_gpa": float(np.abs(stress).max()),
        "chgnet_mag_density": float(np.abs(magmom).sum() / volume),
        "chgnet_magmom_abs_mean": float(np.abs(magmom).mean()),
        "chgnet_magmom_abs_max": float(np.abs(magmom).max()),
    }
    for atomic_number, fraction in enumerate(fractions, start=1):
        output[f"element_fraction_z{atomic_number:03d}"] = float(fraction)
    if not all(math.isfinite(value) for value in output.values()):
        raise ValueError("non-finite structure feature")
    return output


def _as_bool(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in BOOLEAN_LABELS:
        if frame[column].dtype != bool:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False})
            )
        if frame[column].isna().any():
            raise ValueError(f"invalid label values in {column}")
    return frame


def load_tasks(
    generation_progress: Path,
    report_root: Path,
) -> list[dict[str, Any]]:
    progress = json.loads(generation_progress.read_text(encoding="utf-8"))
    task_index = {
        (str(row["config"]), int(row["seed"])): row
        for row in progress["tasks"]
        if row["status"] == "success"
    }
    tasks: list[dict[str, Any]] = []
    for method in METHODS:
        label_path = report_root / method / "official_metrics_per_structure.csv"
        labels = _as_bool(pd.read_csv(label_path))
        if len(labels) != 256:
            raise ValueError(f"{method} requires 256 frozen labels")
        for label in labels.to_dict(orient="records"):
            seed = int(label["seed"])
            source = task_index[(method, seed)]
            input_path = Path(source["output_dir"]) / "generated_crystals.extxyz"
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            tasks.append(
                {
                    "method": method,
                    "seed": seed,
                    "split": split_for_seed(seed),
                    "input_path": str(input_path),
                    **{
                        key: label[key]
                        for key in (
                            "energy_above_hull_per_atom",
                            "rmsd_from_relaxation",
                            *BOOLEAN_LABELS,
                        )
                    },
                }
            )
    return sorted(tasks, key=lambda row: (row["seed"], row["method"]))


def extract(
    *,
    generation_progress: Path,
    report_root: Path,
    output_csv: Path,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    from chgnet.model.model import CHGNet

    tasks = load_tasks(generation_progress, report_root)
    completed = pd.DataFrame()
    completed_keys: set[tuple[str, int]] = set()
    if output_csv.is_file():
        completed = pd.read_csv(output_csv)
        completed_keys = {
            (str(row.method), int(row.seed))
            for row in completed.itertuples(index=False)
        }
    pending = [
        row
        for row in tasks
        if (str(row["method"]), int(row["seed"])) not in completed_keys
    ]
    if not pending:
        return completed.sort_values(["seed", "method"]).reset_index(drop=True)

    model = CHGNet.load(
        model_name="0.3.0",
        verbose=False,
        use_device=device,
    )
    rows = completed.to_dict(orient="records")
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        atoms = [ase.io.read(row["input_path"]) for row in chunk]
        structures = [AseAtomsAdaptor.get_structure(value) for value in atoms]
        predictions = model.predict_structure(
            structures,
            task="efsm",
            return_site_energies=False,
            batch_size=batch_size,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        if len(predictions) != len(chunk):
            raise RuntimeError("CHGNet prediction count mismatch")
        for metadata, structure, prediction in zip(
            chunk, atoms, predictions, strict=True
        ):
            rows.append(
                {
                    **metadata,
                    **structure_features(structure, prediction),
                }
            )
        frame = pd.DataFrame(rows).sort_values(["seed", "method"])
        atomic_csv(output_csv, frame)
        print(
            json.dumps(
                {
                    "completed": len(frame),
                    "total": len(tasks),
                    "device": device,
                }
            ),
            flush=True,
        )
    return pd.DataFrame(rows).sort_values(["seed", "method"]).reset_index(
        drop=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-progress",
        type=Path,
        default=Path(
            "/data/dxl/results/formal_256/progress/generation_progress.json"
        ),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("/data/dxl/reports/formal_256"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "/data/dxl/results/postgen_fastgate/features/"
            "historical_features.csv"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = extract(
        generation_progress=args.generation_progress,
        report_root=args.report_root,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "rows": len(frame),
                "splits": frame["split"].value_counts().to_dict(),
                "methods": frame["method"].value_counts().to_dict(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
