"""Label clean-x0 structures with frozen CHGNet 0.3.0 geometry proxies."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chgnet.model.model import CHGNet
from pymatgen.core import Lattice, Structure

from mattergen.common.data.dataset import CrystalDatasetBuilder
from mattergen.common.data.transform import symmetrize_lattice


def to_structure(sample) -> Structure:
    return Structure(
        lattice=Lattice(sample.cell.squeeze(0).cpu().numpy()),
        species=sample.atomic_numbers.cpu().numpy(),
        coords=sample.pos.cpu().numpy(),
        coords_are_cartesian=False,
    )


def min_distance(structure: Structure) -> float:
    if len(structure) < 2:
        return math.inf
    distances = np.asarray(structure.distance_matrix, dtype=float)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p30": float(np.quantile(values, 0.30)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p70": float(np.quantile(values, 0.70)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "std": float(values.std(ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CHGNet.from_file(args.model_path).to(device)
    all_rows: list[dict[str, object]] = []
    for split in ("train", "val"):
        dataset = CrystalDatasetBuilder.from_cache_path(
            str(args.data_root / "cache" / split),
            transforms=[symmetrize_lattice],
            properties=["dft_mag_density"],
        ).build()
        structures = [to_structure(dataset[index]) for index in range(len(dataset))]
        predictions = model.predict_structure(
            structures, task="efsm", batch_size=args.batch_size
        )
        if isinstance(predictions, dict):
            predictions = [predictions]
        for index, (structure, prediction) in enumerate(zip(structures, predictions)):
            forces = np.asarray(prediction["f"], dtype=float).reshape(len(structure), 3)
            norms = np.linalg.norm(forces, axis=1)
            all_rows.append(
                {
                    "split": split,
                    "split_index": index,
                    "structure_id": str(dataset.structure_id[index]),
                    "formula": structure.composition.reduced_formula,
                    "chemical_system": structure.composition.chemical_system,
                    "num_atoms": len(structure),
                    "dft_mag_density": float(dataset.properties["dft_mag_density"][index]),
                    "chgnet_energy_per_atom_ev": float(np.asarray(prediction["e"]).reshape(-1)[0]),
                    "chgnet_force_rms_ev_ang": float(np.sqrt(np.mean(forces**2))),
                    "chgnet_force_mean_ev_ang": float(norms.mean()),
                    "chgnet_force_max_ev_ang": float(norms.max()),
                    "minimum_distance_angstrom": min_distance(structure),
                    "volume_per_atom_ang3": float(structure.volume / len(structure)),
                }
            )

    frame = pd.DataFrame(all_rows)
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("CHGNet/geometry labels contain NaN or Inf")
    train_force = frame.loc[
        frame["split"] == "train", "chgnet_force_max_ev_ang"
    ].to_numpy(float)
    low, high = np.quantile(train_force, [0.30, 0.70])
    frame["quality_weight"] = np.where(
        frame["chgnet_force_max_ev_ang"] <= low,
        1.25,
        np.where(frame["chgnet_force_max_ev_ang"] >= high, 0.75, 1.0),
    )
    frame["quality_bin"] = np.where(
        frame["chgnet_force_max_ev_ang"] <= low,
        "low_force",
        np.where(frame["chgnet_force_max_ev_ang"] >= high, "high_force", "middle"),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    train = frame[frame["split"] == "train"]
    low_group = train[train["quality_bin"] == "low_force"]
    high_group = train[train["quality_bin"] == "high_force"]
    summary = {
        "label_model": "CHGNet 0.3.0",
        "label_model_path": str(args.model_path.resolve()),
        "label_device": device,
        "n": len(frame),
        "n_train": int((frame["split"] == "train").sum()),
        "n_validation": int((frame["split"] == "val").sum()),
        "all_finite": True,
        "weight_scheme": {"low_30pct": 1.25, "middle_40pct": 1.0, "high_30pct": 0.75},
        "train_force_thresholds": {"p30": float(low), "p70": float(high)},
        "distributions": {
            column: describe(frame[column].to_numpy(float))
            for column in (
                "chgnet_energy_per_atom_ev",
                "chgnet_force_rms_ev_ang",
                "chgnet_force_mean_ev_ang",
                "chgnet_force_max_ev_ang",
                "minimum_distance_angstrom",
                "volume_per_atom_ang3",
            )
        },
        "composition_check": {
            "train_unique_chemical_systems": int(train["chemical_system"].nunique()),
            "low_force_unique_chemical_systems": int(low_group["chemical_system"].nunique()),
            "high_force_unique_chemical_systems": int(high_group["chemical_system"].nunique()),
            "low_force_max_chemical_system_share": float(low_group["chemical_system"].value_counts(normalize=True).max()),
            "high_force_max_chemical_system_share": float(high_group["chemical_system"].value_counts(normalize=True).max()),
        },
        "signal_checks": {
            "force_max_nonconstant": bool(np.unique(frame["chgnet_force_max_ev_ang"].round(10)).size > 10),
            "force_max_p90_over_p10": float(
                np.quantile(frame["chgnet_force_max_ev_ang"], 0.90)
                / max(np.quantile(frame["chgnet_force_max_ev_ang"], 0.10), 1e-12)
            ),
            "low_high_bins_span_multiple_chemical_systems": bool(
                low_group["chemical_system"].nunique() >= 5
                and high_group["chemical_system"].nunique() >= 5
            ),
        },
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
