"""Compute the CHGNet magnetic-density proxy in the isolated teacher env."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import ase.io
import numpy as np
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    target = float(payload["target_density"])
    structures = [
        AseAtomsAdaptor.get_structure(ase.io.read(task["structure_path"]))
        for task in tasks
    ]
    model = CHGNet.load(model_name="0.3.0", verbose=False, use_device="cuda")
    rows = []
    for start in range(0, len(tasks), 32):
        selected_tasks = tasks[start : start + 32]
        selected_structures = structures[start : start + 32]
        predictions = model.predict_structure(
            selected_structures,
            task="efsm",
            return_site_energies=False,
            batch_size=32,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        for task, structure, prediction in zip(
            selected_tasks,
            selected_structures,
            predictions,
            strict=True,
        ):
            moments = np.asarray(prediction["m"], dtype=float).reshape(-1)
            density = float(np.abs(moments).sum() / float(structure.volume))
            error = abs(density - target)
            rows.append(
                {
                    "config": task["config"],
                    "seed": int(task["seed"]),
                    "chgnet_mag_density": density,
                    "target_error": error,
                    "hit_0.01": error <= 0.01,
                    "hit_0.02": error <= 0.02,
                    "hit_0.05": error <= 0.05,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
