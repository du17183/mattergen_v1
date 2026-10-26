"""Hard PASS/FAIL gate for historical Formal256 seed 740000 reproduction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from ase import io
from ase.filters import ExpCellFilter
import ase.constraints
import numpy as np
import torch
import torch_geometric
import torch_scatter  # noqa: F401: historical CHGNet import order
ase.constraints.ExpCellFilter = ExpCellFilter
sys.path.append("/mnt/datasets-livsyn/dxl/mattergen_v1/.venv/lib/python3.10/site-packages")
from chgnet.model.model import CHGNet
from mattersim.forcefield.potential import MatterSimCalculator
from pymatgen.io.ase import AseAtomsAdaptor

from experiments.mattersim_late_force_guidance_p0.late_force_sampler import _load_potential


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
HISTORY = PROJECT / "experiments/mattersim_late_force_guidance_formal256"
CHGNET_PATH = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar")
MATTERSIM_PATH = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth")
EXPECTED_HASHES = {
    "C0": "d4b2f381d3de5625e36af86e7355eb4e5e5079c4e1f0866a65b61405e1f56433",
    "F0": "765e297599299fda973a6d6651f40b88f39805401fb5b84eb0e5a6206a7d1407",
}
GEOMETRY_ATOL = 1e-7
METRIC_ATOL = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def geometry_difference(original, replay) -> dict:
    same_numbers = np.array_equal(original.numbers, replay.numbers)
    same_pbc = np.array_equal(original.pbc, replay.pbc)
    cell = float(np.max(np.abs(original.cell.array - replay.cell.array)))
    fractional_delta = original.get_scaled_positions() - replay.get_scaled_positions()
    fractional_delta -= np.round(fractional_delta)
    cartesian = float(np.max(np.abs(fractional_delta @ original.cell.array)))
    return {
        "same_atomic_numbers": bool(same_numbers),
        "same_pbc": bool(same_pbc),
        "max_abs_cell_a": cell,
        "max_abs_periodic_position_a": cartesian,
        "pass": bool(same_numbers and same_pbc and cell <= GEOMETRY_ATOL and cartesian <= GEOMETRY_ATOL),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-dir", choices=("replay_740000", "alm_replay_740000"),
        default="replay_740000",
    )
    args = parser.parse_args()
    replay_root = ROOT / args.replay_dir
    gate_path = replay_root / "gate.json"
    if gate_path.exists():
        raise FileExistsError(f"replay gate already exists: {gate_path}")
    with (HISTORY / "per_structure_metrics.csv").open(newline="", encoding="utf-8") as stream:
        historical = {
            row["method"]: row for row in csv.DictReader(stream)
            if int(row["seed"]) == 740000 and row["method"] in EXPECTED_HASHES
        }
    if set(historical) != set(EXPECTED_HASHES):
        raise RuntimeError("historical seed metrics are incomplete")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chgnet = CHGNet.from_file(CHGNET_PATH).to(device)
    chgnet.eval()
    potential = _load_potential(str(MATTERSIM_PATH), device)
    calculator = MatterSimCalculator(potential=potential, compute_stress=False, device=device)
    results = {}
    for method, expected_hash in EXPECTED_HASHES.items():
        old_path = HISTORY / "generation" / method / "740000/generated_crystals.extxyz"
        if args.replay_dir == "alm_replay_740000":
            new_path = replay_root / "generation" / method / "740000/generated_crystals.extxyz"
        else:
            new_path = replay_root / method / "generated_crystals.extxyz"
        old_hash, new_hash = sha256(old_path), sha256(new_path)
        if old_hash != expected_hash:
            raise RuntimeError(f"historical {method} artifact hash changed")
        original, replay = io.read(old_path), io.read(new_path)
        geom = geometry_difference(original, replay)
        structure = AseAtomsAdaptor.get_structure(replay)
        prediction = chgnet.predict_structure(structure, task="efsm")
        if isinstance(prediction, list):
            prediction = prediction[0]
        property_density = float(np.abs(np.asarray(prediction["m"], dtype=float)).sum() / structure.volume)
        property_mae = abs(property_density - 0.2)
        replay.calc = calculator
        with torch.enable_grad():
            force = np.asarray(replay.get_forces(), dtype=float)
        norms = np.linalg.norm(force, axis=1)
        force_max, force_mean = float(norms.max()), float(norms.mean())
        expected = historical[method]
        values = {
            "property_mae": (property_mae, float(expected["mag_absolute_error_a3"])),
            "mattersim_max_force": (force_max, float(expected["maxF_ev_per_a"])),
            "mattersim_mean_force": (force_mean, float(expected["atomic_force_mean_ev_per_a"])),
        }
        metrics = {
            name: {"replay": measured, "historical": old, "abs_delta": abs(measured - old),
                   "pass": abs(measured - old) <= METRIC_ATOL}
            for name, (measured, old) in values.items()
        }
        results[method] = {
            "historical_sha256": old_hash,
            "replay_sha256": new_hash,
            "byte_hash_match": old_hash == new_hash,
            "geometry": geom,
            "metrics": metrics,
            "pass": bool((old_hash == new_hash or geom["pass"]) and all(v["pass"] for v in metrics.values())),
        }
    payload = {
        "status": "PASS" if all(x["pass"] for x in results.values()) else "FAIL",
        "seed": 740000,
        "replay_dir": args.replay_dir,
        "geometry_tolerance_a": GEOMETRY_ATOL,
        "metric_abs_tolerance": METRIC_ATOL,
        "methods": results,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
    }
    gate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
