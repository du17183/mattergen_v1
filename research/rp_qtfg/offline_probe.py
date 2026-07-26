from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor

from research.rp_qtfg.common import RESULTS, atomic_json, now, set_stage, stop_requested


SEEDS = tuple(range(20000, 20064))
SOURCE = Path("/data/dxl/results/formal_256/generation/A0")
OUT = RESULTS / "offline_probe"
STRUCTURES = OUT / "structures"
MANIFEST = OUT / "probe_manifest.csv"
SUMMARY = OUT / "probe_summary.json"
VARIANTS = ("baseline", "pos_1", "pos_3", "pos_5", "poscell_1", "poscell_3")


@dataclass(frozen=True)
class ProbeConfig:
    position_eta: float = 0.01
    position_radius_angstrom: float = 0.02
    cell_eta_per_gpa: float = 0.00025
    cell_strain_radius: float = 0.003
    backtrack_max: int = 3
    minimum_distance_angstrom: float = 0.5
    batch_size: int = 128


def _as_list(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else [value]


def _predict(
    model: CHGNet,
    atoms_list: list[Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]
    return _as_list(
        model.predict_structure(
            structures,
            task="efs",
            batch_size=min(batch_size, len(structures)),
        )
    )


def _minimum_distance(atoms: Any) -> float:
    if len(atoms) < 2:
        return math.inf
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _finite_and_safe(atoms: Any, minimum_distance: float) -> bool:
    volume = float(atoms.get_volume())
    return (
        np.isfinite(atoms.positions).all()
        and np.isfinite(atoms.cell.array).all()
        and math.isfinite(volume)
        and volume > 0.1
        and _minimum_distance(atoms) >= minimum_distance
    )


def _proposal(
    atoms: Any,
    prediction: dict[str, Any],
    config: ProbeConfig,
    *,
    cell: bool,
    scale: float,
) -> Any:
    candidate = atoms.copy()
    forces = np.asarray(prediction["f"], dtype=float)
    displacement = config.position_eta * scale * forces
    norms = np.linalg.norm(displacement, axis=1)
    cap = config.position_radius_angstrom * scale
    displacement *= np.minimum(1.0, cap / np.maximum(norms, 1e-12))[:, None]

    if cell:
        stress = np.asarray(prediction["s"], dtype=float)
        stress = 0.5 * (stress + stress.T)
        strain = -config.cell_eta_per_gpa * scale * stress
        strain_norm = float(np.linalg.norm(strain))
        strain_cap = config.cell_strain_radius * scale
        if strain_norm > strain_cap:
            strain *= strain_cap / strain_norm
        candidate.set_cell(
            candidate.cell.array @ (np.eye(3) + strain).T,
            scale_atoms=True,
        )

    candidate.positions[:] = candidate.positions + displacement
    candidate.wrap()
    return candidate


def _advance(
    model: CHGNet,
    atoms_list: list[Any],
    config: ProbeConfig,
    *,
    cell: bool,
    counters: list[dict[str, int]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    old_predictions = _predict(model, atoms_list, config.batch_size)
    unresolved = list(range(len(atoms_list)))
    accepted: dict[int, tuple[Any, dict[str, Any], int]] = {}
    for backtrack in range(config.backtrack_max):
        if not unresolved:
            break
        scale = 0.5**backtrack
        candidates = [
            _proposal(
                atoms_list[index],
                old_predictions[index],
                config,
                cell=cell,
                scale=scale,
            )
            for index in unresolved
        ]
        safe_local = [
            index
            for index, candidate in enumerate(candidates)
            if _finite_and_safe(candidate, config.minimum_distance_angstrom)
        ]
        safe_predictions = (
            _predict(
                model,
                [candidates[index] for index in safe_local],
                config.batch_size,
            )
            if safe_local
            else []
        )
        by_local = dict(zip(safe_local, safe_predictions, strict=True))
        remaining: list[int] = []
        for local_index, global_index in enumerate(unresolved):
            prediction = by_local.get(local_index)
            old_energy = float(
                np.asarray(old_predictions[global_index]["e"]).reshape(-1)[0]
            )
            new_energy = (
                float(np.asarray(prediction["e"]).reshape(-1)[0])
                if prediction is not None
                else math.inf
            )
            if math.isfinite(new_energy) and new_energy <= old_energy + 1e-7:
                accepted[global_index] = (
                    candidates[local_index],
                    prediction,
                    backtrack,
                )
            else:
                remaining.append(global_index)
        unresolved = remaining

    outputs: list[Any] = []
    final_predictions: list[dict[str, Any]] = []
    for index, atoms in enumerate(atoms_list):
        if index in accepted:
            candidate, prediction, backtrack = accepted[index]
            outputs.append(candidate)
            final_predictions.append(prediction)
            counters[index]["accepted_steps"] += 1
            counters[index]["backtracking_count"] += backtrack
        else:
            outputs.append(atoms.copy())
            final_predictions.append(old_predictions[index])
            counters[index]["fallback_count"] += 1
    return outputs, final_predictions


def _load_inputs() -> list[Any]:
    atoms_list = []
    for seed in SEEDS:
        path = SOURCE / str(seed) / "generated_crystals.extxyz"
        if not path.is_file():
            raise FileNotFoundError(path)
        atoms = ase.io.read(path)
        if len(atoms) == 0 or not _finite_and_safe(atoms, 0.5):
            raise RuntimeError(f"invalid frozen A0 structure: {path}")
        atoms.info["rp_qtfg_source_seed"] = seed
        atoms_list.append(atoms)
    return atoms_list


def _write_snapshot(
    method: str,
    atoms_list: list[Any],
    predictions: list[dict[str, Any]],
    initial_numbers: list[np.ndarray],
    counters: list[dict[str, int]],
) -> list[dict[str, Any]]:
    rows = []
    method_dir = STRUCTURES / method
    method_dir.mkdir(parents=True, exist_ok=True)
    for index, (seed, atoms, prediction) in enumerate(
        zip(SEEDS, atoms_list, predictions, strict=True)
    ):
        if not np.array_equal(atoms.numbers, initial_numbers[index]):
            raise RuntimeError(
                f"atomic sequence changed for seed {seed}, method {method}"
            )
        path = method_dir / f"{seed}.extxyz"
        ase.io.write(path, atoms, format="extxyz")
        check = ase.io.read(path)
        if not np.array_equal(check.numbers, initial_numbers[index]):
            raise RuntimeError(f"written atomic sequence changed: {path}")
        forces = np.asarray(prediction["f"], dtype=float)
        rows.append(
            {
                "method": method,
                "seed": seed,
                "path": str(path),
                "n_atoms": len(atoms),
                "volume": float(atoms.get_volume()),
                "minimum_distance_angstrom": _minimum_distance(atoms),
                "chgnet_energy_per_atom_ev": float(
                    np.asarray(prediction["e"]).reshape(-1)[0]
                ),
                "chgnet_max_force_ev_ang": float(
                    np.linalg.norm(forces, axis=1).max()
                ),
                **counters[index],
            }
        )
    return rows


def run(config: ProbeConfig | None = None) -> dict[str, Any]:
    config = config or ProbeConfig()
    if SUMMARY.is_file() and MANIFEST.is_file():
        summary = json.loads(SUMMARY.read_text())
        if summary.get("status") == "success":
            return summary

    set_stage(
        "offline_direction_probe",
        "running",
        "Generating small trust-region CHGNet position and position+cell probes for 64 frozen A0 structures.",
        {"structures": len(SEEDS), "variants": list(VARIANTS)},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    model = CHGNet.load(
        model_name="0.3.0",
        verbose=False,
        use_device="cuda",
    )
    baseline = _load_inputs()
    initial_numbers = [atoms.numbers.copy() for atoms in baseline]
    baseline_predictions = _predict(model, baseline, config.batch_size)
    zero_counters = [
        {"accepted_steps": 0, "fallback_count": 0, "backtracking_count": 0}
        for _ in baseline
    ]
    rows = _write_snapshot(
        "baseline",
        baseline,
        baseline_predictions,
        initial_numbers,
        zero_counters,
    )

    position_atoms = [atoms.copy() for atoms in baseline]
    position_predictions = baseline_predictions
    position_counters = [
        {"accepted_steps": 0, "fallback_count": 0, "backtracking_count": 0}
        for _ in baseline
    ]
    for step in range(1, 6):
        if stop_requested():
            raise KeyboardInterrupt("STOP_REQUESTED")
        position_atoms, position_predictions = _advance(
            model,
            position_atoms,
            config,
            cell=False,
            counters=position_counters,
        )
        if step in {1, 3, 5}:
            rows.extend(
                _write_snapshot(
                    f"pos_{step}",
                    position_atoms,
                    position_predictions,
                    initial_numbers,
                    position_counters,
                )
            )

    position_cell_atoms = [atoms.copy() for atoms in baseline]
    position_cell_predictions = baseline_predictions
    position_cell_counters = [
        {"accepted_steps": 0, "fallback_count": 0, "backtracking_count": 0}
        for _ in baseline
    ]
    for step in range(1, 4):
        if stop_requested():
            raise KeyboardInterrupt("STOP_REQUESTED")
        position_cell_atoms, position_cell_predictions = _advance(
            model,
            position_cell_atoms,
            config,
            cell=True,
            counters=position_cell_counters,
        )
        if step in {1, 3}:
            rows.extend(
                _write_snapshot(
                    f"poscell_{step}",
                    position_cell_atoms,
                    position_cell_predictions,
                    initial_numbers,
                    position_cell_counters,
                )
            )

    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "success",
        "created_at": now(),
        "structures": len(SEEDS),
        "variants": list(VARIANTS),
        "records": len(rows),
        "model": "CHGNet 0.3.0",
        "config": asdict(config),
        "manifest": str(MANIFEST),
        "atomic_numbers_unchanged": True,
    }
    atomic_json(SUMMARY, summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
