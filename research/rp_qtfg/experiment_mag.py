from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor

from research.rp_qtfg.experiment_config import EIGHT_SEED_CONFIGS, EIGHT_SEEDS, THIRTY_TWO_SEEDS

from research.rp_qtfg.common import (
    REPORTS,
    RESULTS,
    atomic_json,
    now,
    set_stage,
    stop_requested,
)


GENERATION = RESULTS / "generation"
ROWS_ROOT = RESULTS / "magnetic_eval"
TARGET = 0.10


def _configs_and_seeds(mode: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if mode == "eight":
        return EIGHT_SEED_CONFIGS, EIGHT_SEEDS
    if mode == "thirtytwo":
        selected = str(json.loads((REPORTS / "eight_seed/selected_candidate.json").read_text())["selected_config"])
        return ("A0", selected), THIRTY_TWO_SEEDS
    raise ValueError(mode)


def _prediction_density(prediction: dict[str, Any], volume: float) -> float:
    magmom = np.asarray(prediction["m"], dtype=float).reshape(-1)
    if not np.isfinite(magmom).all() or volume <= 0 or not math.isfinite(volume):
        raise RuntimeError("invalid CHGNet magmom or structure volume")
    return float(np.abs(magmom).sum() / volume)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(mode: str, batch_size: int = 32) -> dict[str, Any]:
    atomic_json(
        REPORTS / "launcher.json",
        {
            "pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgid(0),
            "sid": os.getsid(0), "user": os.environ.get("USER", ""),
            "cwd": os.getcwd(), "exe": os.readlink(f"/proc/{os.getpid()}/exe"),
            "command": " ".join(os.sys.argv), "mode": mode, "started_at": now(),
        },
    )
    configs, seeds = _configs_and_seeds(mode)
    rows_path = ROWS_ROOT / f"{mode}.csv"
    out = REPORTS / ("eight_seed" if mode == "eight" else "thirty_two")
    report_path = out / "magnetic_metrics.json"
    expected = len(configs) * len(seeds)
    if report_path.is_file() and rows_path.is_file():
        existing = json.loads(report_path.read_text())
        if existing.get("total") == expected:
            return existing
    tasks = [(config, seed) for config in configs for seed in seeds]
    structures = []
    for config, seed in tasks:
        path = GENERATION / config / str(seed) / "generated_crystals.extxyz"
        structures.append(AseAtomsAdaptor.get_structure(ase.io.read(path)))
    model = CHGNet.load(model_name="0.3.0", verbose=False, use_device="cuda")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(tasks), batch_size):
        if stop_requested():
            raise KeyboardInterrupt("RP-QTFG stop requested")
        selected_tasks = tasks[start:start + batch_size]
        selected_structures = structures[start:start + batch_size]
        predictions = model.predict_structure(selected_structures, task="efsm", return_site_energies=False, batch_size=batch_size)
        if not isinstance(predictions, list):
            predictions = [predictions]
        for (config, seed), structure, prediction in zip(selected_tasks, selected_structures, predictions, strict=True):
            density = _prediction_density(prediction, float(structure.volume))
            error = abs(density - TARGET)
            rows.append({
                "config_id": config, "seed": seed, "num_atoms": len(structure),
                "volume_ang3": float(structure.volume), "chgnet_mag_density": density,
                "target_error": error, "hit_0.01": error <= 0.01,
                "hit_0.02": error <= 0.02, "hit_0.05": error <= 0.05,
            })
        _write_rows(rows_path, rows)
    summaries = []
    for config in configs:
        selected = [row for row in rows if row["config_id"] == config]
        summaries.append({
            "config_id": config, "n": len(selected),
            "mean_target_error": float(np.mean([row["target_error"] for row in selected])),
            "median_target_error": float(np.median([row["target_error"] for row in selected])),
            "hit_0.01": float(np.mean([row["hit_0.01"] for row in selected])),
            "hit_0.02": float(np.mean([row["hit_0.02"] for row in selected])),
            "hit_0.05": float(np.mean([row["hit_0.05"] for row in selected])),
        })
    result = {
        "created_at": now(), "mode": mode, "total": len(rows),
        "formula": "sum(abs(CHGNet site magmom)) / volume",
        "target": TARGET, "CHGNET_MAG_ORACLE_GO": True,
        "magnetic_property_dft_verified": False, "summaries": summaries,
    }
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(report_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eight", "thirtytwo"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args.mode, batch_size=args.batch_size)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
