from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from chgnet.model.model import CHGNet
from pymatgen.core import Lattice, Structure
from scipy.stats import spearmanr

from research.rp_qtfg.common import (
    REPORTS,
    RESULTS,
    atomic_json,
    now,
    set_stage,
    stop_requested,
)


DATA = Path("/data/dxl/datasets/cache/mp_20/test")
ROWS = RESULTS / "mag_oracle/mag_oracle_predictions.csv"
REPORT = REPORTS / "mag_oracle_report.json"
REPORT_MD = REPORTS / "mag_oracle_report.md"
MODEL_CHECKPOINT = Path(
    "/data/dxl/envs/fn_pra_teacher/lib/python3.10/site-packages/"
    "chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar"
)
TARGET = 0.10


def _load_cache() -> tuple[np.ndarray, ...]:
    num_atoms = np.load(DATA / "num_atoms.npy")
    cells = np.load(DATA / "cell.npy")
    positions = np.load(DATA / "pos.npy")
    atomic_numbers = np.load(DATA / "atomic_numbers.npy")
    structure_ids = np.load(DATA / "structure_id.npy")
    payload = json.loads((DATA / "dft_mag_density.json").read_text(encoding="utf-8"))
    labels = np.asarray(payload["values"], dtype=float)
    if len(num_atoms) != len(cells) or len(num_atoms) != len(labels):
        raise RuntimeError("MP-20 test cache arrays are not aligned")
    return num_atoms, cells, positions, atomic_numbers, structure_ids, labels


def _structures(
    indices: np.ndarray,
    num_atoms: np.ndarray,
    cells: np.ndarray,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
) -> list[Structure]:
    offsets = np.concatenate([[0], np.cumsum(num_atoms)])
    values: list[Structure] = []
    for index in indices:
        span = slice(int(offsets[index]), int(offsets[index + 1]))
        values.append(
            Structure(
                Lattice(cells[index]),
                atomic_numbers[span].tolist(),
                positions[span],
                coords_are_cartesian=False,
            )
        )
    return values


def _safe_float(value: Any) -> float:
    output = float(value)
    if not math.isfinite(output):
        raise RuntimeError(f"non-finite CHGNet output: {output}")
    return output


def _prediction_density(prediction: dict[str, Any], volume: float) -> float:
    magmom = np.asarray(prediction["m"], dtype=float).reshape(-1)
    if not np.isfinite(magmom).all() or volume <= 0 or not math.isfinite(volume):
        raise RuntimeError("invalid CHGNet magmom or structure volume")
    return float(np.abs(magmom).sum() / volume)


def _write_rows(rows: list[dict[str, Any]]) -> None:
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    temporary = ROWS.with_name(f".{ROWS.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, ROWS)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = np.asarray([row["dft_mag_density"] for row in rows], dtype=float)
    predicted = np.asarray([row["chgnet_mag_density"] for row in rows], dtype=float)
    residual = predicted - observed
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]
    residual = residual[finite]
    target_001 = np.abs(observed - TARGET) <= 0.01
    target_002 = np.abs(observed - TARGET) <= 0.02
    target_005 = np.abs(observed - TARGET) <= 0.05
    predicted_error = np.abs(predicted - TARGET)

    enrichment: dict[str, Any] = {}
    baseline = float(target_002.mean())
    order = np.argsort(predicted_error)
    for k in (50, 100, 200, 500):
        selected = order[: min(k, len(order))]
        rate = float(target_002[selected].mean())
        enrichment[str(k)] = {
            "selected": int(len(selected)),
            "hit_rate_0.02": rate,
            "random_hit_rate_0.02": baseline,
            "enrichment": rate / baseline if baseline > 0 else None,
        }

    target_region = target_005
    target_region_mae = float(np.mean(np.abs(residual[target_region])))
    rank = float(spearmanr(predicted, observed, nan_policy="omit").statistic)
    ss_res = float(np.square(residual).sum())
    ss_tot = float(np.square(observed - observed.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("-inf")
    main_enrichment = enrichment["100"]["enrichment"]
    oracle_go = bool(
        rank >= 0.60
        and target_region_mae <= 0.03
        and main_enrichment is not None
        and main_enrichment >= 2.0
    )
    return {
        "created_at": now(),
        "formula": "sum(abs(CHGNet site magmom [mu_B])) / cell volume [A^3]",
        "target": TARGET,
        "n": int(len(observed)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": r2,
        "spearman": rank,
        "target_region_definition": "abs(dft_mag_density - 0.10) <= 0.05",
        "target_region_n": int(target_region.sum()),
        "target_region_mae": target_region_mae,
        "actual_hit_rate_0.01": float(target_001.mean()),
        "actual_hit_rate_0.02": float(target_002.mean()),
        "actual_hit_rate_0.05": float(target_005.mean()),
        "top_k_target_enrichment": enrichment,
        "gate_thresholds": {
            "spearman_min": 0.60,
            "target_region_mae_max": 0.03,
            "top_100_enrichment_min": 2.0,
        },
        "CHGNET_MAG_ORACLE_GO": oracle_go,
        "model_name": "CHGNet 0.3.0",
        "model_checkpoint": str(MODEL_CHECKPOINT),
    }


def run(batch_size: int = 128, limit: int | None = None) -> dict[str, Any]:
    if REPORT.exists() and ROWS.exists():
        return json.loads(REPORT.read_text(encoding="utf-8"))
    set_stage(
        "mag_oracle_validation",
        "running",
        "Evaluating CHGNet 0.3.0 site-moment density on held-out MP-20.",
        {"batch_size": batch_size, "limit": limit},
    )
    num_atoms, cells, positions, atomic_numbers, structure_ids, labels = _load_cache()
    indices = np.flatnonzero(np.isfinite(labels))
    if limit is not None:
        indices = indices[:limit]
    model = CHGNet.load(model_name="0.3.0", use_device="cuda")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(indices), batch_size):
        if stop_requested():
            raise KeyboardInterrupt("RP-QTFG stop requested")
        selected = indices[start : start + batch_size]
        structures = _structures(
            selected, num_atoms, cells, positions, atomic_numbers
        )
        predictions = model.predict_structure(
            structures,
            task="efsm",
            return_site_energies=False,
            batch_size=batch_size,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        for index, structure, prediction in zip(selected, structures, predictions, strict=True):
            force = np.asarray(prediction["f"], dtype=float)
            rows.append(
                {
                    "index": int(index),
                    "structure_id": str(structure_ids[index]),
                    "num_atoms": int(num_atoms[index]),
                    "volume_ang3": _safe_float(structure.volume),
                    "dft_mag_density": _safe_float(labels[index]),
                    "chgnet_mag_density": _prediction_density(
                        prediction, structure.volume
                    ),
                    "chgnet_energy_ev_atom": _safe_float(prediction["e"]),
                    "chgnet_max_force_ev_ang": _safe_float(
                        np.linalg.norm(force, axis=1).max()
                    ),
                }
            )
        if len(rows) % (batch_size * 8) == 0 or len(rows) == len(indices):
            _write_rows(rows)
            atomic_json(
                RESULTS / "mag_oracle/progress.json",
                {
                    "updated_at": now(),
                    "completed": len(rows),
                    "total": int(len(indices)),
                    "status": "running",
                },
            )
    metrics = _metrics(rows)
    atomic_json(REPORT, metrics)
    REPORT_MD.write_text(
        "\n".join(
            [
                "# CHGNet magnetic-density oracle validation",
                "",
                f"- Samples: {metrics['n']}",
                f"- MAE: {metrics['mae']:.6f} A^-3",
                f"- RMSE: {metrics['rmse']:.6f} A^-3",
                f"- R2: {metrics['r2']:.6f}",
                f"- Spearman: {metrics['spearman']:.6f}",
                f"- Target-region MAE: {metrics['target_region_mae']:.6f} A^-3",
                "- Top-100 enrichment: "
                f"{metrics['top_k_target_enrichment']['100']['enrichment']}",
                f"- CHGNET_MAG_ORACLE_GO: {metrics['CHGNET_MAG_ORACLE_GO']}",
                "",
                "The CHGNet signal is only a candidate surrogate and is not treated "
                "as independent proof of the generated structures' DFT magnetic density.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    set_stage(
        "mag_oracle_validation",
        "success",
        "CHGNet magnetic-density candidate oracle validation completed.",
        metrics,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(batch_size=args.batch_size, limit=args.limit)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
