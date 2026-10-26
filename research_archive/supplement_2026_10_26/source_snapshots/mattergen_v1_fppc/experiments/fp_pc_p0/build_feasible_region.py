"""Freeze FP-PC feasibility bounds from the official Alex-MP train split.

This script deliberately never looks at generated outcomes.  It takes a
deterministic reservoir sample from reference structures and freezes the
0.5/99.5 percentiles requested by the preregistration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from pymatgen.core import Structure


SHIFTS = np.asarray(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible(row: dict[str, str]) -> bool:
    try:
        target = float(row["dft_mag_density"])
        num_sites = int(float(row["num_sites"]))
    except (KeyError, TypeError, ValueError):
        return False
    return bool(row.get("cif")) and math.isfinite(target) and 0 < num_sites <= 20


def reservoir_rows(
    zip_path: Path, member: str, *, count: int, seed: int
) -> tuple[list[dict[str, str]], int]:
    rng = random.Random(seed)
    chosen: list[dict[str, str]] = []
    eligible_count = 0
    with ZipFile(zip_path) as archive, archive.open(member, "r") as binary:
        reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8", newline=""))
        for source_row, row in enumerate(reader):
            if not eligible(row):
                continue
            row["source_row"] = str(source_row)
            eligible_count += 1
            if len(chosen) < count:
                chosen.append(row)
            else:
                replacement = rng.randrange(eligible_count)
                if replacement < count:
                    chosen[replacement] = row
    if len(chosen) != count:
        raise RuntimeError(f"requested {count}, found {len(chosen)}")
    return chosen, eligible_count


def minimum_periodic_distance(structure: Structure) -> float:
    """Minimum site/image distance using the same 27-image rule as B0."""

    frac = np.remainder(np.asarray(structure.frac_coords, dtype=float), 1.0)
    cell = np.asarray(structure.lattice.matrix, dtype=float)
    delta = frac[:, None, None, :] - frac[None, :, None, :] + SHIFTS[None, None, :, :]
    cart = np.einsum("...i,ij->...j", delta, cell)
    distance = np.linalg.norm(cart, axis=-1)
    diagonal = np.arange(len(frac))
    zero_shift = int(np.flatnonzero(np.all(SHIFTS == 0, axis=1))[0])
    distance[diagonal, diagonal, zero_shift] = np.inf
    return float(distance.min())


def describe(values: np.ndarray) -> dict[str, float]:
    quantiles = (0.0, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 1.0)
    return {f"q{q:g}": float(np.quantile(values, q)) for q in quantiles} | {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    rows, eligible_count = reservoir_rows(
        args.zip_path, "alex_mp_20/train.csv", count=args.count, seed=args.seed
    )
    distance: list[float] = []
    volume_per_atom: list[float] = []
    parse_failures: list[dict[str, str]] = []
    for row in rows:
        try:
            structure = Structure.from_str(row["cif"], fmt="cif")
            d_min = minimum_periodic_distance(structure)
            vn = float(structure.volume / len(structure))
            if not (math.isfinite(d_min) and math.isfinite(vn) and d_min > 0 and vn > 0):
                raise ValueError(f"non-finite/non-positive metrics d={d_min}, V/N={vn}")
            distance.append(d_min)
            volume_per_atom.append(vn)
        except Exception as exc:  # retain an auditable failure count, never substitute data
            parse_failures.append(
                {"source_row": row["source_row"], "material_id": row.get("material_id", ""),
                 "error": f"{type(exc).__name__}: {exc}"}
            )

    d = np.asarray(distance, dtype=float)
    vn = np.asarray(volume_per_atom, dtype=float)
    if len(d) < int(0.99 * args.count):
        raise RuntimeError(f"too many parse failures: {len(parse_failures)}/{args.count}")
    result = {
        "schema_version": 1,
        "frozen": True,
        "frozen_before_diagnostic_outcomes": True,
        "source": "official MatterGen Alex-MP training split",
        "source_zip": str(args.zip_path.resolve()),
        "source_member": "alex_mp_20/train.csv",
        "source_zip_sha256": sha256(args.zip_path),
        "eligibility": "finite dft_mag_density, CIF present, 1-20 sites",
        "eligible_source_rows": eligible_count,
        "reservoir_seed": args.seed,
        "reservoir_requested": args.count,
        "reference_structures_used": int(len(d)),
        "parse_failure_count": len(parse_failures),
        "parse_failures": parse_failures,
        "periodic_distance_algorithm": "27 images (-1,0,1)^3; includes self images except zero",
        "quantile_rule": {
            "d_min_floor": "reference q0.005",
            "VN_low": "reference q0.005",
            "VN_high": "reference q0.995",
        },
        "d_min_floor_angstrom": float(np.quantile(d, 0.005)),
        "VN_low_angstrom3_per_atom": float(np.quantile(vn, 0.005)),
        "VN_high_angstrom3_per_atom": float(np.quantile(vn, 0.995)),
        "reference_distributions": {
            "minimum_periodic_distance_angstrom": describe(d),
            "volume_per_atom_angstrom3": describe(vn),
        },
        "heldout_generation_metrics_consulted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"parse_failures"}}, indent=2))


if __name__ == "__main__":
    main()
