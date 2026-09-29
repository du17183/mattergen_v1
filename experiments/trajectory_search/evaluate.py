"""Frozen-surrogate evaluation stages for new trajectory-search cohorts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import ase.constraints
from ase.filters import ExpCellFilter
from ase.io import read, write
import numpy as np
import pandas as pd
import torch
import torch_geometric  # noqa: F401
import torch_scatter  # noqa: F401

ase.constraints.ExpCellFilter = ExpCellFilter
sys.path.append("/mnt/datasets-livsyn/dxl/mattergen_v1/.venv/lib/python3.10/site-packages")
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor

from experiments.trajectory_search.run_generation import ROOT, PROJECT, registry, sha256


RESULTS = PROJECT / "results"
CHGNET = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar")
CHGNET_SHA = "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"
MATTERSIM = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth")
MATTERSIM_SHA = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
REFERENCE = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/data-release/alex-mp/reference_MP2020correction.gz")
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"
TARGET = 0.1
INVALID_OBJECTIVE = 1.0
GROUPS = {
    "search_baseline": ("C0", "GPulse", "PPulse", "Independent_C0"),
    "budget_scaling": ("C0", "GPulse", "PPulse", "APulse", "CPulse"),
}


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def valid_geometry(atoms) -> tuple[bool, float, str]:
    cell = np.asarray(atoms.cell.array, float)
    if len(atoms) < 1 or not np.isfinite(cell).all():
        return False, float("nan"), "empty_or_nonfinite_cell"
    determinant = float(np.linalg.det(cell))
    if not np.isfinite(determinant) or determinant <= 0.1:
        return False, float("nan"), "nonpositive_or_small_cell"
    if len(atoms) == 1:
        return True, float("inf"), ""
    distances = np.asarray(atoms.get_all_distances(mic=True), float)
    np.fill_diagonal(distances, np.inf)
    minimum = float(distances.min())
    return (bool(np.isfinite(minimum) and minimum >= 0.5), minimum, "" if minimum >= 0.5 else "minimum_distance_below_0.5A")


def prepare(cohort: str) -> None:
    if sha256(CHGNET) != CHGNET_SHA:
        raise RuntimeError("CHGNet checkpoint hash changed")
    output = RESULTS / cohort
    if (output / "property_metrics_raw.csv").exists():
        raise FileExistsError("property evaluation already frozen")
    seeds, _ = registry(cohort)
    atoms_list = []
    manifest_hash = sha256(ROOT / "protocol/seed_manifest.json")
    for seed in seeds:
        folder = output / "generation" / str(seed)
        summary = json.loads((folder / "run_summary.json").read_text())
        if not summary["success"] or not summary["reference_reproduction"]["success"] or summary["manifest_sha256"] != manifest_hash:
            raise RuntimeError(f"failed or mismatched generation {seed}")
        atoms_list.extend(read(folder / "candidates.extxyz", index=":"))
    if len(atoms_list) != len(seeds) * len(GROUPS[cohort]):
        raise RuntimeError("incomplete candidate matrix")
    write(output / "candidates_all.extxyz", atoms_list)
    model = CHGNet.from_file(CHGNET).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    rows, structures, indices = [], [], []
    for index, atoms in enumerate(atoms_list):
        good, minimum, reason = valid_geometry(atoms)
        group = str(atoms.info["policy_id"])
        row = {
            "branch_id": str(atoms.info["branch_id"]), "seed": int(atoms.info["seed"]),
            "policy_id": group, "formula": atoms.get_chemical_formula(), "num_atoms": len(atoms),
            "geometric_valid": good, "minimum_periodic_distance_angstrom": minimum,
            "validity_reason": reason, "chgnet_success": False, "chgnet_mag_density": float("nan"),
            "target_mag_density": TARGET, "property_absolute_error": INVALID_OBJECTIVE,
            "structure_valid": False, "surrogate_property_eval": True, "dft_verified": False,
        }
        rows.append(row)
        if good:
            try:
                structure = AseAtomsAdaptor.get_structure(atoms)
            except BaseException as error:
                row["validity_reason"] = f"pymatgen_conversion:{type(error).__name__}"
            else:
                indices.append(index)
                structures.append(structure)
    for start in range(0, len(structures), 32):
        batch = structures[start:start + 32]
        try:
            predictions = model.predict_structure(batch, task="efsm", batch_size=32)
            if isinstance(predictions, dict):
                predictions = [predictions]
        except BaseException:
            predictions = []
            for structure in batch:
                try:
                    predictions.append(model.predict_structure(structure, task="efsm"))
                except BaseException as error:
                    predictions.append(error)
        for index, structure, prediction in zip(indices[start:start + 32], batch, predictions):
            row = rows[index]
            if isinstance(prediction, BaseException):
                row["validity_reason"] = f"chgnet:{type(prediction).__name__}"
                continue
            value = float(np.abs(np.asarray(prediction["m"], float)).sum() / structure.volume)
            if not np.isfinite(value):
                row["validity_reason"] = "chgnet_nonfinite"
                continue
            row.update(chgnet_success=True, chgnet_mag_density=value, property_absolute_error=abs(value - TARGET), structure_valid=True, validity_reason="")
    csv_write(output / "property_metrics_raw.csv", rows)
    folder = output / "evaluation_structures"
    folder.mkdir(exist_ok=False)
    lookup = {row["branch_id"]: row for row in rows}
    for group in GROUPS[cohort]:
        selected = [atoms for atoms in atoms_list if str(atoms.info["policy_id"]) == group and lookup[str(atoms.info["branch_id"])]["structure_valid"]]
        if selected:
            write(folder / f"{group}.extxyz", selected)
    (output / "property_summary.json").write_text(json.dumps({"n": len(rows), "valid": sum(x["structure_valid"] for x in rows), "invalid": sum(not x["structure_valid"] for x in rows), "surrogate_only": True}, indent=2) + "\n")


def evaluate_group(cohort: str, mode: str, group: str, gpu: int, reference_fd: int) -> None:
    output = RESULTS / cohort
    structure = output / ("evaluation_structures" if mode == "branches" else "mixed_evaluation_structures") / f"{group}.extxyz"
    if not structure.exists():
        raise FileNotFoundError(f"no valid structures in {group}; explicit evaluation failure")
    quality = output / ("quality_branches" if mode == "branches" else "quality_mixed") / group
    if (quality / "official_detailed.json.gz").exists():
        return
    if quality.exists():
        raise FileExistsError(f"partial quality output retained: {quality}")
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=str(output / "runtime_tmp"), XDG_CACHE_HOME="/mnt/datasets-livsyn/dxl/mattergen_v1/.cache", HF_HOME="/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/huggingface", TORCH_HOME="/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/torch", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    cmd = [PYTHON, "-m", "research.corrector_distillation.evaluate_quality", "--method", group, "--structures-path", str(structure), "--potential-path", str(MATTERSIM), "--reference-memory-fd", str(reference_fd), "--output-dir", str(quality), "--temporary-dir", str(output / "runtime_tmp"), "--device", "cuda"]
    with (logs / f"quality_{mode}_{group}_gpu{gpu}.log").open("x") as stream:
        result = subprocess.run(cmd, cwd=Path("/mnt/datasets-livsyn/dxl/mattergen_v1_field_cfg"), env=env, stdout=stream, stderr=subprocess.STDOUT, pass_fds=(reference_fd,))
    if result.returncode:
        raise RuntimeError(f"quality failed {mode}/{group}: code {result.returncode}")


def quality(cohort: str, mode: str, gpus: tuple[int, ...]) -> None:
    if sha256(MATTERSIM) != MATTERSIM_SHA:
        raise RuntimeError("MatterSim checkpoint changed")
    output = RESULTS / cohort
    source = output / ("evaluation_structures" if mode == "branches" else "mixed_evaluation_structures")
    groups = sorted(path.stem for path in source.glob("*.extxyz"))
    if not groups:
        raise RuntimeError("nothing to evaluate")
    (output / ("quality_branches" if mode == "branches" else "quality_mixed")).mkdir(exist_ok=True)
    (output / "runtime_tmp").mkdir(exist_ok=True)
    fd = os.memfd_create(f"trajectory_{cohort}_{mode}_reference", os.MFD_ALLOW_SEALING)
    with gzip.open(REFERENCE, "rb") as src, os.fdopen(os.dup(fd), "wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL)
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(evaluate_group, cohort, mode, group, gpus[index % len(gpus)], fd) for index, group in enumerate(groups)]
            for future in futures:
                future.result()
    finally:
        os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=tuple(GROUPS), required=True)
    parser.add_argument("--stage", choices=("prepare", "quality-branches", "quality-mixed"), required=True)
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.cohort)
    else:
        quality(args.cohort, "branches" if args.stage == "quality-branches" else "mixed", tuple(int(x) for x in args.gpus.split(",")))
