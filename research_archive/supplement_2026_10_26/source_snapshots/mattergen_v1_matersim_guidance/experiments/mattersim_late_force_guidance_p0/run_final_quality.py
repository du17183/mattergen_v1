"""Run the frozen Alex-MP/MatterGen evaluator for A0 final structures."""
from __future__ import annotations

import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1")
REFERENCE = ARTIFACT_ROOT / "data-release/alex-mp/reference_MP2020correction.gz"
POTENTIAL = ARTIFACT_ROOT / "checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth"


def _merge_detailed_metrics() -> None:
    output = ROOT / "diagnostic_final_quality"
    per_structure = pd.read_csv(output / "per_structure.csv")
    with gzip.open(output / "official_detailed.json.gz", "rt", encoding="utf-8") as stream:
        detailed = json.load(stream)
    for key in (
        "energy_above_hull_per_atom", "rmsd_from_relaxation", "stable",
        "novel_unique_stable", "novel", "unique",
    ):
        if len(detailed[key]) != len(per_structure):
            raise RuntimeError(f"detailed metric count mismatch: {key}")
        per_structure[key] = detailed[key]
    per_structure.to_csv(output / "per_structure.csv", index=False)


def main() -> None:
    (PROJECT_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    output = ROOT / "diagnostic_final_quality"
    if output.exists():
        raise FileExistsError(output)
    memory_fd = os.memfd_create("late_force_reference", os.MFD_ALLOW_SEALING)
    with gzip.open(REFERENCE, "rb") as source:
        with os.fdopen(os.dup(memory_fd), "wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    fcntl.fcntl(
        memory_fd, fcntl.F_ADD_SEALS,
        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL,
    )
    command = [
        sys.executable, "-m", "research.corrector_distillation.evaluate_quality",
        "--method", "A0_diagnostic_B0",
        "--structures-path", str(ROOT / "diagnostic_final_generated.extxyz"),
        "--potential-path", str(POTENTIAL),
        "--reference-memory-fd", str(memory_fd),
        "--output-dir", str(output),
        "--temporary-dir", str(ROOT / "runtime_tmp"),
        "--device", "cpu",
        "--precomputed-relaxation-dir", str(ROOT / "diagnostic_final_relaxation"),
    ]
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="", TMPDIR=str(PROJECT_ROOT / ".tmp"),
        XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
        HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
        TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"), OMP_NUM_THREADS="4",
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1",
    )
    try:
        with (ROOT / "diagnostic_final_quality.log").open("x", encoding="utf-8") as stream:
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment,
                           stdout=stream, stderr=subprocess.STDOUT,
                           pass_fds=(memory_fd,), check=True)
    finally:
        os.close(memory_fd)
    _merge_detailed_metrics()


if __name__ == "__main__":
    main()
