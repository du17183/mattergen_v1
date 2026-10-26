"""Apply the frozen Alex-MP/MatterGen quality evaluator to diagnostic outputs."""
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
    output = ROOT / "diagnostic_quality"
    per_structure = pd.read_csv(output / "per_structure.csv")
    with gzip.open(output / "official_detailed.json.gz", "rt", encoding="utf-8") as stream:
        detailed = json.load(stream)
    mapping = {
        "energy_above_hull_per_atom": "energy_above_hull_per_atom",
        "rmsd_from_relaxation": "rmsd_from_relaxation",
        "stable": "stable",
        "novel_unique_stable": "novel_unique_stable",
        "novel": "novel",
        "unique": "unique",
    }
    for source, destination in mapping.items():
        values = detailed[source]
        if len(values) != len(per_structure):
            raise RuntimeError(f"detailed metric count mismatch: {source}")
        per_structure[destination] = values
    per_structure.to_csv(output / "per_structure.csv", index=False)


def main() -> None:
    (PROJECT_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    output = ROOT / "diagnostic_quality"
    if output.exists():
        raise FileExistsError(output)
    memory_fd = os.memfd_create("field_diagnostic_reference", os.MFD_ALLOW_SEALING)
    print("decompressing frozen Alex-MP reference once", flush=True)
    with gzip.open(REFERENCE, "rb") as source:
        with os.fdopen(os.dup(memory_fd), "wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    fcntl.fcntl(
        memory_fd, fcntl.F_ADD_SEALS,
        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL,
    )
    log_path = ROOT / "diagnostic_quality.log"
    command = [
        sys.executable, "-m", "research.corrector_distillation.evaluate_quality",
        "--method", "diagnostic_B0",
        "--structures-path", str(ROOT / "generated_diagnostic.extxyz"),
        "--potential-path", str(POTENTIAL),
        "--reference-memory-fd", str(memory_fd),
        "--output-dir", str(output),
        "--temporary-dir", str(ROOT / "runtime_tmp"),
        "--device", "cpu",
        "--precomputed-relaxation-dir", str(ROOT / "diagnostic_relaxation"),
    ]
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="", TMPDIR=str(PROJECT_ROOT / ".tmp"),
        XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
        HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
        TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"), OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1",
    )
    try:
        with log_path.open("x", encoding="utf-8") as stream:
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, pass_fds=(memory_fd,), check=True)
    finally:
        os.close(memory_fd)
    _merge_detailed_metrics()
    print(f"Frozen quality evaluation complete: {output}")


if __name__ == "__main__":
    main()
