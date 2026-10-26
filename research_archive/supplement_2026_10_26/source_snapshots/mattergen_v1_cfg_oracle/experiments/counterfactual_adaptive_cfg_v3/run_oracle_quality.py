"""Run official MatterGen surrogate quality evaluation for Oracle branch groups."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/counterfactual_adaptive_cfg_v3"
ORACLE = ROOT / "oracle"
REFERENCE = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/data-release/alex-mp/"
    "reference_MP2020correction.gz"
)
POTENTIAL = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/"
    "mattersim-v1.0.0-5M.pth"
)
POTENTIAL_SHA256 = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
GROUPS = (
    "SHORT_DOWN", "SHORT_KEEP", "SHORT_UP",
    "FULL_DOWN", "FULL_KEEP", "FULL_UP",
)
GPU_POOL = (0, 1, 3, 4, 5, 6)
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_quality(group: str, gpu: int, memory_fd: int) -> None:
    structure_path = ORACLE / "evaluation_structures" / f"{group}.extxyz"
    if not structure_path.exists():
        print(f"skipping empty valid group {group}", flush=True)
        return
    output = ORACLE / "quality" / group
    if (output / "official_detailed.json.gz").exists():
        print(f"already complete {group}", flush=True)
        return
    if output.exists():
        raise FileExistsError(f"retained incomplete quality output: {output}")
    log_path = ROOT / "logs" / f"oracle_quality_{group}_gpu_{gpu}.log"
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        CFG_ORACLE_PHYSICAL_GPU=str(gpu),
        TMPDIR=str(ORACLE / "runtime_tmp"),
        XDG_CACHE_HOME=str(PROJECT / ".cache"),
        HF_HOME=str(PROJECT / ".cache/huggingface"),
        TORCH_HOME=str(PROJECT / ".cache/torch"),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
    )
    command = [
        PYTHON,
        "-m",
        "research.corrector_distillation.evaluate_quality",
        "--method",
        group,
        "--structures-path",
        str(structure_path),
        "--potential-path",
        str(POTENTIAL),
        "--reference-memory-fd",
        str(memory_fd),
        "--output-dir",
        str(output),
        "--temporary-dir",
        str(ORACLE / "runtime_tmp"),
        "--device",
        "cuda",
    ]
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            pass_fds=(memory_fd,),
        )
    print(f"quality {group} gpu={gpu} return_code={result.returncode}", flush=True)
    if result.returncode:
        raise RuntimeError(f"quality failed for {group}; see {log_path}")


def main() -> None:
    if sha256(POTENTIAL) != POTENTIAL_SHA256:
        raise RuntimeError("MatterSim checkpoint SHA256 mismatch")
    (ORACLE / "quality").mkdir(exist_ok=True)
    (ORACLE / "runtime_tmp").mkdir(exist_ok=True)
    memory_fd = os.memfd_create("cfg_oracle_reference", os.MFD_ALLOW_SEALING)
    print("decompressing frozen Alex-MP reference once", flush=True)
    with gzip.open(REFERENCE, "rb") as source:
        with os.fdopen(os.dup(memory_fd), "wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    fcntl.fcntl(
        memory_fd,
        fcntl.F_ADD_SEALS,
        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL,
    )
    try:
        with ThreadPoolExecutor(max_workers=len(GROUPS)) as pool:
            futures = [
                pool.submit(run_quality, group, gpu, memory_fd)
                for group, gpu in zip(GROUPS, GPU_POOL)
            ]
            for future in futures:
                future.result()
    finally:
        os.close(memory_fd)


if __name__ == "__main__":
    main()
