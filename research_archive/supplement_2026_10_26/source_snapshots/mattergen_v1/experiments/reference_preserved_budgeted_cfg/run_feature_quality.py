"""Run frozen MatterSim quality evaluation for policy or final mixed batches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import subprocess


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/reference_preserved_budgeted_cfg"
FEATURE_ROOT = ROOT / "feature_study"
REFERENCE = PROJECT / "data-release/alex-mp/reference_MP2020correction.gz"
POTENTIAL = PROJECT / "checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth"
POTENTIAL_SHA256 = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"
POLICY_IDS = ("C0", "GPulse", "APulse", "PPulse", "CPulse")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_group(mode: str, group: str, gpu: int, memory_fd: int) -> None:
    if mode == "policies":
        structure_path = FEATURE_ROOT / "evaluation_structures" / f"{group}.extxyz"
        quality_root = FEATURE_ROOT / "quality_policies"
    else:
        structure_path = FEATURE_ROOT / "mixed_evaluation_structures" / f"{group}.extxyz"
        quality_root = FEATURE_ROOT / "quality_mixed"
    if not structure_path.exists():
        print(f"skip empty {mode}/{group}", flush=True)
        return
    output = quality_root / group
    if (output / "official_detailed.json.gz").exists():
        print(f"already complete {mode}/{group}", flush=True)
        return
    if output.exists():
        raise FileExistsError(output)
    log_path = ROOT / "logs" / f"quality_{mode}_{group}_gpu_{gpu}.log"
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu), RPB_CFG_PHYSICAL_GPU=str(gpu),
        TMPDIR=str(FEATURE_ROOT / "runtime_tmp"), XDG_CACHE_HOME=str(PROJECT / ".cache"),
        HF_HOME=str(PROJECT / ".cache/huggingface"), TORCH_HOME=str(PROJECT / ".cache/torch"),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1",
    )
    command = [
        PYTHON, "-m", "research.corrector_distillation.evaluate_quality",
        "--method", group, "--structures-path", str(structure_path),
        "--potential-path", str(POTENTIAL), "--reference-memory-fd", str(memory_fd),
        "--output-dir", str(output), "--temporary-dir", str(FEATURE_ROOT / "runtime_tmp"),
        "--device", "cuda",
    ]
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT, pass_fds=(memory_fd,))
    if result.returncode:
        raise RuntimeError(f"quality failed {mode}/{group}: {log_path}")


def worker(mode: str, gpu: int, groups: tuple[str, ...], memory_fd: int) -> None:
    for group in groups:
        run_group(mode, group, gpu, memory_fd)


def main(mode: str, gpus: tuple[int, ...]) -> None:
    if sha256(POTENTIAL) != POTENTIAL_SHA256:
        raise RuntimeError("MatterSim hash mismatch")
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("invalid GPU list")
    if mode == "policies":
        groups = POLICY_IDS
        quality_root = FEATURE_ROOT / "quality_policies"
    else:
        groups = tuple(sorted(path.stem for path in (FEATURE_ROOT / "mixed_evaluation_structures").glob("*.extxyz")))
        quality_root = FEATURE_ROOT / "quality_mixed"
    quality_root.mkdir(exist_ok=True)
    (FEATURE_ROOT / "runtime_tmp").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    assignments = {gpu: tuple(group for index, group in enumerate(groups) if index % len(gpus) == rank) for rank, gpu in enumerate(gpus)}
    memory_fd = os.memfd_create(f"rpb_cfg_phase_b_{mode}_reference", os.MFD_ALLOW_SEALING)
    with gzip.open(REFERENCE, "rb") as source:
        with os.fdopen(os.dup(memory_fd), "wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    fcntl.fcntl(memory_fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL)
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, mode, gpu, assigned, memory_fd) for gpu, assigned in assignments.items()]
            for future in futures:
                future.result()
    finally:
        os.close(memory_fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("policies", "mixed"), required=True)
    parser.add_argument("--gpus", required=True)
    arguments = parser.parse_args()
    main(arguments.mode, tuple(int(value) for value in arguments.gpus.split(",") if value))
