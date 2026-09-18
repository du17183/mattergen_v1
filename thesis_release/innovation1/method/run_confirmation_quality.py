"""Run frozen MatterSim evaluation for branch or final mixed batches."""

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


PROJECT = Path(__file__).resolve().parents[2]; ROOT = PROJECT / "experiments/frozen_linear_k2_confirmation"
REFERENCE = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/data-release/alex-mp/reference_MP2020correction.gz")
POTENTIAL = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth")
POTENTIAL_SHA256 = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"; PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def run_group(phase: str, mode: str, group: str, gpu: int, memory_fd: int) -> None:
    phase_root = ROOT / phase; structure_root = phase_root / ("evaluation_structures" if mode == "branches" else "mixed_evaluation_structures"); quality_root = phase_root / ("quality_branches" if mode == "branches" else "quality_mixed")
    structure_path = structure_root / f"{group}.extxyz"; output = quality_root / group
    if not structure_path.exists(): print(f"skip empty {phase}/{mode}/{group}", flush=True); return
    if (output / "official_detailed.json.gz").exists(): print(f"already complete {phase}/{mode}/{group}", flush=True); return
    if output.exists(): raise FileExistsError(output)
    log_path = ROOT / "logs" / f"quality_{phase}_{mode}_{group}_gpu_{gpu}.log"; environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), LINEAR_K2_PHYSICAL_GPU=str(gpu), TMPDIR=str(phase_root / "runtime_tmp"), XDG_CACHE_HOME="/mnt/datasets-livsyn/dxl/mattergen_v1/.cache", HF_HOME="/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/huggingface", TORCH_HOME="/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/torch", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    command = [PYTHON, "-m", "research.corrector_distillation.evaluate_quality", "--method", group, "--structures-path", str(structure_path), "--potential-path", str(POTENTIAL), "--reference-memory-fd", str(memory_fd), "--output-dir", str(output), "--temporary-dir", str(phase_root / "runtime_tmp"), "--device", "cuda"]
    with log_path.open("x", encoding="utf-8") as stream: result = subprocess.run(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT, pass_fds=(memory_fd,))
    if result.returncode: raise RuntimeError(f"quality failed {phase}/{mode}/{group}: {log_path}")


def worker(phase: str, mode: str, gpu: int, groups: tuple[str, ...], fd: int) -> None:
    for group in groups: run_group(phase, mode, group, gpu, fd)


def main(phase: str, mode: str, gpus: tuple[int, ...]) -> None:
    if sha256(POTENTIAL) != POTENTIAL_SHA256: raise RuntimeError("MatterSim hash mismatch")
    if not gpus or len(set(gpus)) != len(gpus): raise ValueError("invalid GPUs")
    phase_root = ROOT / phase; structure_root = phase_root / ("evaluation_structures" if mode == "branches" else "mixed_evaluation_structures"); groups = tuple(sorted(path.stem for path in structure_root.glob("*.extxyz"))); quality_root = phase_root / ("quality_branches" if mode == "branches" else "quality_mixed")
    quality_root.mkdir(exist_ok=True); (phase_root / "runtime_tmp").mkdir(exist_ok=True); (ROOT / "logs").mkdir(exist_ok=True)
    assignments = {gpu: tuple(group for index, group in enumerate(groups) if index % len(gpus) == rank) for rank, gpu in enumerate(gpus)}
    fd = os.memfd_create(f"linear_k2_{phase}_{mode}_reference", os.MFD_ALLOW_SEALING)
    with gzip.open(REFERENCE, "rb") as source:
        with os.fdopen(os.dup(fd), "wb") as destination: shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL)
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, phase, mode, gpu, assigned, fd) for gpu, assigned in assignments.items()]
            for future in futures: future.result()
    finally: os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("c1_128", "c2_128", "pooled256"), required=True); parser.add_argument("--mode", choices=("branches", "mixed"), required=True); parser.add_argument("--gpus", required=True); args = parser.parse_args(); main(args.phase, args.mode, tuple(int(v) for v in args.gpus.split(",") if v))
