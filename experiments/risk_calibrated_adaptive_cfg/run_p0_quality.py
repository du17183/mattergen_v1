"""Run MatterSim quality evaluation for fresh P0 C0/T0/A0 groups."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/risk_calibrated_adaptive_cfg"
P0 = ROOT / "p0"
REFERENCE = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/data-release/alex-mp/reference_MP2020correction.gz")
POTENTIAL = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth")
POTENTIAL_SHA256 = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def run(policy: str, gpu: int, memory_fd: int) -> None:
    source = P0 / "evaluation_structures" / f"{policy}.extxyz"; output = P0 / "quality" / policy
    if not source.exists(): return
    if (output / "official_detailed.json.gz").exists(): return
    if output.exists(): raise FileExistsError(f"retained incomplete quality output: {output}")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=str(P0 / "runtime_tmp"),
                       XDG_CACHE_HOME=str(PROJECT / ".cache"), HF_HOME=str(PROJECT / ".cache/huggingface"),
                       TORCH_HOME=str(PROJECT / ".cache/torch"), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    command = [PYTHON, "-m", "research.corrector_distillation.evaluate_quality", "--method", policy,
               "--structures-path", str(source), "--potential-path", str(POTENTIAL),
               "--reference-memory-fd", str(memory_fd), "--output-dir", str(output),
               "--temporary-dir", str(P0 / "runtime_tmp"), "--device", "cuda"]
    with (ROOT / "logs" / f"p0_quality_{policy}_gpu_{gpu}.log").open("x") as stream:
        result = subprocess.run(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT, pass_fds=(memory_fd,))
    if result.returncode: raise RuntimeError(f"P0 quality failed: {policy}")


def main() -> None:
    if sha256(POTENTIAL) != POTENTIAL_SHA256: raise RuntimeError("MatterSim checkpoint mismatch")
    (P0 / "quality").mkdir(exist_ok=True); (P0 / "runtime_tmp").mkdir(exist_ok=True)
    memory_fd = os.memfd_create("risk_cfg_p0_reference", os.MFD_ALLOW_SEALING)
    with gzip.open(REFERENCE, "rb") as source:
        with os.fdopen(os.dup(memory_fd), "wb") as destination: shutil.copyfileobj(source, destination, 16 * 1024 * 1024)
    fcntl.fcntl(memory_fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run, policy, gpu, memory_fd) for policy, gpu in zip(("C0", "T0", "A0"), (0, 1, 3))]
            for future in futures: future.result()
    finally: os.close(memory_fd)


if __name__ == "__main__": main()
