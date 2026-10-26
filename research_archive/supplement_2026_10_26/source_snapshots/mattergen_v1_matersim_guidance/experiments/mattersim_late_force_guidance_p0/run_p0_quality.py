"""Compute official P0 MatterGen metrics from one frozen Alex-MP reference."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/mattersim_late_force_guidance_p0"
REFERENCE = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/data-release/alex-mp/"
    "reference_MP2020correction.gz"
)
POTENTIAL = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/"
    "mattersim-v1.0.0-5M.pth"
)
METHODS = ("B0", "F0")
POTENTIAL_SHA256 = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sha256(POTENTIAL) != POTENTIAL_SHA256:
        raise RuntimeError("MatterSim checkpoint SHA256 mismatch")
    memory_fd = os.memfd_create("mattersim_guidance_p0_reference", os.MFD_ALLOW_SEALING)
    print("decompressing frozen Alex-MP reference once", flush=True)
    with gzip.open(REFERENCE, "rb") as source:
        with os.fdopen(os.dup(memory_fd), "wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    fcntl.fcntl(
        memory_fd, fcntl.F_ADD_SEALS,
        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL,
    )
    running = []
    try:
        for method in METHODS:
            output = ROOT / "p0_quality" / method
            if output.exists():
                raise FileExistsError(output)
            log_path = ROOT / "logs" / f"p0_quality_{method}.log"
            stream = log_path.open("x", encoding="utf-8")
            command = [
                sys.executable, "-m", "research.corrector_distillation.evaluate_quality",
                "--method", method,
                "--structures-path", str(ROOT / "p0_structures" / f"{method}_generated.extxyz"),
                "--potential-path", str(POTENTIAL),
                "--reference-memory-fd", str(memory_fd),
                "--output-dir", str(output),
                "--temporary-dir", str(ROOT / "runtime_tmp"),
                "--device", "cpu",
                "--precomputed-relaxation-dir", str(ROOT / "p0_relaxation" / method),
            ]
            environment = dict(os.environ)
            environment.update(
                CUDA_VISIBLE_DEVICES="", TMPDIR=str(PROJECT_ROOT / ".tmp"),
                XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
                HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
                TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"),
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                PYTHONUNBUFFERED="1",
            )
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, pass_fds=(memory_fd,),
            )
            running.append((method, process, stream, log_path))
            print(f"started {method} pid={process.pid}", flush=True)
        failures = []
        for method, process, stream, log_path in running:
            code = process.wait()
            stream.close()
            print(f"completed {method} return_code={code}", flush=True)
            if code:
                failures.append((method, code, str(log_path)))
        if failures:
            raise RuntimeError(f"P0 quality failures: {failures}")
    finally:
        os.close(memory_fd)


if __name__ == "__main__":
    main()
