"""Compute quality metrics from the shared reference and paired relaxations."""
from __future__ import annotations

import fcntl
import gzip
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/cell_atom_fusion_p0"
REFERENCE = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/data-release/"
    "alex-mp/reference_MP2020correction.gz"
)
POTENTIAL = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/"
    "mattersim/mattersim-v1.0.0-5M.pth"
)
METHODS = ("B0", "C0")


def main() -> None:
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    memory_fd = os.memfd_create("cell_atom_p0_reference", os.MFD_ALLOW_SEALING)
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
            output = ROOT / "quality" / method
            if output.exists():
                raise FileExistsError(output)
            log_path = ROOT / "logs" / f"quality_{method}.log"
            stream = log_path.open("x", encoding="utf-8")
            command = [
                sys.executable, "-m", "research.corrector_distillation.evaluate_quality",
                "--method", method,
                "--structures-path", str(ROOT / "structures" / f"{method}_generated.extxyz"),
                "--potential-path", str(POTENTIAL),
                "--reference-memory-fd", str(memory_fd),
                "--output-dir", str(output),
                "--temporary-dir", str(ROOT / "runtime_tmp"), "--device", "cpu",
                "--precomputed-relaxation-dir", str(ROOT / "relaxation" / method),
            ]
            environment = dict(os.environ)
            environment.update(
                CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1",
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
            raise RuntimeError(f"quality metric failures: {failures}")
    finally:
        os.close(memory_fd)


if __name__ == "__main__":
    main()
