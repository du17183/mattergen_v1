"""Run the existing GPU MatterSim and official metrics pipelines for V3."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import os
from pathlib import Path
import shutil
import subprocess
import sys

from research.corrector_distillation.v3_anchor_protocol import (
    ASSET_PATHS,
    PROJECT_ROOT,
    methods_for,
    output_root_for,
)


def run_jobs(jobs: list[tuple[str, list[str], Path, str]], pass_fds=()) -> None:
    processes = []
    for method, command, log_path, gpu in jobs:
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=gpu,
            TMPDIR=str(PROJECT_ROOT / ".tmp"),
            XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
            HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
            TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            PYTHONUNBUFFERED="1",
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("x", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
            pass_fds=pass_fds,
        )
        processes.append((method, process, stream, log_path))
        print(f"started {method} pid={process.pid} log={log_path}", flush=True)
    failures = []
    for method, process, stream, log_path in processes:
        return_code = process.wait()
        stream.close()
        print(f"completed {method} return_code={return_code}", flush=True)
        if return_code:
            failures.append((method, return_code, str(log_path)))
    if failures:
        raise RuntimeError(f"quality pipeline failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stage-b", "stage-c"), required=True)
    parser.add_argument("--phase", choices=("relax", "metrics"), required=True)
    args = parser.parse_args()
    root = output_root_for(args.stage)
    jobs = []
    memory_fd = None
    if args.phase == "metrics":
        memory_fd = os.memfd_create("mattergen_alex_mp_reference", os.MFD_ALLOW_SEALING)
        print("Decompressing the frozen reference once into shared anonymous RAM", flush=True)
        with gzip.open(ASSET_PATHS["alex_mp_reference"], "rb") as source:
            with os.fdopen(os.dup(memory_fd), "wb") as destination:
                shutil.copyfileobj(source, destination, length=16*1024*1024)
        fcntl.fcntl(memory_fd, fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL)
        print(f"Shared immutable LMDB ready: {os.fstat(memory_fd).st_size} bytes", flush=True)
    for index, method in enumerate(methods_for(args.stage)):
        structures = root / "structures" / f"{method}_generated.extxyz"
        if not structures.is_file():
            raise FileNotFoundError(structures)
        relaxed = root / "relaxation/precomputed" / method
        if args.phase == "relax":
            output = relaxed
            complete = output / "relaxation_summary.json"
            command = [
                sys.executable, "-m",
                "research.corrector_distillation.relax_many_individual",
                "--structures-path", str(structures),
                "--potential-path", str(ASSET_PATHS["mattersim_checkpoint"]),
                "--output-dir", str(output),
                "--device", "cuda", "--cuda-3x3-det-workaround",
            ]
            gpu = str(index + 1)
        else:
            output = root / "quality" / method
            complete = output / "quality_summary.json"
            command = [
                sys.executable, "-m",
                "research.corrector_distillation.evaluate_quality",
                "--method", method, "--structures-path", str(structures),
                "--potential-path", str(ASSET_PATHS["mattersim_checkpoint"]),
                "--reference-memory-fd", str(memory_fd),
                "--output-dir", str(output),
                "--temporary-dir", str(root / "runtime_tmp"),
                "--device", "cpu", "--precomputed-relaxation-dir", str(relaxed),
            ]
            gpu = ""
        if complete.is_file():
            print(f"already complete {method}: {complete}", flush=True)
            continue
        if output.exists():
            raise RuntimeError(f"inspect incomplete output before retrying: {output}")
        jobs.append((method, command, root / "logs" / f"{args.phase}_{method}.log", gpu))
    try:
        run_jobs(jobs, pass_fds=() if memory_fd is None else (memory_fd,))
    finally:
        if memory_fd is not None:
            os.close(memory_fd)


if __name__ == "__main__":
    main()
