"""Run MatterSim quality evaluation for paired C0/T0/S0 fresh outputs."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import os
from pathlib import Path
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

from run_quality import POTENTIAL, POTENTIAL_SHA256, PYTHON, REFERENCE, sha256


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/calibrated_safe_selection_cfg"
POLICIES = ("C0", "T0", "S0")


def run_policy(cohort: str, policy: str, gpu: int, memory_fd: int) -> None:
    cohort_root = ROOT / cohort
    structure_path = cohort_root / "evaluation_structures" / f"{policy}.extxyz"
    if not structure_path.exists():
        print(f"skipping empty {cohort} {policy}", flush=True)
        return
    output = cohort_root / "quality" / policy
    if (output / "official_detailed.json.gz").exists():
        print(f"already complete {cohort} {policy}", flush=True)
        return
    if output.exists():
        raise FileExistsError(f"retained incomplete quality output: {output}")
    log_path = ROOT / "logs" / f"quality_{cohort}_{policy}_gpu_{gpu}.log"
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu), SAFE_CFG_PHYSICAL_GPU=str(gpu),
        TMPDIR=str(cohort_root / "runtime_tmp"), XDG_CACHE_HOME=str(PROJECT / ".cache"),
        HF_HOME=str(PROJECT / ".cache/huggingface"), TORCH_HOME=str(PROJECT / ".cache/torch"),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1",
    )
    command = [
        PYTHON, "-m", "research.corrector_distillation.evaluate_quality",
        "--method", f"{cohort}_{policy}", "--structures-path", str(structure_path),
        "--potential-path", str(POTENTIAL), "--reference-memory-fd", str(memory_fd),
        "--output-dir", str(output), "--temporary-dir", str(cohort_root / "runtime_tmp"),
        "--device", "cuda",
    ]
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command, cwd=PROJECT, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, pass_fds=(memory_fd,),
        )
    print(f"quality {cohort} {policy} gpu={gpu} return_code={result.returncode}", flush=True)
    if result.returncode:
        raise RuntimeError(f"quality failed for {cohort} {policy}; see {log_path}")


def worker(cohort: str, gpu: int, policies: tuple[str, ...], memory_fd: int) -> None:
    for policy in policies:
        run_policy(cohort, policy, gpu, memory_fd)


def main(cohort: str, gpus: tuple[int, ...]) -> None:
    if not gpus or any(gpu == 2 for gpu in gpus):
        raise ValueError("a non-empty GPU set excluding GPU2 is required")
    if sha256(POTENTIAL) != POTENTIAL_SHA256:
        raise RuntimeError("MatterSim checkpoint SHA256 mismatch")
    cohort_root = ROOT / cohort
    (cohort_root / "quality").mkdir(exist_ok=True)
    (cohort_root / "runtime_tmp").mkdir(exist_ok=True)
    assignments = {
        gpu: tuple(POLICIES[index] for index in range(len(POLICIES)) if index % len(gpus) == rank)
        for rank, gpu in enumerate(gpus)
    }
    memory_fd = os.memfd_create(f"safe_cfg_{cohort}_reference", os.MFD_ALLOW_SEALING)
    with gzip.open(REFERENCE, "rb") as source:
        with os.fdopen(os.dup(memory_fd), "wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    fcntl.fcntl(
        memory_fd, fcntl.F_ADD_SEALS,
        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL,
    )
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, cohort, gpu, policies, memory_fd) for gpu, policies in assignments.items()]
            for future in futures:
                future.result()
    finally:
        os.close(memory_fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("p0", "formal256"), required=True)
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args()
    main(args.cohort, tuple(int(value) for value in args.gpus.split(",") if value))
