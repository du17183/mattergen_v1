"""Run the frozen A0 cohort over explicitly selected physical GPUs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "generate_diagnostic_one.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,3,4,5,6,7")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    gpus = [int(item) for item in args.gpus.split(",")]
    if 2 in gpus:
        raise ValueError("GPU2 is frozen out of this experiment")
    seeds = [710000] if args.smoke else list(range(710000, 710032))
    processes: list[tuple[int, int, subprocess.Popen]] = []
    failures: list[tuple[int, int]] = []
    for index, seed in enumerate(seeds):
        gpu = gpus[index % len(gpus)]
        if index >= len(gpus):
            prior_seed, prior_gpu, prior = processes[index - len(gpus)]
            code = prior.wait()
            if code:
                failures.append((prior_seed, code))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["A_PHYSICAL_GPU"] = str(gpu)
        log = (ROOT / "logs")
        log.mkdir(parents=True, exist_ok=True)
        stream = (log / f"diagnostic_seed_{seed}_gpu_{gpu}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--seed", str(seed)],
            cwd=ROOT.parents[1], env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
        processes.append((seed, gpu, process))
    for seed, gpu, process in processes[max(0, len(processes) - len(gpus)):]:
        code = process.wait()
        if code:
            failures.append((seed, code))
    if failures:
        raise RuntimeError(f"diagnostic generation failures retained: {failures}")


if __name__ == "__main__":
    main()
