"""Run frozen P0 pairs on the currently available A-only H20 pool."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="5,6,7")
    parser.add_argument("--exclude-seeds", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    gpus = [int(item) for item in args.gpus.split(",")]
    if set(gpus) - {5, 6, 7}:
        raise ValueError("after external contention, A is restricted to pool {5,6,7}")
    excluded = {int(item) for item in args.exclude_seeds.split(",") if item}
    seeds = [710032] if args.smoke else list(range(710032, 710048))
    seeds = [seed for seed in seeds if seed not in excluded]
    # Preserve successful smoke/completed pairs exactly; rerunning a pair is
    # unnecessary and would blur the fresh-pair provenance in the manifest.
    seeds = [
        seed for seed in seeds
        if not (
            (ROOT / "p0_generation" / "B0" / str(seed) / "run_summary.json").is_file()
            and (ROOT / "p0_generation" / "F0" / str(seed) / "run_summary.json").is_file()
        )
    ]
    running = []
    failures = []
    for index, seed in enumerate(seeds):
        gpu = gpus[index % len(gpus)]
        if index >= len(gpus):
            prior_seed, prior, stream = running[index - len(gpus)]
            code = prior.wait()
            stream.close()
            if code:
                failures.append((prior_seed, code))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), A_PHYSICAL_GPU=str(gpu))
        (ROOT / "logs").mkdir(parents=True, exist_ok=True)
        stream = (ROOT / "logs" / f"p0_pair_{seed}_gpu_{gpu}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "generate_p0_pair.py"), "--seed", str(seed)],
            cwd=ROOT.parents[1], env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
        running.append((seed, process, stream))
    for seed, process, stream in running[max(0, len(running) - len(gpus)):]:
        code = process.wait()
        stream.close()
        if code:
            failures.append((seed, code))
    if failures:
        raise RuntimeError(f"P0 pair failures retained: {failures}")


if __name__ == "__main__":
    main()
