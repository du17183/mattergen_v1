"""Resume only incomplete preregistered probe trajectories on selected GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/self_correcting_search_p0"
DATASET = ROOT / "probe_dataset"
WORKER = ROOT / "collect_probe_trajectories.py"
ALL_SEEDS = tuple(range(98000, 98192))
REQUIRED = (
    "generated_crystals.extxyz",
    "generated_crystals_cif.zip",
    "hidden_states.npz",
    "run_summary.json",
)


def split_for(seed: int) -> str:
    if seed < 98128:
        return "train"
    if seed < 98160:
        return "validation"
    return "test"


def complete(seed: int) -> bool:
    run_dir = DATASET / split_for(seed) / str(seed)
    return all((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0
               for name in REQUIRED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,4,5,6,7")
    args = parser.parse_args()
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus:
        raise ValueError("no GPUs selected")

    missing = tuple(seed for seed in ALL_SEEDS if not complete(seed))
    print(json.dumps({"event": "resume_audit", "complete": 192 - len(missing),
                      "missing": len(missing), "seeds": missing}), flush=True)
    if not missing:
        return

    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for worker_index, gpu in enumerate(gpus):
        seeds = missing[worker_index::len(gpus)]
        if not seeds:
            continue
        log_path = logs / f"probe_collection_resume_gpu_{gpu}.log"
        stream = log_path.open("a", encoding="utf-8")
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        command = [sys.executable, str(WORKER), "--seeds", *map(str, seeds)]
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT,
        )
        jobs.append((gpu, seeds, process, stream, log_path))
        print(json.dumps({"event": "worker_started", "gpu": gpu,
                          "pid": process.pid, "seeds": seeds}), flush=True)

    failures = []
    for gpu, seeds, process, stream, log_path in jobs:
        code = process.wait()
        stream.close()
        print(json.dumps({"event": "worker_complete", "gpu": gpu,
                          "return_code": code, "seeds": seeds}), flush=True)
        if code:
            failures.append({"gpu": gpu, "return_code": code,
                             "log": str(log_path)})
    if failures:
        raise RuntimeError(f"probe collection failures: {failures}")

    remaining = tuple(seed for seed in ALL_SEEDS if not complete(seed))
    print(json.dumps({"event": "resume_complete", "complete": 192 - len(remaining),
                      "missing": len(remaining), "seeds": remaining}), flush=True)
    if remaining:
        raise RuntimeError(f"incomplete trajectories remain: {remaining}")


if __name__ == "__main__":
    main()
