"""Launch resumable probe-trajectory collection on explicitly selected free GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/self_correcting_search_p0"
WORKER = ROOT / "collect_probe_trajectories.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,6,7")
    args = parser.parse_args()
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus:
        raise ValueError("no GPUs selected")
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for shard, gpu in enumerate(gpus):
        log_path = logs / f"probe_collection_shard_{shard}.log"
        stream = log_path.open("a", encoding="utf-8")
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        process = subprocess.Popen(
            [sys.executable, str(WORKER), "--shard-index", str(shard),
             "--num-shards", str(len(gpus))],
            cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
        jobs.append((shard, gpu, process, stream, log_path))
        print(json.dumps({"event": "shard_started", "shard": shard,
                          "gpu": gpu, "pid": process.pid}), flush=True)
    failures = []
    for shard, gpu, process, stream, log_path in jobs:
        code = process.wait()
        stream.close()
        print(json.dumps({"event": "shard_complete", "shard": shard,
                          "gpu": gpu, "return_code": code}), flush=True)
        if code:
            failures.append({"shard": shard, "gpu": gpu, "return_code": code,
                             "log": str(log_path)})
    if failures:
        raise RuntimeError(f"probe collection failures: {failures}")


if __name__ == "__main__":
    main()
