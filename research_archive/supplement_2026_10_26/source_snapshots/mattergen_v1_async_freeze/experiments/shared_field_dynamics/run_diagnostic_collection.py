"""Launch the fixed 64-trajectory diagnostic over explicitly selected GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/shared_field_dynamics"
WORKER = ROOT / "collect_diagnostic_trajectories.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,5,7")
    args = parser.parse_args()
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus:
        raise ValueError("no GPUs selected")
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for shard, gpu in enumerate(gpus):
        log_path = logs / f"diagnostic_shard_{shard}_gpu_{gpu}.log"
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
        print(json.dumps({"event": "started", "shard": shard, "gpu": gpu,
                          "pid": process.pid}), flush=True)
    failures = []
    for shard, gpu, process, stream, log_path in jobs:
        code = process.wait()
        stream.close()
        print(json.dumps({"event": "finished", "shard": shard, "gpu": gpu,
                          "return_code": code}), flush=True)
        if code:
            failures.append({"shard": shard, "gpu": gpu, "code": code,
                             "log": str(log_path)})
    if failures:
        raise RuntimeError(f"diagnostic collection failed: {failures}")


if __name__ == "__main__":
    main()
