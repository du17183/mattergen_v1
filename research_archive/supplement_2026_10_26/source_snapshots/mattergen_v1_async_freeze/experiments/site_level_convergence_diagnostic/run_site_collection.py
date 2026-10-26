"""Launch one calibration or validation site-diagnostic cohort on selected GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=("calibration", "validation"), required=True)
    parser.add_argument("--gpus", default="1,5,7")
    args = parser.parse_args()
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for shard, gpu in enumerate(gpus):
        path = logs / f"{args.subset}_shard_{shard}_gpu_{gpu}.log"
        stream = path.open("a", encoding="utf-8")
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8")
        process = subprocess.Popen([
            sys.executable, str(ROOT / "collect_site_trajectories.py"),
            "--subset", args.subset, "--shard-index", str(shard),
            "--num-shards", str(len(gpus)),
        ], cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        jobs.append((shard, gpu, process, stream, path))
        print(json.dumps({"event": "started", "subset": args.subset,
                          "shard": shard, "gpu": gpu, "pid": process.pid}), flush=True)
    failures = []
    for shard, gpu, process, stream, path in jobs:
        code = process.wait()
        stream.close()
        print(json.dumps({"event": "finished", "subset": args.subset,
                          "shard": shard, "gpu": gpu, "return_code": code}), flush=True)
        if code:
            failures.append({"shard": shard, "gpu": gpu, "code": code, "log": str(path)})
    if failures:
        raise RuntimeError(f"site collection failures: {failures}")


if __name__ == "__main__":
    main()
