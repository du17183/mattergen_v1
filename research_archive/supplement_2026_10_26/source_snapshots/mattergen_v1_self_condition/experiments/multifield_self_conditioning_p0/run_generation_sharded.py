"""Run one persistent checkpoint-loading worker per GPU and method."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/multifield_self_conditioning_p0"
MODEL_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")
SC_CHECKPOINT = ROOT / "checkpoint/sc_p0.pt"
WORKER = ROOT / "generate_shard.py"
METHODS = ("B0", "S-off", "S-on")
SEEDS = tuple(range(95000, 95032))


def run_method(method: str, gpu_ids: tuple[int, ...]) -> None:
    pending = []
    for seed in SEEDS:
        run_dir = ROOT / "generation" / method / str(seed)
        if (run_dir / "run_summary.json").is_file():
            continue
        if run_dir.exists():
            raise RuntimeError(f"incomplete run exists: {run_dir}")
        pending.append(seed)
    shards = [pending[index::len(gpu_ids)] for index in range(len(gpu_ids))]
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = []
    for gpu, seeds in zip(gpu_ids, shards):
        if not seeds:
            continue
        log_path = logs / f"generation_shard_{method}_{seeds[0]}_{seeds[-1]}.log"
        stream = log_path.open("x", encoding="utf-8")
        command = [
            sys.executable, str(WORKER),
            "--checkpoint-root", str(MODEL_ROOT),
            "--output-root", str(ROOT),
            "--method", method,
            "--seeds", *map(str, seeds),
        ]
        if method != "B0":
            command.extend(["--sc-checkpoint", str(SC_CHECKPOINT)])
        environment = dict(os.environ)
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            MPLCONFIGDIR="/tmp/matplotlib-self-condition",
            PYTHONPATH=(existing_pythonpath + ":" + str(PROJECT_ROOT) + ":" + str(ROOT)).strip(":"),
            PYTHONUNBUFFERED="1",
        )
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        running.append((gpu, seeds, process, stream, log_path))
        print(json.dumps({"event": "shard_started", "method": method, "gpu": gpu, "seeds": seeds, "pid": process.pid}), flush=True)
    failures = []
    for gpu, seeds, process, stream, log_path in running:
        return_code = process.wait()
        stream.close()
        print(json.dumps({"event": "shard_complete", "method": method, "gpu": gpu, "seeds": seeds, "return_code": return_code}), flush=True)
        if return_code:
            failures.append((gpu, seeds, return_code, str(log_path)))
    if failures:
        raise RuntimeError(f"generation shard failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3,4")
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    gpu_ids = tuple(int(value) for value in args.gpus.split(",") if value)
    methods = tuple(value for value in args.methods.split(",") if value)
    for method in methods:
        run_method(method, gpu_ids)


if __name__ == "__main__":
    main()
