"""Run resumable paired B0/S-off/S-on generation waves."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/multifield_self_conditioning_p0"
MODEL_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/"
    "hf_mattergen/checkpoints/dft_mag_density"
)
SC_CHECKPOINT = ROOT / "checkpoint/sc_p0.pt"
GENERATOR = ROOT / "generate_one.py"
METHODS = ("B0", "S-off", "S-on")
SEEDS = tuple(range(95000, 95032))


def command(method: str, seed: int) -> list[str]:
    result = [
        sys.executable,
        str(GENERATOR),
        "--checkpoint-root", str(MODEL_ROOT),
        "--output-root", str(ROOT),
        "--method", method,
        "--seed", str(seed),
        "--target", "0.2",
        "--guidance-scale", "2.0",
    ]
    if method != "B0":
        result.extend(["--sc-checkpoint", str(SC_CHECKPOINT)])
    return result


def run_method(method: str, gpu_ids: tuple[int, ...]) -> None:
    pending = []
    for seed in SEEDS:
        run_dir = ROOT / "generation" / method / str(seed)
        if (run_dir / "run_summary.json").is_file():
            continue
        if run_dir.exists():
            raise RuntimeError(f"incomplete run exists: {run_dir}")
        pending.append(seed)
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    method_started = time.perf_counter()
    while pending:
        wave = pending[:len(gpu_ids)]
        pending = pending[len(gpu_ids):]
        running = []
        for gpu, seed in zip(gpu_ids, wave):
            log_path = logs / f"generation_{method}_{seed}.log"
            stream = log_path.open("x", encoding="utf-8")
            environment = dict(os.environ)
            existing_pythonpath = environment.get("PYTHONPATH", "")
            environment.update(
                CUDA_VISIBLE_DEVICES=str(gpu),
                MPLCONFIGDIR="/tmp/matplotlib-self-condition",
                PYTHONPATH=(existing_pythonpath + ":" + str(PROJECT_ROOT)).strip(":"),
                PYTHONUNBUFFERED="1",
            )
            process = subprocess.Popen(
                command(method, seed), cwd=PROJECT_ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT,
            )
            running.append((gpu, seed, process, stream, log_path))
            print(json.dumps({"event": "started", "method": method, "seed": seed, "gpu": gpu, "pid": process.pid}), flush=True)
        failures = []
        for gpu, seed, process, stream, log_path in running:
            return_code = process.wait()
            stream.close()
            print(json.dumps({"event": "complete", "method": method, "seed": seed, "gpu": gpu, "return_code": return_code}), flush=True)
            if return_code:
                failures.append((seed, return_code, str(log_path)))
        if failures:
            raise RuntimeError(f"generation failures: {failures}")
    print(json.dumps({"event": "method_complete", "method": method, "wall_seconds": time.perf_counter() - method_started}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    gpu_ids = tuple(int(value) for value in args.gpus.split(",") if value)
    methods = tuple(value for value in args.methods.split(",") if value)
    if not gpu_ids or any(method not in METHODS for method in methods):
        raise ValueError("invalid GPU or method selection")
    if not SC_CHECKPOINT.is_file() and any(method != "B0" for method in methods):
        raise FileNotFoundError(SC_CHECKPOINT)
    for method in methods:
        run_method(method, gpu_ids)


if __name__ == "__main__":
    main()
