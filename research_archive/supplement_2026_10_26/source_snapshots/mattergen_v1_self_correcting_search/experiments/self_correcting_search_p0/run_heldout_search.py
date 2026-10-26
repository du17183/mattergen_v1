"""Run the 8-seed paired B0/S0 P0 after the probe gate passes."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/self_correcting_search_p0"
WORKER = ROOT / "generate_search_one.py"
SEEDS = tuple(range(98200, 98208))


def run_method(method: str, gpus: tuple[str, ...]) -> None:
    pending = [seed for seed in SEEDS if not (
        ROOT / "search_generation" / method / str(seed) / "run_summary.json"
    ).is_file()]
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending:
        wave, pending = pending[:len(gpus)], pending[len(gpus):]
        jobs = []
        for gpu, seed in zip(gpus, wave):
            log_path = logs / f"search_{method}_{seed}.log"
            stream = log_path.open("a", encoding="utf-8")
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                       CUBLAS_WORKSPACE_CONFIG=":4096:8")
            process = subprocess.Popen(
                [sys.executable, str(WORKER), "--method", method, "--seed", str(seed)],
                cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            )
            jobs.append((gpu, seed, process, stream, log_path))
            print(json.dumps({"event": "started", "method": method,
                              "seed": seed, "gpu": gpu, "pid": process.pid}), flush=True)
        failures = []
        for gpu, seed, process, stream, log_path in jobs:
            code = process.wait()
            stream.close()
            print(json.dumps({"event": "complete", "method": method,
                              "seed": seed, "gpu": gpu, "return_code": code}), flush=True)
            if code:
                failures.append((seed, code, str(log_path)))
        if failures:
            raise RuntimeError(f"heldout generation failures: {failures}")


def aggregate_trace() -> None:
    rows = []
    for seed in SEEDS:
        path = ROOT / "search_generation" / "S0" / str(seed) / "search_trace.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    output = ROOT / "search_trace.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "seed", "timestep", "current_pred", "proposal_pred", "current_distance",
            "proposal_distance", "score_delta", "accept_probability", "accepted",
        ], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,6,7")
    args = parser.parse_args()
    checkpoint = torch.load(ROOT / "probe_checkpoint.pt", map_location="cpu", weights_only=False)
    if not checkpoint.get("probe_go", False):
        raise RuntimeError("heldout search must not run after Probe P0 failure")
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus:
        raise ValueError("no GPUs selected")
    for method in ("B0", "S0"):
        run_method(method, gpus)
    aggregate_trace()


if __name__ == "__main__":
    main()
