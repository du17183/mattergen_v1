"""Run paired MatterSim-5M relaxations on two explicitly selected GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/cell_atom_fusion_p0"
POTENTIAL = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/"
    "mattersim/mattersim-v1.0.0-5M.pth"
)
METHODS = ("B0", "C0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,5")
    args = parser.parse_args()
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if len(gpus) < len(METHODS):
        raise ValueError("relaxation requires two distinct GPUs")
    running = []
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    for method, gpu in zip(METHODS, gpus):
        output = ROOT / "relaxation" / method
        if output.exists():
            raise FileExistsError(output)
        log_path = ROOT / "logs" / f"relaxation_{method}.log"
        stream = log_path.open("x", encoding="utf-8")
        command = [
            sys.executable, "-m", "research.corrector_distillation.relax_many_individual",
            "--structures-path", str(ROOT / "structures" / f"{method}_generated.extxyz"),
            "--potential-path", str(POTENTIAL), "--output-dir", str(output),
            "--device", "cuda", "--fmax", "0.05", "--max-natoms-per-batch", "512",
            "--cuda-3x3-det-workaround",
        ]
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2"
        )
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
        running.append((method, gpu, process, stream, log_path))
        print(json.dumps({"event": "started", "method": method,
                          "gpu": gpu, "pid": process.pid}), flush=True)
    failures = []
    for method, gpu, process, stream, log_path in running:
        code = process.wait()
        stream.close()
        print(json.dumps({"event": "finished", "method": method,
                          "gpu": gpu, "return_code": code}), flush=True)
        if code:
            failures.append({"method": method, "gpu": gpu, "log": str(log_path)})
    if failures:
        raise RuntimeError(f"MatterSim relaxation failures: {failures}")


if __name__ == "__main__":
    main()
