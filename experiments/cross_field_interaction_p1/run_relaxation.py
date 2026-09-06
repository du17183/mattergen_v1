"""Run frozen P1 MatterSim-5M relaxations on three H20 GPUs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/cross_field_interaction_p1"
POTENTIAL = PROJECT_ROOT / "checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth"
METHODS = ("C0", "MLP", "CFI")


def main() -> None:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = []
    for gpu, method in enumerate(METHODS):
        output = ROOT / "relaxation" / method
        if output.exists():
            raise FileExistsError(output)
        log_path = logs / f"relaxation_{method}.log"
        stream = log_path.open("x", encoding="utf-8")
        command = [
            sys.executable,
            "-m", "research.corrector_distillation.relax_many_individual",
            "--structures-path", str(ROOT / "structures" / f"{method}_generated.extxyz"),
            "--potential-path", str(POTENTIAL),
            "--output-dir", str(output),
            "--device", "cuda",
            "--fmax", "0.05",
            "--max-natoms-per-batch", "512",
            "--cuda-3x3-det-workaround",
        ]
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            TMPDIR=str(PROJECT_ROOT / ".tmp"),
            XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
            HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
            TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"),
            PYTHONUNBUFFERED="1",
        )
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
        running.append((method, process, stream, log_path))
        print(json.dumps({"event": "relaxation_started", "method": method, "gpu": gpu, "pid": process.pid}), flush=True)
    failures = []
    for method, process, stream, log_path in running:
        return_code = process.wait()
        stream.close()
        print(json.dumps({"event": "relaxation_complete", "method": method, "return_code": return_code}), flush=True)
        if return_code:
            failures.append((method, return_code, str(log_path)))
    if failures:
        raise RuntimeError(f"MatterSim relaxation failures: {failures}")


if __name__ == "__main__":
    main()
