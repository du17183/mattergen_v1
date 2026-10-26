"""Run frozen P0 MatterSim-5M relaxations on A-reserved GPUs 5 and 6."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/mattersim_late_force_guidance_p0"
POTENTIAL = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/"
    "mattersim-v1.0.0-5M.pth"
)
JOBS = (("B0", 5), ("F0", 6))
POTENTIAL_SHA256 = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sha256(POTENTIAL) != POTENTIAL_SHA256:
        raise RuntimeError("MatterSim checkpoint SHA256 mismatch")
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = []
    for method, physical_gpu in JOBS:
        output = ROOT / "p0_relaxation" / method
        if output.exists():
            raise FileExistsError(output)
        log_path = logs / f"p0_relaxation_{method}_gpu_{physical_gpu}.log"
        stream = log_path.open("x", encoding="utf-8")
        command = [
            sys.executable, "-m", "research.corrector_distillation.relax_many_individual",
            "--structures-path", str(ROOT / "p0_structures" / f"{method}_generated.extxyz"),
            "--potential-path", str(POTENTIAL),
            "--output-dir", str(output),
            "--device", "cuda", "--fmax", "0.05",
            "--max-natoms-per-batch", "512", "--cuda-3x3-det-workaround",
        ]
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(physical_gpu),
            A_PHYSICAL_GPU=str(physical_gpu),
            TMPDIR=str(PROJECT_ROOT / ".tmp"),
            XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
            HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
            TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"),
            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
            PYTHONUNBUFFERED="1",
        )
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
        running.append((method, physical_gpu, process, stream, log_path))
        print(json.dumps({"event": "started", "method": method,
                          "physical_gpu": physical_gpu, "pid": process.pid}), flush=True)
    failures = []
    for method, physical_gpu, process, stream, log_path in running:
        code = process.wait()
        stream.close()
        print(json.dumps({"event": "complete", "method": method,
                          "physical_gpu": physical_gpu, "return_code": code}), flush=True)
        if code:
            failures.append((method, code, str(log_path)))
    if failures:
        raise RuntimeError(f"P0 MatterSim relaxation failures: {failures}")


if __name__ == "__main__":
    main()
