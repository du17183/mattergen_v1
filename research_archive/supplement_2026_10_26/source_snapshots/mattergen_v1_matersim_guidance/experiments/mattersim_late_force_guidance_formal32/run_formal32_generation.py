"""Run the immutable Formal32 seed cohort on an explicit GPU pool."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "generate_formal32_pair.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="5,6,7")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    gpus = [int(item) for item in args.gpus.split(",")]
    if not gpus or any(gpu in {0, 1, 2, 3, 4} for gpu in gpus):
        raise ValueError("Formal32 currently reserves only audited-free GPUs 5,6,7")
    seeds = [730000] if args.smoke else list(range(730000, 730032))
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    active: list[tuple[int, int, subprocess.Popen, object]] = []
    failures: list[dict] = []
    for index, seed in enumerate(seeds):
        gpu = gpus[index % len(gpus)]
        if index >= len(gpus):
            old_seed, old_gpu, process, stream = active[index - len(gpus)]
            code = process.wait()
            stream.close()
            if code:
                failures.append({"seed": old_seed, "gpu": old_gpu, "return_code": code})
        log_path = logs / f"formal32_pair_{seed}_gpu_{gpu}.log"
        mode = "a" if log_path.exists() else "x"
        stream = log_path.open(mode, encoding="utf-8")
        environment = dict(os.environ)
        inherited_pythonpath = environment.get("PYTHONPATH", "")
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu), FORMAL32_PHYSICAL_GPU=str(gpu),
            TMPDIR=str(ROOT.parents[1] / ".tmp"),
            XDG_CACHE_HOME=str(ROOT.parents[1] / ".cache"),
            HF_HOME=str(ROOT.parents[1] / ".cache/huggingface"),
            TORCH_HOME=str(ROOT.parents[1] / ".cache/torch"),
            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", PYTHONUNBUFFERED="1",
            PYTHONPATH=(
                str(ROOT.parents[1])
                + ((":" + inherited_pythonpath) if inherited_pythonpath else "")
            ),
        )
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--seed", str(seed)],
            cwd=ROOT.parents[1], env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
        active.append((seed, gpu, process, stream))
        print(json.dumps({"event": "started", "seed": seed, "gpu": gpu,
                          "pid": process.pid}), flush=True)
    for seed, gpu, process, stream in active[max(0, len(active) - len(gpus)):]:
        code = process.wait()
        stream.close()
        if code:
            failures.append({"seed": seed, "gpu": gpu, "return_code": code})
    if failures:
        (ROOT / "generation_failures.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"Formal32 generation failures retained: {failures}")


if __name__ == "__main__":
    main()
