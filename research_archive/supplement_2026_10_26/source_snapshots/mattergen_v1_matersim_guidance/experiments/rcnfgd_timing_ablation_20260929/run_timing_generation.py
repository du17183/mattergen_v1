"""One seed process per free GPU; keep every failed seed and every log."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "timing_formal32"
ALM_PYTHON = Path("/mnt/datasets-livsyn/dxl/alm/.venv/bin/python")
LOCK = threading.Lock()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU indices")
    parser.add_argument("--attempt-id", default="attempt1")
    args = parser.parse_args()
    if not args.attempt_id.replace("_", "").isalnum():
        raise ValueError("attempt-id must be alphanumeric or underscore")
    gpus = [int(item) for item in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu not in range(2, 8) for gpu in gpus):
        raise ValueError("Use distinct audited-free GPUs in {2,3,4,5,6,7}")
    if Path(sys.executable).resolve() != ALM_PYTHON.resolve():
        raise RuntimeError(f"ALM interpreter required: {sys.executable}")
    protocol = json.loads((ROOT / "timing_protocol.json").read_text())
    seeds = list(range(protocol["seed_start"], protocol["seed_stop_exclusive"]))
    if len(seeds) != protocol["paired_seed_count"] or protocol["methods"] != ["C0", "Early", "Middle", "Late"]:
        raise RuntimeError("preregistered cohort/methods changed")
    OUT.mkdir(exist_ok=True)
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    failures: list[dict] = []
    completed: list[int] = []
    started_utc = time.time()

    def run_shard(gpu: int, shard: list[int]) -> None:
        for seed in shard:
            if all((OUT / "generation" / method / str(seed) / "run_summary.json").is_file()
                   for method in protocol["methods"]):
                with LOCK:
                    completed.append(seed)
                continue
            log = logs / f"seed_{seed}_gpu_{gpu}_{args.attempt_id}.log"
            if log.exists():
                raise FileExistsError(f"Existing incomplete seed log retained: {log}")
            env = dict(os.environ)
            env.pop("PYTHONHASHSEED", None)
            env.update(
                CUDA_VISIBLE_DEVICES=str(gpu), FORMAL32_PHYSICAL_GPU=str(gpu),
                PYTHONPATH=str(PROJECT), TMPDIR=str(PROJECT / ".tmp"),
                XDG_CACHE_HOME=str(PROJECT / ".cache"),
                HF_HOME=str(PROJECT / ".cache/huggingface"),
                TORCH_HOME=str(PROJECT / ".cache/torch"),
                TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
            )
            command = [sys.executable, "-m", "experiments.rcnfgd_timing_ablation_20260929.generate_timing_seed",
                       "--seed", str(seed)]
            print(json.dumps({"event": "seed_start", "seed": seed, "gpu": gpu, "log": str(log)}), flush=True)
            with log.open("x", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            with LOCK:
                if result.returncode:
                    failures.append({"seed": seed, "gpu": gpu, "return_code": result.returncode, "log": str(log)})
                else:
                    completed.append(seed)
                atomic_json(OUT / f"generation_status_{args.attempt_id}.json", {
                    "expected_paired_seeds": seeds,
                    "completed_paired_seeds": sorted(completed),
                    "failures": sorted(failures, key=lambda item: item["seed"]),
                    "gpus": gpus, "started_epoch": started_utc, "updated_epoch": time.time(),
                    "status": "FAIL" if failures else "RUNNING",
                })
            print(json.dumps({"event": "seed_complete", "seed": seed, "gpu": gpu,
                              "return_code": result.returncode}), flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_shard, gpu, seeds[index::len(gpus)])
                   for index, gpu in enumerate(gpus)]
        for future in futures:
            future.result()
    complete = sorted(completed) == seeds and not failures
    atomic_json(OUT / f"generation_status_{args.attempt_id}.json", {
        "expected_paired_seeds": seeds, "completed_paired_seeds": sorted(completed),
        "failures": sorted(failures, key=lambda item: item["seed"]),
        "gpus": gpus, "started_epoch": started_utc, "updated_epoch": time.time(),
        "status": "COMPLETE" if complete else "FAIL",
    })
    if not complete:
        raise RuntimeError(f"Timing generation incomplete: {len(completed)}/{len(seeds)} paired seeds, {len(failures)} failures")
    print(json.dumps({"event": "all_generation_complete", "seeds": len(seeds), "arms": 4}), flush=True)


if __name__ == "__main__":
    main()
