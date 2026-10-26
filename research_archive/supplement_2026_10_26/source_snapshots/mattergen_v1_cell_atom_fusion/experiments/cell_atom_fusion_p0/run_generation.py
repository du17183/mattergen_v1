"""Run the 16-seed paired B0/C0 smoke across explicitly selected GPUs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/cell_atom_fusion_p0"
MODEL_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/"
    "hf_mattergen/checkpoints/dft_mag_density"
)
FUSION_CHECKPOINT = ROOT / "checkpoints/best_fusion.pt"
METHODS = ("B0", "C0")
SMOKE_SEEDS = tuple(range(110000, 110016))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_command(method: str, seed: int) -> list[str]:
    command = [
        sys.executable, str(ROOT / "generate_one.py"),
        "--checkpoint-root", str(MODEL_ROOT),
        "--output-root", str(ROOT), "--method", method,
        "--seed", str(seed), "--target", "0.2", "--guidance-scale", "2.0",
    ]
    if method == "C0":
        command.extend(["--fusion-checkpoint", str(FUSION_CHECKPOINT)])
    return command


def worker(args: argparse.Namespace) -> None:
    gpu = args.gpus.split(",")[args.worker_index]
    jobs = [
        (method, seed) for method in METHODS for seed in SMOKE_SEEDS
        if (SMOKE_SEEDS.index(seed) % args.num_workers) == args.worker_index
    ]
    for method, seed in jobs:
        final_dir = ROOT / "generation" / method / str(seed)
        if (final_dir / "run_summary.json").is_file():
            print(json.dumps({"event": "skip", "method": method, "seed": seed}), flush=True)
            continue
        if final_dir.exists() or final_dir.with_name(final_dir.name + ".partial").exists():
            raise RuntimeError(f"incomplete generation output retained for {method}/{seed}")
        log_path = ROOT / "logs" / f"generation_{method}_{seed}.log"
        with log_path.open("x", encoding="utf-8") as stream:
            environment = dict(os.environ)
            environment.update(
                CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
            )
            started = time.perf_counter()
            result = subprocess.run(
                generate_command(method, seed), cwd=PROJECT_ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
        print(json.dumps({
            "event": "complete", "method": method, "seed": seed,
            "gpu": gpu, "return_code": result.returncode,
            "wall_seconds": time.perf_counter() - started,
        }), flush=True)
        if result.returncode:
            raise RuntimeError(f"generation failed; inspect {log_path}")


def preflight(gpus: tuple[str, ...]) -> dict:
    if len(set(gpus)) != len(gpus) or not gpus:
        raise ValueError("GPU list must contain unique entries")
    if not FUSION_CHECKPOINT.is_file():
        raise FileNotFoundError(FUSION_CHECKPOINT)
    summary = json.loads((ROOT / "checkpoints/training_summary.json").read_text())
    if not summary["identity_check"]["passed"] or not summary["generation_allowed"]:
        raise RuntimeError("training summary does not allow generation")
    if sha256(FUSION_CHECKPOINT) != summary["checkpoint_sha256"]:
        raise RuntimeError("fusion checkpoint hash mismatch")
    return summary


def launch(args: argparse.Namespace) -> None:
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    summary = preflight(gpus)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    running = []
    for index, gpu in enumerate(gpus):
        log_path = ROOT / "logs" / f"generation_worker_{index}_gpu_{gpu}.log"
        stream = log_path.open("x", encoding="utf-8")
        environment = dict(os.environ)
        environment.update(PYTHONUNBUFFERED="1")
        command = [
            sys.executable, str(Path(__file__).resolve()), "--gpus", args.gpus,
            "--worker-index", str(index), "--num-workers", str(len(gpus)),
        ]
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
        running.append((index, gpu, process, stream, log_path))
        print(json.dumps({"event": "worker_started", "index": index,
                          "gpu": gpu, "pid": process.pid}), flush=True)
    failures = []
    for index, gpu, process, stream, log_path in running:
        code = process.wait()
        stream.close()
        print(json.dumps({"event": "worker_finished", "index": index,
                          "gpu": gpu, "return_code": code}), flush=True)
        if code:
            failures.append({"index": index, "gpu": gpu, "log": str(log_path)})
    if failures:
        raise RuntimeError(f"generation worker failures: {failures}")

    rows = []
    for method in METHODS:
        for seed in SMOKE_SEEDS:
            run_summary = json.loads(
                (ROOT / "generation" / method / str(seed) / "run_summary.json").read_text()
            )
            rows.append({
                "method": method, "seed": seed, "success": run_summary["success"],
                "formula": run_summary["formula"], "num_atoms": run_summary["num_atoms"],
                "elapsed_seconds": run_summary["elapsed_seconds"],
                "peak_allocated_bytes": run_summary["peak_allocated_bytes"],
                "mattergen_score_calls": run_summary["mattergen_score_calls"],
                "sampling_steps": run_summary["sampling_steps"],
                "corrector_steps_per_timestep": run_summary["corrector_steps_per_timestep"],
                "guidance_scale": run_summary["guidance_scale"],
                "target_dft_mag_density": run_summary["target"]["dft_mag_density"],
                "base_checkpoint_sha256": run_summary["base_checkpoint_sha256"],
                "fusion_checkpoint_sha256": run_summary["fusion_checkpoint_sha256"] or "",
                "surrogate_property_eval": True, "dft_verified": False,
            })
    output = ROOT / "generation_results.csv"
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"event": "generation_complete", "n": len(rows),
                      "training_signal": summary["validation_signal"]}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,5,7")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_index is None:
        launch(parsed)
    else:
        if parsed.num_workers is None:
            raise ValueError("worker mode requires --num-workers")
        worker(parsed)
