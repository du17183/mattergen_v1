"""Run frozen C0/MLP/CFI P1 generation over 32 independent paired seeds."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/cross_field_interaction_p1"
P0_ROOT = PROJECT_ROOT / "experiments/cross_field_interaction_p0"
MLP_ROOT = PROJECT_ROOT / "experiments/global_transformer_adapter_p0"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
GENERATOR = P0_ROOT / "generate_one.py"
METHODS = ("C0", "MLP", "CFI")
SEEDS = tuple(range(79000, 79032))
CHECKPOINTS = {
    "MLP": MLP_ROOT / "checkpoints/MLP/adapter.pt",
    "CFI": P0_ROOT / "checkpoints/CFI/adapter.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight() -> dict[str, str]:
    if SEEDS != tuple(range(79000, 79032)):
        raise RuntimeError("P1 seed range changed")
    if not (MODEL_ROOT / "checkpoints/last.ckpt").is_file():
        raise FileNotFoundError(MODEL_ROOT / "checkpoints/last.ckpt")
    training = {
        row["method"]: row
        for row in csv.DictReader((P0_ROOT / "training_summary.csv").open())
    }
    hashes: dict[str, str] = {}
    for method, path in CHECKPOINTS.items():
        actual = sha256(path)
        expected = training[method]["checkpoint_sha256"]
        if actual != expected:
            raise RuntimeError(f"{method} checkpoint hash mismatch")
        hashes[method] = actual
    return hashes


def command(method: str, seed: int) -> list[str]:
    result = [
        sys.executable,
        str(GENERATOR),
        "--checkpoint-root", str(MODEL_ROOT),
        "--output-root", str(ROOT),
        "--method", method,
        "--seed", str(seed),
        "--target", "0.1",
        "--guidance-scale", "2.0",
        "--cpu-threads", "2",
    ]
    if method != "C0":
        result.extend(["--adapter-checkpoint", str(CHECKPOINTS[method])])
    return result


def run_chunk(method: str, seeds: tuple[int, ...], chunk_index: int) -> None:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = []
    started = time.perf_counter()
    for gpu, seed in enumerate(seeds):
        run_dir = ROOT / "generation" / method / str(seed)
        summary_path = run_dir / "run_summary.json"
        if summary_path.is_file():
            print(json.dumps({"event": "sample_preexisting", "method": method, "seed": seed}), flush=True)
            continue
        if run_dir.exists():
            raise RuntimeError(f"incomplete run exists at {run_dir}")
        log_path = logs / f"generation_{method}_{seed}.log"
        stream = log_path.open("x", encoding="utf-8")
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
            command(method, seed), cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
        running.append((gpu, seed, process, stream, log_path))
        print(json.dumps({
            "event": "sample_started", "method": method, "chunk": chunk_index,
            "gpu": gpu, "seed": seed, "pid": process.pid,
        }), flush=True)
    failures = []
    for gpu, seed, process, stream, log_path in running:
        return_code = process.wait()
        stream.close()
        print(json.dumps({
            "event": "sample_complete", "method": method, "chunk": chunk_index,
            "gpu": gpu, "seed": seed, "return_code": return_code,
        }), flush=True)
        if return_code:
            failures.append((seed, return_code, str(log_path)))
    print(json.dumps({
        "event": "chunk_complete", "method": method, "chunk": chunk_index,
        "seeds": seeds, "wall_seconds": time.perf_counter() - started,
    }), flush=True)
    if failures:
        raise RuntimeError(f"generation failures: {failures}")


def write_results() -> None:
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            path = ROOT / "generation" / method / str(seed) / "run_summary.json"
            summary = json.loads(path.read_text())
            rows.append({
                "method": method,
                "seed": seed,
                "success": summary["success"],
                "formula": summary["formula"],
                "num_atoms": summary["num_atoms"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "sampling_steps": summary["sampling_steps"],
                "corrector_steps_per_time": summary["corrector_steps_per_time"],
                "guidance_scale": summary["guidance_scale"],
                "target_dft_mag_density": summary["target"]["dft_mag_density"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "adapter_checkpoint_sha256": summary["adapter_checkpoint_sha256"] or "",
                "dft_verified": False,
            })
    output = ROOT / "generation_results.csv"
    if output.exists():
        raise FileExistsError(output)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    hashes = preflight()
    print(json.dumps({
        "seeds": SEEDS, "methods": METHODS, "adapter_hashes": hashes,
        "sampling": {"target": 0.1, "cfg": 2.0, "steps": 1000, "corrector": 1},
    }), flush=True)
    for method in METHODS:
        for chunk_index, offset in enumerate(range(0, len(SEEDS), 8), start=1):
            run_chunk(method, SEEDS[offset:offset + 8], chunk_index)
    write_results()


if __name__ == "__main__":
    main()
