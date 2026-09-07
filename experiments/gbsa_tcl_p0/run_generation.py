"""Generate C0, frozen TCL, and GBSA-TCL on eight new paired seeds."""
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
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_p0"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
GENERATOR = ROOT / "generate_one.py"
METHODS = ("C0", "TCL", "GBSA-TCL")
SEEDS = tuple(range(84000, 84008))
GPU_IDS = tuple(range(8))
CHECKPOINTS = {
    "TCL": PROJECT_ROOT / "experiments/tcl_dml_p0/checkpoints/TCL/model.pt",
    "GBSA-TCL": ROOT / "checkpoints/GBSA-TCL/model.pt",
}
EXPECTED_HASHES = {
    "TCL": "8dd9879f11edea7f25f587ae72f928a6f3ab2e5c81fc9400c13642364ad90443",
    "GBSA-TCL": "1edc1c0c7848e9edaff9d73b135642fcfaa36e23591e4e2f336c39b3df3dcf0f",
}
EXPECTED_BASE_HASH = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight() -> dict[str, str]:
    if SEEDS != tuple(range(84000, 84008)) or len(SEEDS) != 8:
        raise RuntimeError("P0 seed range changed")
    forbidden = (range(81000, 81032), range(82000, 82064), range(83000, 83256))
    if any(seed in blocked for seed in SEEDS for blocked in forbidden):
        raise RuntimeError("P0 seeds overlap a frozen range")
    base_hash = sha256(MODEL_ROOT / "checkpoints/last.ckpt")
    if base_hash != EXPECTED_BASE_HASH:
        raise RuntimeError(f"C0 checkpoint changed: {base_hash}")
    actual = {"C0": base_hash}
    for method, path in CHECKPOINTS.items():
        value = sha256(path)
        if value != EXPECTED_HASHES[method]:
            raise RuntimeError(f"{method} checkpoint changed: {value}")
        actual[method] = value
    return actual


def command(method: str, seed: int) -> list[str]:
    result = [
        sys.executable, str(GENERATOR),
        "--checkpoint-root", str(MODEL_ROOT),
        "--output-root", str(ROOT),
        "--method", method,
        "--seed", str(seed),
        "--target", "0.1",
        "--guidance-scale", "2.0",
        "--cpu-threads", "2",
    ]
    if method != "C0":
        result.extend(["--finetuned-checkpoint", str(CHECKPOINTS[method])])
    return result


def run_chunk(method: str, seeds: tuple[int, ...], chunk_index: int) -> None:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = []
    started = time.perf_counter()
    for gpu, seed in zip(GPU_IDS, seeds):
        run_dir = ROOT / "generation" / method / str(seed)
        if (run_dir / "run_summary.json").is_file():
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
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
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
        "wall_seconds": time.perf_counter() - started,
    }), flush=True)
    if failures:
        raise RuntimeError(f"generation failures: {failures}")


def write_results() -> None:
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            summary = json.loads(
                (ROOT / "generation" / method / str(seed) / "run_summary.json").read_text()
            )
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
                "base_checkpoint_sha256": summary["base_checkpoint_sha256"],
                "finetuned_checkpoint_sha256": summary["finetuned_checkpoint_sha256"] or "",
                "dft_verified": False,
            })
    with (ROOT / "generation_results.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print(json.dumps({
        "methods": METHODS,
        "seeds": SEEDS,
        "historical_overlap": False,
        "checkpoint_hashes": preflight(),
        "sampling": {"target": 0.1, "cfg": 2.0, "steps": 1000, "corrector": 1},
        "gpu_ids": GPU_IDS,
    }), flush=True)
    for method in METHODS:
        for start in range(0, len(SEEDS), len(GPU_IDS)):
            run_chunk(method, SEEDS[start:start + len(GPU_IDS)], start // len(GPU_IDS) + 1)
    write_results()


if __name__ == "__main__":
    main()
