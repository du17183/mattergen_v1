"""Run frozen P1 C0/M1/M2 generation on 32 paired seeds across eight GPUs."""
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
P0_ROOT = PROJECT_ROOT / "experiments/distribution_constrained_quality_adapter_p0"
P1_ROOT = PROJECT_ROOT / "experiments/distribution_constrained_quality_adapter_p1"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
GENERATOR = P0_ROOT / "generate_one.py"
METHODS = ("C0", "M1", "M2")
SEEDS = tuple(range(70000, 70032))
EXPECTED_ADAPTER_HASHES = {
    "M1": "7b55836bd69eb315465c2bb91230c2a6e04498d22054d39127805dd9196bc7bc",
    "M2": "9bdbaa254e8cec9cf3710e6028d003575b6d54f3a9f4ae6cd19313e21657bfd1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_path(method: str) -> Path:
    return P0_ROOT / "checkpoints" / method / "adapter.pt"


def command(method: str, seed: int) -> list[str]:
    result = [
        sys.executable,
        str(GENERATOR),
        "--checkpoint-root",
        str(MODEL_ROOT),
        "--output-root",
        str(P1_ROOT),
        "--method",
        method,
        "--seed",
        str(seed),
        "--target",
        "0.1",
        "--guidance-scale",
        "2.0",
    ]
    if method != "C0":
        result.extend(["--adapter-checkpoint", str(adapter_path(method))])
    return result


def preflight() -> None:
    if len(SEEDS) != 32 or len(set(SEEDS)) != 32:
        raise RuntimeError("P1 requires exactly 32 unique seeds")
    for method, expected in EXPECTED_ADAPTER_HASHES.items():
        actual = sha256(adapter_path(method))
        if actual != expected:
            raise RuntimeError(f"frozen {method} checkpoint hash changed: {actual}")
    if not (MODEL_ROOT / "checkpoints/last.ckpt").is_file():
        raise FileNotFoundError(MODEL_ROOT / "checkpoints/last.ckpt")


def run_wave(method: str, wave: int) -> dict[str, object]:
    assignments = [(gpu, 70000 + gpu + 8 * wave) for gpu in range(8)]
    running = []
    events = []
    wave_started = time.perf_counter()
    logs = P1_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for gpu, seed in assignments:
        run_dir = P1_ROOT / "generation" / method / str(seed)
        summary_path = run_dir / "run_summary.json"
        if summary_path.is_file():
            events.append({"gpu": gpu, "seed": seed, "return_code": 0, "preexisting": True})
            continue
        if run_dir.exists():
            raise RuntimeError(f"incomplete run exists at {run_dir}")
        log_path = logs / f"generation_{method}_{seed}.log"
        if log_path.exists():
            raise FileExistsError(log_path)
        stream = log_path.open("x", encoding="utf-8")
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            TMPDIR=str(PROJECT_ROOT / ".tmp"),
            PYTHONUNBUFFERED="1",
        )
        started = time.perf_counter()
        process = subprocess.Popen(
            command(method, seed),
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        running.append((gpu, seed, process, stream, log_path, started))
    for gpu, seed, process, stream, log_path, started in running:
        return_code = process.wait()
        stream.close()
        event = {
            "gpu": gpu,
            "seed": seed,
            "return_code": return_code,
            "preexisting": False,
            "wall_seconds": time.perf_counter() - started,
            "log_path": str(log_path),
        }
        events.append(event)
        print(json.dumps({"event": "sample_complete", "method": method, "wave": wave, **event}), flush=True)
    failures = [event for event in events if int(event["return_code"]) != 0]
    result = {
        "method": method,
        "wave": wave,
        "assignments": assignments,
        "wall_seconds": time.perf_counter() - wave_started,
        "failures": failures,
    }
    print(json.dumps({"event": "wave_complete", **result}), flush=True)
    if failures:
        raise RuntimeError(f"generation failures: {failures}")
    return result


def write_results() -> None:
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            path = P1_ROOT / "generation" / method / str(seed) / "run_summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
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
                }
            )
    output = P1_ROOT / "generation_results.csv"
    if output.exists():
        raise FileExistsError(output)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    (PROJECT_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    P1_ROOT.mkdir(parents=True, exist_ok=True)
    preflight()
    print(json.dumps({"seeds": SEEDS, "methods": METHODS, "frozen_hashes": EXPECTED_ADAPTER_HASHES}), flush=True)
    for wave in range(4):
        for method in METHODS:
            run_wave(method, wave)
    write_results()


if __name__ == "__main__":
    main()
