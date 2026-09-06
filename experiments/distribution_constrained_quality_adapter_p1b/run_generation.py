"""Run the five-method P1b 16-seed paired generation screen."""

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
ROOT = PROJECT_ROOT / "experiments/distribution_constrained_quality_adapter_p1b"
STAGE_ROOT = ROOT / "screen16"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
GENERATOR = ROOT / "generate_one.py"
METHODS = ("C0", "M1", "M2", "M2-L1", "M2-L2")
SEEDS = tuple(range(71000, 71016))
CHECKPOINTS = {
    "M1": P0_ROOT / "checkpoints/M1/adapter.pt",
    "M2": P0_ROOT / "checkpoints/M2/adapter.pt",
    "M2-L1": ROOT / "checkpoints/M2-L1/adapter.pt",
    "M2-L2": ROOT / "checkpoints/M2-L2/adapter.pt",
}
EXPECTED_ADAPTER_HASHES = {
    "M1": "7b55836bd69eb315465c2bb91230c2a6e04498d22054d39127805dd9196bc7bc",
    "M2": "9bdbaa254e8cec9cf3710e6028d003575b6d54f3a9f4ae6cd19313e21657bfd1",
    "M2-L1": "76eecaf64ccf81c08fa34839366ee352b63704434c023f15d5cd04acf7222693",
    "M2-L2": "83421ca821f44baee339cb08ddbb8b3c77be4bb97952001e8be1e491d858dcb2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(method: str, seed: int) -> list[str]:
    result = [
        sys.executable,
        str(GENERATOR),
        "--checkpoint-root",
        str(MODEL_ROOT),
        "--output-root",
        str(STAGE_ROOT),
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
        result.extend(["--adapter-checkpoint", str(CHECKPOINTS[method])])
    return result


def preflight() -> None:
    if SEEDS != tuple(range(71000, 71016)):
        raise RuntimeError("screen must use exactly 71000-71015")
    for method, expected in EXPECTED_ADAPTER_HASHES.items():
        actual = sha256(CHECKPOINTS[method])
        if actual != expected:
            raise RuntimeError(
                f"{method} checkpoint hash mismatch: {actual} != {expected}"
            )
    if not (MODEL_ROOT / "checkpoints/last.ckpt").is_file():
        raise FileNotFoundError(MODEL_ROOT / "checkpoints/last.ckpt")


def run_wave(method: str, wave: int) -> dict[str, object]:
    assignments = [(gpu, 71000 + gpu + 8 * wave) for gpu in range(8)]
    running = []
    events = []
    wave_started = time.perf_counter()
    logs = STAGE_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for gpu, seed in assignments:
        run_dir = STAGE_ROOT / "generation" / method / str(seed)
        summary_path = run_dir / "run_summary.json"
        if summary_path.is_file():
            events.append(
                {
                    "gpu": gpu,
                    "seed": seed,
                    "return_code": 0,
                    "preexisting": True,
                }
            )
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
            XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
            HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
            TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"),
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
        print(
            json.dumps(
                {
                    "event": "sample_complete",
                    "method": method,
                    "wave": wave,
                    **event,
                }
            ),
            flush=True,
        )
    failures = [event for event in events if int(event["return_code"]) != 0]
    print(
        json.dumps(
            {
                "event": "wave_complete",
                "method": method,
                "wave": wave,
                "assignments": assignments,
                "wall_seconds": time.perf_counter() - wave_started,
                "failures": failures,
            }
        ),
        flush=True,
    )
    if failures:
        raise RuntimeError(f"generation failures: {failures}")
    return {"method": method, "wave": wave, "failures": failures}


def write_results() -> None:
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            path = (
                STAGE_ROOT
                / "generation"
                / method
                / str(seed)
                / "run_summary.json"
            )
            summary = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "stage": "screen16",
                    "method": method,
                    "seed": seed,
                    "success": summary["success"],
                    "formula": summary["formula"],
                    "num_atoms": summary["num_atoms"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "sampling_steps": summary["sampling_steps"],
                    "corrector_steps_per_time": summary[
                        "corrector_steps_per_time"
                    ],
                    "guidance_scale": summary["guidance_scale"],
                    "target_dft_mag_density": summary["target"][
                        "dft_mag_density"
                    ],
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "adapter_checkpoint_sha256": (
                        summary["adapter_checkpoint_sha256"] or ""
                    ),
                    "dft_verified": False,
                }
            )
    output = STAGE_ROOT / "generation_results.csv"
    if output.exists():
        raise FileExistsError(output)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    (PROJECT_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    preflight()
    print(
        json.dumps(
            {
                "seeds": SEEDS,
                "methods": METHODS,
                "frozen_hashes": EXPECTED_ADAPTER_HASHES,
            }
        ),
        flush=True,
    )
    for wave in range(2):
        for method in METHODS:
            run_wave(method, wave)
    write_results()


if __name__ == "__main__":
    main()
