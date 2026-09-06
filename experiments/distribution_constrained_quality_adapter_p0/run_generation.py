"""Run paired C0/M1/M2 generation waves across eight GPUs."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/distribution_constrained_quality_adapter_p0"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
SEEDS = tuple(range(78000, 78008))
METHODS = ("C0", "M1", "M2")


def command(method: str, seed: int) -> list[str]:
    result = [
        sys.executable,
        str(EXPERIMENT_ROOT / "generate_one.py"),
        "--checkpoint-root",
        str(MODEL_ROOT),
        "--output-root",
        str(EXPERIMENT_ROOT),
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
        result.extend(
            [
                "--adapter-checkpoint",
                str(EXPERIMENT_ROOT / "checkpoints" / method / "adapter.pt"),
            ]
        )
    return result


def run_wave(method: str) -> dict[str, object]:
    logs = EXPERIMENT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = []
    started = time.perf_counter()
    for gpu, seed in enumerate(SEEDS):
        run_dir = EXPERIMENT_ROOT / "generation" / method / str(seed)
        if run_dir.exists():
            raise FileExistsError(f"refusing to overwrite {run_dir}")
        log_path = logs / f"generation_{method}_{seed}.log"
        stream = log_path.open("x", encoding="utf-8")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["TMPDIR"] = str(PROJECT_ROOT / ".tmp")
        process = subprocess.Popen(
            command(method, seed),
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        running.append((gpu, seed, process, stream, log_path, time.perf_counter()))

    events = []
    for gpu, seed, process, stream, log_path, process_started in running:
        return_code = process.wait()
        stream.close()
        event = {
            "gpu": gpu,
            "seed": seed,
            "return_code": return_code,
            "wall_seconds": time.perf_counter() - process_started,
            "log_path": str(log_path),
        }
        events.append(event)
        print(json.dumps({"event": "sample_complete", "method": method, **event}), flush=True)
    failures = [event for event in events if event["return_code"] != 0]
    result = {
        "method": method,
        "wave_wall_seconds": time.perf_counter() - started,
        "events": events,
        "failures": failures,
    }
    print(json.dumps({"event": "wave_complete", **result}), flush=True)
    if failures:
        raise RuntimeError(f"{method} generation failures: {failures}")
    return result


def write_results(waves: list[dict[str, object]]) -> None:
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            path = EXPERIMENT_ROOT / "generation" / method / str(seed) / "run_summary.json"
            with path.open(encoding="utf-8") as stream:
                summary = json.load(stream)
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "formula": summary["formula"],
                    "num_atoms": summary["num_atoms"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "sampling_steps": summary["sampling_steps"],
                    "guidance_scale": summary["guidance_scale"],
                    "target_dft_mag_density": summary["target"]["dft_mag_density"],
                    "adapter_parameter_l2": summary["adapter_parameter_l2"],
                    "peak_allocated_bytes": summary["peak_allocated_bytes"],
                }
            )
    path = EXPERIMENT_ROOT / "generation_results.csv"
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "paired methods; one fresh seed per GPU; 1000-step PC; CFG=2.0; target=0.1",
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "waves": waves,
    }
    (EXPERIMENT_ROOT / "generation_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    (PROJECT_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    waves = [run_wave(method) for method in METHODS]
    write_results(waves)


if __name__ == "__main__":
    main()
