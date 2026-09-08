"""Generate the two frozen arms on 32 prospective paired seeds over eight H20s."""
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
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_prospective32"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
GENERATOR = ROOT / "generate_one.py"
METHODS = ("C0", "GBSA-TCL")
SEEDS = tuple(range(87000, 87032))
GPU_IDS = tuple(range(8))
GBSA = PROJECT_ROOT / "experiments/gbsa_tcl_p0/checkpoints/GBSA-TCL/model.pt"
EXPECTED = {
    "C0": "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e",
    "GBSA-TCL": "1edc1c0c7848e9edaff9d73b135642fcfaa36e23591e4e2f336c39b3df3dcf0f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight() -> dict[str, str]:
    if SEEDS != tuple(range(87000, 87032)) or len(SEEDS) != 32:
        raise RuntimeError("prospective seed range changed")
    forbidden = (
        range(61000, 70000), range(81000, 81032), range(82000, 82064),
        range(83000, 83256), range(84000, 84008), range(85000, 85032),
    )
    if any(seed in blocked for seed in SEEDS for blocked in forbidden):
        raise RuntimeError("prospective seeds overlap a frozen range")
    actual = {
        "C0": sha256(MODEL_ROOT / "checkpoints/last.ckpt"),
        "GBSA-TCL": sha256(GBSA),
    }
    if actual != EXPECTED:
        raise RuntimeError(f"checkpoint hash mismatch: {actual}")
    return actual


def command(method: str, seed: int) -> list[str]:
    result = [
        sys.executable, str(GENERATOR), "--checkpoint-root", str(MODEL_ROOT),
        "--output-root", str(ROOT), "--method", method, "--seed", str(seed),
        "--target", "0.1", "--guidance-scale", "2.0", "--cpu-threads", "2",
    ]
    if method == "GBSA-TCL":
        result.extend(["--finetuned-checkpoint", str(GBSA)])
    return result


def run_chunk(method: str, seeds: tuple[int, ...], block: int) -> None:
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
            raise RuntimeError(f"incomplete run retained at {run_dir}")
        log_path = logs / f"generation_{method}_{seed}.log"
        stream = log_path.open("x", encoding="utf-8")
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=str(PROJECT_ROOT / ".tmp"),
            XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
            HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
            TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"),
            CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1",
        )
        process = subprocess.Popen(
            command(method, seed), cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
        running.append((gpu, seed, process, stream, log_path))
        print(json.dumps({"event": "sample_started", "method": method, "block": block,
                          "gpu": gpu, "seed": seed, "pid": process.pid}), flush=True)
    failures = []
    for gpu, seed, process, stream, log_path in running:
        return_code = process.wait()
        stream.close()
        print(json.dumps({"event": "sample_complete", "method": method, "block": block,
                          "gpu": gpu, "seed": seed, "return_code": return_code}), flush=True)
        if return_code:
            failures.append({"seed": seed, "return_code": return_code, "log": str(log_path)})
    print(json.dumps({"event": "block_complete", "method": method, "block": block,
                      "wall_seconds": time.perf_counter() - started}), flush=True)
    if failures:
        raise RuntimeError(f"generation failures retained: {failures}")


def write_results() -> None:
    rows = []
    for seed in SEEDS:
        for method in METHODS:
            summary = json.loads((ROOT / "generation" / method / str(seed) / "run_summary.json").read_text())
            rows.append({
                "method": method, "seed": seed, "success": summary["success"],
                "technical_rerun": False, "formula": summary["formula"],
                "num_atoms": summary["num_atoms"], "elapsed_seconds": summary["elapsed_seconds"],
                "sampling_steps": summary["sampling_steps"],
                "corrector_steps_per_time": summary["corrector_steps_per_time"],
                "guidance_scale": summary["guidance_scale"],
                "target_dft_mag_density": summary["target"]["dft_mag_density"],
                "base_checkpoint_sha256": summary["base_checkpoint_sha256"],
                "finetuned_checkpoint_sha256": summary["finetuned_checkpoint_sha256"] or "",
                "prospective_training_steps": 0, "dft_verified": False,
            })
    with (ROOT / "generation_results.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    hashes = preflight()
    print(json.dumps({
        "methods": METHODS, "seed_range": [SEEDS[0], SEEDS[-1]], "seed_count": len(SEEDS),
        "historical_overlap": False, "training_steps": 0, "checkpoint_hashes": hashes,
        "sampling": {"target": 0.1, "cfg": 2.0, "steps": 1000, "corrector": 1},
    }), flush=True)
    for start in range(0, len(SEEDS), len(GPU_IDS)):
        block = start // len(GPU_IDS) + 1
        seeds = SEEDS[start:start + len(GPU_IDS)]
        for method in METHODS:
            run_chunk(method, seeds, block)
    write_results()


if __name__ == "__main__":
    main()
