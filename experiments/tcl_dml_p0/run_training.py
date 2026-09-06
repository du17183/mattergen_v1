"""Run FT0, TCL, and DML independently on three H20 GPUs."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/tcl_dml_p0"
TRAINER = ROOT / "train_method.py"
METHODS = ("FT0", "TCL", "DML")


def main() -> None:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = []
    for gpu, method in enumerate(METHODS):
        output = ROOT / "checkpoints" / method
        if output.exists():
            raise FileExistsError(output)
        log_path = logs / f"training_{method}.log"
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
        command = [
            sys.executable, str(TRAINER),
            "--method", method,
            "--steps", "1000",
            "--batch-size", "16",
            "--learning-rate", "0.0001",
            "--weight-decay", "0.0001",
            "--validation-every", "100",
            "--seed", "20260908",
        ]
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
        )
        running.append((method, process, stream, log_path))
        print(json.dumps({
            "event": "training_started", "method": method,
            "gpu": gpu, "pid": process.pid,
        }), flush=True)
    failures = []
    for method, process, stream, log_path in running:
        return_code = process.wait()
        stream.close()
        print(json.dumps({
            "event": "training_complete", "method": method,
            "return_code": return_code,
        }), flush=True)
        if return_code:
            failures.append((method, return_code, str(log_path)))
    if failures:
        raise RuntimeError(f"training failures: {failures}")

    summaries = [
        json.loads((ROOT / "checkpoints" / method / "training_summary.json").read_text())
        for method in METHODS
    ]
    if len({summary["trainable_parameter_names_sha256"] for summary in summaries}) != 1:
        raise RuntimeError("trainable parameter names differ across FT0/TCL/DML")
    rows = []
    for summary in summaries:
        train = summary["final_train"]
        validation = summary["final_validation_original_fixed_loss"]
        rows.append({
            "method": summary["method"],
            "model_trainable_params": summary["model_trainable_params"],
            "scheduler_params_in_equal_scope": summary["scheduler_trainable_params_in_scope"],
            "total_trainable_params_in_scope": summary["total_trainable_params_in_scope"],
            "trainable_parameter_names_sha256": summary["trainable_parameter_names_sha256"],
            "steps": summary["steps"],
            "effective_batch_size": summary["effective_batch_size"],
            "training_seed": summary["training_seed"],
            "final_train_total": train["loss_total"],
            "final_train_base": train["loss_base"],
            "final_train_atomic": train["loss_atomic_numbers"],
            "final_train_pos": train["loss_pos"],
            "final_train_cell": train["loss_cell"],
            "final_consistency": train["consistency"],
            "final_dynamic_loss": train["loss_dynamic"],
            "validation_original_total": validation["loss_total"],
            "validation_original_atomic": validation["loss_atomic_numbers"],
            "validation_original_pos": validation["loss_pos"],
            "validation_original_cell": validation["loss_cell"],
            "all_losses_finite": summary["all_losses_finite"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "peak_cuda_memory_mb": summary["peak_cuda_memory_mb"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
        })
    with (ROOT / "training_summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
