"""Train the sole M2-DB candidate with the frozen original-M2 configuration."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/distribution_balanced_quality_adapter"
P0_ROOT = PROJECT_ROOT / "experiments/distribution_constrained_quality_adapter_p0"
TRAINER = P0_ROOT / "train_adapter.py"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
DATA_ROOT = P0_ROOT / "data"
SOURCE_LABELS = P0_ROOT / "quality_labels.csv"
LABELS = ROOT / "runtime_tmp/m2_db_quality_labels.csv"
OUTPUT = ROOT / "checkpoints/M2-DB"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight() -> dict:
    mechanism = json.loads((ROOT / "runtime_tmp/mechanism_summary.json").read_text())
    if not mechanism["may_train_m2_db"]:
        raise RuntimeError("mechanism analysis did not authorize M2-DB training")
    source = pd.read_csv(SOURCE_LABELS)
    balanced = pd.read_csv(LABELS)
    shared = [column for column in source if column not in {"quality_weight", "quality_bin"}]
    pd.testing.assert_frame_equal(
        source[shared],
        balanced[shared],
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    if sorted(balanced["quality_weight"].unique()) != [0.75, 1.0, 1.25]:
        raise RuntimeError("unexpected M2-DB weights")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    return mechanism


def main() -> None:
    mechanism = preflight()
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(TRAINER),
        "--method", "M2",
        "--model-root", str(MODEL_ROOT),
        "--data-root", str(DATA_ROOT),
        "--labels", str(LABELS),
        "--output-dir", str(OUTPUT),
        "--steps", "200",
        "--batch-size", "16",
        "--learning-rate", "0.0003",
        "--anchor-lambda", "0.05",
        "--replay-fraction", "0.5",
        "--seed", "20260906",
        "--validation-every", "20",
    ]
    environment = dict(os.environ)
    environment.update(
        TMPDIR=str(PROJECT_ROOT / ".tmp"),
        XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
        HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
        TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"),
        PYTHONUNBUFFERED="1",
    )
    log_path = ROOT / "logs/training_M2-DB.log"
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"M2-DB training failed; inspect {log_path}")

    summary = json.loads((OUTPUT / "training_summary.json").read_text())
    curve = pd.read_csv(OUTPUT / "training_curve.csv")
    validation = curve[curve["split"] == "validation"].sort_values("step")
    row = {
        "method": "M2-DB",
        "selected_k": mechanism["selected_k"],
        "high_quality_low_force_weight": 1.25,
        "middle_weight": 1.0,
        "low_quality_high_force_weight": 0.75,
        "steps": summary["steps"],
        "batch_size": summary["batch_size"],
        "learning_rate": summary["learning_rate"],
        "replay_fraction": summary["replay_fraction"],
        "anchor_lambda": summary["anchor_lambda"],
        "trainable_params": summary["trainable_params"],
        "all_losses_finite": summary["all_losses_finite"],
        "zero_init_passed": summary["zero_init_check"]["passed"],
        "frozen_parameters_unchanged": summary["frozen_parameters_unchanged"],
        "adapter_gradient_parameter_count": len(
            summary["first_step_gradient_check"]["adapter_parameters_with_gradient"]
        ),
        "frozen_gradient_parameter_count": len(
            summary["first_step_gradient_check"]["frozen_parameters_with_gradient"]
        ),
        "first_20_train_loss_mean": summary["first_20_train_loss_mean"],
        "last_20_train_loss_mean": summary["last_20_train_loss_mean"],
        "validation_loss_step_1": float(validation.iloc[0]["loss_total"]),
        "validation_loss_step_200": float(validation.iloc[-1]["loss_total"]),
        "best_recorded_validation_loss": float(validation["loss_total"].min()),
        "elapsed_seconds": summary["elapsed_seconds"],
        "peak_cuda_memory_mb": summary["peak_cuda_memory_mb"],
        "checkpoint_sha256": sha256(OUTPUT / "adapter.pt"),
    }
    with (ROOT / "training_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
