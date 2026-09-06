"""Train the two frozen-design P1b quality-contrast candidates."""

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
P0_ROOT = PROJECT_ROOT / "experiments/distribution_constrained_quality_adapter_p0"
ROOT = PROJECT_ROOT / "experiments/distribution_constrained_quality_adapter_p1b"
TRAINER = P0_ROOT / "train_adapter.py"
MODEL_ROOT = PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
DATA_ROOT = P0_ROOT / "data"
SOURCE_LABELS = P0_ROOT / "quality_labels.csv"
ORIGINAL_WEIGHTS = {"low_force": 1.25, "middle": 1.0, "high_force": 0.75}
CANDIDATES = {
    "M2-L1": {"low_force": 1.125, "middle": 1.0, "high_force": 0.875},
    "M2-L2": {"low_force": 1.075, "middle": 1.0, "high_force": 0.925},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_labels(candidate: str) -> Path:
    frame = pd.read_csv(SOURCE_LABELS)
    observed = {
        quality_bin: sorted(group["quality_weight"].unique().tolist())
        for quality_bin, group in frame.groupby("quality_bin")
    }
    expected = {
        quality_bin: [weight] for quality_bin, weight in ORIGINAL_WEIGHTS.items()
    }
    if observed != expected:
        raise RuntimeError(f"frozen M2 weight mapping changed: {observed}")
    before = frame.drop(columns=["quality_weight"]).copy()
    frame["quality_weight"] = frame["quality_bin"].map(CANDIDATES[candidate])
    if frame["quality_weight"].isna().any():
        raise RuntimeError("unrecognized quality bin")
    if not before.equals(frame.drop(columns=["quality_weight"])):
        raise RuntimeError("a non-weight label column changed")
    output = ROOT / "runtime_tmp" / f"{candidate}_quality_labels.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def command(candidate: str, labels: Path) -> list[str]:
    return [
        sys.executable,
        str(TRAINER),
        "--method",
        "M2",
        "--model-root",
        str(MODEL_ROOT),
        "--data-root",
        str(DATA_ROOT),
        "--labels",
        str(labels),
        "--output-dir",
        str(ROOT / "checkpoints" / candidate),
        "--steps",
        "200",
        "--batch-size",
        "16",
        "--learning-rate",
        "0.0003",
        "--anchor-lambda",
        "0.05",
        "--replay-fraction",
        "0.5",
        "--seed",
        "20260906",
        "--validation-every",
        "20",
    ]


def write_summary() -> None:
    rows = []
    for candidate, weights in CANDIDATES.items():
        run_root = ROOT / "checkpoints" / candidate
        summary = json.loads((run_root / "training_summary.json").read_text())
        curve = pd.read_csv(run_root / "training_curve.csv")
        validation = curve[curve["split"] == "validation"].sort_values("step")
        rows.append(
            {
                "candidate": candidate,
                "high_quality_low_force_weight": weights["low_force"],
                "middle_weight": weights["middle"],
                "low_quality_high_force_weight": weights["high_force"],
                "contrast_fraction_vs_original": (
                    (weights["low_force"] - weights["high_force"])
                    / (
                        ORIGINAL_WEIGHTS["low_force"]
                        - ORIGINAL_WEIGHTS["high_force"]
                    )
                ),
                "steps": summary["steps"],
                "batch_size": summary["batch_size"],
                "learning_rate": summary["learning_rate"],
                "replay_fraction": summary["replay_fraction"],
                "anchor_lambda": summary["anchor_lambda"],
                "trainable_params": summary["trainable_params"],
                "all_losses_finite": summary["all_losses_finite"],
                "zero_init_passed": summary["zero_init_check"]["passed"],
                "frozen_parameters_unchanged": summary[
                    "frozen_parameters_unchanged"
                ],
                "adapter_gradient_parameter_count": len(
                    summary["first_step_gradient_check"][
                        "adapter_parameters_with_gradient"
                    ]
                ),
                "frozen_gradient_parameter_count": len(
                    summary["first_step_gradient_check"][
                        "frozen_parameters_with_gradient"
                    ]
                ),
                "first_20_train_loss_mean": summary[
                    "first_20_train_loss_mean"
                ],
                "last_20_train_loss_mean": summary[
                    "last_20_train_loss_mean"
                ],
                "validation_loss_step_1": float(
                    validation.iloc[0]["loss_total"]
                ),
                "validation_loss_step_200": float(
                    validation.iloc[-1]["loss_total"]
                ),
                "best_recorded_validation_loss": float(
                    validation["loss_total"].min()
                ),
                "elapsed_seconds": summary["elapsed_seconds"],
                "peak_cuda_memory_mb": summary["peak_cuda_memory_mb"],
                "checkpoint_sha256": sha256(run_root / "adapter.pt"),
            }
        )
    with (ROOT / "training_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu, candidate in enumerate(CANDIDATES):
        output = ROOT / "checkpoints" / candidate
        if output.exists():
            raise FileExistsError(output)
        labels = prepare_labels(candidate)
        log_path = logs / f"training_{candidate}.log"
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
            command(candidate, labels),
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((candidate, process, stream, log_path))
        print(
            json.dumps(
                {
                    "event": "training_started",
                    "candidate": candidate,
                    "gpu": gpu,
                    "pid": process.pid,
                }
            ),
            flush=True,
        )
    failures = []
    for candidate, process, stream, log_path in processes:
        return_code = process.wait()
        stream.close()
        print(
            json.dumps(
                {
                    "event": "training_complete",
                    "candidate": candidate,
                    "return_code": return_code,
                }
            ),
            flush=True,
        )
        if return_code:
            failures.append((candidate, return_code, str(log_path)))
    if failures:
        raise RuntimeError(f"candidate training failures: {failures}")
    write_summary()
    print((ROOT / "training_summary.csv").read_text(), flush=True)


if __name__ == "__main__":
    main()
