"""Recalibrate global and field risks on held-out on-policy trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from mattergen.diffusion.sampling.residual_distillation import (
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from research.corrector_distillation.train_adapter import (
    calibrate_uncertainty,
    manifest_seeds,
    manifests,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--risk-analysis", type=Path, required=True)
    parser.add_argument("--batch-records", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint.expanduser().resolve()
    loaded = load_adapter_checkpoint(checkpoint, device)
    calibration_paths = manifests(args.calibration_root)
    calibration_seeds = manifest_seeds(calibration_paths)
    training_seeds = set(loaded.metadata.get("train_teacher_seeds", []))
    if training_seeds & set(calibration_seeds):
        raise ValueError("on-policy calibration seeds overlap Adapter training seeds")
    target_scales = loaded.metadata.get("target_residual_rms")
    if not target_scales:
        raise ValueError("checkpoint is missing target_residual_rms")
    calibration = calibrate_uncertainty(
        loaded.model.eval(),
        calibration_paths,
        batch_records=args.batch_records,
        device=device,
        target_scales=target_scales,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(loaded.metadata)
    metadata.update(
        {
            "pre_recalibration_checkpoint": str(checkpoint),
            "pre_recalibration_checkpoint_sha256": sha256(checkpoint),
            "risk_calibration_root": str(args.calibration_root.expanduser().resolve()),
            "risk_calibration_seeds": calibration_seeds,
            "risk_calibration_kind": "held-out on-policy DAgger states with exact teacher targets",
        }
    )
    output_checkpoint = save_adapter_checkpoint(
        output_dir / "residual_adapter.pt",
        model=loaded.model,
        calibration=calibration,
        metadata=metadata,
    )
    canonical = json.dumps(calibration, sort_keys=True, allow_nan=True).encode()
    risk_model_sha256 = hashlib.sha256(canonical).hexdigest()
    rows = []
    for field, model in calibration["field_risk_models"].items():
        for stage, correlation in model["correlations"].items():
            rows.append(
                {
                    "row_type": "correlation",
                    "field": field,
                    "stage": stage,
                    "coverage_target": "",
                    "threshold": "",
                    "marginal_quantile": "",
                    "joint_coverage": "",
                    "correlation": correlation,
                }
            )
    for coverage, thresholds in calibration["field_coverage_thresholds"].items():
        details = calibration["field_coverage_calibration"][coverage]
        for field, threshold in thresholds.items():
            rows.append(
                {
                    "row_type": "threshold",
                    "field": field,
                    "stage": "overall",
                    "coverage_target": coverage,
                    "threshold": threshold,
                    "marginal_quantile": details["marginal_quantile"],
                    "joint_coverage": details["joint_coverage"],
                    "correlation": "",
                }
            )
    risk_path = args.risk_analysis.expanduser().resolve()
    risk_path.parent.mkdir(parents=True, exist_ok=True)
    if risk_path.exists():
        raise FileExistsError(risk_path)
    with risk_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(output_checkpoint),
        "checkpoint_sha256": sha256(output_checkpoint),
        "risk_model_sha256": risk_model_sha256,
        "calibration_seeds": calibration_seeds,
        "risk_analysis": str(risk_path),
        "global_risk_correlation": calibration["risk_validation_correlation"],
        "field_correlations": {
            field: model["correlations"]
            for field, model in calibration["field_risk_models"].items()
        },
    }
    with (output_dir / "recalibration_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
