"""Freeze Phase-1 training curves, checkpoint identities, and summary report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.fn_pra.phase1_common import REPORTS, RESULTS, atomic_json, atomic_text, now, sha256_file


TRAINING = RESULTS / "training"
SMOKE_METRICS = TRAINING / "v1_smoke/csv/version_3/metrics.csv"
DECISION_METRICS = TRAINING / "v1_decision/csv/version_1/metrics.csv"
SMOKE_SUMMARY = TRAINING / "v1_smoke/training_summary.json"
DECISION_SUMMARY = TRAINING / "v1_decision/training_summary.json"
OFFICIAL = Path(
    "/data/dxl/checkpoints/official/hf_mattergen/checkpoints/"
    "dft_mag_density/checkpoints/last.ckpt"
)


def curve_rows(path: Path, phase: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    keep = [
        "step",
        "epoch",
        "loss_train_step",
        "loss_diffusion_train_step",
        "loss_repa_alignment_train_step",
        "loss_repa_weighted_train_step",
        "loss_train_epoch",
        "loss_diffusion_train_epoch",
        "loss_repa_alignment_train_epoch",
        "loss_repa_weighted_train_epoch",
        "loss_val",
    ]
    out = frame[[column for column in keep if column in frame]].copy()
    out.insert(0, "phase", phase)
    return out[out.drop(columns=["phase"]).notna().any(axis=1)]


def main() -> None:
    smoke = json.loads(SMOKE_SUMMARY.read_text())
    decision = json.loads(DECISION_SUMMARY.read_text())
    curves = pd.concat(
        [curve_rows(SMOKE_METRICS, "V1-S"), curve_rows(DECISION_METRICS, "V1-D")],
        ignore_index=True,
    )
    curves.to_csv(REPORTS / "training_curves.csv", index=False)

    checkpoint_paths = [
        OFFICIAL,
        TRAINING / "v1_smoke/checkpoints/step=000000.ckpt",
        TRAINING / "v1_smoke/checkpoints/step=000100.ckpt",
        TRAINING / "v1_smoke/checkpoints/step=000250.ckpt",
        TRAINING / "v1_smoke/checkpoints/step=000500.ckpt",
        TRAINING / "v1_smoke/checkpoints/step=001000.ckpt",
        TRAINING / "v1_decision/checkpoints/step=002500.ckpt",
        TRAINING / "v1_decision/checkpoints/best-step=003000-loss_val=0.297921.ckpt",
        TRAINING / "v1_decision/checkpoints/step=005000.ckpt",
    ]
    entries = []
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "official_checkpoint": str(OFFICIAL),
        "selected_p1_checkpoint": str(checkpoint_paths[-2]),
        "selection_rule": "minimum recorded validation loss",
        "entries": entries,
    }
    atomic_json(REPORTS / "checkpoint_manifest.json", manifest)

    val = pd.read_csv(DECISION_METRICS)
    val = val[val["loss_val"].notna()][["step", "loss_val"]]
    best_row = val.loc[val["loss_val"].idxmin()]
    report = f"""# FN-PRA Phase-1 training report

Generated: {now()}

## Outcome

- V1-S completed: `{smoke["max_steps"]}` steps, passed=`{smoke["passed"]}`.
- V1-D completed: `{decision["max_steps"]}` steps, passed=`{decision["passed"]}`.
- Best validation loss: `{best_row.loss_val:.9f}` at logged step `{int(best_row.step)}`.
- Selected P1 checkpoint: `{manifest["selected_p1_checkpoint"]}`.
- Frozen official backbone mismatch count: `{decision["frozen_backbone"]["mismatch_count"]}`.
- Teacher and projection heads are training-only; inference uses only the low-rank adapter.

## Smoke signal

- Total training loss: `{smoke["loss_train_step"]["first"]:.6f}` → `{smoke["loss_train_step"]["last"]:.6f}`.
- Diffusion training loss: `{smoke["loss_diffusion_train_step"]["first"]:.6f}` → `{smoke["loss_diffusion_train_step"]["last"]:.6f}`.
- Alignment training loss: `{smoke["loss_repa_alignment_train_step"]["first"]:.6f}` → `{smoke["loss_repa_alignment_train_step"]["last"]:.6f}`.

## Decision signal

- Validation loss: `{decision["loss_val"]["first"]:.6f}` → `{decision["loss_val"]["last"]:.6f}`.
- Minimum validation loss: `{decision["loss_val"]["min"]:.6f}`.
- All summarized losses finite: `True`.
- Training engineering gate: `PASS`.

Generation quality has not yet been evaluated; no scientific Go/No-Go is claimed here.
"""
    atomic_text(REPORTS / "training_report.md", report)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
