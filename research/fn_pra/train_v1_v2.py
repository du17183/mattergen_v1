"""Safe V1 trainer wrapper with rank-0 pre-DDP step-0 capture."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch

from research.fn_pra import train_v1 as base


def safe_verify_frozen_checkpoint(
    trained_path: Path,
    official_state: dict[str, torch.Tensor],
) -> dict:
    trained = torch.load(trained_path, map_location="cpu", weights_only=False)["state_dict"]
    mismatches = []
    for name, value in official_state.items():
        if name not in trained or not torch.equal(value.cpu(), trained[name].cpu()):
            mismatches.append(name)
    return {
        "official_keys": len(official_state),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "passed": not mismatches,
    }


def no_step_zero_callback(
    _self: base.ExactStepCheckpoint,
    _trainer: pl.Trainer,
    _pl_module: pl.LightningModule,
) -> None:
    return None


def argument_value(name: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def capture_step_zero() -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    run_name = argument_value("--run-name", "v1_smoke")
    learning_rate = float(argument_value("--learning-rate", "1e-4"))
    output = base.TRAINING_ROOT / run_name / "checkpoints" / "step=000000.ckpt"
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(20260725, workers=True)
    model, info, _base_config, lightning_config, _checkpoint = base.initialize_model(learning_rate)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "global_step": 0,
            "fn_pra_step_zero": True,
            "official_checkpoint": info.checkpoint_path,
            "lightning_module_config": lightning_config,
        },
        output,
    )
    del model


def main() -> None:
    base.verify_frozen_checkpoint = safe_verify_frozen_checkpoint
    base.ExactStepCheckpoint.on_train_start = no_step_zero_callback
    capture_step_zero()
    base.main()


if __name__ == "__main__":
    main()
