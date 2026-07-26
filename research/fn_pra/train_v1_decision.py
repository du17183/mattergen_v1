"""Continue the passed V1-S checkpoint to the 5000-step V1-D decision run."""

from __future__ import annotations

from pathlib import Path

import torch

from research.fn_pra import train_v1 as base
from research.fn_pra import train_v1_v4 as final_compat
from research.fn_pra.train_v1_v6 import drop_last_train_dataloader


_Trainer = final_compat._Trainer
_trainer_fit = _Trainer.fit
_torch_load = torch.load
_checkpoint_init = base.ExactStepCheckpoint.__init__
SMOKE_CHECKPOINT = (
    base.TRAINING_ROOT / "v1_smoke" / "checkpoints" / "step=001000.ckpt"
)


def trusted_local_load(*args, **kwargs):
    """PyTorch 2.6+ compatibility for project-generated trusted checkpoints."""
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


def decision_fit(self, *args, **kwargs):
    if not SMOKE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Passed V1-S checkpoint missing: {SMOKE_CHECKPOINT}")
    kwargs.setdefault("ckpt_path", str(SMOKE_CHECKPOINT))
    return _trainer_fit(self, *args, **kwargs)


def decision_checkpoint_init(self, output_dir: Path, steps: list[int]) -> None:
    _checkpoint_init(self, output_dir, sorted(set(steps) | {1000, 2500, 5000}))


def main() -> None:
    torch.load = trusted_local_load
    base.CrystDataModule.train_dataloader = drop_last_train_dataloader
    base.ExactStepCheckpoint.__init__ = decision_checkpoint_init
    _Trainer.fit = decision_fit
    final_compat.main()


if __name__ == "__main__":
    main()
