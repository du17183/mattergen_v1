"""Resume-safe FN-PRA trainer with singleton-tail protection."""

from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import DataLoader

from mattergen.common.data.collate import collate
from mattergen.common.data.datamodule import worker_init_fn
from research.fn_pra import train_v1 as base
from research.fn_pra import train_v1_v4 as final_compat


_Trainer = final_compat._Trainer
_trainer_fit = _Trainer.fit


def drop_last_train_dataloader(self, shuffle: bool = True) -> DataLoader:
    """Avoid the unsupported per-rank singleton batch in wrapped-normal loss."""
    return DataLoader(
        self.train_dataset,
        shuffle=shuffle,
        batch_size=self.batch_size.train,
        num_workers=self.num_workers.train,
        worker_init_fn=worker_init_fn,
        collate_fn=collate,
        drop_last=True,
    )


def argument_value(name: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def resume_fit(self, *args, **kwargs):
    run_name = argument_value("--run-name", "v1_smoke")
    max_steps = int(argument_value("--max-steps", "1000"))
    resume_path = (
        base.TRAINING_ROOT
        / run_name
        / "checkpoints"
        / f"step={min(500, max_steps):06d}.ckpt"
    )
    final_summary = base.TRAINING_ROOT / run_name / "training_summary.json"
    if max_steps >= 500 and resume_path.exists() and not final_summary.exists():
        kwargs.setdefault("ckpt_path", str(resume_path))
    return _trainer_fit(self, *args, **kwargs)


def main() -> None:
    base.CrystDataModule.train_dataloader = drop_last_train_dataloader
    _Trainer.fit = resume_fit
    final_compat.main()


if __name__ == "__main__":
    main()
