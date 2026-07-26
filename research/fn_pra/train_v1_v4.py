"""Final compatibility wrapper for static FN-PRA V1 training.

Lightning requires ``check_val_every_n_epoch=None`` when an integer
``val_check_interval`` is intended to span multiple epochs.  The Phase-1
configuration uses a large global batch, so one epoch contains fewer than 250
optimizer steps.
"""

from __future__ import annotations

from research.fn_pra import train_v1 as base
from research.fn_pra import train_v1_v3 as compatible


_Trainer = base.pl.Trainer


def step_based_trainer(*args, **kwargs):
    kwargs["check_val_every_n_epoch"] = None
    return _Trainer(*args, **kwargs)


def main() -> None:
    base.pl.Trainer = step_based_trainer
    compatible.main()


if __name__ == "__main__":
    main()
