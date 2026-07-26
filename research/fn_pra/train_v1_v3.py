"""PyTorch-compatible safe V1 trainer.

This wrapper keeps the rank-0-only step-zero checkpoint handling from V2 and
accepts the legacy ``verbose`` scheduler keyword present in MatterGen's frozen
Hydra configuration.  PyTorch 2.8 removed that keyword from
``ReduceLROnPlateau``.
"""

from __future__ import annotations

import torch

from research.fn_pra import train_v1_v2 as safe


_ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau


class CompatibleReduceLROnPlateau(_ReduceLROnPlateau):
    def __init__(self, optimizer, *args, verbose=None, **kwargs):
        del verbose
        super().__init__(optimizer, *args, **kwargs)


def main() -> None:
    torch.optim.lr_scheduler.ReduceLROnPlateau = CompatibleReduceLROnPlateau
    safe.main()


if __name__ == "__main__":
    main()
