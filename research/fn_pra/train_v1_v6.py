"""Static FN-PRA trainer with all required runtime compatibility fixes."""

from __future__ import annotations

from torch.utils.data import DataLoader

from mattergen.common.data.collate import collate
from mattergen.common.data.datamodule import worker_init_fn
from research.fn_pra import train_v1 as base
from research.fn_pra import train_v1_v4 as final_compat


def drop_last_train_dataloader(self, shuffle: bool = True) -> DataLoader:
    """Drop the per-rank singleton tail unsupported by wrapped-normal loss."""
    return DataLoader(
        self.train_dataset,
        shuffle=shuffle,
        batch_size=self.batch_size.train,
        num_workers=self.num_workers.train,
        worker_init_fn=worker_init_fn,
        collate_fn=collate,
        drop_last=True,
    )


def main() -> None:
    base.CrystDataModule.train_dataloader = drop_last_train_dataloader
    final_compat.main()


if __name__ == "__main__":
    main()
