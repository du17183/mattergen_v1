from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from mattergen.common.data.dataset import CrystalDataset


class RepaCrystalDataset(Dataset):
    """Attach strictly mapped, atom-wise teacher representations to MP-20 rows."""

    def __init__(self, base: CrystalDataset, teacher_cache_path: str | Path) -> None:
        self.base = base
        self.teacher_cache_path = Path(teacher_cache_path)
        self.features = np.load(self.teacher_cache_path / "features.f16.npy", mmap_mode="r")
        self.teacher_atomic_numbers = np.load(
            self.teacher_cache_path / "atomic_numbers.npy", mmap_mode="r"
        )
        self.offsets = np.load(self.teacher_cache_path / "offsets.npy")
        self.num_atoms = np.load(self.teacher_cache_path / "num_atoms.npy")
        self.structure_ids = np.load(self.teacher_cache_path / "structure_ids.npy")
        self.dataset_indices = np.load(self.teacher_cache_path / "dataset_indices.npy")
        expected_indices = np.arange(len(base), dtype=self.dataset_indices.dtype)
        if len(base) != len(self.structure_ids):
            raise ValueError(
                f"Dataset/cache structure count mismatch: {len(base)} != "
                f"{len(self.structure_ids)}"
            )
        if not np.array_equal(expected_indices, self.dataset_indices):
            raise ValueError("Teacher cache does not cover the full dataset in canonical order")
        if not np.array_equal(
            np.asarray(base.structure_id, dtype=str), self.structure_ids.astype(str)
        ):
            raise ValueError("Dataset/cache structure_id order mismatch")
        if not np.array_equal(np.asarray(base.num_atoms, dtype=np.int32), self.num_atoms):
            raise ValueError("Dataset/cache num_atoms mismatch")
        if int(self.offsets[-1]) != len(self.features):
            raise ValueError("Teacher cache offsets do not cover all feature rows")
        self.properties = base.properties
        self.transforms = base.transforms

    @classmethod
    def from_cache_path(
        cls,
        cache_path: str,
        teacher_cache_path: str,
        transforms: list[Any] | None = None,
        properties: list[str] | None = None,
        dataset_transforms: list[Any] | None = None,
    ) -> "RepaCrystalDataset":
        base = CrystalDataset.from_cache_path(
            cache_path=cache_path,
            transforms=transforms,
            properties=properties,
            dataset_transforms=dataset_transforms,
        )
        return cls(base=base, teacher_cache_path=teacher_cache_path)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        graph = self.base[index]
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        cached_atomic_numbers = np.asarray(
            self.teacher_atomic_numbers[start:stop], dtype=np.int64
        )
        graph_atomic_numbers = (
            graph["atomic_numbers"].detach().cpu().numpy().astype(np.int64)
        )
        if not np.array_equal(cached_atomic_numbers, graph_atomic_numbers):
            raise RuntimeError(f"Teacher cache atom order mismatch at local index {index}")
        teacher_features = torch.from_numpy(
            np.asarray(self.features[start:stop], dtype=np.float16).copy()
        )
        teacher_atomic_numbers = torch.from_numpy(cached_atomic_numbers.copy())
        return graph.replace(
            teacher_features=teacher_features,
            teacher_atomic_numbers=teacher_atomic_numbers,
            teacher_cache_index=torch.tensor(index, dtype=torch.long),
        )
