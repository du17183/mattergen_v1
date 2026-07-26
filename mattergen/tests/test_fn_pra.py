from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch_geometric.data import Batch

from mattergen.common.data.dataset_transform import filter_sparse_properties
from mattergen.common.data.transform import set_chemical_system_string, symmetrize_lattice
from mattergen.fn_pra.data import RepaCrystalDataset
from mattergen.fn_pra.model import (
    LowRankAtomAdapter,
    _gather_teacher_rows,
    element_aware_nce,
)


def test_low_rank_adapter_is_exact_identity_at_initialization() -> None:
    torch.manual_seed(7)
    adapter = LowRankAtomAdapter(hidden_dim=32, rank=8)
    hidden = torch.randn(17, 32)
    output = adapter(hidden)
    assert torch.equal(output, hidden)
    assert torch.count_nonzero(adapter.up.weight) == 0


def test_element_aware_nce_excludes_same_element_off_diagonal() -> None:
    student = torch.tensor(
        [[1.0, 0.2, 0.1], [0.7, -0.4, 0.3], [-0.2, 0.8, 0.5]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[0.8, 0.1, 0.4], [0.4, -0.7, 0.5], [0.0, 0.9, 0.2]],
        requires_grad=True,
    )
    elements = torch.tensor([6, 6, 8])
    temperature = 0.2
    actual = element_aware_nce(student, teacher, elements, temperature)
    normalized_student = F.normalize(student, dim=-1)
    normalized_teacher = F.normalize(teacher, dim=-1)
    logits = normalized_student @ normalized_teacher.T / temperature
    allowed = elements[:, None].ne(elements[None, :])
    allowed.fill_diagonal_(True)
    expected = F.cross_entropy(logits.masked_fill(~allowed, -torch.inf), torch.arange(3))
    assert torch.allclose(actual, expected, atol=0, rtol=0)
    actual.backward()
    assert torch.isfinite(student.grad).all()
    assert torch.isfinite(teacher.grad).all()
    assert student.grad.abs().sum() > 0
    assert teacher.grad.abs().sum() > 0


def test_element_aware_nce_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        element_aware_nce(torch.ones(2, 3), torch.ones(3, 3), torch.ones(2, dtype=torch.long))
    with pytest.raises(ValueError):
        element_aware_nce(torch.ones(2, 3), torch.ones(2, 3), torch.ones(3, dtype=torch.long))


def test_real_teacher_cache_mapping_and_collate() -> None:
    dataset_root = Path("/data/dxl/datasets/cache/mp_20/train")
    teacher_root = Path("/data/dxl/data/fn_pra_teacher_cache/train")
    if not dataset_root.exists() or not teacher_root.exists():
        pytest.skip("FN-PRA cache is not available")
    dataset = RepaCrystalDataset.from_cache_path(
        cache_path=str(dataset_root),
        teacher_cache_path=str(teacher_root),
        properties=["dft_mag_density"],
        transforms=[symmetrize_lattice, set_chemical_system_string],
        dataset_transforms=[filter_sparse_properties],
    )
    assert len(dataset) == 26117
    samples = [dataset[index] for index in (0, len(dataset) // 2, len(dataset) - 1)]
    for sample in samples:
        assert sample.teacher_features.shape == (int(sample.num_atoms), 64)
        assert torch.equal(sample.teacher_atomic_numbers, sample.atomic_numbers)
        assert torch.isfinite(sample.teacher_features).all()
    batch = Batch.from_data_list(samples)
    assert batch.teacher_features.shape[0] == int(batch.num_atoms.sum())
    assert torch.equal(batch.teacher_atomic_numbers, batch.atomic_numbers)


def _distributed_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        count = 2 + rank
        torch.manual_seed(100 + rank)
        teacher = torch.randn(count, 5, requires_grad=True)
        student = torch.randn(count, 5, requires_grad=True)
        elements = torch.tensor(([6, 8] if rank == 0 else [6, 14, 8]), dtype=torch.long)
        gathered, gathered_elements, offset = _gather_teacher_rows(teacher, elements)
        assert gathered.shape == (5, 5)
        assert gathered_elements.tolist() == [6, 8, 6, 14, 8]
        assert offset == (0 if rank == 0 else 2)
        loss = element_aware_nce(student, teacher, elements)
        assert torch.isfinite(loss)
        loss.backward()
        assert teacher.grad is not None and torch.isfinite(teacher.grad).all()
        assert student.grad is not None and torch.isfinite(student.grad).all()
    finally:
        dist.destroy_process_group()


def test_variable_atom_ddp_gather_preserves_rank_offsets(tmp_path: Path) -> None:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("Nested multiprocessing is not used under xdist")
    init_file = str(tmp_path / "dist_init")
    mp.spawn(_distributed_worker, args=(2, init_file), nprocs=2, join=True)
