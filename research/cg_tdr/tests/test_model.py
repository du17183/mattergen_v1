from __future__ import annotations

import inspect

import torch

from research.cg_tdr.model import CGTDRConfig, CGTDRRefiner, _build_periodic_edges
from research.cg_tdr.sampler import CGTDRGuidedPredictorCorrector


def _inputs(node_dim: int = 12):
    torch.manual_seed(7)
    frac_pos = torch.tensor(
        [[0.02, 0.10, 0.20], [0.98, 0.10, 0.20], [0.40, 0.55, 0.60]],
        dtype=torch.float32,
    )
    cell = torch.tensor(
        [[[5.0, 0.0, 0.0], [0.2, 5.2, 0.0], [0.1, 0.3, 4.8]]],
        dtype=torch.float32,
    )
    batch = torch.zeros(3, dtype=torch.long)
    node = torch.randn(3, node_dim)
    convergence = torch.randn(1, 8)
    return node, frac_pos, cell, batch, convergence


def _active_model(node_dim: int = 12, enable_cell: bool = True) -> CGTDRRefiner:
    model = CGTDRRefiner(
        CGTDRConfig(
            node_dim=node_dim,
            hidden_dim=16,
            num_rbf=8,
            cutoff=4.0,
            enable_cell=enable_cell,
        )
    )
    with torch.no_grad():
        model.position_output.weight.fill_(0.02)
        model.position_gate[-1].bias.fill_(8.0)
        model.cell_head[-1].weight.fill_(2.0e-4)
        model.cell_gate[-1].bias.fill_(8.0)
    return model


def test_zero_initialized_refiner_is_exact_identity():
    node, pos, cell, batch, convergence = _inputs()
    model = CGTDRRefiner(CGTDRConfig(node_dim=12, hidden_dim=16, num_rbf=8))
    output = model(
        node_features=node,
        frac_pos=pos,
        cell=cell,
        batch_idx=batch,
        convergence=convergence,
    )
    assert torch.equal(output.frac_pos, pos)
    assert torch.equal(output.cell, cell)
    assert torch.count_nonzero(output.position_residual_cart) == 0
    assert torch.count_nonzero(output.strain) == 0


def test_position_head_is_rotation_equivariant():
    node, pos, cell, batch, convergence = _inputs()
    model = _active_model(enable_cell=False)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    original = model(
        node_features=node,
        frac_pos=pos,
        cell=cell,
        batch_idx=batch,
        convergence=convergence,
    )
    rotated = model(
        node_features=node,
        frac_pos=pos,
        cell=cell @ rotation.T,
        batch_idx=batch,
        convergence=convergence,
    )
    assert torch.allclose(
        rotated.position_residual_cart,
        original.position_residual_cart @ rotation.T,
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_position_head_is_translation_invariant():
    node, pos, cell, batch, convergence = _inputs()
    model = _active_model(enable_cell=False)
    shifted = torch.remainder(pos + torch.tensor([0.17, 0.29, 0.31]), 1.0)
    first = model(
        node_features=node,
        frac_pos=pos,
        cell=cell,
        batch_idx=batch,
        convergence=convergence,
    )
    second = model(
        node_features=node,
        frac_pos=shifted,
        cell=cell,
        batch_idx=batch,
        convergence=convergence,
    )
    assert torch.allclose(
        first.position_residual_cart,
        second.position_residual_cart,
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_periodic_edge_uses_minimum_image():
    _, pos, cell, batch, _ = _inputs()
    edge_index, _, distance = _build_periodic_edges(pos[:2], cell, batch[:2], cutoff=1.0)
    assert edge_index.shape == (2, 2)
    assert torch.allclose(distance, torch.full_like(distance, 0.2), atol=1.0e-6)


def test_trust_regions_and_symmetric_strain_hold():
    node, pos, cell, batch, convergence = _inputs()
    model = _active_model(enable_cell=True)
    with torch.no_grad():
        model.position_output.weight.fill_(100.0)
        model.cell_head[-1].weight.fill_(100.0)
    output = model(
        node_features=node,
        frac_pos=pos,
        cell=cell,
        batch_idx=batch,
        convergence=convergence,
    )
    assert output.position_residual_cart.norm(dim=-1).max() <= 0.080001
    assert torch.allclose(output.strain, output.strain.transpose(-1, -2))
    assert torch.linalg.matrix_norm(output.strain, ord="fro") <= 0.010001
    assert torch.linalg.det(output.cell).item() > 0


def test_zero_confidence_returns_exact_a0_even_with_nonzero_heads():
    node, pos, cell, batch, convergence = _inputs()
    model = _active_model(enable_cell=True)
    with torch.no_grad():
        model.position_gate[-1].bias.fill_(-1000.0)
        model.cell_gate[-1].bias.fill_(-1000.0)
    output = model(
        node_features=node,
        frac_pos=pos,
        cell=cell,
        batch_idx=batch,
        convergence=convergence,
    )
    assert torch.equal(output.frac_pos, pos)
    assert torch.equal(output.cell, cell)


def test_inference_sampler_has_no_teacher_or_evaluator_dependency():
    source = inspect.getsource(CGTDRGuidedPredictorCorrector)
    assert "import chgnet" not in source.lower()
    assert "import mattersim" not in source.lower()
