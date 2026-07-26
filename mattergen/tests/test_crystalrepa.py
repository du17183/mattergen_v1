from __future__ import annotations

import torch

from mattergen.crystalrepa.model import (
    ResidualProjection,
    symmetric_element_aware_nce,
)
from research.crystalrepa_repro.run_paired_sample import safe_composition_validity


def test_same_element_off_diagonal_is_excluded() -> None:
    student = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    teacher = student.clone()
    elements = torch.tensor([6, 6, 8])
    loss, cosine = symmetric_element_aware_nce(
        student, teacher, elements, temperature=0.1
    )
    # The two carbon rows cannot act as negatives for each other.
    # If the duplicate-carbon rows were negatives, the loss would contain log(2).
    assert loss.item() < 1e-4
    assert cosine.item() == 1.0


def test_symmetric_ea_nce_is_finite_and_differentiable() -> None:
    torch.manual_seed(7)
    student = torch.randn(9, 5, requires_grad=True)
    teacher = torch.randn(9, 5)
    elements = torch.tensor([1, 6, 8, 6, 14, 8, 1, 26, 7])
    loss, cosine = symmetric_element_aware_nce(student, teacher, elements)
    assert torch.isfinite(loss)
    assert torch.isfinite(cosine)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.norm() > 0


def test_projection_shape_and_gradient() -> None:
    projection = ResidualProjection(16, 6)
    value = torch.randn(11, 16, requires_grad=True)
    output = projection(value)
    assert output.shape == (11, 6)
    output.square().mean().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_missing_smact_element_is_composition_invalid_not_generation_failure() -> None:
    def missing_oxidation_state(_structure):
        raise TypeError("element is absent from SMACT dictionary")

    valid, error = safe_composition_validity(
        object(), checker=missing_oxidation_state
    )
    assert valid is False
    assert error == "TypeError: element is absent from SMACT dictionary"
