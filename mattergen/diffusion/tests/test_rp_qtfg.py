from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from research.rp_qtfg.analyze_experiment import _comparison
from research.rp_qtfg.physics_guidance import (
    RPQTFGConfig,
    _objective,
    _safe_structure,
    clean_estimate,
    clip_score_correction,
    cosine_similarity,
    periodic_delta,
    score_correction_from_clean_delta,
)


class _LinearSDE:
    def __init__(self, alpha: float, sigma: float, wrapped: bool = False):
        self.alpha = alpha
        self.sigma = sigma
        self.wrapped = wrapped

    def mean_coeff_and_std(self, x, t, batch_idx, batch):
        return torch.full_like(x, self.alpha), torch.full_like(x, self.sigma)

    def wrap(self, x):
        if not self.wrapped:
            return x
        return torch.remainder(x, 1.0)


@pytest.mark.parametrize("wrapped", [False, True])
def test_clean_estimate_recovers_x0(wrapped):
    clean = torch.tensor([[0.2, 0.4, 0.8]])
    alpha = 0.7
    sigma = 0.3
    noise = torch.tensor([[0.1, -0.2, 0.3]])
    noisy = alpha * clean + sigma * noise
    score = -noise / sigma
    recovered, mean, std = clean_estimate(
        noisy=noisy,
        score=score,
        corruption=_LinearSDE(alpha, sigma, wrapped=wrapped),
        t=torch.tensor([0.5]),
        batch_idx=None,
        batch=None,
    )
    assert torch.allclose(recovered, clean)
    assert torch.all(mean == alpha)
    assert torch.all(std == sigma)


def test_clean_estimate_wraps_fractional_coordinates():
    recovered, _, _ = clean_estimate(
        noisy=torch.tensor([[1.2, -0.1, 0.4]]),
        score=torch.zeros(1, 3),
        corruption=_LinearSDE(1.0, 0.2, wrapped=True),
        t=torch.tensor([0.1]),
        batch_idx=None,
        batch=None,
    )
    assert torch.allclose(recovered, torch.tensor([[0.2, 0.9, 0.4]]))


def test_periodic_delta_uses_shortest_image():
    delta = torch.tensor([[0.9, -0.8, 0.4]])
    assert torch.allclose(
        periodic_delta(delta),
        torch.tensor([[-0.1, 0.2, 0.4]]),
    )


def test_score_correction_is_inverse_clean_delta():
    delta = torch.tensor([[0.01, -0.02, 0.03]])
    mean = torch.full_like(delta, 0.8)
    std = torch.full_like(delta, 0.4)
    correction = score_correction_from_clean_delta(
        clean_delta=delta,
        mean_coeff=mean,
        std=std,
        eps=1e-8,
    )
    assert torch.allclose(std.square() * correction / mean, delta)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([0.0, 0.0], [1.0, 0.0], 1.0),
    ],
)
def test_residual_cosine(left, right, expected):
    value = cosine_similarity(torch.tensor(left), torch.tensor(right))
    assert value == pytest.approx(expected, abs=1e-7)


def test_score_correction_is_clipped_fieldwise():
    correction = torch.ones(10)
    guided = torch.ones(10)
    residual = torch.full((10,), 0.5)
    clipped, was_clipped = clip_score_correction(
        correction,
        guided,
        residual,
        ratio_max=0.25,
        eps=1e-8,
    )
    assert was_clipped
    assert torch.sqrt(torch.mean(clipped.square())) == pytest.approx(0.25)


def test_small_score_correction_is_unchanged():
    correction = torch.full((10,), 0.01)
    guided = torch.ones(10)
    residual = torch.ones(10)
    output, was_clipped = clip_score_correction(
        correction,
        guided,
        residual,
        ratio_max=0.25,
        eps=1e-8,
    )
    assert not was_clipped
    assert torch.equal(output, correction)


@pytest.mark.parametrize("fields", ["position", "position_cell"])
def test_valid_config(fields):
    config = RPQTFGConfig(enabled=True, guidance_fields=fields)
    assert config.uses_cell == (fields == "position_cell")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"guidance_fields": "atomic"},
        {"start_progress": -0.1},
        {"start_progress": 1.1},
        {"backtrack_max": 0},
        {"backtrack_max": 4},
        {"conflict_threshold": -1.1},
        {"score_ratio_max": 0.0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        RPQTFGConfig(**kwargs)


def test_structure_safety_accepts_valid_crystal():
    config = RPQTFGConfig()
    structure = _safe_structure(
        np.eye(3) * 4.0,
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        np.array([14, 14]),
        config,
    )
    assert structure is not None


def test_structure_safety_rejects_short_bond():
    config = RPQTFGConfig()
    structure = _safe_structure(
        np.eye(3) * 4.0,
        np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]]),
        np.array([14, 14]),
        config,
    )
    assert structure is None


def test_structure_safety_rejects_atomic_zero():
    config = RPQTFGConfig()
    structure = _safe_structure(
        np.eye(3) * 4.0,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([0]),
        config,
    )
    assert structure is None


def test_physical_objective_uses_energy_force_and_stress():
    config = RPQTFGConfig(
        force_loss_weight=0.1,
        stress_loss_weight=0.01,
    )
    structure = _safe_structure(
        np.eye(3) * 4.0,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([14]),
        config,
    )
    assert structure is not None
    prediction = {
        "e": np.array([-1.0]),
        "f": np.array([[1.0, 0.0, 0.0]]),
        "s": np.eye(3),
    }
    value = _objective(prediction, structure, config)
    expected = -1.0 + 0.1 / 3.0 + 0.01 / 3.0
    assert value == pytest.approx(expected)


def test_eight_seed_gate_rejects_tiny_force_change():
    baseline = pd.DataFrame(
        {
            "seed": range(8),
            "initial_energy_per_atom_ev": [1.0] * 8,
            "initial_max_force_ev_ang": [1.0] * 8,
            "rmsd_from_relaxation": [1.0] * 8,
            "energy_above_hull_per_atom": [0.1] * 8,
        }
    )
    candidate = baseline.copy()
    candidate["initial_max_force_ev_ang"] = [0.99] * 4 + [1.004] * 4
    candidate["rmsd_from_relaxation"] = [1.2] * 8
    base_summary = {
        "method": "A0", "generation_success": 1.0,
        "generation_elapsed_median": 100.0, "initial_max_force_mean": 1.0,
        "relaxation_rmsd_mean": 1.0, "average_ehull": 0.1,
        "stable_rate": 0.5, "composition_validity": 1.0,
        "structure_validity": 1.0, "novel_rate": 0.5,
        "unique_rate": 1.0, "nus_rate": 0.5,
        "relaxation_failure_rate": 0.0, "severe_short_bond_count": 0,
        "atomic_numbers_modified": False,
    }
    candidate_summary = {
        **base_summary, "method": "G1", "generation_elapsed_median": 120.0,
        "initial_max_force_mean": 0.997, "relaxation_rmsd_mean": 1.2,
        "average_ehull": 0.095,
    }
    comparison = _comparison(
        baseline, candidate, base_summary, candidate_summary, "eight"
    )
    assert not comparison["clear_improvement_direction"]
    assert not comparison["gate_go"]
