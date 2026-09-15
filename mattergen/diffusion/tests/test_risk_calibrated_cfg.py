from __future__ import annotations

import numpy as np
import torch
from ase import Atoms

from mattergen.diffusion.sampling.risk_calibrated_cfg import (
    ACTION_BY_SCALE,
    FullOutcomeCounterfactualSampler,
    _cosine,
    _tensor_digest,
)


def test_frozen_action_mapping() -> None:
    assert ACTION_BY_SCALE == {1.75: "DOWN", 2.0: "KEEP", 2.25: "UP"}


def test_cosine_edge_cases() -> None:
    assert _cosine(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == 1.0
    assert _cosine(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])) == -1.0
    assert _cosine(torch.zeros(2), torch.ones(2)) == 0.0


def test_state_digest_is_deterministic_and_field_sensitive() -> None:
    state = {
        "atomic_numbers": torch.tensor([1, 2]),
        "pos": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "cell": torch.eye(3).reshape(1, 3, 3),
    }
    first = _tensor_digest(state)
    assert first == _tensor_digest({key: value.clone() for key, value in state.items()})
    changed = {key: value.clone() for key, value in state.items()}
    changed["pos"][0, 0] += 0.01
    assert first != _tensor_digest(changed)


def test_structural_features_are_online_finite_summaries() -> None:
    atoms = Atoms(
        numbers=[14, 8, 8],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        cell=np.eye(3) * 5,
        pbc=True,
    )
    values = FullOutcomeCounterfactualSampler._structural_features(atoms)
    assert values["x0_num_atoms"] == 3
    assert values["x0_num_elements"] == 2
    assert values["x0_structural_valid"] == 1
    assert all(np.isfinite(value) for value in values.values())
