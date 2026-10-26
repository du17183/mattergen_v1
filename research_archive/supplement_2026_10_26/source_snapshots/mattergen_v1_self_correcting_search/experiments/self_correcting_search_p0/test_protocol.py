"""Small deterministic protocol tests that do not require a MatterGen checkpoint."""
from __future__ import annotations

import math

import pytest
import torch

from experiments.self_correcting_search_p0.self_correcting_sampler import (
    LinearProbe,
    TinyMLPProbe,
    acceptance_probability,
)


def test_acceptance_probability() -> None:
    assert acceptance_probability(-0.001) == 1.0
    assert acceptance_probability(0.0) == 1.0
    assert acceptance_probability(0.01) == pytest.approx(math.exp(-1.0))
    with pytest.raises(ValueError):
        acceptance_probability(0.1, 0.0)


@pytest.mark.parametrize("probe", [LinearProbe(), TinyMLPProbe()])
def test_probe_shape(probe: torch.nn.Module) -> None:
    assert probe(torch.zeros(7, 513)).shape == (7,)
