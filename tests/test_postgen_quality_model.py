from __future__ import annotations

import torch

from research.postgen_fastgate.model import QualityNetwork


def test_quality_network_shapes() -> None:
    model = QualityNetwork(input_dim=17, hidden_dim=32, dropout=0.0)
    output = model(torch.randn(7, 17))
    assert output["continuous"].shape == (7, 2)
    assert output["binary_logits"].shape == (7, 5)
    assert output["embedding"].shape == (7, 16)


def test_quality_network_is_deterministic_in_eval_mode() -> None:
    torch.manual_seed(5)
    model = QualityNetwork(input_dim=4, hidden_dim=16, dropout=0.2)
    features = torch.randn(3, 4)
    model.eval()
    first = model(features)["continuous"]
    second = model(features)["continuous"]
    assert torch.equal(first, second)
