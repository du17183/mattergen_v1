from __future__ import annotations

import torch

from research.postgen_fastgate.setrank import SetRankNetwork


def test_setrank_shape_and_permutation_equivariance() -> None:
    torch.manual_seed(17)
    model = SetRankNetwork(input_dim=11, hidden_dim=24, dropout=0.0)
    model.eval()
    pools = torch.randn(3, 4, 11)
    permutation = torch.tensor([2, 0, 3, 1])
    original = model(pools)
    permuted = model(pools[:, permutation])
    assert original.shape == (3, 4)
    assert torch.allclose(permuted, original[:, permutation], atol=1.0e-6)


def test_setrank_rejects_non_pool_input() -> None:
    model = SetRankNetwork(input_dim=3, hidden_dim=12, dropout=0.0)
    try:
        model(torch.randn(4, 3))
    except ValueError as error:
        assert "batch, pool, feature" in str(error)
    else:
        raise AssertionError("expected ValueError")
