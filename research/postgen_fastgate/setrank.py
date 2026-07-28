"""Pool-aware novelty-stability selector."""

from __future__ import annotations

import torch
from torch import nn


class SetRankNetwork(nn.Module):
    """Permutation-equivariant candidate scorer with pooled context."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pools: torch.Tensor) -> torch.Tensor:
        if pools.ndim != 3:
            raise ValueError("expected [batch, pool, feature] tensor")
        hidden = self.candidate_encoder(pools)
        mean = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
        maximum = hidden.max(dim=1, keepdim=True).values.expand_as(hidden)
        context = torch.cat([hidden, mean, maximum], dim=-1)
        return self.score_head(context).squeeze(-1)
