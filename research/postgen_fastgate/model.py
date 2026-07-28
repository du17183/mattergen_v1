"""Small post-generation quality network; MatterGen and CHGNet stay frozen."""

from __future__ import annotations

import torch
from torch import nn


class QualityNetwork(nn.Module):
    """Shared encoder with continuous and binary quality heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.continuous_head = nn.Linear(hidden_dim // 2, 2)
        self.binary_head = nn.Linear(hidden_dim // 2, 5)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(features)
        return {
            "continuous": self.continuous_head(hidden),
            "binary_logits": self.binary_head(hidden),
            "embedding": hidden,
        }
