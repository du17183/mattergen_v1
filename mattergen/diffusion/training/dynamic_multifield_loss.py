"""Noise-aware constrained dynamic weighting of original MatterGen losses."""
from __future__ import annotations

import torch
from torch import nn


FIELD_ORDER = ("atomic_numbers", "pos", "cell")


class NoiseAwareFieldScheduler(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(FIELD_ORDER)),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, normalized_timestep: torch.Tensor) -> torch.Tensor:
        routing_logits = self.network(normalized_timestep.unsqueeze(-1))
        return 1.0 + 0.5 * torch.tanh(routing_logits)


def dynamic_weighted_loss(
    scheduler: NoiseAwareFieldScheduler,
    per_field: dict[str, torch.Tensor],
    timestep: torch.Tensor,
    base_weights: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    multipliers = scheduler(timestep)
    base = timestep.new_tensor([base_weights[field] for field in FIELD_ORDER])
    unnormalized = multipliers * base.unsqueeze(0)
    weights = unnormalized * (base.sum() / unnormalized.sum(dim=-1, keepdim=True))
    field_losses = torch.stack([per_field[field] for field in FIELD_ORDER], dim=-1)
    return (weights * field_losses).sum(dim=-1).mean(), multipliers, weights


def multiplier_anchor(low: torch.Tensor, high: torch.Tensor, coefficient: float = 0.01):
    return coefficient * torch.cat((low, high), dim=0).sub(1.0).square().mean()
