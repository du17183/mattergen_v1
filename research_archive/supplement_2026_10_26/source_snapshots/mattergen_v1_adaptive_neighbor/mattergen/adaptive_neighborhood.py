"""Timestep-conditioned three-band radial gate for the isolated A experiment."""
from __future__ import annotations

import torch
from torch import nn


class TimestepRadialGate(nn.Module):
    """Return one [0.5, 1.5] edge weight from a shared 1->16->3 time MLP."""

    def __init__(
        self,
        hidden_dim: int = 16,
        near_upper: float = 3.5,
        mid_upper: float = 5.5,
        cutoff: float = 7.0,
    ) -> None:
        super().__init__()
        self.near_upper = float(near_upper)
        self.mid_upper = float(mid_upper)
        self.cutoff = float(cutoff)
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def gate_values(self, timestep: torch.Tensor) -> torch.Tensor:
        logits = self.network(timestep.reshape(-1, 1).float())
        return 1.0 + 0.5 * torch.tanh(logits)

    def forward(
        self,
        *,
        timestep: torch.Tensor,
        edge_distances: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> torch.Tensor:
        if bool((edge_distances > self.cutoff + 1.0e-5).any()):
            raise RuntimeError("radial gate received an edge outside the frozen cutoff")
        band = torch.where(
            edge_distances <= self.near_upper,
            torch.zeros_like(edge_distances, dtype=torch.long),
            torch.where(
                edge_distances <= self.mid_upper,
                torch.ones_like(edge_distances, dtype=torch.long),
                torch.full_like(edge_distances, 2, dtype=torch.long),
            ),
        )
        graph_gates = self.gate_values(timestep).to(edge_distances)
        return graph_gates[edge_batch, band].unsqueeze(-1)
