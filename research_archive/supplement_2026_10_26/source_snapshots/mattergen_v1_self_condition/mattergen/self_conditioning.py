"""Multi-field self-conditioning components for the isolated P0 experiment."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class MultiFieldSelfConditioner(nn.Module):
    """Encode detached atom/position/cell clean estimates into one node residual."""

    def __init__(
        self,
        hidden_dim: int = 512,
        position_hidden_dim: int = 64,
        cell_hidden_dim: int = 64,
        gate_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.position_encoder = nn.Sequential(
            nn.Linear(6, position_hidden_dim),
            nn.SiLU(),
            nn.Linear(position_hidden_dim, hidden_dim),
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(9, cell_hidden_dim),
            nn.SiLU(),
            nn.Linear(cell_hidden_dim, hidden_dim),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 3),
        )
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.zeros_(self.gate_network[-1].bias)
        self.enabled = True
        self.last_gate_values: torch.Tensor | None = None

    def gate_values(self, time_embedding: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.gate_network(time_embedding))

    @staticmethod
    def _repeat_to_batch(
        value: torch.Tensor, repeat_factor: int
    ) -> torch.Tensor:
        if repeat_factor == 1:
            return value
        return value.repeat((repeat_factor,) + (1,) * (value.ndim - 1))

    def forward(
        self,
        *,
        current_pos: torch.Tensor,
        current_cell: torch.Tensor,
        num_atoms: torch.Tensor,
        batch: torch.Tensor,
        time_embedding: torch.Tensor,
        atom_embedding_weight: torch.Tensor,
        self_condition: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not self.enabled or self_condition is None:
            self.last_gate_values = torch.zeros(
                (len(current_cell), 3),
                device=current_pos.device,
                dtype=current_pos.dtype,
            )
            return torch.zeros(
                (len(current_pos), self.hidden_dim),
                device=current_pos.device,
                dtype=atom_embedding_weight.dtype,
            )

        sc_cell = self_condition["cell"].to(current_cell)
        sc_graphs = len(sc_cell)
        if len(current_cell) % sc_graphs:
            raise ValueError("self-condition batch cannot align to model batch")
        repeat_factor = len(current_cell) // sc_graphs
        sc_cell = self._repeat_to_batch(sc_cell, repeat_factor)
        sc_pos = self._repeat_to_batch(
            self_condition["pos"].to(current_pos), repeat_factor
        )
        atom_probabilities = self._repeat_to_batch(
            self_condition["atomic_probabilities"].to(atom_embedding_weight),
            repeat_factor,
        )
        sample_mask = self_condition.get(
            "sample_mask",
            torch.ones(sc_graphs, device=current_cell.device, dtype=current_cell.dtype),
        ).to(device=current_cell.device, dtype=current_cell.dtype)
        sample_mask = self._repeat_to_batch(sample_mask, repeat_factor)
        if len(sc_pos) != len(current_pos) or len(atom_probabilities) != len(current_pos):
            raise ValueError("self-condition atom rows cannot align to current batch")
        if atom_probabilities.shape[1] != atom_embedding_weight.shape[0]:
            raise ValueError("atomic probability and embedding class counts differ")

        atom_encoding = atom_probabilities @ atom_embedding_weight.detach()
        delta_pos = sc_pos - current_pos
        delta_pos = delta_pos - torch.round(delta_pos)
        periodic_position = torch.cat(
            [torch.sin(2.0 * torch.pi * delta_pos), torch.cos(2.0 * torch.pi * delta_pos)],
            dim=-1,
        )
        position_encoding = self.position_encoder(periodic_position)

        cell_scale = num_atoms.to(current_cell).clamp_min(1.0).pow(1.0 / 3.0)
        normalized_cell_delta = (sc_cell - current_cell) / cell_scale[:, None, None]
        normalized_cell_delta = normalized_cell_delta.clamp(-5.0, 5.0)
        cell_encoding = self.cell_encoder(normalized_cell_delta.flatten(start_dim=1))

        gates = self.gate_values(time_embedding)
        self.last_gate_values = gates.detach()
        node_gates = gates[batch]
        residual = (
            node_gates[:, [0]] * atom_encoding
            + node_gates[:, [1]] * position_encoding
            + node_gates[:, [2]] * cell_encoding[batch]
        )
        return residual * sample_mask[batch, None]


def _continuous_clean_prediction(
    *,
    corruption: Any,
    noisy_batch: Any,
    t: torch.Tensor,
    field: str,
    output: torch.Tensor,
    output_is_score: bool,
) -> torch.Tensor:
    noisy = noisy_batch[field]
    batch_idx = noisy_batch.get_batch_idx(field)
    zeros = torch.zeros_like(noisy)
    ones = torch.ones_like(noisy)
    mean_zero, std = corruption.marginal_prob(
        zeros, t=t, batch_idx=batch_idx, batch=noisy_batch
    )
    mean_one, _ = corruption.marginal_prob(
        ones, t=t, batch_idx=batch_idx, batch=noisy_batch
    )
    alpha = mean_one - mean_zero
    signed_floor = torch.where(
        alpha >= 0,
        torch.full_like(alpha, 1.0e-6),
        torch.full_like(alpha, -1.0e-6),
    )
    safe_alpha = torch.where(alpha.abs() >= 1.0e-6, alpha, signed_floor)
    correction = std.square() * output if output_is_score else std * output
    return (noisy - mean_zero + correction) / safe_alpha


@torch.no_grad()
def clean_prediction(
    *,
    diffusion: Any,
    noisy_batch: Any,
    t: torch.Tensor,
    output: Any,
    output_is_score: bool,
    sample_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Convert raw score-times-std outputs or true scores into detached x0 estimates."""

    atomic_probabilities = torch.softmax(output["atomic_numbers"].float(), dim=-1)
    pos = _continuous_clean_prediction(
        corruption=diffusion.corruption.corruptions["pos"],
        noisy_batch=noisy_batch,
        t=t,
        field="pos",
        output=output["pos"],
        output_is_score=output_is_score,
    ).remainder(1.0)
    cell = _continuous_clean_prediction(
        corruption=diffusion.corruption.corruptions["cell"],
        noisy_batch=noisy_batch,
        t=t,
        field="cell",
        output=output["cell"],
        output_is_score=output_is_score,
    )
    cell = 0.5 * (cell + cell.transpose(-1, -2))
    if sample_mask is None:
        sample_mask = torch.ones(
            noisy_batch.get_batch_size(), device=cell.device, dtype=cell.dtype
        )
    return {
        "atomic_probabilities": atomic_probabilities.detach(),
        "pos": pos.detach(),
        "cell": cell.detach(),
        "sample_mask": sample_mask.detach(),
    }
