"""Small residual adapters and P0 quality-weighted diffusion utilities."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply


class QualityResidualBottleneck(nn.Module):
    """A zero-initialized 512 -> bottleneck -> 512 residual branch."""

    def __init__(self, hidden_dim: int = 512, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(self.norm(h))))


class QualityAdapterStack(nn.Module):
    """Inject residual bottlenecks after selected GemNet interaction blocks."""

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        bottleneck_dim: int = 64,
        block_indices: tuple[int, ...] = (1, 2),
        stage_aware: bool = False,
    ) -> None:
        super().__init__()
        self.block_indices = tuple(block_indices)
        self.adapters = nn.ModuleDict(
            {
                str(index): QualityResidualBottleneck(hidden_dim, bottleneck_dim)
                for index in self.block_indices
            }
        )
        self.stage_gates = (
            nn.ModuleDict(
                {str(index): nn.Linear(hidden_dim, 1) for index in self.block_indices}
            )
            if stage_aware
            else None
        )
        self.enabled = True

    def forward(
        self,
        *,
        h: torch.Tensor,
        time_embedding: torch.Tensor,
        batch: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        key = str(block_index)
        if not self.enabled or key not in self.adapters:
            return h
        residual = self.adapters[key](h)
        if self.stage_gates is not None:
            gate = torch.sigmoid(self.stage_gates[key](time_embedding))[batch]
            residual = gate * residual
        return h + residual

    @contextmanager
    def disabled(self) -> Iterator[None]:
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous


def weighted_materials_loss(
    *,
    base_loss,
    multi_corruption: MultiCorruption,
    batch,
    noisy_batch,
    score_model_output,
    t: torch.Tensor,
    quality_weight: torch.Tensor,
    replay_fraction: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply graph-level quality weights before the batch mean."""
    batch_size = batch.get_batch_size()
    if quality_weight.shape != (batch_size,):
        raise ValueError(
            f"quality_weight must have shape {(batch_size,)}, got {quality_weight.shape}"
        )
    batch_idx = {key: batch.get_batch_idx(key) for key in base_loss.loss_fns}
    per_field = apply(
        fns=base_loss.loss_fns,
        corruption=multi_corruption.corruptions,
        x=batch,
        noisy_x=noisy_batch,
        score_model_output=score_model_output,
        batch_idx=batch_idx,
        broadcast=dict(t=t, batch_size=batch_size, batch=batch),
        node_is_unmasked={key: None for key in base_loss.loss_fns},
    )
    aggregate = torch.stack(
        [base_loss.loss_weights[key] * value for key, value in per_field.items()]
    ).sum(0)

    effective_weight = quality_weight.clone()
    replay_count = min(batch_size, max(0, round(batch_size * replay_fraction)))
    if replay_count:
        replay_indices = torch.randperm(batch_size, device=quality_weight.device)[
            :replay_count
        ]
        effective_weight[replay_indices] = 1.0
    loss = (effective_weight * aggregate).mean()
    metrics = {f"diffusion_{key}": value.mean() for key, value in per_field.items()}
    metrics.update(
        {
            "diffusion_total": loss.detach(),
            "effective_weight_mean": effective_weight.mean().detach(),
            "replay_fraction_actual": torch.tensor(
                replay_count / batch_size, device=quality_weight.device
            ),
        }
    )
    return loss, metrics


def output_anchor_loss(student, teacher) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Anchor all three model outputs on the identical noisy state."""
    atomic = F.kl_div(
        F.log_softmax(student.atomic_numbers, dim=-1),
        F.softmax(teacher.atomic_numbers, dim=-1),
        reduction="batchmean",
    )

    def relative_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        scale = right.float().square().mean().detach().clamp_min(1e-6)
        return (left.float() - right.float()).square().mean() / scale

    position = relative_mse(student.pos, teacher.pos)
    cell = relative_mse(student.cell, teacher.cell)
    total = atomic + 0.1 * position + cell
    return total, {"anchor_atomic": atomic, "anchor_pos": position, "anchor_cell": cell}
