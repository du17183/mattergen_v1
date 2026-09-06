"""Periodic geometry-aware global and matched local residual adapters."""
from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterator

import torch
from torch import nn


class PeriodicDistanceBias(nn.Module):
    """Map scalar periodic distances to one additive bias per attention head."""

    def __init__(self, *, cutoff: float, num_rbf: int, num_heads: int) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        spacing = cutoff / max(num_rbf - 1, 1)
        self.gamma = 1.0 / max(spacing**2, 1e-8)
        self.projection = nn.Linear(num_rbf, num_heads)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(
            -self.gamma * (distances[..., None] - self.centers) ** 2
        )
        return self.projection(rbf).permute(0, 3, 1, 2).contiguous()


class GeometryBiasedSelfAttention(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, d_model = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        scores = scores + attention_bias.to(scores.dtype)
        scores = scores.masked_fill(
            key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, d_model
        )
        return self.output(attended)


class GlobalTransformerBlock(nn.Module):
    def __init__(
        self, *, d_model: int, num_heads: int, ffn_dim: int
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = GeometryBiasedSelfAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        valid = (~key_padding_mask)[..., None].to(tokens.dtype)
        tokens = tokens + self.attention(
            self.attention_norm(tokens),
            key_padding_mask=key_padding_mask,
            attention_bias=attention_bias,
        )
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens * valid


class GlobalTransformerAdapter(nn.Module):
    """One global atom-cell-time attention block after GemNet block index 1."""

    architecture = "global_transformer"

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        time_dim: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 768,
        num_rbf: int = 32,
        cutoff: float = 7.0,
        block_index: int = 1,
    ) -> None:
        super().__init__()
        self.block_index = block_index
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.cutoff = cutoff
        self.atom_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, d_model)
        )
        self.cell_projection = nn.Sequential(
            nn.Linear(9, 128), nn.SiLU(), nn.Linear(128, d_model)
        )
        self.time_projection = nn.Sequential(
            nn.LayerNorm(time_dim), nn.Linear(time_dim, d_model)
        )
        self.token_type_embedding = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.token_type_embedding, std=0.02)
        self.distance_bias = PeriodicDistanceBias(
            cutoff=cutoff, num_rbf=num_rbf, num_heads=num_heads
        )
        self.transformer = GlobalTransformerBlock(
            d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim
        )
        self.output_projection = nn.Linear(d_model, hidden_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.enabled = True

    def cell_descriptors(
        self, lattice: torch.Tensor, num_atoms: torch.Tensor
    ) -> torch.Tensor:
        # MatterGen stores lattice vectors as rows (cart = frac @ lattice).
        # L L^T is invariant to a global Cartesian rotation L -> L R.
        gram = lattice @ lattice.transpose(-1, -2)
        components = torch.stack(
            (
                gram[:, 0, 0],
                gram[:, 1, 1],
                gram[:, 2, 2],
                gram[:, 0, 1],
                gram[:, 0, 2],
                gram[:, 1, 2],
            ),
            dim=-1,
        )
        components = torch.sign(components) * torch.log1p(components.abs())
        # Equivalent differentiable scalar triple product avoids the H20/CUDA
        # batched 3x3 determinant driver failure.
        volume = (
            lattice[:, 0]
            * torch.cross(lattice[:, 1], lattice[:, 2], dim=-1)
        ).sum(dim=-1).abs().clamp_min(1e-8)
        atom_count = num_atoms.to(lattice.dtype).clamp_min(1.0)
        return torch.cat(
            (
                components,
                torch.log1p(volume)[:, None],
                torch.log1p(volume / atom_count)[:, None],
                torch.log1p(atom_count)[:, None],
            ),
            dim=-1,
        )

    def dense_periodic_distances(
        self,
        *,
        batch: torch.Tensor,
        num_atoms: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(num_atoms.shape[0])
        max_atoms = int(num_atoms.max().item())
        distances = edge_distances.new_full(
            (batch_size, max_atoms, max_atoms), self.cutoff
        )
        starts = torch.cumsum(num_atoms, dim=0) - num_atoms
        source, target = edge_index
        edge_batch = batch[source]
        local_source = source - starts[edge_batch]
        local_target = target - starts[edge_batch]
        flat_index = (
            edge_batch * max_atoms * max_atoms
            + local_source * max_atoms
            + local_target
        )
        distances.view(-1).scatter_reduce_(
            0,
            flat_index,
            edge_distances.clamp(max=self.cutoff),
            reduce="amin",
            include_self=True,
        )
        atom_range = torch.arange(max_atoms, device=num_atoms.device)
        valid = atom_range[None, :] < num_atoms[:, None]
        diagonal = torch.arange(max_atoms, device=num_atoms.device)
        distances[:, diagonal, diagonal] = torch.where(
            valid, torch.zeros_like(distances[:, diagonal, diagonal]), self.cutoff
        )
        return distances

    def forward(
        self,
        *,
        h: torch.Tensor,
        time_embedding: torch.Tensor,
        batch: torch.Tensor,
        num_atoms: torch.Tensor,
        lattice: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        if not self.enabled or block_index != self.block_index:
            return h
        batch_size = int(num_atoms.shape[0])
        max_atoms = int(num_atoms.max().item())
        starts = torch.cumsum(num_atoms, dim=0) - num_atoms
        local_index = torch.arange(h.shape[0], device=h.device) - torch.repeat_interleave(
            starts, num_atoms
        )
        sequence_length = max_atoms + 2
        tokens = h.new_zeros((batch_size, sequence_length, self.d_model))
        padding_mask = torch.ones(
            (batch_size, sequence_length), device=h.device, dtype=torch.bool
        )
        tokens[batch, local_index] = (
            self.atom_projection(h) + self.token_type_embedding[0]
        )
        padding_mask[batch, local_index] = False
        tokens[:, max_atoms] = self.cell_projection(
            self.cell_descriptors(lattice, num_atoms)
        ) + self.token_type_embedding[1]
        tokens[:, max_atoms + 1] = self.time_projection(
            time_embedding
        ) + self.token_type_embedding[2]
        padding_mask[:, max_atoms:] = False

        atom_distances = self.dense_periodic_distances(
            batch=batch,
            num_atoms=num_atoms,
            edge_index=edge_index,
            edge_distances=edge_distances,
        )
        atom_bias = self.distance_bias(atom_distances)
        attention_bias = h.new_zeros(
            (batch_size, self.num_heads, sequence_length, sequence_length)
        )
        attention_bias[:, :, :max_atoms, :max_atoms] = atom_bias
        output = self.transformer(
            tokens,
            key_padding_mask=padding_mask,
            attention_bias=attention_bias,
        )
        atom_output = output[batch, local_index]
        return h + self.output_projection(atom_output)

    @contextmanager
    def disabled(self) -> Iterator[None]:
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous


class MatchedMLPAdapter(nn.Module):
    """Parameter-matched atom-local control at the identical GemNet position."""

    architecture = "matched_mlp"

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        mlp_dim: int = 650,
        block_index: int = 1,
    ) -> None:
        super().__init__()
        self.block_index = block_index
        self.mlp_dim = mlp_dim
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
        )
        self.output_projection = nn.Linear(mlp_dim, hidden_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.enabled = True

    def forward(self, *, h: torch.Tensor, block_index: int, **_) -> torch.Tensor:
        if not self.enabled or block_index != self.block_index:
            return h
        return h + self.output_projection(self.network(h))

    @contextmanager
    def disabled(self) -> Iterator[None]:
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous


def build_global_adapter(architecture: str) -> nn.Module:
    if architecture == "Transformer":
        return GlobalTransformerAdapter()
    if architecture == "MLP":
        return MatchedMLPAdapter()
    raise ValueError(f"unknown architecture: {architecture}")
