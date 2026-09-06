"""Lightweight cross-field feature interaction for MatterGen P0 experiments."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from torch_scatter import scatter


class AtomicFieldEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 512, field_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, field_dim), nn.SiLU()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.network(h)


class PeriodicGeometrySummary(nn.Module):
    """Aggregate scalar RBF features of GemNet's existing periodic edges."""

    def __init__(
        self, *, field_dim: int = 128, num_rbf: int = 32, cutoff: float = 7.0
    ) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.cutoff = cutoff
        spacing = cutoff / max(num_rbf - 1, 1)
        self.gamma = 1.0 / max(spacing**2, 1e-8)
        self.projection = nn.Linear(num_rbf, field_dim)

    def forward(
        self,
        *,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        rbf = torch.exp(
            -self.gamma
            * (edge_distances.clamp(max=self.cutoff)[:, None] - self.centers)
            ** 2
        )
        edge_features = self.projection(rbf)
        return scatter(
            edge_features,
            edge_index[1],
            dim=0,
            dim_size=num_nodes,
            reduce="mean",
        )


class PositionFieldEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        field_dim: int = 128,
        num_rbf: int = 32,
        cutoff: float = 7.0,
    ) -> None:
        super().__init__()
        self.hidden_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, field_dim)
        )
        self.geometry = PeriodicGeometrySummary(
            field_dim=field_dim, num_rbf=num_rbf, cutoff=cutoff
        )
        self.activation = nn.SiLU()

    def forward(
        self,
        h: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
    ) -> torch.Tensor:
        geometry = self.geometry(
            edge_index=edge_index,
            edge_distances=edge_distances,
            num_nodes=h.shape[0],
        )
        return self.activation(self.hidden_projection(h) + geometry)


class CellFieldEncoder(nn.Module):
    def __init__(
        self, *, hidden_dim: int = 512, field_dim: int = 128
    ) -> None:
        super().__init__()
        descriptor_dim = hidden_dim + 9
        self.network = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, 256),
            nn.SiLU(),
            nn.Linear(256, field_dim),
            nn.SiLU(),
        )

    @staticmethod
    def cell_descriptors(
        lattice: torch.Tensor, num_atoms: torch.Tensor
    ) -> torch.Tensor:
        # MatterGen stores lattice vectors as rows (cart = frac @ lattice).
        # L L^T removes global Cartesian rotation, but is not claimed to be
        # invariant to an arbitrary change of lattice basis.
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
        # Scalar triple product avoids the H20 batched 3x3 determinant issue.
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

    def forward(
        self,
        h: torch.Tensor,
        *,
        batch: torch.Tensor,
        num_atoms: torch.Tensor,
        lattice: torch.Tensor,
    ) -> torch.Tensor:
        pooled_h = scatter(
            h,
            batch,
            dim=0,
            dim_size=num_atoms.shape[0],
            reduce="mean",
        )
        descriptors = self.cell_descriptors(lattice, num_atoms)
        return self.network(torch.cat((pooled_h, descriptors), dim=-1))


class GatedFieldExchange(nn.Module):
    """One message and elementwise sigmoid gate over concatenated field context."""

    def __init__(self, field_dim: int = 128, context_fields: int = 4) -> None:
        super().__init__()
        input_dim = field_dim * context_fields
        self.message = nn.Linear(input_dim, field_dim)
        self.gate = nn.Linear(input_dim, field_dim)
        self.activation = nn.SiLU()

    def forward(self, base: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        message = self.activation(self.message(context))
        gate = torch.sigmoid(self.gate(context))
        return base + gate * message


class CrossFieldInteraction(nn.Module):
    """One bidirectional atomic-position-cell gated exchange."""

    def __init__(self, field_dim: int = 128) -> None:
        super().__init__()
        self.atomic_exchange = GatedFieldExchange(field_dim)
        self.position_exchange = GatedFieldExchange(field_dim)
        self.cell_exchange = GatedFieldExchange(field_dim)

    def forward(
        self,
        *,
        atomic: torch.Tensor,
        position: torch.Tensor,
        cell: torch.Tensor,
        timestep: torch.Tensor,
        batch: torch.Tensor,
        num_crystals: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        atomic_context = torch.cat(
            (atomic, position, cell[batch], timestep[batch]), dim=-1
        )
        position_context = torch.cat(
            (position, atomic, cell[batch], timestep[batch]), dim=-1
        )
        pooled_atomic = scatter(
            atomic, batch, dim=0, dim_size=num_crystals, reduce="mean"
        )
        pooled_position = scatter(
            position, batch, dim=0, dim_size=num_crystals, reduce="mean"
        )
        cell_context = torch.cat(
            (cell, pooled_atomic, pooled_position, timestep), dim=-1
        )
        return (
            self.atomic_exchange(atomic, atomic_context),
            self.position_exchange(position, position_context),
            self.cell_exchange(cell, cell_context),
        )


class CrossFieldAdapter(nn.Module):
    """Field-specific encoding, one gated exchange, and zero-init h residual."""

    architecture = "cross_field_interaction"

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        time_dim: int = 512,
        field_dim: int = 128,
        fusion_dim: int = 256,
        num_rbf: int = 32,
        cutoff: float = 7.0,
        block_index: int = 1,
    ) -> None:
        super().__init__()
        self.block_index = block_index
        self.field_dim = field_dim
        self.fusion_dim = fusion_dim
        self.atomic_encoder = AtomicFieldEncoder(hidden_dim, field_dim)
        self.position_encoder = PositionFieldEncoder(
            hidden_dim=hidden_dim,
            field_dim=field_dim,
            num_rbf=num_rbf,
            cutoff=cutoff,
        )
        self.cell_encoder = CellFieldEncoder(
            hidden_dim=hidden_dim, field_dim=field_dim
        )
        self.timestep_projection = nn.Sequential(
            nn.LayerNorm(time_dim), nn.Linear(time_dim, field_dim), nn.SiLU()
        )
        self.interaction = CrossFieldInteraction(field_dim)
        self.fusion = nn.Sequential(
            nn.Linear(3 * field_dim, fusion_dim), nn.SiLU()
        )
        self.output_projection = nn.Linear(fusion_dim, hidden_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.enabled = True

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
        atomic = self.atomic_encoder(h)
        position = self.position_encoder(
            h, edge_index=edge_index, edge_distances=edge_distances
        )
        cell = self.cell_encoder(
            h, batch=batch, num_atoms=num_atoms, lattice=lattice
        )
        timestep = self.timestep_projection(time_embedding)
        atomic, position, cell = self.interaction(
            atomic=atomic,
            position=position,
            cell=cell,
            timestep=timestep,
            batch=batch,
            num_crystals=num_atoms.shape[0],
        )
        fused = self.fusion(torch.cat((atomic, position, cell[batch]), dim=-1))
        return h + self.output_projection(fused)

    @contextmanager
    def disabled(self) -> Iterator[None]:
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous
