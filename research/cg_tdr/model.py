"""Lightweight equivariant terminal refiner used by the CG-TDR experiment.

The module consumes frozen MatterGen node features and applies one deterministic
post-denoising correction.  It never predicts atomic identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CGTDRConfig:
    node_dim: int = 512
    convergence_dim: int = 8
    hidden_dim: int = 128
    num_rbf: int = 32
    cutoff: float = 6.0
    max_position_angstrom: float = 0.08
    max_strain_norm: float = 0.01
    max_relative_volume_change: float = 0.03
    max_condition_ratio: float = 1.25
    min_lattice_length: float = 1.0
    min_volume: float = 1.0
    gate_bias: float = -4.0
    enable_cell: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class CGTDROutput:
    frac_pos: torch.Tensor
    cell: torch.Tensor
    position_residual_cart: torch.Tensor
    strain: torch.Tensor
    position_gate: torch.Tensor
    cell_gate: torch.Tensor
    position_clipped: torch.Tensor
    cell_fallback: torch.Tensor
    edge_index: torch.Tensor
    edge_distance: torch.Tensor


def _build_periodic_edges(
    frac_pos: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build directed minimum-image edges for a small batched crystal graph."""

    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    vectors: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    for graph_index in range(cell.shape[0]):
        atom_index = torch.nonzero(batch_idx == graph_index, as_tuple=False).flatten()
        if atom_index.numel() < 2:
            continue
        source = atom_index.repeat_interleave(atom_index.numel())
        target = atom_index.repeat(atom_index.numel())
        keep = source != target
        source = source[keep]
        target = target[keep]
        frac_vector = frac_pos[source] - frac_pos[target]
        frac_vector = frac_vector - torch.round(frac_vector)
        cart_vector = frac_vector @ cell[graph_index]
        distance = torch.linalg.vector_norm(cart_vector, dim=-1)
        keep = (distance > 1.0e-7) & (distance <= cutoff)
        sources.append(source[keep])
        targets.append(target[keep])
        vectors.append(cart_vector[keep])
        distances.append(distance[keep])

    if not sources:
        empty_index = torch.empty((2, 0), dtype=torch.long, device=frac_pos.device)
        empty_vector = frac_pos.new_empty((0, 3))
        empty_distance = frac_pos.new_empty((0,))
        return empty_index, empty_vector, empty_distance
    edge_index = torch.stack([torch.cat(sources), torch.cat(targets)])
    return edge_index, torch.cat(vectors), torch.cat(distances)


def _graph_mean(values: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    output = values.new_zeros((batch_size, values.shape[-1]))
    output.index_add_(0, batch_idx, values)
    counts = torch.bincount(batch_idx, minlength=batch_size).to(values.dtype).clamp_min_(1)
    return output / counts[:, None]


class CGTDRRefiner(nn.Module):
    """Rotation-equivariant position head with a weak lattice-basis strain head."""

    def __init__(self, config: CGTDRConfig | None = None):
        super().__init__()
        self.config = config or CGTDRConfig()
        cfg = self.config
        edge_input_dim = 2 * cfg.node_dim + cfg.num_rbf + cfg.convergence_dim
        graph_input_dim = cfg.node_dim + cfg.convergence_dim + 3
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.position_output = nn.Linear(cfg.hidden_dim, 1)
        self.position_gate = nn.Sequential(
            nn.Linear(graph_input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.cell_head = nn.Sequential(
            nn.Linear(graph_input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 6),
        )
        self.cell_gate = nn.Sequential(
            nn.Linear(graph_input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.node_norm = nn.LayerNorm(cfg.node_dim)
        self.register_buffer("rbf_centers", torch.linspace(0.0, cfg.cutoff, cfg.num_rbf))
        self.register_buffer("convergence_mean", torch.zeros(cfg.convergence_dim))
        self.register_buffer("convergence_std", torch.ones(cfg.convergence_dim))
        self.reset_identity_parameters()

    def reset_identity_parameters(self) -> None:
        """Make an untrained module numerically identical to the A0 terminal state."""

        nn.init.zeros_(self.position_output.weight)
        nn.init.zeros_(self.position_output.bias)
        nn.init.zeros_(self.position_gate[-1].weight)
        nn.init.constant_(self.position_gate[-1].bias, self.config.gate_bias)
        nn.init.zeros_(self.cell_head[-1].weight)
        nn.init.zeros_(self.cell_head[-1].bias)
        nn.init.zeros_(self.cell_gate[-1].weight)
        nn.init.constant_(self.cell_gate[-1].bias, self.config.gate_bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def set_convergence_normalization(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        if mean.shape != self.convergence_mean.shape or std.shape != self.convergence_std.shape:
            raise ValueError("convergence normalization shape mismatch")
        self.convergence_mean.copy_(mean.detach().to(self.convergence_mean))
        self.convergence_std.copy_(std.detach().to(self.convergence_std).clamp_min(1.0e-6))

    def _radial_basis(self, distance: torch.Tensor) -> torch.Tensor:
        spacing = self.config.cutoff / max(self.config.num_rbf - 1, 1)
        gamma = 1.0 / max(spacing * spacing, 1.0e-8)
        return torch.exp(-gamma * (distance[:, None] - self.rbf_centers[None]) ** 2)

    @staticmethod
    def _strain_matrix(components: torch.Tensor) -> torch.Tensor:
        strain = components.new_zeros((components.shape[0], 3, 3))
        strain[:, 0, 0] = components[:, 0]
        strain[:, 1, 1] = components[:, 1]
        strain[:, 2, 2] = components[:, 2]
        strain[:, 0, 1] = strain[:, 1, 0] = components[:, 3]
        strain[:, 0, 2] = strain[:, 2, 0] = components[:, 4]
        strain[:, 1, 2] = strain[:, 2, 1] = components[:, 5]
        return strain

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        frac_pos: torch.Tensor,
        cell: torch.Tensor,
        batch_idx: torch.Tensor,
        convergence: torch.Tensor,
        enable_cell: bool | None = None,
    ) -> CGTDROutput:
        cfg = self.config
        batch_size = cell.shape[0]
        if convergence.shape != (batch_size, cfg.convergence_dim):
            raise ValueError(
                f"convergence must have shape {(batch_size, cfg.convergence_dim)}, "
                f"got {tuple(convergence.shape)}"
            )
        edge_index, edge_vector, edge_distance = _build_periodic_edges(
            frac_pos, cell, batch_idx, cfg.cutoff
        )
        node_features = self.node_norm(node_features)
        convergence = (convergence - self.convergence_mean) / self.convergence_std
        source, target = edge_index
        if edge_distance.numel():
            unit_vector = edge_vector / edge_distance[:, None].clamp_min(1.0e-8)
            edge_features = torch.cat(
                [
                    node_features[source],
                    node_features[target],
                    self._radial_basis(edge_distance),
                    convergence[batch_idx[target]],
                ],
                dim=-1,
            )
            edge_weight = self.position_output(self.edge_mlp(edge_features))
            raw_position = frac_pos.new_zeros(frac_pos.shape)
            raw_position.index_add_(0, target, edge_weight * unit_vector)
        else:
            raw_position = frac_pos.new_zeros(frac_pos.shape)

        # Local geometry statistics are invariant scalars.
        geometry = cell.new_zeros((batch_size, 3))
        for graph_index in range(batch_size):
            graph_distances = edge_distance[batch_idx[target] == graph_index]
            if graph_distances.numel():
                geometry[graph_index] = torch.stack(
                    [
                        graph_distances.min(),
                        graph_distances.mean(),
                        graph_distances.max(),
                    ]
                )
        pooled = _graph_mean(node_features, batch_idx, batch_size)
        graph_features = torch.cat([pooled, convergence, geometry], dim=-1)
        position_gate = torch.sigmoid(self.position_gate(graph_features))
        cell_gate = torch.sigmoid(self.cell_gate(graph_features))

        raw_norm = torch.linalg.vector_norm(raw_position, dim=-1, keepdim=True)
        position_scale = torch.clamp(
            cfg.max_position_angstrom / raw_norm.clamp_min(1.0e-12), max=1.0
        )
        bounded_position = raw_position * position_scale
        position_clipped = raw_norm.squeeze(-1) > cfg.max_position_angstrom
        position_residual = bounded_position * position_gate[batch_idx]

        cell_enabled = cfg.enable_cell if enable_cell is None else bool(enable_cell)
        strain_components = self.cell_head(graph_features)
        strain = self._strain_matrix(strain_components)
        strain_norm = torch.linalg.matrix_norm(strain, ord="fro", dim=(-2, -1), keepdim=True)
        strain = strain * torch.clamp(
            cfg.max_strain_norm / strain_norm.clamp_min(1.0e-12), max=1.0
        )
        if not cell_enabled:
            cell_gate = torch.zeros_like(cell_gate)
        strain = strain * cell_gate[:, None]
        identity = torch.eye(3, dtype=cell.dtype, device=cell.device).expand(batch_size, -1, -1)
        candidate_cell = torch.bmm(identity + strain, cell)

        old_volume = torch.linalg.det(cell)
        new_volume = torch.linalg.det(candidate_cell)
        relative_volume = (new_volume / old_volume.clamp_min(1.0e-12) - 1.0).abs()
        old_condition = torch.linalg.cond(cell)
        new_condition = torch.linalg.cond(candidate_cell)
        lattice_lengths = torch.linalg.vector_norm(candidate_cell, dim=-1)
        cell_fallback = (
            (~torch.isfinite(candidate_cell).all(dim=-1).all(dim=-1))
            | (new_volume <= cfg.min_volume)
            | (relative_volume > cfg.max_relative_volume_change)
            | (new_condition > old_condition * cfg.max_condition_ratio)
            | (lattice_lengths.min(dim=-1).values < cfg.min_lattice_length)
        )
        refined_cell = torch.where(cell_fallback[:, None, None], cell, candidate_cell)
        applied_strain = torch.where(cell_fallback[:, None, None], torch.zeros_like(strain), strain)
        zero_cell_update = applied_strain.abs().amax(dim=(-2, -1)) == 0
        refined_cell = torch.where(zero_cell_update[:, None, None], cell, refined_cell)

        cart_pos = torch.bmm(
            frac_pos[:, None, :], cell[batch_idx]
        ).squeeze(1)
        refined_cart = cart_pos + position_residual
        refined_frac = torch.bmm(
            refined_cart[:, None, :], torch.linalg.inv(refined_cell[batch_idx])
        ).squeeze(1)
        refined_frac = torch.remainder(refined_frac, 1.0)
        zero_position_update = position_residual.abs().amax(dim=-1) == 0
        identity_graph = zero_cell_update[batch_idx] & zero_position_update
        refined_frac = torch.where(identity_graph[:, None], frac_pos, refined_frac)

        return CGTDROutput(
            frac_pos=refined_frac,
            cell=refined_cell,
            position_residual_cart=position_residual,
            strain=applied_strain,
            position_gate=position_gate,
            cell_gate=cell_gate,
            position_clipped=position_clipped,
            cell_fallback=cell_fallback,
            edge_index=edge_index,
            edge_distance=edge_distance,
        )
