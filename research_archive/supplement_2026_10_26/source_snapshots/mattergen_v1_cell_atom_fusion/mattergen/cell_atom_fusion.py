"""Lightweight zero-init global Cell<->Atom feature fusion."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch_scatter import scatter


def signed_log1p(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


class CellAtomBidirectionalFusion(nn.Module):
    """Fuse a row-lattice Gram descriptor with the Block-3 atom stream.

    The two scalar gains are exactly zero at construction, so the module is an
    exact identity.  Directional MLPs retain ordinary non-zero initialization;
    zeroing both their outputs and their scalar gains would create a dead path.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        bottleneck_dim: int = 128,
        cell_feature_mean: torch.Tensor | None = None,
        cell_feature_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.block_index = 2
        self.cell_encoder = nn.Sequential(
            nn.Linear(6, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.atom_to_cell = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.cell_to_atom = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.alpha_a2c = nn.Parameter(torch.zeros(()))
        self.alpha_c2a = nn.Parameter(torch.zeros(()))
        mean = torch.zeros(6) if cell_feature_mean is None else cell_feature_mean.float()
        std = torch.ones(6) if cell_feature_std is None else cell_feature_std.float()
        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError("cell normalization tensors must have shape [6]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid cell normalization")
        self.register_buffer("cell_feature_mean", mean.clone())
        self.register_buffer("cell_feature_std", std.clamp_min(1e-6).clone())
        self.forward_calls = 0
        self.last_stats: dict[str, float] = {}

    @staticmethod
    def raw_cell_features(lattice: torch.Tensor) -> torch.Tensor:
        if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3):
            raise ValueError(f"expected lattice [B,3,3], got {tuple(lattice.shape)}")
        # MatterGen stores lattice vectors as rows (cart = frac @ L).  LL^T is
        # therefore the rotation-invariant basis Gram under L -> L R.
        gram = lattice @ lattice.transpose(-1, -2)
        features = torch.stack(
            (gram[:, 0, 0], gram[:, 1, 1], gram[:, 2, 2],
             gram[:, 0, 1], gram[:, 0, 2], gram[:, 1, 2]),
            dim=-1,
        )
        return signed_log1p(features)

    def normalized_cell_features(self, lattice: torch.Tensor) -> torch.Tensor:
        raw = self.raw_cell_features(lattice)
        return (raw - self.cell_feature_mean.to(raw)) / self.cell_feature_std.to(raw)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError("normalization tensors must have shape [6]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("normalization must be finite with positive std")
        with torch.no_grad():
            self.cell_feature_mean.copy_(mean.to(self.cell_feature_mean))
            self.cell_feature_std.copy_(std.clamp_min(1e-6).to(self.cell_feature_std))

    def forward(
        self,
        *,
        h: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor,
        num_atoms: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        if int(block_index) != self.block_index:
            raise ValueError(f"fusion must run only after Block3, got index {block_index}")
        n_graphs = lattice.shape[0]
        if h.ndim != 2 or h.shape[1] != self.hidden_dim:
            raise ValueError(f"expected atom hidden [M,{self.hidden_dim}]")
        if batch.ndim != 1 or len(batch) != len(h) or int(num_atoms.sum()) != len(h):
            raise ValueError("atom/batch/num_atoms shape mismatch")
        if len(num_atoms) != n_graphs or (num_atoms <= 0).any():
            raise ValueError("graph/lattice/num_atoms shape mismatch")
        if int(batch.min()) != 0 or int(batch.max()) + 1 != n_graphs:
            raise ValueError("batch graph ids must be contiguous")

        cell_features = self.normalized_cell_features(lattice)
        if not torch.isfinite(cell_features).all():
            raise FloatingPointError("non-finite normalized cell descriptor")
        h_cell = self.cell_encoder(cell_features)
        h_atom_global = scatter(h, batch, dim=0, dim_size=n_graphs, reduce="mean")
        delta_cell = self.atom_to_cell(h_atom_global)
        h_cell_fused = h_cell + self.alpha_a2c * delta_cell
        delta_atom = self.cell_to_atom(h_cell_fused)
        residual = self.alpha_c2a * delta_atom[batch]
        output = h + residual
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite Cell-Atom fusion output")

        self.forward_calls += 1
        with torch.no_grad():
            h_rms = torch.sqrt(torch.mean(h.float().square())).clamp_min(1e-12)
            residual_rms = torch.sqrt(torch.mean(residual.float().square()))
            self.last_stats = {
                "alpha_a2c": float(self.alpha_a2c.detach().cpu()),
                "alpha_c2a": float(self.alpha_c2a.detach().cpu()),
                "delta_cell_rms": float(torch.sqrt(torch.mean(delta_cell.float().square())).cpu()),
                "delta_atom_rms": float(torch.sqrt(torch.mean(delta_atom.float().square())).cpu()),
                "residual_rms": float(residual_rms.cpu()),
                "hidden_rms": float(h_rms.cpu()),
                "residual_hidden_ratio": float((residual_rms / h_rms).cpu()),
                "cell_feature_abs_max": float(cell_features.abs().max().cpu()),
            }
        return output

    @property
    def audit_metadata(self) -> Mapping[str, int | str | bool]:
        return {
            "architecture": "cell_atom_bidirectional_fusion_v1",
            "hidden_dim": self.hidden_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "block_index_zero_based": self.block_index,
            "row_lattice_metric": "L_times_L_transpose",
            "time_gate": False,
            "mean_pooling": True,
            "alpha_zero_initialized": True,
        }
