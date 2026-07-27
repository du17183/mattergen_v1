from __future__ import annotations

import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from research.spg_static_mvp.common import read_json
from research.spg_static_mvp.reference_graph import cell_repetitions


@dataclass(frozen=True)
class StaticBucketConfig:
    num_atoms_min: int
    num_atoms_max: int
    max_rep_a1: int
    max_rep_a2: int
    max_rep_a3: int
    max_raw_edge_capacity: int
    max_edge_capacity: int
    max_triplet_capacity: int
    cutoff: float = 7.0
    max_neighbors: int = 50

    @classmethod
    def from_json(cls, path: str | Path) -> "StaticBucketConfig":
        data = read_json(Path(path))
        return cls(
            num_atoms_min=int(data["num_atoms_min"]),
            num_atoms_max=int(data["num_atoms_max"]),
            max_rep_a1=int(data["max_rep_a1"]),
            max_rep_a2=int(data["max_rep_a2"]),
            max_rep_a3=int(data["max_rep_a3"]),
            max_raw_edge_capacity=int(data["max_raw_edge_capacity"]),
            max_edge_capacity=int(data["max_edge_capacity"]),
            max_triplet_capacity=int(data["max_triplet_capacity"]),
        )

    @property
    def half_edge_capacity(self) -> int:
        if self.max_edge_capacity % 2:
            raise ValueError("symmetric edge capacity must be even")
        return self.max_edge_capacity // 2

    @property
    def periodic_image_capacity(self) -> int:
        return (
            (2 * self.max_rep_a1 + 1)
            * (2 * self.max_rep_a2 + 1)
            * (2 * self.max_rep_a3 + 1)
        )

    @property
    def candidate_capacity(self) -> int:
        return (
            self.num_atoms_max**2
            * self.periodic_image_capacity
        )


@dataclass
class StaticGraphResult:
    used_static: bool
    fallback_reason: str | None
    raw_count: int
    half_edge_count: int
    triplet_count: int
    builder: "StaticPeriodicGraphBuilder"
    batch_copies: int
    atoms_per_structure: int

    @property
    def edge_count(self) -> int:
        return 2 * self.half_edge_count

    def raw_compact(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        owner = self.builder
        count = self.raw_count
        return (
            owner.raw_edge_index[:, :count],
            owner.raw_cell_offsets[:count],
            owner.raw_edge_distances[:count],
        )

    def compact_gemnet_tuple(self):
        if not self.used_static:
            raise RuntimeError("fallback results have no static compact tuple")
        owner = self.builder
        half = self.half_edge_count
        second_start = owner.config.half_edge_capacity
        first = slice(0, half)
        second = slice(second_start, second_start + half)
        edge_index = torch.cat(
            [owner.edge_index[:, first], owner.edge_index[:, second]],
            dim=1,
        )
        cell_offsets = torch.cat(
            [owner.edge_cell_offsets[first], owner.edge_cell_offsets[second]],
            dim=0,
        )
        distances = torch.cat(
            [owner.edge_distances[first], owner.edge_distances[second]],
            dim=0,
        )
        vectors = torch.cat(
            [owner.edge_vectors[first], owner.edge_vectors[second]],
            dim=0,
        )
        neighbors = torch.tensor(
            [2 * half],
            dtype=torch.long,
            device=edge_index.device,
        )
        id_swap = torch.cat(
            [
                torch.arange(half, 2 * half, device=edge_index.device),
                torch.arange(0, half, device=edge_index.device),
            ]
        )
        count = self.triplet_count
        id3_ba = owner.id3_ba[:count]
        id3_ca = owner.id3_ca[:count]
        id3_ragged_idx = owner.id3_ragged_idx[:count]
        if self.batch_copies == 2:
            edge_count = edge_index.shape[1]
            edge_index = torch.cat(
                [edge_index, edge_index + self.atoms_per_structure], dim=1
            )
            cell_offsets = torch.cat([cell_offsets, cell_offsets], dim=0)
            distances = torch.cat([distances, distances], dim=0)
            vectors = torch.cat([vectors, vectors], dim=0)
            neighbors = torch.cat([neighbors, neighbors], dim=0)
            id_swap = torch.cat([id_swap, id_swap + edge_count], dim=0)
            id3_ba = torch.cat([id3_ba, id3_ba + edge_count], dim=0)
            id3_ca = torch.cat([id3_ca, id3_ca + edge_count], dim=0)
            id3_ragged_idx = torch.cat(
                [id3_ragged_idx, id3_ragged_idx], dim=0
            )
        return (
            edge_index,
            neighbors,
            distances,
            vectors,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            cell_offsets,
        )


class StaticPeriodicGraphBuilder(torch.nn.Module):
    """B1 fixed-capacity periodic graph workspace for one frozen shape bucket."""

    def __init__(self, config: StaticBucketConfig, device: torch.device | str):
        super().__init__()
        self.config = config
        device = torch.device(device)
        max_atoms = config.num_atoms_max
        offsets = torch.cartesian_prod(
            torch.arange(
                -config.max_rep_a1,
                config.max_rep_a1 + 1,
                dtype=torch.float32,
            ),
            torch.arange(
                -config.max_rep_a2,
                config.max_rep_a2 + 1,
                dtype=torch.float32,
            ),
            torch.arange(
                -config.max_rep_a3,
                config.max_rep_a3 + 1,
                dtype=torch.float32,
            ),
        )
        pair_target = torch.arange(max_atoms).repeat_interleave(max_atoms)
        pair_source = torch.arange(max_atoms).repeat(max_atoms)
        image_count = offsets.shape[0]
        candidate_target = pair_target.repeat_interleave(image_count)
        candidate_source = pair_source.repeat_interleave(image_count)
        candidate_offsets = offsets.repeat(max_atoms * max_atoms, 1)
        self.register_buffer("candidate_target", candidate_target.to(device))
        self.register_buffer("candidate_source", candidate_source.to(device))
        self.register_buffer("candidate_offsets", candidate_offsets.to(device))
        self.register_buffer("offset_template", offsets.to(device))
        self.register_buffer(
            "candidate_rank",
            torch.arange(config.candidate_capacity, device=device),
        )
        self.register_buffer(
            "raw_rank",
            torch.arange(config.max_raw_edge_capacity, device=device),
        )
        self.register_buffer(
            "packed_candidate_indices",
            torch.zeros(
                config.max_raw_edge_capacity,
                dtype=torch.long,
                device=device,
            ),
        )
        self.register_buffer(
            "edge_slot_rank",
            torch.arange(config.max_edge_capacity, device=device),
        )
        self.register_buffer(
            "representative_indices_buffer",
            torch.zeros(
                config.half_edge_capacity,
                dtype=torch.long,
                device=device,
            ),
        )
        for atoms in range(config.num_atoms_min, config.num_atoms_max + 1):
            mask = (candidate_target < atoms) & (candidate_source < atoms)
            self.register_buffer(f"atom_mask_{atoms}", mask.to(device))
            actual_target = torch.arange(atoms).repeat_interleave(atoms)
            actual_source = torch.arange(atoms).repeat(atoms)
            self.register_buffer(
                f"pair_target_{atoms}", actual_target.to(device)
            )
            self.register_buffer(
                f"pair_source_{atoms}", actual_source.to(device)
            )
            self.register_buffer(
                f"pair_slots_{atoms}",
                (actual_target * max_atoms + actual_source).to(device),
            )
        for rep_a1 in range(config.max_rep_a1 + 1):
            for rep_a2 in range(config.max_rep_a2 + 1):
                for rep_a3 in range(config.max_rep_a3 + 1):
                    offset_mask = (
                        (candidate_offsets[:, 0].abs() <= rep_a1)
                        & (candidate_offsets[:, 1].abs() <= rep_a2)
                        & (candidate_offsets[:, 2].abs() <= rep_a3)
                    )
                    self.register_buffer(
                        f"offset_mask_{rep_a1}_{rep_a2}_{rep_a3}",
                        offset_mask.to(device),
                    )
                    active_offset_mask = (
                        (offsets[:, 0].abs() <= rep_a1)
                        & (offsets[:, 1].abs() <= rep_a2)
                        & (offsets[:, 2].abs() <= rep_a3)
                    )
                    self.register_buffer(
                        f"offset_indices_{rep_a1}_{rep_a2}_{rep_a3}",
                        torch.nonzero(
                            active_offset_mask, as_tuple=False
                        ).flatten().to(device),
                    )

        candidate_capacity = config.candidate_capacity
        self.register_buffer(
            "padded_cart_positions",
            torch.zeros(max_atoms, 3, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "candidate_vectors",
            torch.zeros(candidate_capacity, 3, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "candidate_shifted_source",
            torch.zeros(candidate_capacity, 3, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "candidate_cart_offsets",
            torch.zeros(candidate_capacity, 3, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "offset_cart_workspace",
            torch.zeros(
                config.periodic_image_capacity,
                3,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "candidate_distance_squared",
            torch.zeros(candidate_capacity, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "candidate_valid",
            torch.zeros(candidate_capacity, dtype=torch.bool, device=device),
        )
        self.register_buffer(
            "candidate_selected",
            torch.zeros(candidate_capacity, dtype=torch.bool, device=device),
        )
        row_width = max_atoms * config.periodic_image_capacity
        self.register_buffer(
            "compact_neighbor_distances",
            torch.full(
                (max_atoms, row_width),
                torch.inf,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "compact_candidate_slots",
            torch.zeros(max_atoms, row_width, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "row_local_slots",
            torch.arange(row_width, device=device).view(1, -1),
        )

        raw_capacity = config.max_raw_edge_capacity
        self.register_buffer(
            "raw_edge_index",
            torch.zeros(2, raw_capacity, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "raw_cell_offsets",
            torch.zeros(raw_capacity, 3, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "raw_edge_distances",
            torch.zeros(raw_capacity, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "raw_edge_vectors",
            torch.zeros(raw_capacity, 3, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "raw_edge_mask",
            torch.zeros(raw_capacity, dtype=torch.bool, device=device),
        )

        edge_capacity = config.max_edge_capacity
        self.register_buffer(
            "edge_index",
            torch.zeros(2, edge_capacity, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "edge_cell_offsets",
            torch.zeros(edge_capacity, 3, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "edge_distances",
            torch.zeros(edge_capacity, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "edge_vectors",
            torch.zeros(edge_capacity, 3, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "edge_mask",
            torch.zeros(edge_capacity, dtype=torch.bool, device=device),
        )
        self.register_buffer(
            "edge_compact_index",
            torch.full((edge_capacity,), -1, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "triplet_candidate_mask",
            torch.zeros(
                edge_capacity,
                edge_capacity,
                dtype=torch.bool,
                device=device,
            ),
        )

        triplet_capacity = config.max_triplet_capacity
        self.register_buffer(
            "id3_ba",
            torch.zeros(triplet_capacity, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "id3_ca",
            torch.zeros(triplet_capacity, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "id3_ragged_idx",
            torch.zeros(triplet_capacity, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "triplet_mask",
            torch.zeros(triplet_capacity, dtype=torch.bool, device=device),
        )
        self.fallback_counts: dict[str, int] = {}
        self.static_calls = 0

    def _fallback(self, reason: str) -> StaticGraphResult:
        self.fallback_counts[reason] = self.fallback_counts.get(reason, 0) + 1
        return StaticGraphResult(
            used_static=False,
            fallback_reason=reason,
            raw_count=0,
            half_edge_count=0,
            triplet_count=0,
            builder=self,
            batch_copies=1,
            atoms_per_structure=0,
        )

    def _clear_outputs(self) -> None:
        self.raw_edge_mask.zero_()
        self.edge_mask.zero_()
        self.triplet_mask.zero_()
        self.edge_compact_index.fill_(-1)

    @torch.no_grad()
    def build(
        self,
        *,
        cart_positions: torch.Tensor,
        cell: torch.Tensor,
        num_atoms: torch.Tensor,
        cutoff: float,
        max_neighbors: int,
        max_cell_images_per_dim: int,
    ) -> StaticGraphResult:
        self._clear_outputs()
        batch_copies = 1
        geometry_cart_positions = cart_positions
        geometry_cell = cell
        if num_atoms.numel() == 2:
            atoms = int(num_atoms[0])
            duplicated_joint = (
                int(num_atoms[1]) == atoms
                and cart_positions.shape[0] == 2 * atoms
                and cell.shape[0] == 2
                and torch.equal(cart_positions[:atoms], cart_positions[atoms:])
                and torch.equal(cell[:1], cell[1:])
            )
            if not duplicated_joint:
                return self._fallback("batch_size_not_one")
            batch_copies = 2
            cart_positions = cart_positions[:atoms]
            cell = cell[:1]
            num_atoms = num_atoms[:1]
        elif num_atoms.numel() == 1:
            atoms = int(num_atoms[0])
        else:
            return self._fallback("batch_size_not_one")
        if not self.config.num_atoms_min <= atoms <= self.config.num_atoms_max:
            return self._fallback("num_atoms_outside_bucket")
        if (
            abs(float(cutoff) - self.config.cutoff) > 1e-12
            or int(max_neighbors) != self.config.max_neighbors
        ):
            return self._fallback("graph_parameters_changed")
        if not bool(torch.isfinite(cell).all().item()):
            return self._fallback("non_finite_cell")
        determinant = torch.linalg.det(cell[0])
        if not bool(torch.isfinite(determinant).item()) or abs(float(determinant)) < 1e-8:
            return self._fallback("singular_cell")
        repetitions = cell_repetitions(
            geometry_cell,
            cutoff,
            max_cell_images_per_dim,
        )
        if (
            repetitions[0] > self.config.max_rep_a1
            or repetitions[1] > self.config.max_rep_a2
            or repetitions[2] > self.config.max_rep_a3
        ):
            return self._fallback("periodic_images_outside_bucket")

        self.padded_cart_positions.zero_()
        self.padded_cart_positions[:atoms].copy_(cart_positions)
        active_offset_indices = getattr(
            self,
            f"offset_indices_{repetitions[0]}_{repetitions[1]}_{repetitions[2]}",
        )
        active_offsets = self.offset_template[active_offset_indices]
        active_offset_cart_batch = torch.bmm(
            torch.transpose(geometry_cell, 1, 2),
            active_offsets.transpose(0, 1).unsqueeze(0).expand(
                batch_copies, -1, -1
            ),
        ).transpose(1, 2)
        active_offset_cart = active_offset_cart_batch[0]
        self.offset_cart_workspace.zero_()
        self.offset_cart_workspace.index_copy_(
            0, active_offset_indices, active_offset_cart
        )
        pair_target = getattr(self, f"pair_target_{atoms}")
        pair_source = getattr(self, f"pair_source_{atoms}")
        pair_slots = getattr(self, f"pair_slots_{atoms}")
        num_cells = int(active_offset_indices.numel())
        if batch_copies == 2:
            pair_count = pair_target.numel()
            joint_pair_target = torch.cat(
                [pair_target, pair_target + atoms], dim=0
            )
            joint_pair_source = torch.cat(
                [pair_source, pair_source + atoms], dim=0
            )
            pos1 = geometry_cart_positions[joint_pair_target].view(
                -1, 3, 1
            ).expand(-1, -1, num_cells)
            pos2 = geometry_cart_positions[joint_pair_source].view(
                -1, 3, 1
            ).expand(-1, -1, num_cells)
            pbc_offsets_per_pair = torch.repeat_interleave(
                active_offset_cart_batch.transpose(1, 2),
                pair_count,
                dim=0,
            )
            joint_distances_squared = torch.sum(
                (pos1 - (pos2 + pbc_offsets_per_pair)) ** 2, dim=1
            ).reshape(batch_copies, -1)
            if not torch.equal(
                joint_distances_squared[0], joint_distances_squared[1]
            ):
                return self._fallback("joint_geometry_rounding_mismatch")
            exact_distances_squared = joint_distances_squared[0]
        else:
            pos1 = cart_positions[pair_target].view(-1, 3, 1).expand(
                -1, -1, num_cells
            )
            pos2 = cart_positions[pair_source].view(-1, 3, 1).expand(
                -1, -1, num_cells
            )
            pbc_offsets_per_pair = active_offset_cart.transpose(
                0, 1
            ).unsqueeze(0).repeat(pair_target.numel(), 1, 1)
            exact_distances_squared = torch.sum(
                (pos1 - (pos2 + pbc_offsets_per_pair)) ** 2, dim=1
            ).reshape(-1)
        candidate_slots = (
            pair_slots[:, None] * self.config.periodic_image_capacity
            + active_offset_indices[None, :]
        ).reshape(-1)
        self.candidate_distance_squared.fill_(torch.inf)
        self.candidate_distance_squared.index_copy_(
            0, candidate_slots, exact_distances_squared
        )
        atom_mask = getattr(self, f"atom_mask_{atoms}")
        offset_mask = getattr(
            self,
            f"offset_mask_{repetitions[0]}_{repetitions[1]}_{repetitions[2]}",
        )
        torch.logical_and(atom_mask, offset_mask, out=self.candidate_valid)
        self.candidate_valid.logical_and_(
            self.candidate_distance_squared <= cutoff * cutoff
        )
        self.candidate_valid.logical_and_(
            self.candidate_distance_squared > 0.0001
        )

        row_width = (
            self.config.num_atoms_max
            * self.config.periodic_image_capacity
        )
        dense_distances = torch.where(
            self.candidate_valid,
            self.candidate_distance_squared,
            torch.inf,
        ).view(self.config.num_atoms_max, row_width)
        sorted_distances, sorted_slots = torch.sort(
            dense_distances,
            dim=1,
            stable=True,
        )
        selected_slots = sorted_slots[:, : self.config.max_neighbors]
        if row_width > self.config.max_neighbors:
            boundary = self.config.max_neighbors
            crossing_tie = (
                torch.isfinite(sorted_distances[:, boundary])
                & (
                    sorted_distances[:, boundary - 1]
                    == sorted_distances[:, boundary]
                )
            )
            if bool(crossing_tie.any().item()):
                # The original OCP code compacts each target row before an
                # unstable CUDA sort. Reproduce that exact dynamic-width sort
                # only for boundary ties; all outputs remain fixed-capacity.
                valid_rows = self.candidate_valid.view(
                    self.config.num_atoms_max, row_width
                )
                compact_positions = torch.cumsum(
                    valid_rows.to(torch.long), dim=1
                ) - 1
                safe_positions = torch.where(
                    valid_rows,
                    compact_positions,
                    row_width - 1,
                )
                self.compact_neighbor_distances.fill_(torch.inf)
                self.compact_neighbor_distances.scatter_(
                    1, safe_positions, dense_distances
                )
                self.compact_candidate_slots.zero_()
                local_slots = self.row_local_slots.expand_as(valid_rows)
                self.compact_candidate_slots.scatter_(
                    1, safe_positions, local_slots
                )
                max_raw_neighbors = int(valid_rows.sum(dim=1).max())
                exact_distances, exact_compact_slots = torch.sort(
                    self.compact_neighbor_distances[
                        :atoms, :max_raw_neighbors
                    ],
                    dim=1,
                    stable=False,
                )
                exact_local_slots = self.compact_candidate_slots[
                    :atoms, :max_raw_neighbors
                ].gather(1, exact_compact_slots)
                selected_slots[:atoms] = exact_local_slots[
                    :, : self.config.max_neighbors
                ]
        row_offsets = (
            torch.arange(
                self.config.num_atoms_max,
                device=cart_positions.device,
            )[:, None]
            * row_width
        )
        selected_flat = (selected_slots + row_offsets).reshape(-1)
        selected_valid = selected_flat[self.candidate_valid[selected_flat]]
        packed_valid_indices = torch.sort(selected_valid).values
        raw_count = packed_valid_indices.numel()
        if raw_count > self.config.max_raw_edge_capacity:
            return self._fallback("raw_edge_capacity")
        self.packed_candidate_indices.zero_()
        self.packed_candidate_indices[:raw_count].copy_(packed_valid_indices)
        packed_candidate_indices = self.packed_candidate_indices
        self.raw_edge_index[0].copy_(
            self.candidate_source[packed_candidate_indices]
        )
        self.raw_edge_index[1].copy_(
            self.candidate_target[packed_candidate_indices]
        )
        self.raw_cell_offsets.copy_(
            self.candidate_offsets[packed_candidate_indices]
        )
        raw_distance_vectors = (
            self.padded_cart_positions[
                self.candidate_source[packed_candidate_indices]
            ]
            - self.padded_cart_positions[
                self.candidate_target[packed_candidate_indices]
            ]
        )
        lattice_edges = cell.expand(self.config.max_raw_edge_capacity, -1, -1)
        output_cart_offsets = torch.einsum(
            "bi,bij->bj",
            self.raw_cell_offsets.float(),
            lattice_edges,
        )
        raw_distance_vectors.add_(output_cart_offsets)
        self.raw_edge_distances.copy_(
            torch.linalg.vector_norm(raw_distance_vectors, dim=-1)
        )
        self.raw_edge_vectors.copy_(
            -raw_distance_vectors
            / self.raw_edge_distances[:, None].clamp_min(1e-12)
        )
        self.raw_edge_mask.copy_(self.raw_rank < raw_count)

        source = self.raw_edge_index[0]
        target = self.raw_edge_index[1]
        offsets = self.raw_cell_offsets
        earlier = (
            (offsets[:, 0] < 0)
            | ((offsets[:, 0] == 0) & (offsets[:, 1] < 0))
            | (
                (offsets[:, 0] == 0)
                & (offsets[:, 1] == 0)
                & (offsets[:, 2] < 0)
            )
        )
        representative = self.raw_edge_mask & (
            (source < target) | ((source == target) & earlier)
        )
        half_count = int(representative.sum())
        half_capacity = self.config.half_edge_capacity
        if half_count > half_capacity:
            return self._fallback("edge_capacity")
        valid_representative_indices = torch.nonzero(
            representative, as_tuple=False
        ).flatten()
        self.representative_indices_buffer.zero_()
        self.representative_indices_buffer[:half_count].copy_(
            valid_representative_indices
        )
        representative_indices = self.representative_indices_buffer
        first = slice(0, half_capacity)
        second = slice(half_capacity, 2 * half_capacity)
        rep_source = source[representative_indices]
        rep_target = target[representative_indices]
        self.edge_index[0, first].copy_(rep_source)
        self.edge_index[1, first].copy_(rep_target)
        self.edge_index[0, second].copy_(rep_target)
        self.edge_index[1, second].copy_(rep_source)
        self.edge_cell_offsets[first].copy_(
            self.raw_cell_offsets[representative_indices]
        )
        self.edge_cell_offsets[second].copy_(
            -self.raw_cell_offsets[representative_indices]
        )
        self.edge_distances[first].copy_(
            self.raw_edge_distances[representative_indices]
        )
        self.edge_distances[second].copy_(
            self.raw_edge_distances[representative_indices]
        )
        self.edge_vectors[first].copy_(
            self.raw_edge_vectors[representative_indices]
        )
        self.edge_vectors[second].copy_(
            -self.raw_edge_vectors[representative_indices]
        )
        valid_half = self.raw_rank[:half_capacity] < half_count
        self.edge_mask[first].copy_(valid_half)
        self.edge_mask[second].copy_(valid_half)
        self.edge_compact_index[:half_count] = torch.arange(
            half_count,
            device=cart_positions.device,
        )
        self.edge_compact_index[
            half_capacity : half_capacity + half_count
        ] = torch.arange(
            half_count,
            2 * half_count,
            device=cart_positions.device,
        )

        target_slots = self.edge_index[1]
        triplet_candidates = (
            self.edge_mask[:, None]
            & self.edge_mask[None, :]
            & (target_slots[:, None] == target_slots[None, :])
        )
        triplet_candidates.fill_diagonal_(False)
        self.triplet_candidate_mask.copy_(triplet_candidates)
        triplet_count = int(triplet_candidates.sum())
        if triplet_count > self.config.max_triplet_capacity:
            return self._fallback("triplet_capacity")
        flat_triplet_slots = torch.nonzero(
            triplet_candidates.reshape(-1),
            as_tuple=False,
        ).flatten()
        ca_slots = torch.div(
            flat_triplet_slots,
            self.config.max_edge_capacity,
            rounding_mode="floor",
        )
        ba_slots = flat_triplet_slots % self.config.max_edge_capacity
        ragged = torch.cumsum(
            triplet_candidates.to(torch.long),
            dim=1,
        ) - 1
        self.id3_ba[:triplet_count].copy_(
            self.edge_compact_index[ba_slots]
        )
        self.id3_ca[:triplet_count].copy_(
            self.edge_compact_index[ca_slots]
        )
        self.id3_ragged_idx[:triplet_count].copy_(
            ragged[ca_slots, ba_slots]
        )
        self.triplet_mask[:triplet_count] = True
        self.static_calls += 1
        return StaticGraphResult(
            used_static=True,
            fallback_reason=None,
            raw_count=raw_count,
            half_edge_count=half_count,
            triplet_count=triplet_count,
            builder=self,
            batch_copies=batch_copies,
            atoms_per_structure=atoms,
        )


@contextmanager
def install_static_builder(
    gemnet,
    builder: StaticPeriodicGraphBuilder,
    counters: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Temporarily route B1 on-the-fly graphs through the static bucket."""

    counters = counters if counters is not None else {}
    counters.setdefault("calls", 0)
    counters.setdefault("static", 0)
    counters.setdefault("fallback", 0)
    counters.setdefault("fallback_reasons", {})
    original = gemnet.generate_interaction_graph

    def wrapped(
        self,
        cart_coords,
        lattice,
        num_atoms,
        edge_index,
        to_jimages,
        num_bonds,
    ):
        counters["calls"] += 1
        if not self.otf_graph or any(
            value is not None for value in (edge_index, to_jimages, num_bonds)
        ):
            counters["fallback"] += 1
            reason = "non_otf_or_explicit_graph"
            counters["fallback_reasons"][reason] = (
                counters["fallback_reasons"].get(reason, 0) + 1
            )
            return original(
                cart_coords,
                lattice,
                num_atoms,
                edge_index,
                to_jimages,
                num_bonds,
            )
        result = builder.build(
            cart_positions=cart_coords,
            cell=lattice,
            num_atoms=num_atoms,
            cutoff=float(self.cutoff),
            max_neighbors=int(self.max_neighbors),
            max_cell_images_per_dim=int(self.max_cell_images_per_dim),
        )
        if not result.used_static:
            counters["fallback"] += 1
            reason = result.fallback_reason or "unknown"
            counters["fallback_reasons"][reason] = (
                counters["fallback_reasons"].get(reason, 0) + 1
            )
            return original(
                cart_coords,
                lattice,
                num_atoms,
                edge_index,
                to_jimages,
                num_bonds,
            )
        counters["static"] += 1
        return result.compact_gemnet_tuple()

    gemnet.generate_interaction_graph = types.MethodType(wrapped, gemnet)
    try:
        yield counters
    finally:
        gemnet.generate_interaction_graph = original


__all__ = [
    "StaticBucketConfig",
    "StaticGraphResult",
    "StaticPeriodicGraphBuilder",
    "install_static_builder",
]
