from __future__ import annotations

from typing import Any

import torch

from mattergen.common.gemnet.utils import repeat_blocks
from mattergen.common.utils.data_utils import (
    get_pbc_distances,
    radius_graph_pbc,
)


def cell_repetitions(
    cell: torch.Tensor,
    radius: float,
    max_cell_images_per_dim: int,
) -> tuple[int, int, int]:
    """Return the exact repetitions used by the B1 all-periodic reference builder."""

    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    volume = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
    cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
    inverse_plane_distances = (
        torch.linalg.vector_norm(cross_a2a3 / volume, dim=-1),
        torch.linalg.vector_norm(cross_a3a1 / volume, dim=-1),
        torch.linalg.vector_norm(cross_a1a2 / volume, dim=-1),
    )
    return tuple(
        min(
            int(torch.ceil(radius * inverse_distance).max()),
            max_cell_images_per_dim,
        )
        for inverse_distance in inverse_plane_distances
    )


def frac_to_cart(
    frac_positions: torch.Tensor,
    cell: torch.Tensor,
    num_atoms: torch.Tensor,
) -> torch.Tensor:
    lattice_nodes = torch.repeat_interleave(cell, num_atoms, dim=0)
    return torch.einsum("bi,bij->bj", frac_positions, lattice_nodes)


def build_reference_graph(
    gemnet,
    *,
    frac_positions: torch.Tensor,
    cell: torch.Tensor,
    num_atoms: torch.Tensor,
) -> dict[str, Any]:
    """Run the exact current radius/reorder/triplet path and expose all intermediates."""

    cart_positions = frac_to_cart(frac_positions, cell, num_atoms)
    raw_edge_index, raw_cell_offsets, raw_neighbors = radius_graph_pbc(
        cart_coords=cart_positions,
        lattice=cell,
        num_atoms=num_atoms,
        radius=gemnet.cutoff,
        max_num_neighbors_threshold=gemnet.max_neighbors,
        max_cell_images_per_dim=gemnet.max_cell_images_per_dim,
    )
    distances = get_pbc_distances(
        cart_positions,
        raw_edge_index,
        cell,
        raw_cell_offsets,
        num_atoms,
        raw_neighbors,
        coord_is_cart=True,
        return_offsets=True,
        return_distance_vec=True,
    )
    raw_distances = distances["distances"]
    raw_vectors = -distances["distance_vec"] / raw_distances[:, None]
    (
        edge_index,
        cell_offsets,
        neighbors,
        edge_distances,
        edge_vectors,
    ) = gemnet.reorder_symmetric_edges(
        distances["edge_index"],
        raw_cell_offsets,
        raw_neighbors,
        raw_distances,
        raw_vectors,
    )
    block_sizes = neighbors // 2
    block_sizes = torch.masked_select(block_sizes, block_sizes > 0)
    if block_sizes.numel() == 0:
        id_swap = edge_index.new_empty(0)
    else:
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )
    id3_ba, id3_ca, id3_ragged_idx = gemnet.get_triplets(
        edge_index,
        num_atoms=num_atoms.sum(),
    )
    return {
        "cart_positions": cart_positions,
        "raw_edge_index": raw_edge_index,
        "raw_cell_offsets": raw_cell_offsets,
        "raw_neighbors": raw_neighbors,
        "raw_distances": raw_distances,
        "edge_index": edge_index,
        "cell_offsets": cell_offsets,
        "neighbors": neighbors,
        "edge_distances": edge_distances,
        "edge_vectors": edge_vectors,
        "id_swap": id_swap,
        "id3_ba": id3_ba,
        "id3_ca": id3_ca,
        "id3_ragged_idx": id3_ragged_idx,
    }
