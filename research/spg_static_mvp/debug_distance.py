from __future__ import annotations

import torch

from mattergen.common.utils.data_utils import get_pbc_distances
from research.spg_static_mvp.common import RESULTS
from research.spg_static_mvp.generation import (
    build_c0_generator,
    configure_determinism,
    find_gemnet,
)
from research.spg_static_mvp.reference_graph import build_reference_graph
from research.spg_static_mvp.static_builder import (
    StaticBucketConfig,
    StaticPeriodicGraphBuilder,
)


def main() -> int:
    configure_determinism()
    state = torch.load(
        RESULTS / "shape_states/seed_24501/states.pt",
        map_location="cpu",
        weights_only=False,
    )[5]
    generator = build_c0_generator(sampling_steps=2)
    gemnet = find_gemnet(generator.model)
    device = next(gemnet.parameters()).device
    frac = state["pos"].to(device)
    cell = state["cell"].to(device)
    num_atoms = state["num_atoms"].to(device)
    reference = build_reference_graph(
        gemnet,
        frac_positions=frac,
        cell=cell,
        num_atoms=num_atoms,
    )
    builder = StaticPeriodicGraphBuilder(
        StaticBucketConfig.from_json(RESULTS / "selected_bucket.json"),
        device,
    )
    result = builder.build(
        cart_positions=reference["cart_positions"],
        cell=cell,
        num_atoms=num_atoms,
        cutoff=float(gemnet.cutoff),
        max_neighbors=int(gemnet.max_neighbors),
        max_cell_images_per_dim=int(gemnet.max_cell_images_per_dim),
    )
    raw_edge, raw_offsets, raw_distances = result.raw_compact()
    recomputed = get_pbc_distances(
        reference["cart_positions"],
        raw_edge,
        cell,
        raw_offsets,
        num_atoms,
        torch.tensor([raw_edge.shape[1]], device=device),
        coord_is_cart=True,
        return_offsets=True,
        return_distance_vec=True,
    )
    errors = torch.abs(raw_distances - reference["raw_distances"])
    index = int(torch.argmax(errors))
    print(
        {
            "max_raw_error": float(errors[index]),
            "index": index,
            "edge": raw_edge[:, index].cpu().tolist(),
            "offset": raw_offsets[index].cpu().tolist(),
            "static_distance": float(raw_distances[index]),
            "reference_distance": float(reference["raw_distances"][index]),
            "recomputed_distance": float(recomputed["distances"][index]),
            "static_vector": builder.raw_edge_vectors[index].cpu().tolist(),
            "reference_vector": reference["edge_vectors"][0].cpu().tolist(),
            "expected_cart_offset": (raw_offsets[index] @ cell[0]).cpu().tolist(),
            "implied_static_cart_offset": (-builder.raw_edge_vectors[index] * raw_distances[index] - (reference["cart_positions"][raw_edge[0,index]] - reference["cart_positions"][raw_edge[1,index]])).cpu().tolist(),
            "cell": cell[0].cpu().tolist(),
        }
    )
    candidate_match = (builder.candidate_source == raw_edge[0,index]) & (builder.candidate_target == raw_edge[1,index]) & (builder.candidate_offsets == raw_offsets[index]).all(dim=1)
    candidate_index = int(torch.nonzero(candidate_match)[0])
    print({"candidate_index": candidate_index, "stored_cart_offset": builder.candidate_cart_offsets[candidate_index].cpu().tolist(), "template_offset": builder.candidate_offsets[candidate_index].cpu().tolist()})
    print(
        {
            "recomputed_matches_reference": float(
                torch.max(
                    torch.abs(
                        recomputed["distances"] - reference["raw_distances"]
                    )
                )
            ),
            "static_matches_recomputed": float(
                torch.max(
                    torch.abs(recomputed["distances"] - raw_distances)
                )
            ),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
