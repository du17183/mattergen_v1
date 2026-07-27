from __future__ import annotations

import torch

from research.spg_static_mvp.common import RESULTS
from research.spg_static_mvp.generation import (
    build_c0_generator,
    configure_determinism,
    find_gemnet,
)
from research.spg_static_mvp.reference_graph import build_reference_graph


def candidate_triplets(edge_index: torch.Tensor, *, sort_by_source: bool):
    edge_count = edge_index.shape[1]
    idx_s, idx_t = edge_index
    rows_ba = []
    rows_ca = []
    for ca in range(edge_count):
        ba = torch.nonzero(idx_t == idx_t[ca], as_tuple=False).flatten()
        ba = ba[ba != ca]
        if sort_by_source:
            key = idx_s[ba] * (edge_count + 1) + ba
            ba = ba[torch.argsort(key)]
        rows_ba.append(ba)
        rows_ca.append(torch.full_like(ba, ca))
    return torch.cat(rows_ba), torch.cat(rows_ca)


def main() -> int:
    configure_determinism()
    states = torch.load(
        RESULTS / "shape_states/seed_24515/states.pt",
        map_location="cpu",
        weights_only=False,
    )
    generator = build_c0_generator(sampling_steps=2)
    gemnet = find_gemnet(generator.model)
    device = next(gemnet.parameters()).device
    state = states[0]
    graph = build_reference_graph(
        gemnet,
        frac_positions=state["pos"].to(device),
        cell=state["cell"].to(device),
        num_atoms=state["num_atoms"].to(device),
    )
    for sort_by_source in (False, True):
        ba, ca = candidate_triplets(
            graph["edge_index"],
            sort_by_source=sort_by_source,
        )
        print(
            {
                "sort_by_source": sort_by_source,
                "ba_match": bool(torch.equal(ba, graph["id3_ba"])),
                "ca_match": bool(torch.equal(ca, graph["id3_ca"])),
                "first_ba_mismatch": int(
                    torch.nonzero(ba != graph["id3_ba"])[0]
                )
                if not torch.equal(ba, graph["id3_ba"])
                else None,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
