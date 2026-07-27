from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import torch

from research.spg_static_mvp.common import RESULTS, atomic_json, now
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


MANIFEST = RESULTS / "equivalence/manifest.json"
OUTPUT = RESULTS / "equivalence/workers"
BUCKET = RESULTS / "selected_bucket.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def canonical_edge_set(
    edge_index: torch.Tensor,
    offsets: torch.Tensor,
) -> set[tuple[int, int, float, float, float]]:
    edges = edge_index.detach().cpu().transpose(0, 1).tolist()
    cells = offsets.detach().cpu().tolist()
    return {
        (
            int(edge[0]),
            int(edge[1]),
            float(offset[0]),
            float(offset[1]),
            float(offset[2]),
        )
        for edge, offset in zip(edges, cells, strict=True)
    }


def compare_state(gemnet, builder, state: dict) -> dict:
    device = next(gemnet.parameters()).device
    frac_positions = state["pos"].to(device)
    cell = state["cell"].to(device)
    num_atoms = state["num_atoms"].to(device)
    reference = build_reference_graph(
        gemnet,
        frac_positions=frac_positions,
        cell=cell,
        num_atoms=num_atoms,
    )
    result = builder.build(
        cart_positions=reference["cart_positions"],
        cell=cell,
        num_atoms=num_atoms,
        cutoff=float(gemnet.cutoff),
        max_neighbors=int(gemnet.max_neighbors),
        max_cell_images_per_dim=int(gemnet.max_cell_images_per_dim),
    )
    row = {
        "seed": int(state["seed"]),
        "state_index": int(state["state_index"]),
        "used_static": result.used_static,
        "fallback_reason": result.fallback_reason,
    }
    if not result.used_static:
        return row

    raw_edge, raw_offsets, raw_distances = result.raw_compact()
    static = result.compact_gemnet_tuple()
    (
        edge_index,
        neighbors,
        distances,
        vectors,
        id_swap,
        id3_ba,
        id3_ca,
        id3_ragged_idx,
        offsets,
    ) = static
    raw_order = bool(torch.equal(raw_edge, reference["raw_edge_index"]))
    raw_offset_order = bool(
        torch.equal(raw_offsets, reference["raw_cell_offsets"])
    )
    edge_order = bool(torch.equal(edge_index, reference["edge_index"]))
    offset_order = bool(torch.equal(offsets, reference["cell_offsets"]))
    row.update(
        {
            "raw_edge_order_match": raw_order,
            "raw_offset_order_match": raw_offset_order,
            "edge_order_match": edge_order,
            "edge_set_match": (
                edge_order and offset_order
                or canonical_edge_set(edge_index, offsets)
                == canonical_edge_set(
                    reference["edge_index"],
                    reference["cell_offsets"],
                )
            ),
            "offset_match": offset_order,
            "neighbors_match": bool(
                torch.equal(neighbors, reference["neighbors"])
            ),
            "distance_max_error": float(
                torch.max(torch.abs(distances - reference["edge_distances"]))
            )
            if distances.numel() == reference["edge_distances"].numel()
            else float("inf"),
            "vector_max_error": float(
                torch.max(torch.abs(vectors - reference["edge_vectors"]))
            )
            if vectors.numel() == reference["edge_vectors"].numel()
            else float("inf"),
            "id_swap_match": bool(
                torch.equal(id_swap, reference["id_swap"])
            ),
            "triplet_set_match": (
                set(zip(id3_ba.cpu().tolist(), id3_ca.cpu().tolist()))
                == set(
                    zip(
                        reference["id3_ba"].cpu().tolist(),
                        reference["id3_ca"].cpu().tolist(),
                    )
                )
            ),
            "triplet_order_match": bool(
                torch.equal(id3_ba, reference["id3_ba"])
                and torch.equal(id3_ca, reference["id3_ca"])
            ),
            "triplet_ragged_match": bool(
                torch.equal(
                    id3_ragged_idx,
                    reference["id3_ragged_idx"],
                )
            ),
            "raw_edge_count": int(raw_edge.shape[1]),
            "edge_count": int(edge_index.shape[1]),
            "triplet_count": int(id3_ba.numel()),
        }
    )
    return row


def main() -> int:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assigned = manifest[args.rank :: args.world_size]
    if args.limit is not None:
        assigned = assigned[: args.limit]
    configure_determinism()
    generator = build_c0_generator(sampling_steps=2)
    gemnet = find_gemnet(generator.model)
    device = next(gemnet.parameters()).device
    config = StaticBucketConfig.from_json(BUCKET)
    builder = StaticPeriodicGraphBuilder(config, device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    state_cache: dict[int, list[dict]] = {}
    first_failure = None
    try:
        for item in assigned:
            seed = int(item["seed"])
            if seed not in state_cache:
                state_cache[seed] = torch.load(
                    RESULTS / f"shape_states/seed_{seed}/states.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            state = state_cache[seed][int(item["state_index"])]
            row = compare_state(gemnet, builder, state)
            rows.append(row)
            required = (
                row.get("used_static")
                and row.get("raw_edge_order_match")
                and row.get("raw_offset_order_match")
                and row.get("edge_order_match")
                and row.get("edge_set_match")
                and row.get("offset_match")
                and row.get("neighbors_match")
                and row.get("id_swap_match")
                and row.get("triplet_set_match")
                and row.get("triplet_order_match")
                and row.get("triplet_ragged_match")
            )
            if not required and first_failure is None:
                first_failure = row
                torch.save(
                    state,
                    OUTPUT / f"minimal_repro_rank{args.rank}.pt",
                )
        summary = {
            "success": True,
            "rank": args.rank,
            "world_size": args.world_size,
            "states": len(rows),
            "static_states": sum(bool(row.get("used_static")) for row in rows),
            "fallback_states": sum(not bool(row.get("used_static")) for row in rows),
            "first_failure": first_failure,
            "rows": rows,
            "finished_at": now(),
        }
        atomic_json(OUTPUT / f"rank_{args.rank}.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    except BaseException:
        atomic_json(
            OUTPUT / f"rank_{args.rank}_error.json",
            {
                "success": False,
                "rank": args.rank,
                "error": traceback.format_exc(),
                "finished_at": now(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
