from __future__ import annotations

import json

import torch

from research.spg_static_mvp.builder_benchmark import (
    prepare_states,
    select_representative_states,
)
from research.spg_static_mvp.common import LOGS, RESULTS, atomic_text
from research.spg_static_mvp.generation import (
    build_c0_generator,
    configure_determinism,
    find_gemnet,
)
from research.spg_static_mvp.static_builder import (
    StaticBucketConfig,
    StaticPeriodicGraphBuilder,
)


def main() -> int:
    configure_determinism()
    generator = build_c0_generator(sampling_steps=2)
    gemnet = find_gemnet(generator.model)
    device = next(gemnet.parameters()).device
    builder = StaticPeriodicGraphBuilder(
        StaticBucketConfig.from_json(RESULTS / "selected_bucket.json"), device
    )
    state = prepare_states(select_representative_states(), device)[4]

    def call():
        result = builder.build(
            cart_positions=state["joint_cart"],
            cell=state["joint_cell"],
            num_atoms=state["joint_num_atoms"],
            cutoff=float(gemnet.cutoff),
            max_neighbors=int(gemnet.max_neighbors),
            max_cell_images_per_dim=int(gemnet.max_cell_images_per_dim),
        )
        if not result.used_static:
            raise RuntimeError(result.fallback_reason)
        return result.compact_gemnet_tuple()

    with torch.inference_mode():
        for _ in range(50):
            call()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            for _ in range(50):
                call()
        torch.cuda.synchronize()
    table = profiler.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )
    output = LOGS / "builder_hotspots_initial.txt"
    atomic_text(output, table + "\n")
    print(json.dumps({"profile": str(output), "state": {
        "seed": state["seed"],
        "state_index": state["state_index"],
        "edge_count": state["edge_count"],
        "triplet_count": state["triplet_count"],
    }}, indent=2))
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
