from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import pandas as pd
import torch

from research.spg_static_mvp.common import (
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
    set_stage,
)
from research.spg_static_mvp.generation import (
    build_c0_generator,
    configure_determinism,
    find_gemnet,
)
from research.spg_static_mvp.reference_graph import frac_to_cart
from research.spg_static_mvp.static_builder import (
    StaticBucketConfig,
    StaticPeriodicGraphBuilder,
)


WARMUP = 50
ITERATIONS = 300
ROUNDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="initial")
    return parser.parse_args()


def select_representative_states() -> list[dict]:
    bucket = json.loads(
        (RESULTS / "selected_bucket.json").read_text(encoding="utf-8")
    )
    frame = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(
                (RESULTS / "shape_states").glob("seed_*/shape_statistics.csv")
            )
        ],
        ignore_index=True,
    )
    mask = frame["num_atoms"].between(
        bucket["num_atoms_min"], bucket["num_atoms_max"]
    )
    mask &= (
        frame[["rep_a1", "rep_a2", "rep_a3"]]
        <= [bucket["max_rep_a1"], bucket["max_rep_a2"], bucket["max_rep_a3"]]
    ).all(axis=1)
    selected = frame.loc[mask].sort_values("triplet_count").reset_index(drop=True)
    indices = np.linspace(0, len(selected) - 1, 9, dtype=np.int64)
    state_cache = {}
    states = []
    for row in selected.iloc[indices].itertuples(index=False):
        seed = int(row.seed)
        if seed not in state_cache:
            state_cache[seed] = torch.load(
                RESULTS / f"shape_states/seed_{seed}/states.pt",
                map_location="cpu",
                weights_only=False,
            )
        state = state_cache[seed][int(row.state_index)]
        state["edge_count"] = int(row.edge_count)
        state["triplet_count"] = int(row.triplet_count)
        states.append(state)
    return states


def prepare_states(states: list[dict], device: torch.device) -> list[dict]:
    prepared = []
    for state in states:
        pos = state["pos"].to(device)
        cell = state["cell"].to(device)
        num_atoms = state["num_atoms"].to(device)
        cart = frac_to_cart(pos, cell, num_atoms)
        prepared.append(
            {
                "seed": int(state["seed"]),
                "state_index": int(state["state_index"]),
                "num_atoms_count": int(num_atoms[0]),
                "single_cart": cart,
                "single_cell": cell,
                "single_num_atoms": num_atoms,
                "joint_cart": torch.cat([cart, cart], dim=0),
                "joint_cell": torch.cat([cell, cell], dim=0),
                "joint_num_atoms": torch.cat([num_atoms, num_atoms], dim=0),
                "edge_count": int(state["edge_count"]),
                "triplet_count": int(state["triplet_count"]),
            }
        )
    return prepared


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def profile_call(function, state: dict) -> dict:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=True,
    ) as profiler:
        for _ in range(10):
            function(state)
    torch.cuda.synchronize()
    events = profiler.events()
    cuda_events = [event for event in events if str(event.device_type).endswith("CUDA")]
    allocator_names = ("aten::empty", "aten::empty_strided", "aten::resize_")
    allocator_calls = sum(
        1 for event in events if any(name in event.name for name in allocator_names)
    )
    synchronization_calls = sum(
        1
        for event in events
        if "Synchronize" in event.name or "_local_scalar_dense" in event.name
    )
    return {
        "kernel_count_per_call": len(cuda_events) / 10.0,
        "allocator_calls_per_call": allocator_calls / 10.0,
        "synchronization_calls_per_call": synchronization_calls / 10.0,
    }


def benchmark(function, states: list[dict]) -> dict:
    with torch.inference_mode():
        for index in range(WARMUP):
            function(states[index % len(states)])
        torch.cuda.synchronize()
        round_results = []
        all_cuda = []
        all_cpu = []
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        for _round in range(ROUNDS):
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERATIONS)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERATIONS)]
            cpu_submission = []
            wall_start = time.perf_counter()
            for index in range(ITERATIONS):
                state = states[index % len(states)]
                cpu_start = time.perf_counter()
                starts[index].record()
                function(state)
                ends[index].record()
                cpu_submission.append((time.perf_counter() - cpu_start) * 1000.0)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0 / ITERATIONS
            cuda_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
            all_cuda.extend(cuda_ms)
            all_cpu.extend(cpu_submission)
            round_results.append(
                {
                    "cuda": distribution(cuda_ms),
                    "cpu_submission": distribution(cpu_submission),
                    "wall_ms_per_call": wall_ms,
                }
            )
        peak_memory = torch.cuda.max_memory_allocated() - baseline_memory
        profile = profile_call(function, states[len(states) // 2])
    return {
        "warmup": WARMUP,
        "iterations_per_round": ITERATIONS,
        "rounds": ROUNDS,
        "cuda": distribution(all_cuda),
        "cpu_submission": distribution(all_cpu),
        "wall_ms_per_call": statistics.median(
            result["wall_ms_per_call"] for result in round_results
        ),
        "peak_incremental_vram_bytes": int(peak_memory),
        "round_results": round_results,
        **profile,
    }


def main() -> int:
    args = parse_args()
    set_stage(
        "builder_microbenchmark",
        "running",
        f"Running {args.label} 300x3 builder benchmark.",
    )
    configure_determinism()
    generator = build_c0_generator(sampling_steps=2)
    gemnet = find_gemnet(generator.model)
    device = next(gemnet.parameters()).device
    config = StaticBucketConfig.from_json(RESULTS / "selected_bucket.json")
    static_builder = StaticPeriodicGraphBuilder(config, device)
    states = prepare_states(select_representative_states(), device)

    def dynamic_joint(state):
        return gemnet.generate_interaction_graph(
            state["joint_cart"],
            state["joint_cell"],
            state["joint_num_atoms"],
            None,
            None,
            None,
        )

    def static_single(builder, state):
        result = builder.build(
            cart_positions=state["single_cart"],
            cell=state["single_cell"],
            num_atoms=state["single_num_atoms"],
            cutoff=float(gemnet.cutoff),
            max_neighbors=int(gemnet.max_neighbors),
            max_cell_images_per_dim=int(gemnet.max_cell_images_per_dim),
        )
        if not result.used_static:
            raise RuntimeError(result.fallback_reason)
        return result.compact_gemnet_tuple()

    def static_no_share(state):
        static_single(static_builder, state)
        single = static_single(static_builder, state)
        edge_count = single[0].shape[1]
        atom_count = state["num_atoms_count"]
        return (
            torch.cat([single[0], single[0] + atom_count], dim=1),
            torch.cat([single[1], single[1]], dim=0),
            torch.cat([single[2], single[2]], dim=0),
            torch.cat([single[3], single[3]], dim=0),
            torch.cat([single[4], single[4] + edge_count], dim=0),
            torch.cat([single[5], single[5] + edge_count], dim=0),
            torch.cat([single[6], single[6] + edge_count], dim=0),
            torch.cat([single[7], single[7]], dim=0),
            torch.cat([single[8], single[8]], dim=0),
        )

    def static_joint_share(state):
        result = static_builder.build(
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

    configurations = {
        "D0_dynamic_joint": dynamic_joint,
        "D1_static_no_share": static_no_share,
        "D2_static_joint_share": static_joint_share,
        "D3_static_persistent": static_joint_share,
    }
    results = {name: benchmark(function, states) for name, function in configurations.items()}
    dynamic_median = results["D0_dynamic_joint"]["cuda"]["median_ms"]
    static_median = results["D3_static_persistent"]["cuda"]["median_ms"]
    speedup = dynamic_median / static_median
    summary = {
        "completed_at": now(),
        "label": args.label,
        "representative_states": [
            {
                key: state[key]
                for key in ("seed", "state_index", "num_atoms_count", "edge_count", "triplet_count")
            }
            for state in states
        ],
        "results": results,
        "DYNAMIC_BUILDER_TIME_MS": dynamic_median,
        "STATIC_BUILDER_TIME_MS": static_median,
        "STATIC_BUILDER_SPEEDUP": speedup,
        "STATIC_BUILDER_PERFORMANCE_GO": speedup >= 2.25,
    }
    output = RESULTS / f"benchmarks/builder_{args.label}.json"
    atomic_json(output, summary)
    lines = [
        f"# SPG builder microbenchmark ({args.label})",
        "",
        f"- Representative states: `{len(states)}`",
        f"- Warmup: `{WARMUP}`",
        f"- Timed calls: `{ITERATIONS} x {ROUNDS}` per configuration",
        f"- Dynamic median: `{dynamic_median:.6f} ms`",
        f"- Static persistent median: `{static_median:.6f} ms`",
        f"- STATIC_BUILDER_SPEEDUP: `{speedup:.6f}x`",
        f"- Gate >=2.25x: `{speedup >= 2.25}`",
        "",
        "| Configuration | CUDA median ms | CUDA P95 ms | Wall ms/call | Kernels | Allocator calls | Sync calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in results.items():
        lines.append(
            f"| {name} | {metrics['cuda']['median_ms']:.6f} | "
            f"{metrics['cuda']['p95_ms']:.6f} | {metrics['wall_ms_per_call']:.6f} | "
            f"{metrics['kernel_count_per_call']:.1f} | {metrics['allocator_calls_per_call']:.1f} | "
            f"{metrics['synchronization_calls_per_call']:.1f} |"
        )
    atomic_text(
        REPORTS / f"builder_benchmark_{args.label}.md",
        "\n".join(lines) + "\n",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
