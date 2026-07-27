from __future__ import annotations

import json
import statistics
import time

import numpy as np
import torch

from research.spg_static_mvp.builder_benchmark import (
    select_representative_states,
)
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
    build_recording_sampler,
    configure_determinism,
    find_gemnet,
    singleton_condition,
)
from research.spg_static_mvp.numerical_equivalence import joint_batch
from research.spg_static_mvp.static_builder import (
    StaticBucketConfig,
    StaticPeriodicGraphBuilder,
    install_static_builder,
)


WARMUP = 50
ITERATIONS = 300
ROUNDS = 3
PROFILE_CALLS = 10
BLOCK_PROFILE_CALLS = 50
FIELDS = ("atomic_numbers", "pos", "cell")


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def prepare_states(
    states: list[dict], sampler, device: torch.device
) -> list[dict]:
    prepared = []
    for state in states:
        seed = int(state["seed"])
        batch = singleton_condition(seed).replace(
            pos=state["pos"],
            cell=state["cell"],
            atomic_numbers=state["atomic_numbers"],
            num_atoms=state["num_atoms"],
        ).to(device)
        timestep = torch.tensor([state["t"]], dtype=torch.float32, device=device)
        prepared.append(
            {
                "seed": seed,
                "state_index": int(state["state_index"]),
                "num_atoms_count": int(state["num_atoms"].sum()),
                "edge_count": int(state["edge_count"]),
                "triplet_count": int(state["triplet_count"]),
                "joint_batch": joint_batch(sampler, batch),
                "joint_timestep": torch.cat([timestep, timestep], dim=0),
            }
        )
    return prepared


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
        for _ in range(PROFILE_CALLS):
            function(state)
    torch.cuda.synchronize()
    events = profiler.events()
    cuda_events = [
        event for event in events if str(event.device_type).endswith("CUDA")
    ]
    allocator_names = ("aten::empty", "aten::empty_strided", "aten::resize_")
    allocator_calls = sum(
        1 for event in events if any(name in event.name for name in allocator_names)
    )
    synchronization_calls = sum(
        1
        for event in events
        if "Synchronize" in event.name or "_local_scalar_dense" in event.name
    )
    operator_rows = []
    scatter_us = 0.0
    for event in profiler.key_averages():
        self_device_us = float(
            getattr(
                event,
                "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
        )
        if self_device_us <= 0.0:
            continue
        operator_rows.append(
            {
                "name": event.key,
                "self_cuda_ms_per_call": self_device_us
                / PROFILE_CALLS
                / 1000.0,
            }
        )
        lowered = event.key.lower()
        if any(
            token in lowered
            for token in ("scatter", "index_add", "index_put", "segment")
        ):
            scatter_us += self_device_us
    operator_rows.sort(key=lambda row: row["self_cuda_ms_per_call"], reverse=True)
    return {
        "profile_calls": PROFILE_CALLS,
        "kernel_count_per_call": len(cuda_events) / PROFILE_CALLS,
        "allocator_calls_per_call": allocator_calls / PROFILE_CALLS,
        "synchronization_calls_per_call": synchronization_calls / PROFILE_CALLS,
        "scatter_self_cuda_ms_per_call": scatter_us / PROFILE_CALLS / 1000.0,
        "top_cuda_operators": operator_rows[:12],
    }


def profile_blocks(function, states: list[dict], gemnet) -> dict:
    starts: dict[int, torch.cuda.Event] = {}
    pairs: dict[int, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        index: [] for index in range(len(gemnet.int_blocks))
    }

    def pre_hook(index: int):
        def hook(_module, _inputs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            starts[index] = event

        return hook

    def post_hook(index: int):
        def hook(_module, _inputs, _output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            pairs[index].append((starts[index], end))

        return hook

    with torch.inference_mode():
        for index in range(5):
            function(states[index % len(states)])
        torch.cuda.synchronize()
        hooks = []
        for index, block in enumerate(gemnet.int_blocks):
            hooks.append(block.register_forward_pre_hook(pre_hook(index)))
            hooks.append(block.register_forward_hook(post_hook(index)))
        try:
            for index in range(BLOCK_PROFILE_CALLS):
                function(states[index % len(states)])
            torch.cuda.synchronize()
        finally:
            for hook in hooks:
                hook.remove()
    return {
        f"block_{index + 1}": distribution(
            [start.elapsed_time(end) for start, end in block_pairs]
        )
        for index, block_pairs in pairs.items()
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
    wall_median = statistics.median(
        result["wall_ms_per_call"] for result in round_results
    )
    cuda_summary = distribution(all_cuda)
    return {
        "warmup": WARMUP,
        "iterations_per_round": ITERATIONS,
        "rounds": ROUNDS,
        "cuda": cuda_summary,
        "cpu_submission": distribution(all_cpu),
        "wall_ms_per_call": wall_median,
        "cpu_launch_gap_ms": max(wall_median - cuda_summary["mean_ms"], 0.0),
        "gpu_active_proxy": min(cuda_summary["mean_ms"] / wall_median, 1.0),
        "peak_incremental_vram_bytes": int(peak_memory),
        "round_results": round_results,
        **profile,
    }


def main() -> int:
    set_stage(
        "forward_microbenchmark",
        "running",
        "Running complete joint-CFG score forward 300x3 benchmark.",
    )
    configure_determinism()
    generator = build_c0_generator(sampling_steps=1000)
    sampler = build_recording_sampler(generator, 0)
    diffusion_module = generator.model.diffusion_module
    gemnet = find_gemnet(generator.model)
    device = next(gemnet.parameters()).device
    states = prepare_states(select_representative_states(), sampler, device)
    builder = StaticPeriodicGraphBuilder(
        StaticBucketConfig.from_json(RESULTS / "selected_bucket.json"), device
    )

    def joint_score(state):
        combined = diffusion_module.score_fn(
            state["joint_batch"], state["joint_timestep"]
        )
        unconditional = combined[0]
        conditional = combined[1]
        return unconditional.replace(
            **{
                field: torch.lerp(
                    unconditional[field],
                    conditional[field],
                    sampler._guidance_scale,
                )
                for field in sampler._multi_corruption.corrupted_fields
            }
        )

    dynamic = benchmark(joint_score, states)
    dynamic_blocks = profile_blocks(joint_score, states, gemnet)
    static_counters: dict = {}
    with install_static_builder(gemnet, builder, static_counters):
        static = benchmark(joint_score, states)
        static_blocks = profile_blocks(joint_score, states, gemnet)

    with torch.inference_mode():
        combined = diffusion_module.score_fn(
            states[0]["joint_batch"], states[0]["joint_timestep"]
        )

    def cfg_mix(_state):
        unconditional = combined[0]
        conditional = combined[1]
        return unconditional.replace(
            **{
                field: torch.lerp(
                    unconditional[field],
                    conditional[field],
                    sampler._guidance_scale,
                )
                for field in FIELDS
            }
        )

    cfg_mix_result = benchmark(cfg_mix, states[:1])
    dynamic_median = dynamic["cuda"]["median_ms"]
    static_median = static["cuda"]["median_ms"]
    speedup = dynamic_median / static_median
    bucket = json.loads(
        (RESULTS / "selected_bucket.json").read_text(encoding="utf-8")
    )
    coverage = float(bucket["state_coverage"])

    def amdahl(bucket_coverage: float) -> float:
        return 1.0 / ((1.0 - bucket_coverage) + bucket_coverage / speedup)

    builder_result = json.loads(
        (RESULTS / "benchmarks/builder_optimized.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "completed_at": now(),
        "representative_states": [
            {
                key: state[key]
                for key in (
                    "seed",
                    "state_index",
                    "num_atoms_count",
                    "edge_count",
                    "triplet_count",
                )
            }
            for state in states
        ],
        "dynamic": dynamic,
        "static": static,
        "dynamic_blocks": dynamic_blocks,
        "static_blocks": static_blocks,
        "cfg_mix": cfg_mix_result,
        "static_counters": static_counters,
        "DYNAMIC_BUILDER_TIME_MS": builder_result["DYNAMIC_BUILDER_TIME_MS"],
        "STATIC_BUILDER_TIME_MS": builder_result["STATIC_BUILDER_TIME_MS"],
        "DYNAMIC_FORWARD_TIME_MS": dynamic_median,
        "STATIC_FORWARD_TIME_MS": static_median,
        "BUCKET_FULL_FORWARD_SPEEDUP": speedup,
        "BUCKET_FULL_FORWARD_PERFORMANCE_GO": speedup >= 1.08,
        "ESTIMATED_GLOBAL_SPEEDUP": amdahl(coverage),
        "ESTIMATED_TWO_BUCKET_SPEEDUP": amdahl(min(2.0 * coverage, 1.0)),
        "ESTIMATED_THREE_BUCKET_SPEEDUP": amdahl(min(3.0 * coverage, 1.0)),
        "amdahl_assumption": (
            "two/three equally sized buckets with identical per-bucket speedup "
            "and non-overlapping coverage"
        ),
    }
    atomic_json(RESULTS / "benchmarks/forward.json", summary)
    lines = [
        "# SPG complete joint-CFG forward microbenchmark",
        "",
        f"- Representative states: `{len(states)}`",
        f"- Warmup: `{WARMUP}`",
        f"- Timed calls: `{ITERATIONS} x {ROUNDS}` per configuration",
        f"- Dynamic CUDA median: `{dynamic_median:.6f} ms`",
        f"- Static CUDA median: `{static_median:.6f} ms`",
        f"- BUCKET_FULL_FORWARD_SPEEDUP: `{speedup:.6f}x`",
        f"- Gate >=1.08x: `{speedup >= 1.08}`",
        f"- Static calls observed: `{static_counters.get('static', 0)}`",
        f"- Static fallbacks observed: `{static_counters.get('fallback', 0)}`",
        "",
        "| Metric | Dynamic | Static |",
        "|---|---:|---:|",
        f"| CUDA median ms | {dynamic_median:.6f} | {static_median:.6f} |",
        f"| CUDA P95 ms | {dynamic['cuda']['p95_ms']:.6f} | {static['cuda']['p95_ms']:.6f} |",
        f"| Wall ms/call | {dynamic['wall_ms_per_call']:.6f} | {static['wall_ms_per_call']:.6f} |",
        f"| Kernel count/call | {dynamic['kernel_count_per_call']:.1f} | {static['kernel_count_per_call']:.1f} |",
        f"| Allocator calls/call | {dynamic['allocator_calls_per_call']:.1f} | {static['allocator_calls_per_call']:.1f} |",
        f"| Sync calls/call | {dynamic['synchronization_calls_per_call']:.1f} | {static['synchronization_calls_per_call']:.1f} |",
        f"| Scatter self CUDA ms/call | {dynamic['scatter_self_cuda_ms_per_call']:.6f} | {static['scatter_self_cuda_ms_per_call']:.6f} |",
        f"| GPU active proxy | {dynamic['gpu_active_proxy']:.2%} | {static['gpu_active_proxy']:.2%} |",
        f"| Peak incremental VRAM bytes | {dynamic['peak_incremental_vram_bytes']} | {static['peak_incremental_vram_bytes']} |",
        "",
        "## Amdahl estimates",
        "",
        f"- Current bucket ({coverage:.4%} coverage): `{summary['ESTIMATED_GLOBAL_SPEEDUP']:.6f}x`",
        f"- Two equal non-overlapping buckets: `{summary['ESTIMATED_TWO_BUCKET_SPEEDUP']:.6f}x`",
        f"- Three equal non-overlapping buckets: `{summary['ESTIMATED_THREE_BUCKET_SPEEDUP']:.6f}x`",
        "",
        "The multi-bucket estimates are theoretical upper bounds under identical",
        "per-bucket speedup and non-overlapping coverage assumptions.",
        "",
    ]
    atomic_text(REPORTS / "forward_benchmark.md", "\n".join(lines))
    set_stage(
        "forward_microbenchmark",
        "success",
        "Completed full joint-CFG forward 300x3 benchmark.",
        {
            "DYNAMIC_FORWARD_TIME_MS": dynamic_median,
            "STATIC_FORWARD_TIME_MS": static_median,
            "BUCKET_FULL_FORWARD_SPEEDUP": speedup,
            "BUCKET_FULL_FORWARD_PERFORMANCE_GO": speedup >= 1.08,
        },
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
