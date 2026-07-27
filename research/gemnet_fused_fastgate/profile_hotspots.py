from __future__ import annotations

import json
import statistics
import types
from collections import defaultdict

import numpy as np
import torch

from mattergen.common.gemnet.gemnet import RBFBasedLatticeUpdateBlock
from mattergen.common.gemnet.layers.atom_update_block import AtomUpdateBlock, OutputBlock
from mattergen.common.gemnet.layers.interaction_block import TripletInteraction
from mattergen.common.gemnet.layers.radial_basis import RadialBasis
from research.gemnet_fused_fastgate.common import (
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
    set_stage,
)
from research.gemnet_fused_fastgate.harness import (
    build_c0_generator,
    build_sampler,
    configure_determinism,
    find_gemnet,
    load_states,
    prepare_joint_states,
    run_joint_score,
)


WARMUP = 20
PROFILE_CALLS = 30
STATE_COUNT = 12


def tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(child) for child in value)
    if isinstance(value, dict):
        return sum(tensor_bytes(child) for child in value.values())
    return 0


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def module_group(name: str, module: torch.nn.Module) -> str | None:
    if type(module) is OutputBlock:
        return "K2_output_update"
    if type(module) is AtomUpdateBlock:
        return "K2_atom_update"
    if type(module) is TripletInteraction:
        return "triplet_interaction_context"
    if type(module) is RadialBasis:
        return "K1_radial_basis"
    if isinstance(module, RBFBasedLatticeUpdateBlock):
        return "K3_lattice_score_head"
    if name in {"out_mlp_E", "out_energy"}:
        return "K3_atomic_score_head"
    return None


def main() -> int:
    set_stage("hotspot_profile", "running", "Profiling original FP32 C0-B1 joint-CFG forward.")
    configure_determinism()
    generator = build_c0_generator(sampling_steps=1000)
    sampler = build_sampler(generator)
    diffusion_module = generator.model.diffusion_module
    gemnet = find_gemnet(generator.model)
    device = next(gemnet.parameters()).device
    states = prepare_joint_states(load_states(STATE_COUNT), sampler, device)

    def call(state):
        return run_joint_score(diffusion_module, sampler, state)

    with torch.inference_mode():
        for index in range(WARMUP):
            call(states[index % len(states)])
        torch.cuda.synchronize()

    start_events: dict[int, torch.cuda.Event] = {}
    group_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
    group_input_bytes: dict[str, list[int]] = defaultdict(list)
    group_output_bytes: dict[str, list[int]] = defaultdict(list)
    source_modules: dict[str, set[str]] = defaultdict(set)
    hooks = []

    for name, module in gemnet.named_modules():
        group = module_group(name, module)
        if group is None:
            continue
        source_modules[group].add(f"{name}:{module.__class__.__name__}")

        def pre_hook(_module, inputs, *, key=id(module), group_name=group):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            start_events[key] = event
            group_input_bytes[group_name].append(tensor_bytes(inputs))

        def post_hook(_module, _inputs, output, *, key=id(module), group_name=group):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            group_pairs[group_name].append((start_events[key], end))
            group_output_bytes[group_name].append(tensor_bytes(output))

        hooks.append(module.register_forward_pre_hook(pre_hook))
        hooks.append(module.register_forward_hook(post_hook))

    original_graph = gemnet.generate_interaction_graph
    graph_pairs = []
    graph_input_bytes = []
    graph_output_bytes = []

    def graph_wrapper(self, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = original_graph(*args, **kwargs)
        end.record()
        graph_pairs.append((start, end))
        graph_input_bytes.append(tensor_bytes(args) + tensor_bytes(kwargs))
        graph_output_bytes.append(tensor_bytes(output))
        return output

    gemnet.generate_interaction_graph = types.MethodType(graph_wrapper, gemnet)
    forward_pairs = []
    try:
        with torch.inference_mode():
            for index in range(PROFILE_CALLS):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                call(states[index % len(states)])
                end.record()
                forward_pairs.append((start, end))
            torch.cuda.synchronize()
    finally:
        gemnet.generate_interaction_graph = original_graph
        for hook in hooks:
            hook.remove()

    module_metrics = {}
    for group, pairs in group_pairs.items():
        elapsed = [start.elapsed_time(end) for start, end in pairs]
        module_metrics[group] = {
            **summarize(elapsed),
            "call_count": len(elapsed),
            "calls_per_forward": len(elapsed) / PROFILE_CALLS,
            "input_bytes_per_call": statistics.mean(group_input_bytes[group]),
            "output_bytes_per_call": statistics.mean(group_output_bytes[group]),
            "modules": sorted(source_modules[group]),
        }
    graph_elapsed = [start.elapsed_time(end) for start, end in graph_pairs]
    module_metrics["K1_graph_geometry"] = {
        **summarize(graph_elapsed),
        "call_count": len(graph_elapsed),
        "calls_per_forward": len(graph_elapsed) / PROFILE_CALLS,
        "input_bytes_per_call": statistics.mean(graph_input_bytes),
        "output_bytes_per_call": statistics.mean(graph_output_bytes),
        "modules": ["mattergen/common/gemnet/gemnet.py:generate_interaction_graph"],
    }
    forward_elapsed = [start.elapsed_time(end) for start, end in forward_pairs]
    forward = summarize(forward_elapsed)

    with torch.inference_mode(), torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        for index in range(PROFILE_CALLS):
            call(states[index % len(states)])
    torch.cuda.synchronize()

    operator_rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        if not event.key.startswith("aten::"):
            continue
        cuda_us = float(getattr(event, "self_device_time_total", 0.0))
        if cuda_us <= 0.0:
            continue
        operator_rows.append(
            {
                "operator": event.key,
                "self_cuda_total_ms": cuda_us / 1000.0,
                "call_count": int(event.count),
                "self_cuda_mean_us": cuda_us / max(int(event.count), 1),
                "input_shapes": str(event.input_shapes),
                "self_device_memory_bytes": int(
                    getattr(event, "self_device_memory_usage", 0)
                ),
            }
        )
    operator_rows.sort(key=lambda row: row["self_cuda_total_ms"], reverse=True)
    top20 = operator_rows[:20]

    total_forward_ms = sum(forward_elapsed)
    k1_ms = sum(start.elapsed_time(end) for start, end in graph_pairs)
    k1_ms += sum(
        start.elapsed_time(end)
        for start, end in group_pairs.get("K1_radial_basis", [])
    )
    k2_ms = sum(
        start.elapsed_time(end)
        for group in ("K2_atom_update", "K2_output_update")
        for start, end in group_pairs.get(group, [])
    )
    k3_ms = sum(
        start.elapsed_time(end)
        for group in ("K3_lattice_score_head", "K3_atomic_score_head")
        for start, end in group_pairs.get(group, [])
    )
    chain_shares = {
        "K1": k1_ms / total_forward_ms,
        "K2": k2_ms / total_forward_ms,
        "K3": k3_ms / total_forward_ms,
    }
    summary = {
        "completed_at": now(),
        "model": "C0-B1 FP32 constant joint CFG, full predictor/corrector semantics",
        "states": STATE_COUNT,
        "warmup": WARMUP,
        "profile_calls": PROFILE_CALLS,
        "forward": forward,
        "chain_shares": chain_shares,
        "module_metrics": module_metrics,
        "top20_cuda_operators": top20,
        "state_coverage": [
            {
                key: state[key]
                for key in (
                    "seed",
                    "state_index",
                    "sampling_step",
                    "phase",
                    "num_atoms",
                    "edge_count",
                    "triplet_count",
                )
            }
            for state in states
        ],
    }
    atomic_json(RESULTS / "profile/hotspot_profile.json", summary)
    lines = [
        "# Original C0-B1 joint-CFG hotspot profile",
        "",
        f"- States: `{STATE_COUNT}`",
        f"- Warmup/profile calls: `{WARMUP}` / `{PROFILE_CALLS}`",
        f"- Forward CUDA median: `{forward['median_ms']:.6f} ms`",
        f"- K1 inclusive share: `{chain_shares['K1']:.3%}`",
        f"- K2 inclusive share: `{chain_shares['K2']:.3%}`",
        f"- K3 inclusive share: `{chain_shares['K3']:.3%}`",
        "",
        "## Top 20 CUDA operators",
        "",
        "| Rank | Operator | CUDA total ms | Calls | Mean us | Input shapes | Device memory bytes |",
        "|---:|---|---:|---:|---:|---|---:|",
    ]
    for index, row in enumerate(top20, 1):
        lines.append(
            f"| {index} | `{row['operator']}` | {row['self_cuda_total_ms']:.6f} | "
            f"{row['call_count']} | {row['self_cuda_mean_us']:.3f} | "
            f"`{row['input_shapes']}` | {row['self_device_memory_bytes']} |"
        )
    lines.extend(["", "## Module/source mapping", ""])
    for group, metrics in sorted(module_metrics.items()):
        lines.append(
            f"- `{group}`: {metrics['call_count']} calls, "
            f"{metrics['mean_ms']:.6f} ms/call, inputs "
            f"{metrics['input_bytes_per_call']:.0f} B, outputs "
            f"{metrics['output_bytes_per_call']:.0f} B; "
            f"modules: {', '.join(metrics['modules'])}"
        )
    lines.extend(
        [
            "",
            "K1/K2/K3 shares use disjoint CUDA-event module boundaries. K2 covers",
            "the repeated dense→gate→scatter→atom/output residual-update family;",
            "triplet interaction is reported separately and is not added to K2.",
            "",
        ]
    )
    atomic_text(REPORTS / "hotspot_profile.md", "\n".join(lines))
    set_stage(
        "hotspot_profile",
        "success",
        "Completed original joint-CFG PyTorch Profiler and CUDA-event audit.",
        {"forward": forward, "chain_shares": chain_shares, "top20": top20},
    )
    print(json.dumps({"forward": forward, "chain_shares": chain_shares}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
