from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch_scatter import scatter

from mattergen.common.gemnet.layers.atom_update_block import AtomUpdateBlock
from research.gemnet_fused_fastgate.common import RESULTS, atomic_json, configure_environment
from research.gemnet_fused_fastgate.harness import (
    build_c0_generator,
    build_sampler,
    find_gemnet,
    load_states,
    prepare_joint_states,
    run_joint_score,
)


def pytorch_aggregation(
    rbf: torch.Tensor,
    messages: torch.Tensor,
    weight: torch.Tensor,
    receiver: torch.Tensor,
    n_atoms: int,
    scale: torch.Tensor,
) -> torch.Tensor:
    projected = F.linear(rbf, weight)
    gated = messages * projected
    output = torch.zeros(
        (n_atoms, gated.shape[-1]),
        dtype=gated.dtype,
        device=gated.device,
    )
    output.scatter_add_(
        0,
        receiver[:, None].expand_as(gated),
        gated,
    )
    return output * scale


def original_aggregation(
    rbf: torch.Tensor,
    messages: torch.Tensor,
    weight: torch.Tensor,
    receiver: torch.Tensor,
    n_atoms: int,
    scale: torch.Tensor,
) -> torch.Tensor:
    projected = F.linear(rbf, weight)
    gated = messages * projected
    return scatter(
        gated,
        receiver,
        dim=0,
        dim_size=n_atoms,
        reduce="sum",
    ) * scale


def timed_ms(fn: Callable[[], torch.Tensor], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / repeats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()

    configure_environment()
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    generator = build_c0_generator()
    sampler = build_sampler(generator)
    diffusion_module = generator.model.diffusion_module.to(device).eval()
    gemnet = find_gemnet(diffusion_module)
    states = prepare_joint_states(load_states(args.states), sampler, device)

    captured: list[tuple[AtomUpdateBlock, tuple[torch.Tensor, ...]]] = []

    def capture(module, inputs):
        captured.append((module, tuple(value.detach() for value in inputs)))

    hooks = [
        module.register_forward_pre_hook(capture)
        for module in gemnet.modules()
        if isinstance(module, AtomUpdateBlock)
    ]
    run_joint_score(diffusion_module, sampler, states[0])
    torch.cuda.synchronize()
    for hook in hooks:
        hook.remove()

    compiled = torch.compile(
        pytorch_aggregation,
        dynamic=True,
        fullgraph=True,
        mode="reduce-overhead",
    )
    records = []
    for index, (module, inputs) in enumerate(captured):
        h, messages, rbf, receiver = inputs
        parameters = (
            rbf,
            messages,
            module.dense_rbf.linear.weight,
            receiver,
            h.shape[0],
            module.scale_sum.scale_factor,
        )
        reference = original_aggregation(*parameters)
        manual = pytorch_aggregation(*parameters)
        optimized = compiled(*parameters)
        torch.cuda.synchronize()
        manual_delta = (manual - reference).abs()
        compiled_delta = (optimized - reference).abs()
        records.append(
            {
                "index": index,
                "module": module.name,
                "n_atoms": int(h.shape[0]),
                "n_edges": int(rbf.shape[0]),
                "id_j_monotonic": bool(torch.all(receiver[1:] >= receiver[:-1]).item()),
                "manual_max_abs": float(manual_delta.max().item()),
                "compiled_max_abs": float(compiled_delta.max().item()),
                "manual_allclose": bool(
                    torch.allclose(manual, reference, atol=1e-6, rtol=1e-5)
                ),
                "compiled_allclose": bool(
                    torch.allclose(optimized, reference, atol=1e-6, rtol=1e-5)
                ),
                "original_ms": timed_ms(
                    lambda: original_aggregation(*parameters),
                    args.warmup,
                    args.repeats,
                ),
                "manual_ms": timed_ms(
                    lambda: pytorch_aggregation(*parameters),
                    args.warmup,
                    args.repeats,
                ),
                "compiled_ms": timed_ms(
                    lambda: compiled(*parameters),
                    args.warmup,
                    args.repeats,
                ),
            }
        )
    for record in records:
        record["manual_speedup"] = record["original_ms"] / record["manual_ms"]
        record["compiled_speedup"] = record["original_ms"] / record["compiled_ms"]

    result = {
        "states_requested": args.states,
        "captured_calls": len(records),
        "records": records,
    }
    output = RESULTS / "fusion" / "k2_aggregation_probe.json"
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
