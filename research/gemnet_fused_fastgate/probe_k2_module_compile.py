from __future__ import annotations

import argparse
import json

import torch

from mattergen.common.gemnet.layers.atom_update_block import AtomUpdateBlock, OutputBlock
from research.gemnet_fused_fastgate.common import RESULTS, atomic_json, configure_environment
from research.gemnet_fused_fastgate.harness import (
    build_c0_generator,
    build_sampler,
    find_gemnet,
    load_states,
    prepare_joint_states,
    run_joint_score,
)


def flatten_output(value):
    if isinstance(value, tuple):
        return value
    return (value,)


def timed_ms(fn, warmup: int, repeats: int) -> float:
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
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()

    configure_environment()
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    generator = build_c0_generator()
    sampler = build_sampler(generator)
    diffusion_module = generator.model.diffusion_module.to(device).eval()
    gemnet = find_gemnet(diffusion_module)
    state = prepare_joint_states(load_states(1), sampler, device)[0]

    captured = {}

    def capture(name):
        def hook(module, inputs):
            captured[name] = tuple(value.detach() for value in inputs)

        return hook

    representatives = {}
    for name, module in gemnet.named_modules():
        if isinstance(module, OutputBlock) and "output" not in representatives:
            representatives["output"] = (name, module)
        elif (
            type(module) is AtomUpdateBlock
            and "atom" not in representatives
        ):
            representatives["atom"] = (name, module)
    hooks = [
        module.register_forward_pre_hook(capture(kind))
        for kind, (_, module) in representatives.items()
    ]
    run_joint_score(diffusion_module, sampler, state)
    torch.cuda.synchronize()
    for hook in hooks:
        hook.remove()

    records = []
    for kind, (name, module) in representatives.items():
        inputs = captured[kind]
        compiled = torch.compile(
            module,
            dynamic=True,
            fullgraph=False,
            mode="reduce-overhead",
        )
        reference = flatten_output(module(*inputs))
        candidate = flatten_output(compiled(*inputs))
        torch.cuda.synchronize()
        max_abs = max(
            float((actual - expected).abs().max().item())
            for actual, expected in zip(candidate, reference)
        )
        allclose = all(
            bool(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
            for actual, expected in zip(candidate, reference)
        )
        original_ms = timed_ms(
            lambda: module(*inputs),
            args.warmup,
            args.repeats,
        )
        compiled_ms = timed_ms(
            lambda: compiled(*inputs),
            args.warmup,
            args.repeats,
        )
        records.append(
            {
                "kind": kind,
                "module": name,
                "n_atoms": int(inputs[0].shape[0]),
                "n_edges": int(inputs[1].shape[0]),
                "max_abs": max_abs,
                "allclose": allclose,
                "original_ms": original_ms,
                "compiled_ms": compiled_ms,
                "speedup": original_ms / compiled_ms,
            }
        )
    result = {"records": records}
    atomic_json(RESULTS / "fusion" / "k2_module_compile_probe.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
