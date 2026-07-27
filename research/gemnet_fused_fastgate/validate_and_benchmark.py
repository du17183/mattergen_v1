from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

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
from research.gemnet_fused_fastgate.k2_local_compile import (
    enable_k2_local_compile,
    iter_k2_modules,
    k2_implementation_manifest,
)


VALIDATION_STATES = 100
CHAIN_WARMUP = 50
CHAIN_REPEATS = 1000
CHAIN_ROUNDS = 3
FORWARD_WARMUP = 50
FORWARD_REPEATS = 300
FORWARD_ROUNDS = 3
ATOL = 1e-6
RTOL = 1e-5
MIN_COSINE = 0.999999


def flatten_named(prefix: str, value: Any) -> list[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, tuple):
        labels = ("node_or_energy", "edge_or_force")
        return [
            (f"{prefix}.{labels[index] if index < len(labels) else index}", child)
            for index, child in enumerate(value)
            if isinstance(child, torch.Tensor)
        ]
    raise TypeError(f"unsupported captured output type: {type(value)!r}")


def tensor_cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_flat = actual.reshape(-1).double()
    expected_flat = expected.reshape(-1).double()
    actual_norm = torch.linalg.vector_norm(actual_flat)
    expected_norm = torch.linalg.vector_norm(expected_flat)
    if actual_norm == 0 and expected_norm == 0:
        return 1.0
    if actual_norm == 0 or expected_norm == 0:
        return 0.0
    return float(F.cosine_similarity(actual_flat, expected_flat, dim=0).item())


def compare_tensor(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    difference = actual.double() - expected.double()
    reference_norm = torch.linalg.vector_norm(expected.double())
    difference_norm = torch.linalg.vector_norm(difference)
    relative_l2 = float(
        difference_norm.item() / max(reference_norm.item(), torch.finfo(torch.float64).tiny)
    )
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "allclose": bool(torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": relative_l2,
        "cosine": tensor_cosine(actual, expected),
        "diff_sq": float(torch.sum(difference * difference).item()),
        "ref_sq": float(torch.sum(expected.double() ** 2).item()),
    }


def capture_outputs(modules: dict[str, torch.nn.Module], store: dict[str, Any]):
    hooks = []
    for name, module in modules.items():
        def hook(_module, _inputs, output, *, key=name):
            if isinstance(output, tuple):
                store[key] = tuple(value.detach() for value in output)
            else:
                store[key] = output.detach()

        hooks.append(module.register_forward_hook(hook))
    return hooks


def cuda_wall_round(fn: Callable[[], Any], repeats: int) -> dict[str, float]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    wall_elapsed = time.perf_counter() - wall_start
    return {
        "cuda_ms_per_call": float(start.elapsed_time(end) / repeats),
        "wall_ms_per_call": float(wall_elapsed * 1000.0 / repeats),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def benchmark(
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    rounds: int,
) -> dict[str, Any]:
    for index in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()
    baseline_rounds = []
    candidate_rounds = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_rounds.append(cuda_wall_round(baseline, repeats))
            candidate_rounds.append(cuda_wall_round(candidate, repeats))
        else:
            candidate_rounds.append(cuda_wall_round(candidate, repeats))
            baseline_rounds.append(cuda_wall_round(baseline, repeats))
    base_cuda = statistics.median(row["cuda_ms_per_call"] for row in baseline_rounds)
    candidate_cuda = statistics.median(
        row["cuda_ms_per_call"] for row in candidate_rounds
    )
    base_wall = statistics.median(row["wall_ms_per_call"] for row in baseline_rounds)
    candidate_wall = statistics.median(
        row["wall_ms_per_call"] for row in candidate_rounds
    )
    return {
        "warmup": warmup,
        "repeats": repeats,
        "rounds": rounds,
        "baseline_rounds": baseline_rounds,
        "candidate_rounds": candidate_rounds,
        "baseline_cuda_median_ms": base_cuda,
        "candidate_cuda_median_ms": candidate_cuda,
        "cuda_speedup": base_cuda / candidate_cuda,
        "baseline_wall_median_ms": base_wall,
        "candidate_wall_median_ms": candidate_wall,
        "wall_speedup": base_wall / candidate_wall,
    }


def count_cuda_events(fn: Callable[[], Any]) -> int:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        fn()
        torch.cuda.synchronize()
    return sum(
        1
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    )


def main() -> int:
    configure_determinism()
    torch.set_grad_enabled(False)
    set_stage(
        "fusion_candidate_selection",
        "success",
        "Selected the unique largest eligible chain: K2 update family.",
        {
            "selected": "K2",
            "forward_share": 0.31834563424073903,
            "call_count_per_forward": 9,
            "selection_threshold": 0.15,
        },
    )
    set_stage(
        "fusion_implementation",
        "running",
        "Enabling strict-FP32 local compilation for only the nine K2 update modules.",
    )

    baseline_generator = build_c0_generator()
    baseline_sampler = build_sampler(baseline_generator)
    baseline_diffusion = baseline_generator.model.diffusion_module.eval()
    baseline_gemnet = find_gemnet(baseline_diffusion)
    device = next(baseline_gemnet.parameters()).device

    candidate_generator = build_c0_generator()
    candidate_sampler = build_sampler(candidate_generator)
    candidate_diffusion = candidate_generator.model.diffusion_module.eval()
    candidate_gemnet = find_gemnet(candidate_diffusion)
    handles = enable_k2_local_compile(candidate_gemnet)
    manifest = k2_implementation_manifest(candidate_gemnet)
    atomic_json(RESULTS / "fusion/implementation_manifest.json", manifest)
    set_stage(
        "fusion_implementation",
        "success",
        "Installed reversible local torch.compile wrappers on the profiled K2 family.",
        manifest,
    )

    raw_states = load_states(VALIDATION_STATES)
    states = prepare_joint_states(raw_states, baseline_sampler, device)
    baseline_modules = dict(iter_k2_modules(baseline_gemnet))
    candidate_modules = dict(iter_k2_modules(candidate_gemnet))
    if list(baseline_modules) != list(candidate_modules):
        raise RuntimeError("baseline/candidate K2 module order differs")

    # Trigger compilation and dynamic-shape guards before collecting validation data.
    with torch.inference_mode():
        for state in states[:4]:
            run_joint_score(candidate_diffusion, candidate_sampler, state)
    torch.cuda.synchronize()

    set_stage(
        "numerical_validation",
        "running",
        f"Comparing K2 intermediates and final score fields on {VALIDATION_STATES} real states.",
    )
    component_accumulator: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "max_abs": 0.0,
            "max_relative_l2": 0.0,
            "min_cosine": 1.0,
            "diff_sq": 0.0,
            "ref_sq": 0.0,
            "allclose": True,
            "finite": True,
        }
    )
    per_state = []
    first_error_operator = None
    baseline_store: dict[str, Any] = {}
    candidate_store: dict[str, Any] = {}
    baseline_hooks = capture_outputs(baseline_modules, baseline_store)
    candidate_hooks = capture_outputs(candidate_modules, candidate_store)
    fields = tuple(baseline_sampler._multi_corruption.corrupted_fields)
    try:
        with torch.inference_mode():
            for state in states:
                baseline_store.clear()
                candidate_store.clear()
                baseline_score = run_joint_score(
                    baseline_diffusion, baseline_sampler, state
                )
                candidate_score = run_joint_score(
                    candidate_diffusion, candidate_sampler, state
                )
                components = []
                for name in baseline_modules:
                    baseline_values = dict(flatten_named(name, baseline_store[name]))
                    candidate_values = dict(flatten_named(name, candidate_store[name]))
                    for label, expected in baseline_values.items():
                        components.append((label, candidate_values[label], expected))
                for field in fields:
                    components.append(
                        (
                            f"final_score.{field}",
                            candidate_score[field],
                            baseline_score[field],
                        )
                    )
                state_max = 0.0
                state_allclose = True
                for label, actual, expected in components:
                    metrics = compare_tensor(actual, expected)
                    aggregate = component_accumulator[label]
                    aggregate["count"] += 1
                    aggregate["max_abs"] = max(aggregate["max_abs"], metrics["max_abs"])
                    aggregate["max_relative_l2"] = max(
                        aggregate["max_relative_l2"], metrics["relative_l2"]
                    )
                    aggregate["min_cosine"] = min(
                        aggregate["min_cosine"], metrics["cosine"]
                    )
                    aggregate["diff_sq"] += metrics["diff_sq"]
                    aggregate["ref_sq"] += metrics["ref_sq"]
                    aggregate["allclose"] &= metrics["allclose"]
                    aggregate["finite"] &= metrics["finite"]
                    state_max = max(state_max, metrics["max_abs"])
                    state_allclose &= bool(metrics["allclose"])
                    if (
                        first_error_operator is None
                        and (
                            not metrics["finite"]
                            or not metrics["allclose"]
                            or metrics["cosine"] < MIN_COSINE
                        )
                    ):
                        first_error_operator = label
                per_state.append(
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
                    | {"max_abs": state_max, "allclose": state_allclose}
                )
    finally:
        for hook in baseline_hooks + candidate_hooks:
            hook.remove()

    component_summary = {}
    for label, aggregate in component_accumulator.items():
        component_summary[label] = {
            key: value
            for key, value in aggregate.items()
            if key not in {"diff_sq", "ref_sq"}
        }
        component_summary[label]["aggregate_relative_l2"] = math.sqrt(
            aggregate["diff_sq"] / max(aggregate["ref_sq"], np.finfo(np.float64).tiny)
        )
    max_abs = max(row["max_abs"] for row in component_summary.values())
    relative_l2 = max(
        row["aggregate_relative_l2"] for row in component_summary.values()
    )
    min_cosine = min(row["min_cosine"] for row in component_summary.values())
    numerical_equivalent = all(
        row["finite"] and row["allclose"] and row["min_cosine"] >= MIN_COSINE
        for row in component_summary.values()
    )
    numerical = {
        "completed_at": now(),
        "states": VALIDATION_STATES,
        "atol": ATOL,
        "rtol": RTOL,
        "cosine_threshold": MIN_COSINE,
        "max_abs_error": max_abs,
        "relative_l2_error": relative_l2,
        "min_cosine": min_cosine,
        "numerical_equivalent": numerical_equivalent,
        "first_error_operator": first_error_operator,
        "components": component_summary,
        "state_coverage": per_state,
    }
    atomic_json(RESULTS / "fusion/numerical_validation.json", numerical)
    set_stage(
        "numerical_validation",
        "success" if numerical_equivalent else "failed",
        "Strict K2 intermediate and final-score validation completed.",
        numerical,
    )

    representative = states[len(states) // 2]
    captured_inputs: dict[str, tuple[torch.Tensor, ...]] = {}
    input_hooks = []
    for name, module in baseline_modules.items():
        def hook(_module, inputs, *, key=name):
            captured_inputs[key] = tuple(value.detach() for value in inputs)

        input_hooks.append(module.register_forward_pre_hook(hook))
    with torch.inference_mode():
        run_joint_score(baseline_diffusion, baseline_sampler, representative)
    for hook in input_hooks:
        hook.remove()

    def baseline_chain():
        for name, module in baseline_modules.items():
            module(*captured_inputs[name])

    def candidate_chain():
        for name, module in candidate_modules.items():
            module(*captured_inputs[name])

    with torch.inference_mode():
        chain = benchmark(
            baseline_chain,
            candidate_chain,
            warmup=CHAIN_WARMUP,
            repeats=CHAIN_REPEATS,
            rounds=CHAIN_ROUNDS,
        )
        chain["baseline_cuda_event_count"] = count_cuda_events(baseline_chain)
        chain["candidate_cuda_event_count"] = count_cuda_events(candidate_chain)
    chain["gate_speedup"] = 1.25
    chain["passed"] = chain["cuda_speedup"] >= chain["gate_speedup"]
    atomic_json(RESULTS / "fusion/chain_microbenchmark.json", chain)
    set_stage(
        "chain_microbenchmark",
        "success" if chain["passed"] else "failed",
        "Completed 50-warmup, 1000-call x 3 K2 chain benchmark.",
        chain,
    )

    forward_states = states[:12]
    base_index = 0
    candidate_index = 0

    def baseline_forward():
        nonlocal base_index
        output = run_joint_score(
            baseline_diffusion,
            baseline_sampler,
            forward_states[base_index % len(forward_states)],
        )
        base_index += 1
        return output

    def candidate_forward():
        nonlocal candidate_index
        output = run_joint_score(
            candidate_diffusion,
            candidate_sampler,
            forward_states[candidate_index % len(forward_states)],
        )
        candidate_index += 1
        return output

    with torch.inference_mode():
        forward = benchmark(
            baseline_forward,
            candidate_forward,
            warmup=FORWARD_WARMUP,
            repeats=FORWARD_REPEATS,
            rounds=FORWARD_ROUNDS,
        )
    forward["gate_speedup"] = 1.08
    forward["passed"] = forward["cuda_speedup"] >= forward["gate_speedup"]
    atomic_json(RESULTS / "fusion/forward_microbenchmark.json", forward)
    set_stage(
        "forward_microbenchmark",
        "success" if forward["passed"] else "failed",
        "Completed 300-call x 3 original versus K2-local-compile joint-CFG benchmark.",
        forward,
    )

    fusion_go = numerical_equivalent and chain["passed"] and forward["passed"]
    decision = {
        "numerical_equivalent": numerical_equivalent,
        "chain_speedup": chain["cuda_speedup"],
        "chain_gate": chain["gate_speedup"],
        "forward_speedup": forward["cuda_speedup"],
        "forward_gate": forward["gate_speedup"],
        "fusion_pre_e2e_go": fusion_go,
    }
    atomic_json(RESULTS / "fusion/pre_e2e_decision.json", decision)
    set_stage(
        "fusion_go_no_go",
        "success" if fusion_go else "failed",
        "Fusion pre-E2E gate passed; proceed to eight seeds."
        if fusion_go
        else "Fusion pre-E2E gate failed; trigger persistent-worker fallback.",
        decision,
    )

    report_lines = [
        "# GemNet K2 local-compile fast gate",
        "",
        f"- Validation states: `{VALIDATION_STATES}`",
        f"- Numerical equivalent: `{numerical_equivalent}`",
        f"- Maximum absolute error: `{max_abs:.9g}`",
        f"- Maximum aggregate relative L2: `{relative_l2:.9g}`",
        f"- Minimum cosine: `{min_cosine:.9g}`",
        f"- K2 chain CUDA speedup: `{chain['cuda_speedup']:.4f}x`",
        f"- Full joint-CFG forward CUDA speedup: `{forward['cuda_speedup']:.4f}x`",
        f"- Pre-E2E gate: `{fusion_go}`",
        "",
        "The implementation locally compiles only the nine profiled AtomUpdate/OutputBlock",
        "forwards. The full GemNet and sampler remain eager; edge ordering and native",
        "scatter-add semantics are unchanged.",
        "",
    ]
    atomic_text(REPORTS / "fusion_fastgate.md", "\n".join(report_lines))
    print(json.dumps(decision, indent=2))
    return 0 if fusion_go else 42


if __name__ == "__main__":
    raise SystemExit(main())
