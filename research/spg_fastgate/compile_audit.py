"""Partial torch.compile feasibility audit on real GemNet inputs."""

from __future__ import annotations

import copy
import time
import traceback
from statistics import median

import torch
from torch._dynamo.utils import counters as dynamo_counters

from mattergen.common.data.collate import collate
from research.spg_fastgate.common import RESULTS, atomic_json, now, set_stage
from research.spg_fastgate.generation import (
    build_generator,
    build_sampler,
    configure_determinism,
    make_condition,
)


RESULT = RESULTS / "compile_audit.json"
MODES = ("default", "reduce-overhead", "max-autotune-no-cudagraphs")


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if hasattr(value, "clone"):
        try:
            return value.clone()
        except TypeError:
            pass
    return value


def compare_tree(reference, candidate) -> dict:
    errors = []

    def visit(left, right):
        if isinstance(left, torch.Tensor):
            difference = (left.detach().float() - right.detach().float()).abs()
            errors.append(
                {
                    "max_abs": float(difference.max()) if difference.numel() else 0.0,
                    "allclose": bool(
                        torch.allclose(left, right, rtol=1e-5, atol=1e-6)
                    ),
                }
            )
            return
        if isinstance(left, (tuple, list)):
            for left_item, right_item in zip(left, right, strict=True):
                visit(left_item, right_item)
            return
        if isinstance(left, dict):
            for key in left:
                visit(left[key], right[key])
            return
        if hasattr(left, "data") and isinstance(left.data, dict):
            visit(left.data, right.data)

    visit(reference, candidate)
    return {
        "tensor_count": len(errors),
        "max_absolute_error": max((row["max_abs"] for row in errors), default=0.0),
        "allclose": all(row["allclose"] for row in errors),
    }


def call_with_rng_state(callable_, args, kwargs, cpu_state, cuda_states):
    """Evaluate a potentially stochastic update from a frozen RNG state."""

    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_states)
    return callable_(*args, **kwargs)


def capture_inputs():
    generator = build_generator("C0", batch_size=1, sampling_steps=10)
    sampler = build_sampler(generator, "C0", [24000])
    condition = collate([make_condition(24000)])
    score_model = generator.model.diffusion_module.model
    gemnet = next(
        module
        for module in generator.model.modules()
        if module.__class__.__name__ in {"GemNetT", "GemNetTCtrl"}
    )
    captures = {}

    def capture_module(name):
        def hook(module, args, kwargs):
            if name not in captures:
                captures[name] = {
                    "callable": module,
                    "args": clone_tree(args),
                    "kwargs": clone_tree(kwargs),
                }

        return hook

    handles = [
        gemnet.int_blocks[0].register_forward_pre_hook(
            capture_module("interaction_block"), with_kwargs=True
        ),
        gemnet.out_blocks[0].register_forward_pre_hook(
            capture_module("score_head"), with_kwargs=True
        ),
    ]
    original_score = sampler._score_fn

    def score_wrapper(x, t):
        if "full_score_model" not in captures:
            captures["full_score_model"] = {
                "callable": score_model,
                "args": (clone_tree(x), clone_tree(t)),
                "kwargs": {},
            }
        return original_score(x, t)

    sampler._score_fn = score_wrapper
    for name, owner, attribute in (
        ("predictor_pos", sampler._predictors["pos"], "update_given_score"),
        ("corrector_pos", sampler._correctors["pos"], "step_given_score"),
    ):
        original = getattr(owner, attribute)

        def wrapper(*args, __name=name, __original=original, **kwargs):
            if __name not in captures:
                captures[__name] = {
                    "callable": __original,
                    "args": clone_tree(args),
                    "kwargs": clone_tree(kwargs),
                }
            return __original(*args, **kwargs)

        setattr(owner, attribute, wrapper)
    sampler.sample(condition, None)
    for handle in handles:
        handle.remove()
    required = {
        "interaction_block",
        "score_head",
        "predictor_pos",
        "corrector_pos",
        "full_score_model",
    }
    missing = required - captures.keys()
    if missing:
        raise RuntimeError(f"failed to capture compile inputs: {sorted(missing)}")
    return generator, captures


def benchmark_callable(name: str, value: dict) -> dict:
    eager = value["callable"]
    args = value["args"]
    kwargs = value["kwargs"]
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    with torch.no_grad():
        reference = call_with_rng_state(
            eager, args, kwargs, cpu_state, cuda_states
        )
    rows = []
    for mode in MODES:
        torch.compiler.reset()
        dynamo_counters.clear()
        compiled = torch.compile(eager, mode=mode, fullgraph=False)
        started = time.monotonic()
        try:
            with torch.no_grad():
                output = call_with_rng_state(
                    compiled, args, kwargs, cpu_state, cuda_states
                )
                torch.cuda.synchronize()
            compile_and_first_seconds = time.monotonic() - started
            comparison = compare_tree(reference, output)
            for _ in range(5):
                with torch.no_grad():
                    compiled(*args, **kwargs)
            torch.cuda.synchronize()
            latencies = []
            calls = 1000 if name in {"predictor_pos", "corrector_pos"} else 100
            for _ in range(calls):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                with torch.no_grad():
                    compiled(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                latencies.append(begin.elapsed_time(end))
            eager_latencies = []
            for _ in range(calls):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                with torch.no_grad():
                    eager(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                eager_latencies.append(begin.elapsed_time(end))
            rows.append(
                {
                    "mode": mode,
                    "success": True,
                    "compile_and_first_seconds": compile_and_first_seconds,
                    "calls": calls,
                    "compiled_median_ms": median(latencies),
                    "eager_median_ms": median(eager_latencies),
                    "speedup": median(eager_latencies) / median(latencies),
                    "comparison": comparison,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "graph_break_count": int(
                        sum(dynamo_counters.get("graph_break", {}).values())
                    ),
                    "unique_graphs": int(
                        dynamo_counters.get("stats", {}).get("unique_graphs", 0)
                    ),
                    "recompile_count": max(
                        0,
                        int(
                            dynamo_counters.get("stats", {}).get(
                                "unique_graphs", 0
                            )
                        )
                        - 1,
                    ),
                    "dynamic_shape_support": "NOT_VERIFIED_SINGLE_CAPTURED_SHAPE",
                }
            )
        except BaseException:
            rows.append(
                {
                    "mode": mode,
                    "success": False,
                    "error": traceback.format_exc(),
                }
            )
    return {"name": name, "modes": rows}


def fullgraph_audit(value: dict) -> dict:
    torch.compiler.reset()
    target = value["callable"]
    args = value["args"]
    kwargs = value["kwargs"]
    try:
        compiled = torch.compile(target, backend="eager", fullgraph=True)
        with torch.no_grad():
            compiled(*args, **kwargs)
        return {"fullgraph_success": True, "error": None}
    except BaseException:
        return {
            "fullgraph_success": False,
            "error": traceback.format_exc(),
        }


def main() -> int:
    configure_determinism()
    set_stage(
        "compile_audit",
        "running",
        "Capturing real GemNet block/head/update inputs and testing three compile modes.",
    )
    _, captures = capture_inputs()
    fullgraph = fullgraph_audit(captures["full_score_model"])
    results = [
        benchmark_callable(name, captures[name])
        for name in (
            "interaction_block",
            "score_head",
            "predictor_pos",
            "corrector_pos",
        )
    ]
    successful = [
        row
        for result in results
        for row in result["modes"]
        if row.get("success") and row["comparison"]["allclose"]
    ]
    block_speedups = [
        row["speedup"]
        for result in results
        if result["name"] == "interaction_block"
        for row in result["modes"]
        if row.get("success") and row["comparison"]["allclose"]
    ]
    other_speedups = [
        row["speedup"]
        for result in results
        if result["name"] != "interaction_block"
        for row in result["modes"]
        if row.get("success") and row["comparison"]["allclose"]
    ]
    partial_works = bool(
        max(block_speedups, default=0.0) >= 1.10
        or max(other_speedups, default=0.0) >= 1.15
    )
    successful_rows = [
        row
        for component in results
        for row in component["modes"]
        if row.get("success")
    ]
    result = {
        "created_at": now(),
        "torch_version": torch.__version__,
        "modes": list(MODES),
        "full_score_model_fullgraph": fullgraph,
        "components": results,
        "successful_numeric_modes": len(successful),
        "best_block_speedup": max(block_speedups, default=0.0),
        "best_other_speedup": max(other_speedups, default=0.0),
        "max_recompiles_observed": max(
            (row.get("recompile_count", 0) for row in successful_rows),
            default=0,
        ),
        "total_graph_breaks_observed": sum(
            row.get("graph_break_count", 0) for row in successful_rows
        ),
        "PARTIAL_COMPILE_WORKS": partial_works,
        "STATIC_GRAPH_REQUIRED": not fullgraph["fullgraph_success"],
    }
    atomic_json(RESULT, result)
    set_stage(
        "compile_audit",
        "success",
        (
            f"Partial compile works={partial_works}; "
            f"static graph required={result['STATIC_GRAPH_REQUIRED']}."
        ),
        result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
