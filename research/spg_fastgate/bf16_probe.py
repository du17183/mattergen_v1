"""Frozen-state FP32/full-BF16/field-safe-BF16 score probe."""

from __future__ import annotations

import math
import time
import types
from contextlib import contextmanager, nullcontext
from statistics import median

import torch
import torch.nn.functional as functional

from mattergen.common.data.collate import collate
from research.spg_fastgate.common import RESULTS, atomic_json, now, set_stage
from research.spg_fastgate.generation import (
    build_generator,
    build_sampler,
    configure_determinism,
    make_condition,
)


RESULT = RESULTS / "bf16_state_probe.json"
FIELDS = ("atomic_numbers", "pos", "cell")


def target_calls(offset: int) -> set[int]:
    # 50 evenly spaced calls over the 2000 corrector/predictor forwards.
    return {
        min(1999, round(index * 1999 / 49) + offset) % 2000
        for index in range(50)
    }


def capture_states(method: str, seed: int, targets: set[int]):
    generator = build_generator(method, batch_size=1, sampling_steps=1000)
    sampler = build_sampler(generator, method, [seed])
    condition = collate([make_condition(seed)])
    captured = []
    original = sampler._score_fn
    call_index = 0

    def wrapped(x, t):
        nonlocal call_index
        if call_index in targets:
            captured.append(
                {
                    "method": method,
                    "call_index": call_index,
                    "progress": call_index / 1999.0,
                    "x": x.clone().to("cpu"),
                    "t": t.detach().clone().to("cpu"),
                }
            )
        call_index += 1
        return original(x, t)

    sampler._score_fn = wrapped
    sampler.sample(condition, None)
    if len(captured) != len(targets):
        raise RuntimeError(
            f"captured {len(captured)} states for {method}, expected {len(targets)}"
        )
    return generator, captured


@contextmanager
def field_safe_linear_bf16(model):
    """Run only nn.Linear GEMMs in BF16 and return their outputs to FP32."""

    restorers = []
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        original = module.forward
        weight = module.weight.detach().to(torch.bfloat16)
        bias = (
            module.bias.detach().to(torch.bfloat16)
            if module.bias is not None
            else None
        )

        def wrapped(
            self,
            value,
            __weight=weight,
            __bias=bias,
        ):
            output = functional.linear(value.to(torch.bfloat16), __weight, __bias)
            return output.to(torch.float32)

        module.forward = types.MethodType(wrapped, module)
        restorers.append((module, original))
    try:
        yield
    finally:
        for module, original in restorers:
            module.forward = original


def precision_context(mode: str, score_model):
    if mode == "fp32":
        return nullcontext()
    if mode == "full_bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mode == "field_safe_bf16":
        return field_safe_linear_bf16(score_model)
    raise ValueError(mode)


def evaluate_mode(generator, states: list[dict], mode: str) -> dict:
    diffusion_module = generator.model.diffusion_module
    score_model = diffusion_module.model
    outputs = []
    latencies = []
    peak_allocated = 0
    with torch.no_grad(), precision_context(mode, score_model):
        for index, state in enumerate(states):
            x = state["x"].to(generator.model.device)
            t = state["t"].to(generator.model.device)
            if index < 3:
                diffusion_module.score_fn(x, t)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            score = diffusion_module.score_fn(x, t)
            finished.record()
            torch.cuda.synchronize()
            latencies.append(started.elapsed_time(finished))
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            outputs.append(
                {
                    field: score[field].detach().float().cpu()
                    for field in FIELDS
                }
            )
    return {
        "mode": mode,
        "outputs": outputs,
        "latency_ms": latencies,
        "median_forward_ms": median(latencies),
        "mean_forward_ms": sum(latencies) / len(latencies),
        "peak_allocated_bytes": peak_allocated,
        "nan_or_inf": any(
            not bool(torch.isfinite(output[field]).all())
            for output in outputs
            for field in FIELDS
        ),
    }


def compare(reference: dict, candidate: dict) -> dict:
    fields = {}
    for field in FIELDS:
        reference_flat = torch.cat(
            [output[field].reshape(-1) for output in reference["outputs"]]
        ).double()
        candidate_flat = torch.cat(
            [output[field].reshape(-1) for output in candidate["outputs"]]
        ).double()
        cosine = torch.nn.functional.cosine_similarity(
            reference_flat, candidate_flat, dim=0, eps=1e-12
        )
        difference = candidate_flat - reference_flat
        fields[field] = {
            "cosine": float(cosine),
            "max_absolute_error": float(difference.abs().max()),
            "relative_l2_error": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference_flat).clamp_min(1e-12)
            ),
        }
    return {
        "fields": fields,
        "speedup": reference["median_forward_ms"] / candidate["median_forward_ms"],
        "candidate_median_forward_ms": candidate["median_forward_ms"],
        "candidate_peak_allocated_bytes": candidate["peak_allocated_bytes"],
        "nan_or_inf": candidate["nan_or_inf"],
    }


def strip_outputs(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "outputs"}


def main() -> int:
    configure_determinism()
    set_stage(
        "bf16_state_probe",
        "running",
        "Capturing 100 real C0/A0 timestep states and probing full/field-safe BF16.",
        {"state_count": 100},
    )
    c0_generator, c0_states = capture_states("C0", 24128, target_calls(0))
    _, a0_states = capture_states("A0", 24129, target_calls(1))
    states = sorted(c0_states + a0_states, key=lambda row: (row["progress"], row["method"]))
    fp32 = evaluate_mode(c0_generator, states, "fp32")
    full = evaluate_mode(c0_generator, states, "full_bf16")
    safe = evaluate_mode(c0_generator, states, "field_safe_bf16")
    full_comparison = compare(fp32, full)
    safe_comparison = compare(fp32, safe)
    score_gate = all(
        row["cosine"] >= 0.9999
        for row in safe_comparison["fields"].values()
    )
    speed_gate = safe_comparison["speedup"] >= 1.10
    safe_go = bool(
        score_gate
        and speed_gate
        and not safe_comparison["nan_or_inf"]
        and all(math.isfinite(row["relative_l2_error"]) for row in safe_comparison["fields"].values())
    )
    result = {
        "created_at": now(),
        "state_count": len(states),
        "methods": {"C0": len(c0_states), "A0": len(a0_states)},
        "noise_coverage": {
            "minimum_progress": min(row["progress"] for row in states),
            "maximum_progress": max(row["progress"] for row in states),
        },
        "fp32": strip_outputs(fp32),
        "full_bf16": strip_outputs(full),
        "field_safe_bf16": strip_outputs(safe),
        "full_bf16_comparison": full_comparison,
        "field_safe_bf16_comparison": safe_comparison,
        "FULL_BF16_GO": bool(
            all(row["cosine"] >= 0.9999 for row in full_comparison["fields"].values())
            and full_comparison["speedup"] >= 1.10
            and not full_comparison["nan_or_inf"]
        ),
        "FIELD_SAFE_BF16_STATE_GO": safe_go,
        "FIELD_SAFE_BF16_GO": False,
        "reason": (
            "State-level score and speed gates passed; 8-seed endpoint validation required."
            if safe_go
            else "State-level score or speed gate failed; endpoint BF16 expansion is stopped."
        ),
    }
    atomic_json(RESULT, result)
    set_stage(
        "bf16_state_probe",
        "success",
        result["reason"],
        result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
