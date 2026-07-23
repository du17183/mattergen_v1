# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import inspect
import pytest
import torch

from mattergen.diffusion.sampling.classifier_free_guidance import score_residual_rms
from mattergen.diffusion.sampling.guidance_schedule import (
    GuidanceController,
    normalize_guidance_schedule,
    piecewise_guidance,
)
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.generator import CrystalGenerator
from mattergen.scripts.generate import main as generate_main


@pytest.fixture(autouse=True)
def seed_random_state() -> None:
    """Keep these CPU-only tests from initializing every shared CUDA device."""


def make_controller(schedule: str = "constant", **kwargs) -> GuidanceController:
    return GuidanceController(
        schedule=schedule,
        base_guidance=kwargs.pop("base_guidance", 2.0),
        warmup_frac=kwargs.pop("warmup_frac", 0.2),
        decay_frac=kwargs.pop("decay_frac", 0.2),
        min_scale=kwargs.pop("min_scale", 0.5),
        max_scale=kwargs.pop("max_scale", 5.0),
        adaptive_alpha=kwargs.pop("adaptive_alpha", 0.5),
        adaptive_ema=kwargs.pop("adaptive_ema", 0.95),
        adaptive_eps=kwargs.pop("adaptive_eps", 1e-6),
        **kwargs,
    )


def test_piecewise_boundaries() -> None:
    for progress, expected in (
        (0.0, 0.5),
        (0.2, 2.0),
        (0.5, 2.0),
        (0.8, 2.0),
        (1.0, 0.5),
    ):
        assert piecewise_guidance(
            progress=progress,
            base_guidance=2.0,
            warmup_frac=0.2,
            decay_frac=0.2,
            min_scale=0.5,
        ) == pytest.approx(expected)


def test_constant_guidance() -> None:
    controller = make_controller("constant")
    decision = controller.evaluate(
        progress=0.75,
        phase="predictor",
        field_deltas={"cell": 100.0, "pos": 0.1, "atomic_numbers": 3.0},
    )
    assert decision.stage_guidance == 2.0
    assert decision.adaptive_multiplier == 1.0
    assert decision.final_guidance == 2.0


def test_adaptive_first_and_later_ema() -> None:
    controller = make_controller("adaptive", adaptive_ema=0.5, adaptive_eps=1e-6)
    first = controller.evaluate(
        progress=0.0, phase="corrector", field_deltas={"cell": 2.0}
    )
    second = controller.evaluate(
        progress=0.1, phase="corrector", field_deltas={"cell": 4.0}
    )
    assert first.ema == 2.0
    assert first.ratio == pytest.approx(2.0 / 2.000001)
    assert second.ema == 3.0
    assert second.ratio == pytest.approx(4.0 / 3.000001)


def test_phase_specific_ema_and_reset() -> None:
    controller = make_controller("adaptive", adaptive_ema=0.5)
    controller.evaluate(progress=0.0, phase="corrector", field_deltas={"cell": 1.0})
    controller.evaluate(progress=0.0, phase="predictor", field_deltas={"cell": 5.0})
    assert controller.ema_by_phase == {"corrector": 1.0, "predictor": 5.0}
    controller.reset()
    assert controller.ema_by_phase == {"corrector": None, "predictor": None}


def test_adaptive_multiplier_and_final_guidance_clamps() -> None:
    controller = make_controller(
        "adaptive", adaptive_alpha=100.0, adaptive_ema=0.99, max_scale=5.0
    )
    controller.evaluate(progress=0.0, phase="predictor", field_deltas={"cell": 0.01})
    decision = controller.evaluate(
        progress=0.1, phase="predictor", field_deltas={"cell": 100.0}
    )
    assert decision.adaptive_multiplier == 4.0
    assert decision.final_guidance == 5.0


def test_stage_adaptive_combines_stage_and_feedback() -> None:
    controller = make_controller("stage_adaptive", adaptive_ema=0.5)
    first = controller.evaluate(
        progress=0.0, phase="predictor", field_deltas={"cell": 1.0}
    )
    second = controller.evaluate(
        progress=0.1, phase="predictor", field_deltas={"cell": 2.0}
    )
    assert first.stage_guidance == 0.5
    assert second.stage_guidance == pytest.approx(1.25)
    assert second.adaptive_multiplier != 1.0
    assert second.final_guidance == pytest.approx(
        min(max(second.stage_guidance * second.adaptive_multiplier, 0.5), 5.0)
    )


def test_differently_shaped_field_residuals() -> None:
    unconditional = {
        "cell": torch.zeros(2, 3, 3),
        "pos": torch.zeros(7, 3),
        "atomic_numbers": torch.zeros(7, 101),
    }
    conditional = {
        "cell": torch.ones(2, 3, 3),
        "pos": torch.full((7, 3), 2.0),
        "atomic_numbers": torch.full((7, 101), 3.0),
    }
    deltas, error = score_residual_rms(
        unconditional_score=unconditional,
        conditional_score=conditional,
        fields=("cell", "pos", "atomic_numbers"),
    )
    assert error is None
    assert deltas == pytest.approx({"cell": 1.0, "pos": 2.0, "atomic_numbers": 3.0})


def test_invalid_residual_falls_back() -> None:
    cases = (
        ({"cell": torch.empty(0)}, "cell:empty"),
        ({"cell": torch.tensor([float("nan")])}, "cell:non_finite"),
        ({"cell": torch.tensor([float("inf")])}, "cell:non_finite"),
        ({"cell": None}, "cell:none"),
    )
    for conditional, expected_error in cases:
        value = conditional["cell"]
        unconditional = {
            "cell": torch.zeros_like(value) if isinstance(value, torch.Tensor) else None
        }
        deltas, error = score_residual_rms(
            unconditional_score=unconditional,
            conditional_score=conditional,
            fields=("cell",),
        )
        controller = make_controller("adaptive")
        decision = controller.evaluate(
            progress=0.5,
            phase="predictor",
            field_deltas=deltas,
            residual_error=error,
        )
        assert error == expected_error
        assert decision.adaptive_multiplier == 1.0
        assert decision.final_guidance == decision.stage_guidance
        assert decision.fallback_reason == expected_error


def test_invalid_parameters() -> None:
    cases = (
        {"schedule": "unknown"},
        {"schedule": "piecewise", "warmup_frac": -0.1},
        {"schedule": "piecewise", "decay_frac": 1.0},
        {"schedule": "piecewise", "warmup_frac": 0.6, "decay_frac": 0.5},
        {"schedule": "adaptive", "min_scale": 6.0, "max_scale": 5.0},
        {"schedule": "adaptive", "adaptive_alpha": -1.0},
        {"schedule": "adaptive", "adaptive_ema": 1.0},
        {"schedule": "adaptive", "adaptive_eps": 0.0},
    )
    for case in cases:
        kwargs = dict(case)
        schedule = kwargs.pop("schedule")
        with pytest.raises(ValueError):
            make_controller(schedule, **kwargs)


def test_stage_adaptive_hyphen_alias() -> None:
    assert normalize_guidance_schedule("stage-adaptive") == "stage_adaptive"


def test_sampling_context_progress_phase_and_call_index() -> None:
    sampler = object.__new__(PredictorCorrector)
    sampler.N = 5
    sampler._sampling_context = {}
    sampler._score_call_index = 0
    sampler._set_sampling_context(sampling_step=2, phase="corrector")
    first = dict(sampler.sampling_context)
    sampler._set_sampling_context(sampling_step=2, phase="predictor")
    second = dict(sampler.sampling_context)
    assert first == {
        "sampling_step": 2,
        "num_steps": 5,
        "progress": 0.5,
        "phase": "corrector",
        "score_call_index": 0,
    }
    assert second["progress"] == first["progress"]
    assert second["phase"] == "predictor"
    assert second["score_call_index"] == 1


def test_seed_and_deterministic_cli_parameters_exist() -> None:
    parameters = inspect.signature(generate_main).parameters
    assert parameters["seed"].default is None
    assert parameters["deterministic"].default is False
    assert parameters["guidance_schedule"].default == "constant"
    assert parameters["guidance_trace_path"].default is None


def test_invalid_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="seed"):
        CrystalGenerator(checkpoint_info=object(), seed=-1)  # type: ignore[arg-type]
