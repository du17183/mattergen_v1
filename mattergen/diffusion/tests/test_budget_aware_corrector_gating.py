from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from mattergen.diffusion.sampling.classifier_free_guidance import (
    GuidedPredictorCorrector,
)
from mattergen.diffusion.sampling.corrector_gating import (
    GATING_FIELDS,
    ConvergenceAwareCorrectorGate,
    CorrectorGateDecision,
)
from mattergen.generator import CrystalGenerator
from mattergen.scripts.generate import main as generate_main


RESIDUALS = {field: 1.0 for field in GATING_FIELDS}
SMALL = {field: 0.01 for field in GATING_FIELDS}


def observe(
    gate: ConvergenceAwareCorrectorGate,
    *,
    predictor_residuals=RESIDUALS,
    corrector_residuals=RESIDUALS,
    predictor_updates=SMALL,
    corrector_updates=SMALL,
) -> None:
    gate.observe_residual(
        phase="corrector", residuals=corrector_residuals
    )
    gate.observe_update(
        phase="corrector", update_norms=corrector_updates
    )
    gate.observe_residual(
        phase="predictor", residuals=predictor_residuals
    )
    gate.observe_update(
        phase="predictor", update_norms=predictor_updates
    )
    gate.finalize_step()


def ready_gate(**kwargs) -> ConvergenceAwareCorrectorGate:
    gate = ConvergenceAwareCorrectorGate(
        warmup_frac=0.0,
        min_progress=kwargs.pop("min_progress", 0.0),
        max_progress=kwargs.pop("max_progress", 1.0),
        convergence_threshold=kwargs.pop(
            "convergence_threshold", 0.05
        ),
        consecutive_stable_steps=kwargs.pop(
            "consecutive_stable_steps", 1
        ),
        calibration_interval=kwargs.pop("calibration_interval", 99),
        max_consecutive_skips=kwargs.pop(
            "max_consecutive_skips", 99
        ),
        fallback_threshold=kwargs.pop("fallback_threshold", 0.20),
        **kwargs,
    )
    observe(gate)
    observe(gate)
    return gate


def schedule(gate: ConvergenceAwareCorrectorGate, count: int = 100):
    decisions = []
    for step in range(count):
        decision = gate.decide(sampling_step=step, num_steps=count)
        decisions.append(decision)
        gate.record_scheduled_decision(decision)
    return decisions


def test_budget_aware_api_defaults_are_disabled() -> None:
    assert (
        inspect.signature(GuidedPredictorCorrector.__init__)
        .parameters["corrector_budget_aware_enabled"]
        .default
        is False
    )
    assert (
        CrystalGenerator.__dataclass_fields__[
            "corrector_budget_aware_enabled"
        ].default
        is False
    )
    assert (
        inspect.signature(generate_main)
        .parameters["corrector_budget_aware_enabled"]
        .default
        is False
    )
    config = Path("sampling_conf/default.yaml").read_text()
    assert "corrector_budget_aware_enabled: false" in config
    assert "corrector_field_aggregation: all_fields" in config


def test_max_skip_ratio_strictly_limits_total_skips() -> None:
    gate = ready_gate(max_skip_ratio=0.10)
    decisions = schedule(gate)
    assert sum(not item.execute_corrector for item in decisions) == 10
    assert gate.skip_budget_total == 10
    assert gate.skip_budget_used == 10
    assert gate.skip_budget_remaining == 0
    assert gate.skip_budget_exhausted


def test_budget_exhaustion_forces_full_corrector_forever() -> None:
    gate = ready_gate(max_skip_ratio=0.01)
    first = gate.decide(sampling_step=0, num_steps=100)
    assert not first.execute_corrector
    gate.record_scheduled_decision(first)
    for step in range(1, 10):
        decision = gate.decide(sampling_step=step, num_steps=100)
        assert decision.execute_corrector
        assert decision.mode == "budget_full_corrector"
        assert decision.budget_exhausted
        gate.record_scheduled_decision(decision)
    assert gate.skip_budget_used == 1


def test_late_stage_gate_never_skips_before_min_progress() -> None:
    gate = ready_gate(min_progress=0.40)
    early = gate.decide(sampling_step=39, num_steps=100)
    late = gate.decide(sampling_step=40, num_steps=100)
    assert early.execute_corrector
    assert early.reason == "outside_gating_window"
    assert not late.execute_corrector


def test_atomic_veto_blocks_when_cell_and_pos_are_stable() -> None:
    gate = ready_gate(
        atomic_veto_enabled=True,
        atomic_stability_threshold=0.02,
        atomic_min_stable_steps=1,
    )
    observe(
        gate,
        predictor_residuals=dict(RESIDUALS, atomic_numbers=1.03),
    )
    assert gate.field_converged["cell"]
    assert gate.field_converged["pos"]
    assert gate.field_converged["atomic_numbers"]
    assert not gate.atomic_stable
    decision = gate.decide(sampling_step=50, num_steps=100)
    assert decision.execute_corrector
    assert decision.atomic_veto
    assert decision.mode == "atomic_veto_corrector"


def test_atomic_minimum_stable_steps_is_a_hard_condition() -> None:
    gate = ready_gate(
        atomic_veto_enabled=True,
        atomic_stability_threshold=0.05,
        atomic_min_stable_steps=3,
    )
    assert gate.atomic_stable_steps == 1
    first_veto = gate.decide(sampling_step=50, num_steps=100)
    assert first_veto.atomic_veto
    observe(gate)
    second_veto = gate.decide(sampling_step=51, num_steps=100)
    assert second_veto.atomic_veto
    observe(gate)
    allowed = gate.decide(sampling_step=52, num_steps=100)
    assert not allowed.execute_corrector


def test_all_fields_is_stricter_than_weighted_rms() -> None:
    strict = ready_gate(field_aggregation="all_fields")
    rms = ready_gate(field_aggregation="weighted_rms")
    changed = dict(RESIDUALS, atomic_numbers=1.06)
    observe(strict, predictor_residuals=changed)
    observe(rms, predictor_residuals=changed)
    assert not strict.global_converged
    assert rms.global_converged
    assert strict.decide(sampling_step=50, num_steps=100).execute_corrector
    assert not rms.decide(
        sampling_step=50, num_steps=100
    ).execute_corrector


def test_adaptive_calibration_interval_grows_while_stable() -> None:
    gate = ready_gate(
        adaptive_calibration_enabled=True,
        calibration_interval_min=2,
        calibration_interval_max=8,
        calibration_interval=2,
    )
    for _ in range(2):
        skipped = gate.decide(sampling_step=50, num_steps=100)
        assert not skipped.execute_corrector
        gate.record_scheduled_decision(skipped)
    calibration = gate.decide(sampling_step=50, num_steps=100)
    assert calibration.forced_calibration
    gate.record_scheduled_decision(calibration)
    assert gate.current_calibration_interval == 3


def test_fallback_and_rescue_reset_adaptive_calibration() -> None:
    gate = ready_gate(
        adaptive_calibration_enabled=True,
        calibration_interval_min=2,
        calibration_interval_max=8,
        calibration_interval=2,
    )
    gate.current_calibration_interval = 6
    observe(
        gate,
        predictor_updates=dict(SMALL, pos=0.30),
    )
    fallback = gate.decide(sampling_step=50, num_steps=100)
    assert fallback.fallback
    gate.record_scheduled_decision(fallback)
    assert gate.current_calibration_interval == 2
    gate.current_calibration_interval = 6
    gate.record_rescue()
    assert gate.current_calibration_interval == 2


def test_original_g3_path_is_compatible_when_new_features_are_off() -> None:
    legacy_kwargs = dict(
        warmup_frac=0.15,
        min_progress=0.15,
        max_progress=0.95,
        convergence_threshold=1.7533084836308386,
        consecutive_stable_steps=2,
        calibration_interval=16,
        max_consecutive_skips=12,
        fallback_threshold=2.6299627254462576,
        rescue_enabled=True,
    )
    legacy = ConvergenceAwareCorrectorGate(**legacy_kwargs)
    explicit_off = ConvergenceAwareCorrectorGate(
        **legacy_kwargs,
        max_skip_ratio=None,
        atomic_veto_enabled=False,
        adaptive_calibration_enabled=False,
        field_aggregation="all_fields",
    )
    for step in range(80):
        observe(legacy)
        observe(explicit_off)
        left = legacy.decide(sampling_step=step, num_steps=100)
        right = explicit_off.decide(sampling_step=step, num_steps=100)
        assert left == right
        legacy.record_scheduled_decision(left)
        explicit_off.record_scheduled_decision(right)
    assert legacy.skip_budget_total is None
    assert explicit_off.skip_budget_total is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_skip_ratio": -0.1},
        {"max_skip_ratio": 1.1},
        {"atomic_min_stable_steps": 0},
        {"calibration_interval_min": 5, "calibration_interval_max": 4},
        {"field_aggregation": "mean"},
    ],
)
def test_budget_aware_parameter_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        ConvergenceAwareCorrectorGate(**kwargs)


def test_budget_state_is_cleared_on_batch_reset() -> None:
    gate = ready_gate(max_skip_ratio=0.25)
    skipped = gate.decide(sampling_step=0, num_steps=100)
    gate.record_scheduled_decision(skipped)
    assert gate.skip_budget_used == 1
    gate.reset()
    assert gate.skip_budget_used == 0
    assert gate.skip_budget_total is None
    assert gate.atomic_stable_steps == 0
