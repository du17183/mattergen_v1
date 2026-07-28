from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from mattergen.diffusion.data.batched_data import SimpleBatchedData
from mattergen.diffusion.sampling.classifier_free_guidance import (
    GuidedPredictorCorrector,
)
from mattergen.diffusion.sampling.corrector_gating import (
    GATING_FIELDS,
    ConvergenceAwareCorrectorGate,
    CorrectorGateDecision,
    CorrectorGatingAccounting,
)
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.generator import CrystalGenerator
from mattergen.scripts.generate import main as generate_main


RESIDUALS = {field: 1.0 for field in GATING_FIELDS}
SMALL_UPDATES = {field: 0.01 for field in GATING_FIELDS}


def observe_step(
    gate: ConvergenceAwareCorrectorGate,
    *,
    residuals=RESIDUALS,
    predictor_updates=SMALL_UPDATES,
    corrector_updates=SMALL_UPDATES,
) -> None:
    gate.observe_residual(phase="corrector", residuals=residuals)
    gate.observe_update(phase="corrector", update_norms=corrector_updates)
    gate.observe_residual(phase="predictor", residuals=residuals)
    gate.observe_update(phase="predictor", update_norms=predictor_updates)
    gate.finalize_step()


def stable_gate(**kwargs) -> ConvergenceAwareCorrectorGate:
    gate = ConvergenceAwareCorrectorGate(
        warmup_frac=0.0,
        min_progress=0.0,
        max_progress=1.0,
        convergence_threshold=0.05,
        consecutive_stable_steps=kwargs.pop(
            "consecutive_stable_steps", 1
        ),
        calibration_interval=kwargs.pop("calibration_interval", 10),
        max_consecutive_skips=kwargs.pop("max_consecutive_skips", 8),
        fallback_threshold=kwargs.pop("fallback_threshold", 0.20),
        **kwargs,
    )
    observe_step(gate)
    observe_step(gate)
    return gate


def test_corrector_gating_defaults_disabled() -> None:
    assert (
        inspect.signature(GuidedPredictorCorrector.__init__)
        .parameters["corrector_gating_enabled"]
        .default
        is False
    )
    assert (
        CrystalGenerator.__dataclass_fields__[
            "corrector_gating_enabled"
        ].default
        is False
    )
    assert (
        inspect.signature(generate_main)
        .parameters["corrector_gating_enabled"]
        .default
        is False
    )


def test_warmup_never_skips_corrector() -> None:
    gate = ConvergenceAwareCorrectorGate(
        warmup_frac=0.10,
        min_progress=0.0,
        max_progress=1.0,
        consecutive_stable_steps=1,
    )
    observe_step(gate)
    observe_step(gate)
    decision = gate.decide(sampling_step=9, num_steps=100)
    assert decision.execute_corrector
    assert decision.mode == "warmup_full"


def test_unconverged_history_does_not_skip() -> None:
    gate = ConvergenceAwareCorrectorGate(
        warmup_frac=0.0,
        min_progress=0.0,
        max_progress=1.0,
        convergence_threshold=0.01,
    )
    observe_step(gate)
    observe_step(gate, residuals=dict(RESIDUALS, pos=2.0))
    decision = gate.decide(sampling_step=50, num_steps=100)
    assert decision.execute_corrector


def test_consecutive_convergence_allows_skip() -> None:
    gate = stable_gate()
    decision = gate.decide(sampling_step=50, num_steps=100)
    assert not decision.execute_corrector
    assert decision.mode == "skip_corrector"


def test_all_three_fields_participate_in_gate() -> None:
    gate = stable_gate()
    observe_step(gate, residuals=dict(RESIDUALS, atomic_numbers=2.0))
    assert gate.field_converged["cell"]
    assert gate.field_converged["pos"]
    assert not gate.field_converged["atomic_numbers"]
    assert not gate.global_converged


def test_calibration_interval_is_enforced() -> None:
    gate = stable_gate(
        calibration_interval=2, max_consecutive_skips=9
    )
    for _ in range(2):
        decision = gate.decide(sampling_step=50, num_steps=100)
        assert not decision.execute_corrector
        gate.record_scheduled_decision(decision)
    calibration = gate.decide(sampling_step=50, num_steps=100)
    assert calibration.execute_corrector
    assert calibration.mode == "forced_calibration"
    assert calibration.reason == "calibration_interval"


def test_max_consecutive_skips_is_enforced() -> None:
    gate = stable_gate(
        calibration_interval=99, max_consecutive_skips=1
    )
    skipped = gate.decide(sampling_step=50, num_steps=100)
    gate.record_scheduled_decision(skipped)
    decision = gate.decide(sampling_step=50, num_steps=100)
    assert decision.execute_corrector
    assert decision.reason == "max_consecutive_skips"


def test_fallback_is_triggered_by_field_change() -> None:
    gate = stable_gate(fallback_threshold=0.15)
    observe_step(
        gate,
        predictor_updates=dict(SMALL_UPDATES, pos=0.30),
    )
    decision = gate.decide(sampling_step=50, num_steps=100)
    assert decision.execute_corrector
    assert decision.mode == "fallback_corrector"
    assert decision.fallback
    assert "pos" in decision.reason


def test_rescue_and_accounting_are_correct() -> None:
    gate = stable_gate(fallback_threshold=0.20, rescue_enabled=True)
    skipped = gate.decide(sampling_step=50, num_steps=100)
    gate.record_scheduled_decision(skipped)
    gate.observe_update(
        phase="predictor",
        update_norms=dict(SMALL_UPDATES, cell=0.50),
    )
    rescue, reason = gate.should_rescue_after_predictor()
    assert rescue and "cell" in reason
    accounting = CorrectorGatingAccounting()
    accounting.record_skip()
    accounting.record_predictor_forward()
    accounting.record_corrector_forward(rescue=True)
    assert accounting.corrector_skipped_count == 1
    assert accounting.corrector_rescue_count == 1
    assert accounting.physical_model_forward_count == 2


def test_batch_reset_clears_history() -> None:
    gate = stable_gate()
    assert gate.stable_count > 0
    gate.reset()
    assert gate.stable_count == 0
    assert gate.consecutive_skip_count == 0
    assert not gate.global_converged


def test_predictor_and_corrector_histories_are_distinct() -> None:
    gate = ConvergenceAwareCorrectorGate()
    gate.observe_residual(phase="corrector", residuals=RESIDUALS)
    gate.observe_residual(
        phase="predictor", residuals=dict(RESIDUALS, cell=3.0)
    )
    snapshot = gate.snapshot()["residuals"]
    assert snapshot["corrector"]["cell"] == 1.0
    assert snapshot["predictor"]["cell"] == 3.0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_history_forces_fallback(value: float) -> None:
    gate = stable_gate()
    gate.observe_residual(
        phase="predictor", residuals=dict(RESIDUALS, pos=value)
    )
    gate.finalize_step()
    decision = gate.decide(sampling_step=50, num_steps=100)
    assert decision.execute_corrector
    assert decision.fallback


def test_physical_forward_accounting() -> None:
    accounting = CorrectorGatingAccounting()
    accounting.record_corrector_forward()
    accounting.record_predictor_forward()
    accounting.record_skip()
    summary = accounting.as_dict()
    assert summary["corrector_forward_count"] == 1
    assert summary["predictor_forward_count"] == 1
    assert summary["physical_model_forward_count"] == 2
    assert summary["joint_batch_forward_count"] == 2
    assert summary["logical_conditional_nfe"] == 2
    assert summary["logical_unconditional_nfe"] == 2


class _DummyMultiCorruption:
    T = 1.0

    @staticmethod
    def _get_batch_indices(batch):
        return {
            field: batch.get_batch_idx(field) for field in GATING_FIELDS
        }


class _DummyDiffusion:
    corruption = _DummyMultiCorruption()


class _RandomCorrector:
    @staticmethod
    def step_given_score(*, x, batch_idx, score, t, dt):
        del batch_idx, score, t, dt
        noise = torch.randn_like(x.float()).to(x.dtype)
        return x + noise, x + noise


class _RandomPredictor:
    @staticmethod
    def update_given_score(*, x, t, dt, batch_idx, score, batch):
        del t, dt, batch_idx, score, batch
        noise = torch.randn_like(x.float()).to(x.dtype)
        return x + noise, x + noise


class _LoopSampler(PredictorCorrector):
    def __init__(self, *, active: bool, execute: bool):
        self._diffusion_module = _DummyDiffusion()
        self._device = torch.device("cpu")
        self._max_t = 1.0
        self._eps_t = 0.1
        self.N = 2
        self._n_steps_corrector = 1
        self._predictors = {
            field: _RandomPredictor() for field in GATING_FIELDS
        }
        self._correctors = {
            field: _RandomCorrector() for field in GATING_FIELDS
        }
        self._active = active
        self._execute = execute
        self._sampling_context = {}
        self._score_call_index = 0
        self.score_calls: list[str] = []

    def _corrector_gating_active(self) -> bool:
        return self._active

    def _corrector_gate_should_execute(self, **kwargs) -> bool:
        del kwargs
        return self._execute

    def _score_fn(self, x, t):
        del t
        self.score_calls.append(self.sampling_context["phase"])
        return x.replace(
            **{field: torch.zeros_like(x[field]) for field in GATING_FIELDS}
        )


def _loop_batch() -> SimpleBatchedData:
    return SimpleBatchedData(
        data={
            "cell": torch.zeros(1, 3, 3),
            "pos": torch.zeros(2, 3),
            "atomic_numbers": torch.zeros(2, 1),
        },
        batch_idx={field: None for field in GATING_FIELDS},
    )


def test_skip_never_calls_corrector_score_and_predictor_still_runs() -> None:
    sampler = _LoopSampler(active=True, execute=False)
    sampler._denoise(batch=_loop_batch(), mask={}, record=False)
    assert sampler.score_calls == ["predictor", "predictor"]


def test_gated_full_loop_matches_frozen_loop_bitwise() -> None:
    baseline = _LoopSampler(active=False, execute=True)
    gated_full = _LoopSampler(active=True, execute=True)
    torch.manual_seed(123)
    baseline_result = baseline._denoise(
        batch=_loop_batch(), mask={}, record=False
    )[0]
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(123)
    gated_result = gated_full._denoise(
        batch=_loop_batch(), mask={}, record=False
    )[0]
    gated_rng = torch.random.get_rng_state().clone()
    for field in GATING_FIELDS:
        assert torch.equal(baseline_result[field], gated_result[field])
    assert torch.equal(baseline_rng, gated_rng)


def test_same_gating_config_and_seed_is_level1() -> None:
    first = _LoopSampler(active=True, execute=False)
    second = _LoopSampler(active=True, execute=False)
    torch.manual_seed(77)
    first_result = first._denoise(
        batch=_loop_batch(), mask={}, record=False
    )[0]
    torch.manual_seed(77)
    second_result = second._denoise(
        batch=_loop_batch(), mask={}, record=False
    )[0]
    for field in GATING_FIELDS:
        assert torch.equal(first_result[field], second_result[field])


def test_update_norms_reduce_each_field_independently() -> None:
    before = _loop_batch()
    after = before.replace(
        cell=torch.ones(1, 3, 3),
        pos=torch.ones(2, 3),
        atomic_numbers=torch.ones(2, 1),
    )
    norms = GuidedPredictorCorrector._normalized_update_rms(before, after)
    assert set(norms) == set(GATING_FIELDS)
    assert all(value is not None for value in norms.values())


def test_trace_row_construction_does_not_change_rng() -> None:
    sampler = object.__new__(GuidedPredictorCorrector)
    sampler._corrector_gate = ConvergenceAwareCorrectorGate()
    sampler._corrector_trace_path = Path("/tmp/not-written-by-unit-test.csv")
    sampler._corrector_trace_rows = []
    sampler._corrector_accounting = CorrectorGatingAccounting()
    sampler._corrector_current_decision = CorrectorGateDecision(
        True, "full_corrector", "unit"
    )
    sampler._corrector_rescue_performed = False
    sampler._corrector_rescue_reason = ""
    sampler._sample_seed = 1
    sampler.N = 2
    before = torch.random.get_rng_state().clone()
    sampler._on_timestep_end(
        batch=_loop_batch(),
        t=torch.ones(1),
        sampling_step=0,
        elapsed_ms=1.0,
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(sampler._corrector_trace_rows) == 1


def test_rescue_trace_reports_real_corrector_execution() -> None:
    sampler = object.__new__(GuidedPredictorCorrector)
    sampler._corrector_gate = ConvergenceAwareCorrectorGate()
    sampler._corrector_trace_path = Path("/tmp/not-written-by-unit-test.csv")
    sampler._corrector_trace_rows = []
    sampler._corrector_accounting = CorrectorGatingAccounting()
    sampler._corrector_accounting.record_predictor_forward()
    sampler._corrector_accounting.record_skip()
    sampler._corrector_accounting.record_corrector_forward(rescue=True)
    sampler._corrector_current_decision = CorrectorGateDecision(
        False, "skip_corrector", "all_fields_converged"
    )
    sampler._corrector_rescue_performed = True
    sampler._corrector_rescue_reason = "predictor_update_large:pos"
    sampler._sample_seed = 1
    sampler.N = 2
    sampler._on_timestep_end(
        batch=_loop_batch(),
        t=torch.ones(1),
        sampling_step=0,
        elapsed_ms=1.0,
    )
    row = sampler._corrector_trace_rows[0]
    assert row["decision"] == "rescue_corrector"
    assert row["corrector_executed"] is True
    assert row["corrector_skipped"] is True
    assert row["fallback"] is True
    assert row["fallback_reason"] == "predictor_update_large:pos"
    assert row["corrector_rescue_count"] == 1
    assert row["physical_forward_count"] == 2
