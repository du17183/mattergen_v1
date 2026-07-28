from __future__ import annotations

import inspect

import pytest
import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.diffusion.sampling.cfg_acceleration import (
    CFG_FIELDS,
    ConvergenceAwareCFGController,
    NFEAccounting,
)
from mattergen.diffusion.sampling.classifier_free_guidance import (
    GuidedPredictorCorrector,
)
from mattergen.diffusion.sampling.guidance_schedule import GuidanceController
from mattergen.generator import CrystalGenerator


RESIDUALS = {"cell": 1.0, "pos": 1.0, "atomic_numbers": 1.0}


def converged_controller(**kwargs) -> ConvergenceAwareCFGController:
    controller = ConvergenceAwareCFGController(
        warmup_frac=0.0,
        convergence_threshold=0.1,
        consecutive_stable_steps=kwargs.pop("consecutive_stable_steps", 1),
        calibration_interval=kwargs.pop("calibration_interval", 10),
        max_reuse_steps=kwargs.pop("max_reuse_steps", 8),
        **kwargs,
    )
    controller.observe_full(phase="predictor", residuals=RESIDUALS)
    controller.observe_full(phase="predictor", residuals=RESIDUALS)
    return controller


def test_acceleration_defaults_disabled() -> None:
    assert (
        inspect.signature(GuidedPredictorCorrector.__init__)
        .parameters["cfg_acceleration_enabled"]
        .default
        is False
    )
    assert CrystalGenerator.__dataclass_fields__["cfg_acceleration_enabled"].default is False
    assert CrystalGenerator.__dataclass_fields__["cfg_trace_mode"].default == "auto"


def test_warmup_only_runs_full_cfg() -> None:
    controller = ConvergenceAwareCFGController(warmup_frac=0.2)
    decision = controller.pre_decision(progress=0.1, phase="corrector", cache_valid=True)
    assert decision.run_full_cfg and decision.reason == "warmup"


def test_unstable_residual_does_not_reuse() -> None:
    controller = ConvergenceAwareCFGController(
        warmup_frac=0.0, convergence_threshold=0.01, consecutive_stable_steps=2
    )
    controller.observe_full(phase="predictor", residuals=RESIDUALS)
    changed = dict(RESIDUALS, pos=10.0)
    controller.observe_full(phase="predictor", residuals=changed)
    assert controller.pre_decision(
        progress=0.5, phase="predictor", cache_valid=True
    ).run_full_cfg


def test_converged_residual_can_reuse() -> None:
    decision = converged_controller().pre_decision(
        progress=0.5, phase="predictor", cache_valid=True
    )
    assert not decision.run_full_cfg and decision.mode == "reuse"


def test_calibration_interval_is_enforced() -> None:
    controller = converged_controller(calibration_interval=2, max_reuse_steps=9)
    controller.observe_reuse(phase="predictor")
    controller.observe_reuse(phase="predictor")
    decision = controller.pre_decision(progress=0.5, phase="predictor", cache_valid=True)
    assert decision.run_full_cfg and decision.mode == "periodic_calibration"


def test_max_reuse_steps_is_enforced() -> None:
    controller = converged_controller(calibration_interval=99, max_reuse_steps=1)
    controller.observe_reuse(phase="predictor")
    decision = controller.pre_decision(progress=0.5, phase="predictor", cache_valid=True)
    assert decision.run_full_cfg and decision.reason == "max_reuse_steps"


def test_cache_error_triggers_fallback() -> None:
    controller = converged_controller(fallback_threshold=0.2)
    observation = controller.observe_full(
        phase="predictor",
        residuals=RESIDUALS,
        cache_relative_errors={"cell": 0.0, "pos": 0.3, "atomic_numbers": 0.0},
        requested_mode="periodic_calibration",
    )
    assert observation.fallback
    assert observation.mode == "fallback_full_cfg"
    assert "pos" in observation.fallback_reason


def test_predictor_corrector_states_are_independent() -> None:
    controller = converged_controller()
    assert not controller.pre_decision(
        progress=0.5, phase="predictor", cache_valid=True
    ).run_full_cfg
    assert controller.pre_decision(
        progress=0.5, phase="corrector", cache_valid=True
    ).run_full_cfg


def test_batch_reset_clears_state() -> None:
    controller = converged_controller()
    controller.reset()
    assert controller.stable_count("predictor") == 0
    assert controller.pre_decision(
        progress=0.5, phase="predictor", cache_valid=False
    ).run_full_cfg


def test_all_fields_participate_in_convergence() -> None:
    controller = ConvergenceAwareCFGController(
        warmup_frac=0.0, convergence_threshold=0.01, consecutive_stable_steps=1
    )
    controller.observe_full(phase="predictor", residuals=RESIDUALS)
    observation = controller.observe_full(
        phase="predictor", residuals=dict(RESIDUALS, atomic_numbers=2.0)
    )
    assert observation.field_converged["cell"]
    assert observation.field_converged["pos"]
    assert not observation.field_converged["atomic_numbers"]
    assert not observation.global_converged


class _DummyCorruption:
    corrupted_fields = CFG_FIELDS


class _DummyDiffusion:
    def __init__(self) -> None:
        self.corruption = _DummyCorruption()
        self.batch_sizes: list[int] = []

    def score_fn(self, x, t):
        self.batch_sizes.append(len(t))
        flag = x.conditional.float()
        node_flag = flag[x.batch]
        return x.replace(
            cell=flag[:, None, None].expand(-1, 3, 3),
            pos=node_flag[:, None].expand(-1, 3),
            atomic_numbers=node_flag[:, None].expand(-1, 4),
        )


def _sampler(enabled: bool) -> GuidedPredictorCorrector:
    sampler = object.__new__(GuidedPredictorCorrector)
    sampler._diffusion_module = _DummyDiffusion()
    sampler._remove_conditioning_fn = lambda x: x.replace(
        conditional=torch.zeros(1, device=x.pos.device)
    )
    sampler._keep_conditioning_fn = lambda x: x.replace(
        conditional=torch.ones(1, device=x.pos.device)
    )
    sampler._guidance_scale = 2.0
    sampler._guidance_controller = GuidanceController(
        schedule="adaptive",
        base_guidance=2.0,
        warmup_frac=0.1,
        decay_frac=0.1,
        min_scale=0.0,
        max_scale=5.0,
        adaptive_alpha=0.25,
        adaptive_ema=0.95,
        adaptive_eps=1e-6,
    )
    sampler._cfg_acceleration_enabled = enabled
    sampler._cfg_controller = ConvergenceAwareCFGController(
        warmup_frac=0.0,
        convergence_threshold=1.0,
        consecutive_stable_steps=1,
        calibration_interval=10,
        max_reuse_steps=8,
    )
    sampler._sample_seed = 1
    sampler._run_id = "unit"
    sampler._trace_enabled = False
    sampler._trace_to_disk = False
    sampler._cfg_trace_mode = "off"
    sampler._cfg_summary_path = None
    sampler._trace_build_cpu_seconds = 0.0
    sampler._trace_write_cpu_seconds = 0.0
    sampler._guidance_trace_rows = []
    sampler._sampling_context = {
        "sampling_step": 1,
        "num_steps": 10,
        "progress": 0.5,
        "phase": "predictor",
        "score_call_index": 1,
    }
    sampler._reset_cfg_acceleration_state()
    return sampler


def _batch():
    return collate(
        [
            ChemGraph(
                cell=torch.zeros(1, 3, 3),
                pos=torch.zeros(2, 3),
                atomic_numbers=torch.ones(2, dtype=torch.long),
                conditional=torch.ones(1),
            )
        ]
    )


def test_reuse_executes_conditional_only_branch() -> None:
    sampler = _sampler(True)
    x = _batch()
    t = torch.ones(1)
    sampler._score_fn(x, t)
    sampler._score_fn(x, t)
    sampler._score_fn(x, t)
    assert sampler._diffusion_module.batch_sizes == [2, 2, 1]
    assert sampler.cfg_nfe_summary["conditional_only_forward_count"] == 1


def test_logical_nfe_accounting() -> None:
    nfe = NFEAccounting()
    nfe.record_full("full_cfg")
    nfe.record_reuse(False)
    assert nfe.conditional_logical_nfe == 2
    assert nfe.unconditional_logical_nfe == 1


def test_physical_forward_accounting() -> None:
    nfe = NFEAccounting()
    nfe.record_full("periodic_calibration")
    nfe.record_reuse(True)
    assert nfe.physical_model_forward_count == 2
    assert nfe.joint_batch_forward_count == 1
    assert nfe.conditional_only_forward_count == 1


def test_trace_disabled_does_not_change_rng() -> None:
    sampler = _sampler(True)
    before = torch.random.get_rng_state().clone()
    sampler._trace_acceleration(
        t=torch.ones(1),
        mode="reuse",
        field_deltas=RESIDUALS,
        decision={"final_guidance": 2.0},
        observation=None,
        calibration_due=False,
        fallback=False,
        fallback_reason="",
        elapsed_ms=1.0,
    )
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_residual_falls_back(value: float) -> None:
    controller = ConvergenceAwareCFGController()
    observation = controller.observe_full(
        phase="predictor", residuals=dict(RESIDUALS, cell=value)
    )
    assert observation.fallback
    assert observation.mode == "fallback_full_cfg"


def test_invalid_cache_forces_full_cfg() -> None:
    controller = converged_controller()
    decision = controller.pre_decision(
        progress=0.5, phase="predictor", cache_valid=False
    )
    assert decision.run_full_cfg and decision.mode == "fallback_full_cfg"


def test_disabled_dispatch_matches_frozen_full_cfg_bitwise() -> None:
    dispatched = _sampler(False)
    direct = _sampler(False)
    x1, x2 = _batch(), _batch()
    t = torch.ones(1)
    a = dispatched._score_fn(x1, t)
    b = direct._score_fn_unaccelerated(x2, t)
    for field in CFG_FIELDS:
        assert torch.equal(a[field], b[field])
