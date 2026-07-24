# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from mattergen.common.data.collate import collate
from mattergen.diffusion.sampling.guidance_schedule import GuidanceController
from mattergen.diffusion.sampling.cfg_acceleration import (
    CFG_FIELDS, AccelerationObservation, ConvergenceAwareCFGController, NFEAccounting,
)
from mattergen.diffusion.sampling.pc_sampler import Diffusable, PredictorCorrector

BatchTransform = Callable[[Diffusable], Diffusable]
TRACE_FIELD_NAMES = (
    "run_id",
    "sample_seed",
    "sampling_step",
    "num_steps",
    "score_call_index",
    "phase",
    "t",
    "progress",
    "guidance_schedule",
    "base_guidance",
    "stage_guidance",
    "delta_cell",
    "delta_pos",
    "delta_atomic_numbers",
    "delta",
    "ema",
    "ratio",
    "adaptive_multiplier",
    "final_guidance",
    "fallback_reason",
    "sample_id",
    "mode",
    "cell_residual",
    "pos_residual",
    "atomic_residual",
    "cell_ema",
    "pos_ema",
    "atomic_ema",
    "cell_converged",
    "pos_converged",
    "atomic_converged",
    "converged",
    "stable_count",
    "reuse_count",
    "calibration_due",
    "fallback",
    "conditional_nfe",
    "unconditional_nfe",
    "physical_forward_count",
    "joint_batch_forward_count",
    "conditional_only_forward_count",
    "elapsed_ms",
)


def identity(x: Diffusable) -> Diffusable:
    """Default function that transforms data to its conditional state."""

    return x


def score_residual_rms(
    *,
    unconditional_score: Diffusable,
    conditional_score: Diffusable,
    fields: tuple[str, ...],
) -> tuple[dict[str, float | None], str | None]:
    """Reduce each differently shaped score residual to an independent scalar RMS."""

    deltas: dict[str, float | None] = {field: None for field in fields}
    errors: list[str] = []
    for field in fields:
        try:
            unconditional = unconditional_score[field]
            conditional = conditional_score[field]
        except (KeyError, TypeError):
            errors.append(f"{field}:missing")
            continue
        if unconditional is None or conditional is None:
            errors.append(f"{field}:none")
            continue
        if not isinstance(unconditional, torch.Tensor) or not isinstance(conditional, torch.Tensor):
            errors.append(f"{field}:not_tensor")
            continue
        if unconditional.shape != conditional.shape:
            errors.append(f"{field}:shape_mismatch")
            continue
        if unconditional.numel() == 0:
            errors.append(f"{field}:empty")
            continue
        residual = conditional - unconditional
        if not bool(torch.isfinite(residual).all().item()):
            errors.append(f"{field}:non_finite")
            continue
        deltas[field] = float(torch.sqrt(torch.mean(residual.float().square())).item())

    return deltas, ";".join(errors) if errors else None


class GuidedPredictorCorrector(PredictorCorrector):
    """Predictor-corrector sampler with constant and scheduled CFG."""

    def __init__(
        self,
        *,
        guidance_scale: float,
        remove_conditioning_fn: BatchTransform,
        keep_conditioning_fn: BatchTransform | None = None,
        guidance_schedule: str = "constant",
        guidance_warmup_frac: float = 0.1,
        guidance_decay_frac: float = 0.1,
        guidance_min_scale: float = 0.0,
        guidance_max_scale: float = 5.0,
        guidance_adaptive_alpha: float = 0.5,
        guidance_adaptive_ema: float = 0.95,
        guidance_adaptive_eps: float = 1e-6,
        guidance_trace_path: str | None = None,
        cfg_acceleration_enabled: bool = False,
        cfg_warmup_frac: float = 0.15,
        cfg_convergence_threshold: float = 0.05,
        cfg_consecutive_stable_steps: int = 3,
        cfg_calibration_interval: int = 10,
        cfg_max_reuse_steps: int = 8,
        cfg_extrapolation_enabled: bool = False,
        cfg_extrapolation_order: int = 1,
        cfg_fallback_threshold: float = 0.20,
        cfg_min_progress: float = 0.0,
        cfg_max_progress: float = 1.0,
        cfg_trace_path: str | None = None,
        sample_seed: int | None = None,
        run_id: str | None = None,
        **kwargs,
    ):
        """
        Args:
            guidance_scale: Base CFG scale.
            remove_conditioning_fn: Transform producing unconditional input.
            keep_conditioning_fn: Transform producing conditional input.
            guidance_schedule: One of constant, piecewise, adaptive, stage_adaptive.
            guidance_trace_path: Optional absolute CSV path. ``GUIDANCE_TRACE_PATH``
                is used when this value is omitted.
            sample_seed: Seed metadata for the trace; RNG seeding is performed once
                by :class:`CrystalGenerator`, not by the sampler.
        """

        super().__init__(**kwargs)
        self._remove_conditioning_fn = remove_conditioning_fn
        self._keep_conditioning_fn = keep_conditioning_fn or identity
        self._guidance_scale = float(guidance_scale)
        self._guidance_controller = GuidanceController(
            schedule=guidance_schedule,
            base_guidance=self._guidance_scale,
            warmup_frac=guidance_warmup_frac,
            decay_frac=guidance_decay_frac,
            min_scale=guidance_min_scale,
            max_scale=guidance_max_scale,
            adaptive_alpha=guidance_adaptive_alpha,
            adaptive_ema=guidance_adaptive_ema,
            adaptive_eps=guidance_adaptive_eps,
        )
        self._sample_seed = sample_seed
        self._run_id = run_id or f"{self._guidance_controller.schedule}_seed_{sample_seed}"
        env_trace_path = os.environ.get("GUIDANCE_TRACE_PATH")
        selected_trace_path = cfg_trace_path or guidance_trace_path or env_trace_path
        save_trace = os.environ.get("SAVE_GUIDANCE_TRACE", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._trace_enabled = save_trace or selected_trace_path is not None
        if self._trace_enabled and selected_trace_path is None:
            raise ValueError(
                "Guidance trace is enabled but no guidance_trace_path or GUIDANCE_TRACE_PATH was set"
            )
        self._guidance_trace_path = (
            Path(selected_trace_path).expanduser() if selected_trace_path is not None else None
        )
        if self._guidance_trace_path is not None and not self._guidance_trace_path.is_absolute():
            raise ValueError("guidance_trace_path must be an absolute path")
        self._guidance_trace_rows: list[dict[str, Any]] = []
        self._cfg_acceleration_enabled = bool(cfg_acceleration_enabled)
        self._cfg_controller = ConvergenceAwareCFGController(
            warmup_frac=cfg_warmup_frac,
            convergence_threshold=cfg_convergence_threshold,
            consecutive_stable_steps=cfg_consecutive_stable_steps,
            calibration_interval=cfg_calibration_interval,
            max_reuse_steps=cfg_max_reuse_steps,
            extrapolation_enabled=cfg_extrapolation_enabled,
            extrapolation_order=cfg_extrapolation_order,
            fallback_threshold=cfg_fallback_threshold,
            min_progress=cfg_min_progress,
            max_progress=cfg_max_progress,
        )
        self._cfg_nfe = NFEAccounting()
        self._cfg_residual_cache: dict[str, dict[str, torch.Tensor | None]] = {}
        self._cfg_previous_residual_cache: dict[str, dict[str, torch.Tensor | None]] = {}
        self._cfg_guidance_cache: dict[str, float | None] = {}
        self._cfg_last_observation: dict[str, AccelerationObservation | None] = {}
        self._reset_cfg_acceleration_state()

    @property
    def guidance_schedule(self) -> str:
        return self._guidance_controller.schedule

    @property
    def guidance_ema_by_phase(self) -> Mapping[str, float | None]:
        return self._guidance_controller.ema_by_phase

    @property
    def cfg_nfe_summary(self) -> Mapping[str, int]:
        return self._cfg_nfe.as_dict()

    def _reset_cfg_acceleration_state(self) -> None:
        self._cfg_controller.reset()
        self._cfg_nfe = NFEAccounting()
        self._cfg_residual_cache = {
            phase: {field: None for field in CFG_FIELDS}
            for phase in ("corrector", "predictor")
        }
        self._cfg_previous_residual_cache = {
            phase: {field: None for field in CFG_FIELDS}
            for phase in ("corrector", "predictor")
        }
        self._cfg_guidance_cache = {"corrector": None, "predictor": None}
        self._cfg_last_observation = {"corrector": None, "predictor": None}

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self._guidance_controller.reset()
        self._reset_cfg_acceleration_state()
        self._guidance_trace_rows = []
        if self._trace_enabled:
            assert self._guidance_trace_path is not None
            if self._guidance_trace_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing guidance trace: {self._guidance_trace_path}"
                )

    def _on_sampling_end(self, error: BaseException | None) -> None:
        if not self._trace_enabled:
            return
        assert self._guidance_trace_path is not None
        self._guidance_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._guidance_trace_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRACE_FIELD_NAMES)
            writer.writeheader()
            writer.writerows(self._guidance_trace_rows)

    def _trace_decision(
        self,
        *,
        t: torch.Tensor,
        field_deltas: Mapping[str, float | None],
        decision: Mapping[str, float | str | None],
    ) -> None:
        if not self._trace_enabled:
            return
        context = self.sampling_context
        self._guidance_trace_rows.append(
            {
                "run_id": self._run_id,
                "sample_seed": self._sample_seed,
                "sampling_step": context.get("sampling_step"),
                "num_steps": context.get("num_steps"),
                "score_call_index": context.get("score_call_index"),
                "phase": context.get("phase"),
                "t": float(t.detach().reshape(-1)[0].cpu().item()),
                "progress": context.get("progress"),
                "guidance_schedule": self._guidance_controller.schedule,
                "base_guidance": self._guidance_scale,
                "stage_guidance": decision["stage_guidance"],
                "delta_cell": field_deltas.get("cell"),
                "delta_pos": field_deltas.get("pos"),
                "delta_atomic_numbers": field_deltas.get("atomic_numbers"),
                "delta": decision["delta"],
                "ema": decision["ema"],
                "ratio": decision["ratio"],
                "adaptive_multiplier": decision["adaptive_multiplier"],
                "final_guidance": decision["final_guidance"],
                "fallback_reason": decision["fallback_reason"],
            }
        )

    def _current_progress_and_phase(self) -> tuple[float, str]:
        context = self.sampling_context
        progress = float(context.get("progress", 0.0))
        phase = str(context.get("phase", "predictor"))
        return progress, phase

    def _cache_valid(self, phase: str) -> bool:
        cache = self._cfg_residual_cache.get(phase, {})
        return all(
            isinstance(cache.get(field), torch.Tensor)
            and cache[field].numel() > 0
            and bool(torch.isfinite(cache[field]).all().item())
            for field in CFG_FIELDS
        )

    def _cached_residual(self, phase: str, extrapolate: bool) -> dict[str, torch.Tensor]:
        current = self._cfg_residual_cache[phase]
        previous = self._cfg_previous_residual_cache[phase]
        result: dict[str, torch.Tensor] = {}
        for field in CFG_FIELDS:
            value = current[field]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"invalid cache for {phase}/{field}")
            if extrapolate and isinstance(previous[field], torch.Tensor):
                value = value + (value - previous[field])
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"non-finite cache for {phase}/{field}")
            result[field] = value
        return result

    @staticmethod
    def _cache_relative_errors(
        predicted: Mapping[str, torch.Tensor], actual: Mapping[str, torch.Tensor]
    ) -> dict[str, float | None]:
        errors: dict[str, float | None] = {}
        for field in CFG_FIELDS:
            try:
                if predicted[field].shape != actual[field].shape:
                    raise ValueError("shape mismatch")
                numerator = torch.sqrt(torch.mean((predicted[field] - actual[field]).float().square()))
                denominator = torch.sqrt(torch.mean(actual[field].float().square())) + 1e-8
                value = numerator / denominator
                errors[field] = float(value.item()) if bool(torch.isfinite(value).item()) else None
            except (KeyError, TypeError, ValueError):
                errors[field] = None
        return errors

    def _trace_acceleration(
        self, *, t: torch.Tensor, mode: str, field_deltas: Mapping[str, float | None],
        decision: Mapping[str, float | str | None], observation: AccelerationObservation | None,
        calibration_due: bool, fallback: bool, fallback_reason: str, elapsed_ms: float,
    ) -> None:
        if not self._trace_enabled:
            return
        context = self.sampling_context
        ema = observation.field_ema if observation else self._cfg_controller.state_for_phase(str(context.get("phase", "predictor"))).residual_ema
        flags = observation.field_converged if observation else {field: True for field in CFG_FIELDS}
        nfe = self._cfg_nfe.as_dict()
        self._guidance_trace_rows.append({
            "run_id": self._run_id, "sample_id": self._run_id, "sample_seed": self._sample_seed,
            "sampling_step": context.get("sampling_step"), "num_steps": context.get("num_steps"),
            "score_call_index": context.get("score_call_index"), "phase": context.get("phase"),
            "t": float(t.detach().reshape(-1)[0].cpu().item()), "progress": context.get("progress"),
            "guidance_schedule": self._guidance_controller.schedule, "base_guidance": self._guidance_scale,
            "stage_guidance": decision.get("stage_guidance"), "delta_cell": field_deltas.get("cell"),
            "delta_pos": field_deltas.get("pos"), "delta_atomic_numbers": field_deltas.get("atomic_numbers"),
            "delta": decision.get("delta"), "ema": decision.get("ema"), "ratio": decision.get("ratio"),
            "adaptive_multiplier": decision.get("adaptive_multiplier"), "final_guidance": decision.get("final_guidance"),
            "fallback_reason": fallback_reason, "mode": mode, "cell_residual": field_deltas.get("cell"),
            "pos_residual": field_deltas.get("pos"), "atomic_residual": field_deltas.get("atomic_numbers"),
            "cell_ema": ema.get("cell"), "pos_ema": ema.get("pos"), "atomic_ema": ema.get("atomic_numbers"),
            "cell_converged": flags.get("cell", False), "pos_converged": flags.get("pos", False),
            "atomic_converged": flags.get("atomic_numbers", False),
            "converged": observation.global_converged if observation else True,
            "stable_count": self._cfg_controller.stable_count(str(context.get("phase", "predictor"))),
            "reuse_count": self._cfg_controller.reuse_count(str(context.get("phase", "predictor"))),
            "calibration_due": calibration_due, "fallback": fallback,
            "conditional_nfe": nfe["conditional_logical_nfe"],
            "unconditional_nfe": nfe["unconditional_logical_nfe"],
            "physical_forward_count": nfe["physical_model_forward_count"],
            "joint_batch_forward_count": nfe["joint_batch_forward_count"],
            "conditional_only_forward_count": nfe["conditional_only_forward_count"],
            "elapsed_ms": elapsed_ms,
        })

    def _score_fn_accelerated(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        fields = tuple(self._multi_corruption.corrupted_fields)
        if not all(field in fields for field in CFG_FIELDS):
            return self._score_fn_unaccelerated(x, t)
        progress, phase = self._current_progress_and_phase()
        pre = self._cfg_controller.pre_decision(
            progress=progress, phase=phase, cache_valid=self._cache_valid(phase)
        )
        started = time.perf_counter()
        if not pre.run_full_cfg:
            conditional = super(GuidedPredictorCorrector, self)._score_fn(
                x=self._keep_conditioning_fn(x), t=t
            )
            extrapolate = pre.mode == "extrapolate"
            cached = self._cached_residual(phase, extrapolate)
            if any(cached[field].shape != conditional[field].shape for field in CFG_FIELDS):
                raise ValueError(f"conditional/cache shape mismatch in {phase}")
            guidance = self._cfg_guidance_cache[phase]
            if guidance is None or not math.isfinite(guidance):
                raise ValueError(f"invalid guidance cache in {phase}")
            result = conditional.replace(**{
                field: conditional[field] + (guidance - 1.0) * cached[field]
                for field in fields
            })
            self._cfg_controller.observe_reuse(phase=phase)
            self._cfg_nfe.record_reuse(extrapolate=extrapolate)
            last = self._cfg_last_observation[phase]
            field_deltas = {field: self._cfg_controller.state_for_phase(phase).previous_residual[field] for field in CFG_FIELDS}
            decision = {"stage_guidance": guidance, "delta": None, "ema": None, "ratio": None,
                        "adaptive_multiplier": None, "final_guidance": guidance}
            self._trace_acceleration(t=t, mode=pre.mode, field_deltas=field_deltas, decision=decision,
                observation=last, calibration_due=False, fallback=False, fallback_reason="",
                elapsed_ms=(time.perf_counter()-started)*1000.0)
            return result

        batch_no_condition = self._remove_conditioning_fn(x)
        batch_with_condition = self._keep_conditioning_fn(x)
        joint_batch = collate([batch_no_condition, batch_with_condition])
        for attr, value in batch_no_condition.items():
            if isinstance(value, list):
                joint_batch[attr] = batch_no_condition[attr] + batch_with_condition[attr]
        combined = super(GuidedPredictorCorrector, self)._score_fn(
            x=joint_batch, t=torch.cat([t, t], dim=0)
        )
        unconditional, conditional = combined[0], combined[1]
        actual = {field: (conditional[field]-unconditional[field]).detach().clone() for field in CFG_FIELDS}
        field_deltas, residual_error = score_residual_rms(
            unconditional_score=unconditional, conditional_score=conditional, fields=fields
        )
        cache_errors = None
        if pre.mode == "periodic_calibration" and self._cache_valid(phase):
            cache_errors = self._cache_relative_errors(
                self._cached_residual(phase, self._cfg_controller.extrapolation_enabled), actual
            )
        decision_obj = self._guidance_controller.evaluate(
            progress=progress, phase=phase, field_deltas=field_deltas, residual_error=residual_error
        )
        observation = self._cfg_controller.observe_full(
            phase=phase, residuals=field_deltas, cache_relative_errors=cache_errors, requested_mode=pre.mode
        )
        self._cfg_previous_residual_cache[phase] = self._cfg_residual_cache[phase]
        self._cfg_residual_cache[phase] = actual
        self._cfg_guidance_cache[phase] = decision_obj.final_guidance
        self._cfg_last_observation[phase] = observation
        self._cfg_nfe.record_full(observation.mode)
        result = unconditional.replace(**{
            field: torch.lerp(unconditional[field], conditional[field], decision_obj.final_guidance)
            for field in fields
        })
        self._trace_acceleration(t=t, mode=observation.mode, field_deltas=field_deltas,
            decision=decision_obj.as_dict(), observation=observation,
            calibration_due=pre.mode == "periodic_calibration", fallback=observation.fallback,
            fallback_reason=observation.fallback_reason or residual_error or "",
            elapsed_ms=(time.perf_counter()-started)*1000.0)
        return result

    def _score_fn(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        if not self._cfg_acceleration_enabled:
            return self._score_fn_unaccelerated(x, t)
        return self._score_fn_accelerated(x, t)

    def _score_fn_unaccelerated(
        self,
        x: Diffusable,
        t: torch.Tensor,
    ) -> Diffusable:
        """Frozen full-CFG implementation used whenever acceleration is disabled."""

        def get_unconditional_score():
            return super(GuidedPredictorCorrector, self)._score_fn(
                x=self._remove_conditioning_fn(x), t=t
            )

        def get_conditional_score():
            return super(GuidedPredictorCorrector, self)._score_fn(
                x=self._keep_conditioning_fn(x), t=t
            )

        # Preserve the two optimized branches of the official constant CFG path.
        if self.guidance_schedule == "constant" and abs(self._guidance_scale - 1) < 1e-15:
            return get_conditional_score()
        if self.guidance_schedule == "constant" and abs(self._guidance_scale) < 1e-15:
            return get_unconditional_score()

        batch_no_condition = self._remove_conditioning_fn(x)
        batch_with_condition = self._keep_conditioning_fn(x)
        joint_batch = collate([batch_no_condition, batch_with_condition])

        for attr, value in batch_no_condition.items():
            if isinstance(value, list):
                joint_batch[attr] = batch_no_condition[attr] + batch_with_condition[attr]

        combined_score = super(GuidedPredictorCorrector, self)._score_fn(
            x=joint_batch,
            t=torch.cat([t, t], dim=0),
        )
        unconditional_score = combined_score[0]
        conditional_score = combined_score[1]

        # This exact branch retains the original torch.lerp call and arithmetic
        # whenever users do not request a new schedule or trace.
        if self.guidance_schedule == "constant" and not self._trace_enabled:
            return unconditional_score.replace(
                **{
                    field: torch.lerp(
                        unconditional_score[field],
                        conditional_score[field],
                        self._guidance_scale,
                    )
                    for field in self._multi_corruption.corrupted_fields
                }
            )

        fields = tuple(self._multi_corruption.corrupted_fields)
        field_deltas, residual_error = score_residual_rms(
            unconditional_score=unconditional_score,
            conditional_score=conditional_score,
            fields=fields,
        )
        progress, phase = self._current_progress_and_phase()
        decision = self._guidance_controller.evaluate(
            progress=progress,
            phase=phase,
            field_deltas=field_deltas,
            residual_error=residual_error,
        )
        decision_dict = decision.as_dict()
        self._trace_decision(t=t, field_deltas=field_deltas, decision=decision_dict)
        return unconditional_score.replace(
            **{
                field: torch.lerp(
                    unconditional_score[field],
                    conditional_score[field],
                    decision.final_guidance,
                )
                for field in fields
            }
        )
