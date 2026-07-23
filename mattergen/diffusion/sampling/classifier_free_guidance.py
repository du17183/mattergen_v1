# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from mattergen.common.data.collate import collate
from mattergen.diffusion.sampling.guidance_schedule import GuidanceController
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
        selected_trace_path = guidance_trace_path or env_trace_path
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

    @property
    def guidance_schedule(self) -> str:
        return self._guidance_controller.schedule

    @property
    def guidance_ema_by_phase(self) -> Mapping[str, float | None]:
        return self._guidance_controller.ema_by_phase

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self._guidance_controller.reset()
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

    def _score_fn(
        self,
        x: Diffusable,
        t: torch.Tensor,
    ) -> Diffusable:
        """Evaluate conditional/unconditional scores and combine them with the active schedule."""

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
