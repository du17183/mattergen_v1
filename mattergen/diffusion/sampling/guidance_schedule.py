# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stateful guidance schedules used by classifier-free sampling.

The controller in this module is deliberately independent of MatterGen batch
types.  It consumes scalar residual summaries, which keeps differently shaped
``cell``, ``pos`` and ``atomic_numbers`` scores separate until after each field
has been reduced to one RMS value.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping


GUIDANCE_SCHEDULES = ("constant", "piecewise", "adaptive", "stage_adaptive")
PHASES = ("corrector", "predictor")


def normalize_guidance_schedule(value: str) -> str:
    """Normalize the CLI-friendly spelling and validate the schedule name."""

    normalized = value.replace("-", "_").lower()
    if normalized not in GUIDANCE_SCHEDULES:
        raise ValueError(
            f"guidance_schedule must be one of {GUIDANCE_SCHEDULES}, got {value!r}"
        )
    return normalized


def piecewise_guidance(
    *,
    progress: float,
    base_guidance: float,
    warmup_frac: float,
    decay_frac: float,
    min_scale: float,
) -> float:
    """Return the warmup/middle/decay guidance value for forward progress."""

    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be finite and in [0, 1], got {progress}")

    if warmup_frac > 0.0 and progress < warmup_frac:
        fraction = progress / warmup_frac
        return min_scale + (base_guidance - min_scale) * fraction

    decay_start = 1.0 - decay_frac
    if decay_frac > 0.0 and progress > decay_start:
        fraction = (progress - decay_start) / decay_frac
        return base_guidance + (min_scale - base_guidance) * fraction

    return base_guidance


@dataclass(frozen=True)
class GuidanceDecision:
    """One schedule decision, ready to be serialized in a guidance trace."""

    stage_guidance: float
    delta: float | None
    ema: float | None
    ratio: float | None
    adaptive_multiplier: float
    final_guidance: float
    fallback_reason: str | None

    def as_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


class GuidanceController:
    """Compute stage/adaptive guidance while isolating corrector/predictor EMA."""

    def __init__(
        self,
        *,
        schedule: str,
        base_guidance: float,
        warmup_frac: float = 0.1,
        decay_frac: float = 0.1,
        min_scale: float = 0.0,
        max_scale: float = 5.0,
        adaptive_alpha: float = 0.5,
        adaptive_ema: float = 0.95,
        adaptive_eps: float = 1e-6,
    ) -> None:
        self.schedule = normalize_guidance_schedule(schedule)
        self.base_guidance = float(base_guidance)
        self.warmup_frac = float(warmup_frac)
        self.decay_frac = float(decay_frac)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.adaptive_alpha = float(adaptive_alpha)
        self.adaptive_ema = float(adaptive_ema)
        self.adaptive_eps = float(adaptive_eps)
        self._validate()
        self.reset()

    @property
    def uses_piecewise_schedule(self) -> bool:
        return self.schedule in ("piecewise", "stage_adaptive")

    @property
    def uses_adaptive_feedback(self) -> bool:
        return self.schedule in ("adaptive", "stage_adaptive")

    @property
    def ema_by_phase(self) -> dict[str, float | None]:
        return dict(self._ema_by_phase)

    def reset(self) -> None:
        """Reset all run-local state; call once before each sampled batch."""

        self._ema_by_phase: dict[str, float | None] = {phase: None for phase in PHASES}

    def _validate(self) -> None:
        finite_values = {
            "base_guidance": self.base_guidance,
            "guidance_warmup_frac": self.warmup_frac,
            "guidance_decay_frac": self.decay_frac,
            "guidance_min_scale": self.min_scale,
            "guidance_max_scale": self.max_scale,
            "guidance_adaptive_alpha": self.adaptive_alpha,
            "guidance_adaptive_ema": self.adaptive_ema,
            "guidance_adaptive_eps": self.adaptive_eps,
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if not 0.0 <= self.warmup_frac < 1.0:
            raise ValueError("guidance_warmup_frac must satisfy 0 <= value < 1")
        if not 0.0 <= self.decay_frac < 1.0:
            raise ValueError("guidance_decay_frac must satisfy 0 <= value < 1")
        if self.warmup_frac + self.decay_frac > 1.0:
            raise ValueError("guidance_warmup_frac + guidance_decay_frac must be <= 1")
        if self.min_scale > self.max_scale:
            raise ValueError("guidance_min_scale must be <= guidance_max_scale")
        if self.schedule != "constant" and not self.min_scale <= self.base_guidance <= self.max_scale:
            raise ValueError(
                "base guidance must lie within [guidance_min_scale, guidance_max_scale] "
                "for non-constant schedules"
            )
        if self.adaptive_alpha < 0.0:
            raise ValueError("guidance_adaptive_alpha must be >= 0")
        if not 0.0 <= self.adaptive_ema < 1.0:
            raise ValueError("guidance_adaptive_ema must satisfy 0 <= value < 1")
        if self.adaptive_eps <= 0.0:
            raise ValueError("guidance_adaptive_eps must be > 0")

    def stage_guidance(self, progress: float) -> float:
        if not self.uses_piecewise_schedule:
            if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
                raise ValueError(f"progress must be finite and in [0, 1], got {progress}")
            return self.base_guidance
        return piecewise_guidance(
            progress=progress,
            base_guidance=self.base_guidance,
            warmup_frac=self.warmup_frac,
            decay_frac=self.decay_frac,
            min_scale=self.min_scale,
        )

    def evaluate(
        self,
        *,
        progress: float,
        phase: str,
        field_deltas: Mapping[str, float | None] | None,
        residual_error: str | None = None,
    ) -> GuidanceDecision:
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")

        stage_guidance = self.stage_guidance(progress)
        if not self.uses_adaptive_feedback:
            return GuidanceDecision(
                stage_guidance=stage_guidance,
                delta=_mean_valid_deltas(field_deltas),
                ema=None,
                ratio=None,
                adaptive_multiplier=1.0,
                final_guidance=stage_guidance,
                fallback_reason=residual_error,
            )

        delta = _mean_valid_deltas(field_deltas)
        if residual_error is not None or delta is None or not math.isfinite(delta):
            return GuidanceDecision(
                stage_guidance=stage_guidance,
                delta=delta,
                ema=self._ema_by_phase[phase],
                ratio=None,
                adaptive_multiplier=1.0,
                final_guidance=stage_guidance,
                fallback_reason=residual_error or "invalid_or_empty_residual",
            )

        previous_ema = self._ema_by_phase[phase]
        # First observation establishes the phase baseline instead of producing
        # an artificial multiplier spike against an all-zero history.
        ema = (
            delta
            if previous_ema is None
            else self.adaptive_ema * previous_ema + (1.0 - self.adaptive_ema) * delta
        )
        if not math.isfinite(ema):
            return GuidanceDecision(
                stage_guidance=stage_guidance,
                delta=delta,
                ema=previous_ema,
                ratio=None,
                adaptive_multiplier=1.0,
                final_guidance=stage_guidance,
                fallback_reason="non_finite_ema",
            )

        self._ema_by_phase[phase] = ema
        ratio = delta / (ema + self.adaptive_eps)
        multiplier = 1.0 + self.adaptive_alpha * (ratio - 1.0)
        multiplier = min(max(multiplier, 0.25), 4.0)
        final_guidance = min(max(stage_guidance * multiplier, self.min_scale), self.max_scale)
        if not all(math.isfinite(x) for x in (ratio, multiplier, final_guidance)):
            return GuidanceDecision(
                stage_guidance=stage_guidance,
                delta=delta,
                ema=ema,
                ratio=None,
                adaptive_multiplier=1.0,
                final_guidance=stage_guidance,
                fallback_reason="non_finite_adaptive_guidance",
            )

        return GuidanceDecision(
            stage_guidance=stage_guidance,
            delta=delta,
            ema=ema,
            ratio=ratio,
            adaptive_multiplier=multiplier,
            final_guidance=final_guidance,
            fallback_reason=None,
        )


def _mean_valid_deltas(field_deltas: Mapping[str, float | None] | None) -> float | None:
    if not field_deltas:
        return None
    values = [value for value in field_deltas.values() if value is not None]
    if not values or not all(math.isfinite(value) for value in values):
        return None
    return sum(values) / len(values)
