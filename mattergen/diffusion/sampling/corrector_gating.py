"""Convergence-aware gating for complete predictor-corrector model calls."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping

GATING_FIELDS = ("cell", "pos", "atomic_numbers")
GATING_PHASES = ("corrector", "predictor")


@dataclass(frozen=True)
class CorrectorGateDecision:
    execute_corrector: bool
    mode: str
    reason: str
    forced_calibration: bool = False
    fallback: bool = False


@dataclass
class _FieldHistory:
    residual: float | None = None
    residual_ema: float | None = None
    residual_relative_change: float | None = None
    update_norm: float | None = None


@dataclass
class CorrectorGatingAccounting:
    predictor_forward_count: int = 0
    corrector_forward_count: int = 0
    corrector_skipped_count: int = 0
    corrector_calibration_count: int = 0
    corrector_fallback_count: int = 0
    corrector_rescue_count: int = 0
    physical_model_forward_count: int = 0
    joint_batch_forward_count: int = 0
    conditional_only_forward_count: int = 0
    logical_conditional_nfe: int = 0
    logical_unconditional_nfe: int = 0

    def record_predictor_forward(self) -> None:
        self.predictor_forward_count += 1
        self._record_joint_forward()

    def record_corrector_forward(self, *, rescue: bool = False) -> None:
        self.corrector_forward_count += 1
        if rescue:
            self.corrector_rescue_count += 1
        self._record_joint_forward()

    def _record_joint_forward(self) -> None:
        self.physical_model_forward_count += 1
        self.joint_batch_forward_count += 1
        self.logical_conditional_nfe += 1
        self.logical_unconditional_nfe += 1

    def record_skip(self) -> None:
        self.corrector_skipped_count += 1

    def record_calibration(self) -> None:
        self.corrector_calibration_count += 1

    def record_fallback(self) -> None:
        self.corrector_fallback_count += 1

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class ConvergenceAwareCorrectorGate:
    """History-only, all-field corrector gate.

    A field is converged only when the latest predictor and corrector residual
    relative changes and normalized state-update norms are all finite and no
    greater than ``convergence_threshold``. The three fields are reduced
    independently and combined using logical AND; differently shaped tensors
    are never stacked.
    """

    def __init__(
        self,
        *,
        warmup_frac: float = 0.15,
        min_progress: float = 0.15,
        max_progress: float = 0.95,
        convergence_threshold: float = 0.05,
        consecutive_stable_steps: int = 3,
        calibration_interval: int = 10,
        max_consecutive_skips: int = 8,
        fallback_threshold: float = 0.20,
        rescue_enabled: bool = True,
        residual_ema_decay: float = 0.90,
        eps: float = 1e-8,
    ):
        if not 0.0 <= warmup_frac <= 1.0:
            raise ValueError("corrector_warmup_frac must be in [0, 1]")
        if not 0.0 <= min_progress <= max_progress <= 1.0:
            raise ValueError("corrector progress bounds are invalid")
        if convergence_threshold < 0.0:
            raise ValueError(
                "corrector_convergence_threshold must be non-negative"
            )
        if consecutive_stable_steps < 1:
            raise ValueError(
                "corrector_consecutive_stable_steps must be >= 1"
            )
        if calibration_interval < 1:
            raise ValueError("corrector_calibration_interval must be >= 1")
        if max_consecutive_skips < 1:
            raise ValueError("corrector_max_consecutive_skips must be >= 1")
        if fallback_threshold < 0.0:
            raise ValueError(
                "corrector_fallback_threshold must be non-negative"
            )
        if not 0.0 <= residual_ema_decay < 1.0:
            raise ValueError("corrector residual EMA decay must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("corrector eps must be positive")
        self.warmup_frac = float(warmup_frac)
        self.min_progress = float(min_progress)
        self.max_progress = float(max_progress)
        self.convergence_threshold = float(convergence_threshold)
        self.consecutive_stable_steps = int(consecutive_stable_steps)
        self.calibration_interval = int(calibration_interval)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.fallback_threshold = float(fallback_threshold)
        self.rescue_enabled = bool(rescue_enabled)
        self.residual_ema_decay = float(residual_ema_decay)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self._history = {
            phase: {field: _FieldHistory() for field in GATING_FIELDS}
            for phase in GATING_PHASES
        }
        self.field_converged = {field: False for field in GATING_FIELDS}
        self.global_converged = False
        self.stable_count = 0
        self.consecutive_skip_count = 0
        self.steps_since_corrector = 0
        self.fallback = False
        self.fallback_reason = ""
        self.last_decision = CorrectorGateDecision(
            execute_corrector=True,
            mode="full_corrector",
            reason="reset",
        )

    @staticmethod
    def _finite_nonnegative(value: float | None) -> bool:
        return (
            value is not None
            and math.isfinite(float(value))
            and float(value) >= 0.0
        )

    def observe_residual(
        self, *, phase: str, residuals: Mapping[str, float | None]
    ) -> None:
        if phase not in self._history:
            raise ValueError(f"unknown corrector gating phase: {phase}")
        for field in GATING_FIELDS:
            item = self._history[phase][field]
            raw = residuals.get(field)
            value = float(raw) if self._finite_nonnegative(raw) else None
            previous = item.residual
            if value is None:
                item.residual = None
                item.residual_relative_change = None
                continue
            item.residual_relative_change = (
                None
                if previous is None
                else abs(value - previous) / (abs(previous) + self.eps)
            )
            item.residual = value
            item.residual_ema = (
                value
                if item.residual_ema is None
                else self.residual_ema_decay * item.residual_ema
                + (1.0 - self.residual_ema_decay) * value
            )

    def observe_update(
        self, *, phase: str, update_norms: Mapping[str, float | None]
    ) -> None:
        if phase not in self._history:
            raise ValueError(f"unknown corrector gating phase: {phase}")
        for field in GATING_FIELDS:
            raw = update_norms.get(field)
            self._history[phase][field].update_norm = (
                float(raw) if self._finite_nonnegative(raw) else None
            )

    def _field_metrics(self, field: str) -> list[float | None]:
        return [
            self._history["predictor"][field].residual_relative_change,
            self._history["corrector"][field].residual_relative_change,
            self._history["predictor"][field].update_norm,
            self._history["corrector"][field].update_norm,
        ]

    def finalize_step(self) -> None:
        invalid: list[str] = []
        excessive: list[str] = []
        converged: dict[str, bool] = {}
        for field in GATING_FIELDS:
            metrics = self._field_metrics(field)
            valid = all(self._finite_nonnegative(value) for value in metrics)
            if not valid:
                invalid.append(field)
                converged[field] = False
                continue
            values = [float(value) for value in metrics if value is not None]
            converged[field] = all(
                value <= self.convergence_threshold for value in values
            )
            if any(value > self.fallback_threshold for value in values):
                excessive.append(field)
        self.field_converged = converged
        self.global_converged = all(converged.values())
        self.fallback = bool(invalid or excessive)
        reasons = []
        if invalid:
            reasons.append("invalid_history:" + ",".join(invalid))
        if excessive:
            reasons.append("history_change:" + ",".join(excessive))
        self.fallback_reason = ";".join(reasons)
        self.stable_count = (
            self.stable_count + 1 if self.global_converged else 0
        )

    def decide(
        self, *, sampling_step: int, num_steps: int
    ) -> CorrectorGateDecision:
        progress = sampling_step / max(num_steps - 1, 1)
        warmup_steps = math.ceil(self.warmup_frac * num_steps)
        if sampling_step < warmup_steps:
            decision = CorrectorGateDecision(
                True, "warmup_full", "warmup"
            )
        elif progress < self.min_progress or progress > self.max_progress:
            decision = CorrectorGateDecision(
                True, "full_corrector", "outside_gating_window"
            )
        elif self.fallback:
            decision = CorrectorGateDecision(
                True,
                "fallback_corrector",
                self.fallback_reason or "history_fallback",
                fallback=True,
            )
        elif (
            self.consecutive_skip_count >= self.max_consecutive_skips
        ):
            decision = CorrectorGateDecision(
                True,
                "forced_calibration",
                "max_consecutive_skips",
                forced_calibration=True,
            )
        elif self.steps_since_corrector >= self.calibration_interval:
            decision = CorrectorGateDecision(
                True,
                "forced_calibration",
                "calibration_interval",
                forced_calibration=True,
            )
        elif self.stable_count >= self.consecutive_stable_steps:
            decision = CorrectorGateDecision(
                False, "skip_corrector", "all_fields_converged"
            )
        else:
            decision = CorrectorGateDecision(
                True, "full_corrector", "not_stably_converged"
            )
        self.last_decision = decision
        return decision

    def record_scheduled_decision(
        self, decision: CorrectorGateDecision
    ) -> None:
        if decision.execute_corrector:
            self.consecutive_skip_count = 0
            self.steps_since_corrector = 0
        else:
            self.consecutive_skip_count += 1
            self.steps_since_corrector += 1

    def should_rescue_after_predictor(self) -> tuple[bool, str]:
        if not self.rescue_enabled or self.last_decision.execute_corrector:
            return False, ""
        invalid = []
        excessive = []
        for field in GATING_FIELDS:
            value = self._history["predictor"][field].update_norm
            if not self._finite_nonnegative(value):
                invalid.append(field)
            elif float(value) > self.fallback_threshold:
                excessive.append(field)
        if invalid:
            return True, "predictor_update_invalid:" + ",".join(invalid)
        if excessive:
            return True, "predictor_update_large:" + ",".join(excessive)
        return False, ""

    def record_rescue(self) -> None:
        self.consecutive_skip_count = 0
        self.steps_since_corrector = 0
        self.stable_count = 0

    def snapshot(self) -> dict[str, object]:
        return {
            "field_converged": dict(self.field_converged),
            "global_converged": self.global_converged,
            "stable_count": self.stable_count,
            "consecutive_skip_count": self.consecutive_skip_count,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "residuals": {
                phase: {
                    field: self._history[phase][field].residual
                    for field in GATING_FIELDS
                }
                for phase in GATING_PHASES
            },
            "residual_ema": {
                phase: {
                    field: self._history[phase][field].residual_ema
                    for field in GATING_FIELDS
                }
                for phase in GATING_PHASES
            },
            "residual_relative_change": {
                phase: {
                    field: self._history[
                        phase
                    ][field].residual_relative_change
                    for field in GATING_FIELDS
                }
                for phase in GATING_PHASES
            },
            "update_norms": {
                phase: {
                    field: self._history[phase][field].update_norm
                    for field in GATING_FIELDS
                }
                for phase in GATING_PHASES
            },
        }
