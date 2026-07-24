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
    budget_exhausted: bool = False
    atomic_veto: bool = False


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
    corrector_atomic_veto_count: int = 0
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

    def record_atomic_veto(self) -> None:
        self.corrector_atomic_veto_count += 1

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class ConvergenceAwareCorrectorGate:
    """History-only gate with optional per-sample budget and hard field vetoes.

    Legacy defaults preserve the original G3 state machine. Budget-aware mode
    can cap skips inside the gateable interval, require discrete atomic-number
    stability, and adapt the calibration interval without adding model calls.
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
        max_skip_ratio: float | None = None,
        atomic_veto_enabled: bool = False,
        atomic_stability_threshold: float | None = None,
        atomic_min_stable_steps: int = 1,
        adaptive_calibration_enabled: bool = False,
        calibration_interval_min: int | None = None,
        calibration_interval_max: int | None = None,
        field_aggregation: str = "all_fields",
        field_weights: Mapping[str, float] | None = None,
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
        if max_skip_ratio is not None and not 0.0 <= max_skip_ratio <= 1.0:
            raise ValueError("corrector_max_skip_ratio must be in [0, 1]")
        if (
            atomic_stability_threshold is not None
            and atomic_stability_threshold < 0.0
        ):
            raise ValueError(
                "corrector_atomic_stability_threshold must be non-negative"
            )
        if atomic_min_stable_steps < 1:
            raise ValueError(
                "corrector_atomic_min_stable_steps must be >= 1"
            )
        selected_interval_min = (
            calibration_interval
            if calibration_interval_min is None
            else calibration_interval_min
        )
        selected_interval_max = (
            calibration_interval
            if calibration_interval_max is None
            else calibration_interval_max
        )
        if (
            selected_interval_min < 1
            or selected_interval_max < selected_interval_min
        ):
            raise ValueError(
                "corrector adaptive calibration interval bounds are invalid"
            )
        if field_aggregation not in (
            "all_fields",
            "weighted_max",
            "weighted_rms",
        ):
            raise ValueError(
                "corrector_field_aggregation must be all_fields, "
                "weighted_max, or weighted_rms"
            )
        selected_weights = {
            field: float((field_weights or {}).get(field, 1.0))
            for field in GATING_FIELDS
        }
        if any(
            not math.isfinite(weight) or weight <= 0.0
            for weight in selected_weights.values()
        ):
            raise ValueError(
                "corrector field weights must be finite and positive"
            )

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
        self.max_skip_ratio = (
            None if max_skip_ratio is None else float(max_skip_ratio)
        )
        self.atomic_veto_enabled = bool(atomic_veto_enabled)
        self.atomic_stability_threshold = float(
            convergence_threshold
            if atomic_stability_threshold is None
            else atomic_stability_threshold
        )
        self.atomic_min_stable_steps = int(atomic_min_stable_steps)
        self.adaptive_calibration_enabled = bool(
            adaptive_calibration_enabled
        )
        self.calibration_interval_min = int(selected_interval_min)
        self.calibration_interval_max = int(selected_interval_max)
        self.field_aggregation = field_aggregation
        self.field_weights = selected_weights
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
        self.atomic_stable = False
        self.atomic_stable_steps = 0
        self.atomic_veto_triggered = False
        self.skip_budget_total: int | None = None
        self.skip_budget_used = 0
        self._budget_num_steps: int | None = None
        self.current_calibration_interval = (
            self.calibration_interval_min
            if self.adaptive_calibration_enabled
            else self.calibration_interval
        )
        self._calibration_interval_sum = 0
        self._calibration_interval_samples = 0
        self.finalized_steps = 0
        self.field_stable_counts = {
            field: 0 for field in GATING_FIELDS
        }
        self.all_fields_stable_count = 0
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

    def _field_is_converged(
        self, *, field: str, values: list[float]
    ) -> bool:
        weighted = [
            value * self.field_weights[field] for value in values
        ]
        score = (
            max(weighted)
            if self.field_aggregation in ("all_fields", "weighted_max")
            else math.sqrt(
                sum(value * value for value in weighted) / len(weighted)
            )
        )
        return score <= self.convergence_threshold

    def _global_is_converged(
        self,
        *,
        converged: Mapping[str, bool],
        valid_values: Mapping[str, list[float]],
    ) -> bool:
        if self.field_aggregation == "all_fields":
            return all(converged.values())
        weighted = [
            value * self.field_weights[field]
            for field, values in valid_values.items()
            for value in values
        ]
        if len(valid_values) != len(GATING_FIELDS) or not weighted:
            return False
        score = (
            max(weighted)
            if self.field_aggregation == "weighted_max"
            else math.sqrt(
                sum(value * value for value in weighted) / len(weighted)
            )
        )
        return score <= self.convergence_threshold

    def finalize_step(self) -> None:
        invalid: list[str] = []
        excessive: list[str] = []
        converged: dict[str, bool] = {}
        valid_values: dict[str, list[float]] = {}
        for field in GATING_FIELDS:
            metrics = self._field_metrics(field)
            valid = all(self._finite_nonnegative(value) for value in metrics)
            if not valid:
                invalid.append(field)
                converged[field] = False
                continue
            values = [float(value) for value in metrics if value is not None]
            valid_values[field] = values
            converged[field] = self._field_is_converged(
                field=field, values=values
            )
            if any(value > self.fallback_threshold for value in values):
                excessive.append(field)
        self.field_converged = converged
        self.global_converged = self._global_is_converged(
            converged=converged, valid_values=valid_values
        )
        atomic_values = valid_values.get("atomic_numbers")
        self.atomic_stable = bool(
            atomic_values
            and all(
                value <= self.atomic_stability_threshold
                for value in atomic_values
            )
        )
        self.atomic_stable_steps = (
            self.atomic_stable_steps + 1 if self.atomic_stable else 0
        )
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
        self.finalized_steps += 1
        for field in GATING_FIELDS:
            self.field_stable_counts[field] += int(
                self.field_converged.get(field, False)
            )
        self.all_fields_stable_count += int(self.global_converged)
        if self.adaptive_calibration_enabled and (
            self.fallback or not self.global_converged
        ):
            self.current_calibration_interval = (
                self.calibration_interval_min
                if self.fallback
                else max(
                    self.calibration_interval_min,
                    self.current_calibration_interval // 2,
                )
            )

    def _ensure_skip_budget(self, *, num_steps: int) -> None:
        if self.max_skip_ratio is None:
            return
        if self._budget_num_steps == num_steps:
            return
        if self.skip_budget_used:
            raise RuntimeError(
                "num_steps changed after corrector skip budget was used"
            )
        warmup_steps = math.ceil(self.warmup_frac * num_steps)
        gateable_steps = 0
        for sampling_step in range(num_steps):
            progress = sampling_step / max(num_steps - 1, 1)
            if (
                sampling_step >= warmup_steps
                and self.min_progress <= progress <= self.max_progress
            ):
                gateable_steps += 1
        self.skip_budget_total = int(
            math.floor(gateable_steps * self.max_skip_ratio + 1e-12)
        )
        self._budget_num_steps = num_steps

    @property
    def skip_budget_remaining(self) -> int | None:
        if self.skip_budget_total is None:
            return None
        return max(self.skip_budget_total - self.skip_budget_used, 0)

    @property
    def skip_budget_exhausted(self) -> bool:
        return bool(
            self.skip_budget_total is not None
            and self.skip_budget_used >= self.skip_budget_total
        )

    def decide(
        self, *, sampling_step: int, num_steps: int
    ) -> CorrectorGateDecision:
        self._ensure_skip_budget(num_steps=num_steps)
        progress = sampling_step / max(num_steps - 1, 1)
        warmup_steps = math.ceil(self.warmup_frac * num_steps)
        atomic_veto = bool(
            self.atomic_veto_enabled
            and self.field_converged.get("cell", False)
            and self.field_converged.get("pos", False)
            and (
                not self.atomic_stable
                or self.atomic_stable_steps
                < self.atomic_min_stable_steps
            )
        )
        self.atomic_veto_triggered = atomic_veto
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
        elif self.skip_budget_exhausted:
            decision = CorrectorGateDecision(
                True,
                "budget_full_corrector",
                "skip_budget_exhausted",
                budget_exhausted=True,
            )
        elif atomic_veto:
            decision = CorrectorGateDecision(
                True,
                "atomic_veto_corrector",
                "atomic_numbers_not_stably_converged",
                atomic_veto=True,
            )
        elif self.consecutive_skip_count >= self.max_consecutive_skips:
            decision = CorrectorGateDecision(
                True,
                "forced_calibration",
                "max_consecutive_skips",
                forced_calibration=True,
            )
        elif (
            self.steps_since_corrector
            >= self.current_calibration_interval
        ):
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
            if self.adaptive_calibration_enabled:
                if decision.fallback or decision.atomic_veto:
                    self.current_calibration_interval = (
                        self.calibration_interval_min
                    )
                elif decision.forced_calibration:
                    self._calibration_interval_sum += (
                        self.current_calibration_interval
                    )
                    self._calibration_interval_samples += 1
                    self.current_calibration_interval = min(
                        self.calibration_interval_max,
                        max(
                            self.current_calibration_interval + 1,
                            math.ceil(
                                self.current_calibration_interval * 1.5
                            ),
                        ),
                    )
        else:
            self.consecutive_skip_count += 1
            self.steps_since_corrector += 1
            self.skip_budget_used += 1
            if (
                self.skip_budget_total is not None
                and self.skip_budget_used > self.skip_budget_total
            ):
                raise RuntimeError("corrector skip budget exceeded")

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
        if self.adaptive_calibration_enabled:
            self.current_calibration_interval = (
                self.calibration_interval_min
            )

    def snapshot(self) -> dict[str, object]:
        atomic_history = self._history["predictor"]["atomic_numbers"]
        return {
            "field_converged": dict(self.field_converged),
            "global_converged": self.global_converged,
            "stable_count": self.stable_count,
            "consecutive_skip_count": self.consecutive_skip_count,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "field_aggregation": self.field_aggregation,
            "skip_budget_total": self.skip_budget_total,
            "skip_budget_used": self.skip_budget_used,
            "skip_budget_remaining": self.skip_budget_remaining,
            "skip_budget_exhausted": self.skip_budget_exhausted,
            "atomic_veto_enabled": self.atomic_veto_enabled,
            "atomic_veto_triggered": self.atomic_veto_triggered,
            "atomic_stable": self.atomic_stable,
            "atomic_stable_steps": self.atomic_stable_steps,
            "atomic_residual": atomic_history.residual,
            "atomic_residual_change": (
                atomic_history.residual_relative_change
            ),
            "calibration_interval_current": (
                self.current_calibration_interval
            ),
            "calibration_interval_mean": (
                self._calibration_interval_sum
                / self._calibration_interval_samples
                if self._calibration_interval_samples
                else None
            ),
            "finalized_steps": self.finalized_steps,
            "cell_stable_rate": (
                self.field_stable_counts["cell"]
                / max(self.finalized_steps, 1)
            ),
            "pos_stable_rate": (
                self.field_stable_counts["pos"]
                / max(self.finalized_steps, 1)
            ),
            "atomic_stable_rate": (
                self.field_stable_counts["atomic_numbers"]
                / max(self.finalized_steps, 1)
            ),
            "all_fields_stable_rate": (
                self.all_fields_stable_count
                / max(self.finalized_steps, 1)
            ),
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
