"""Training-free robust multi-field residual guidance controllers.

The frozen V2 controller standardizes each conditional residual against a
fixed-CFG, timestep-binned calibration, takes the median field consensus, and
adapts only when a magnitude-and-agreement confidence rule supports it.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median, pstdev
from typing import Mapping

from mattergen.diffusion.sampling.guidance_schedule import GuidanceDecision, PHASES


FIELD_ALIASES = {
    "atomic": "atomic_numbers",
    "position": "pos",
    "cell": "cell",
}


class RobustAdaptiveGuidanceController:
    """Apply a frozen normalized-median controller, with optional V2 safeguards."""

    def __init__(
        self,
        *,
        calibration_json: str | Path,
        calibration_csv: str | Path,
        robust: bool,
        base_guidance: float = 2.0,
        min_scale: float = 1.5,
        max_scale: float = 2.5,
        adaptation_amplitude: float = 0.5,
        slew_rate_cap: float = 0.05,
    ) -> None:
        metadata = json.loads(Path(calibration_json).read_text())
        confidence = metadata["confidence"]
        online = metadata["online_statistics"]
        transform = metadata["transform"]
        self.schedule = "robust_adaptive" if robust else "normalized_adaptive"
        self.robust = bool(robust)
        self.base_guidance = float(base_guidance)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.adaptation_amplitude = float(adaptation_amplitude)
        self.slew_rate_cap = float(slew_rate_cap)
        self.time_bins = int(transform["time_bins"])
        self.log_eps = float(transform["log_eps"])
        self.z_clip = float(transform["z_clip"])
        self.ema_beta = float(online["ema_beta"])
        self.signal_deadzone = float(confidence["signal_deadzone"])
        self.signal_scale = float(confidence["signal_scale"])
        self.agreement_k = float(confidence["agreement_k"])
        self._calibration: dict[tuple[str, str, int], tuple[float, float]] = {}
        with Path(calibration_csv).open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["phase"], row["field"], int(row["time_bin"]))
                self._calibration[key] = (
                    float(row["mu_log_residual"]),
                    float(row["sigma_log_residual"]),
                )
        expected = len(PHASES) * len(FIELD_ALIASES) * self.time_bins
        if len(self._calibration) != expected:
            raise ValueError(f"calibration has {len(self._calibration)} rows, expected {expected}")
        if not (
            0 <= self.ema_beta < 1
            and self.min_scale <= self.base_guidance <= self.max_scale
            and self.signal_scale > 0
            and self.slew_rate_cap > 0
        ):
            raise ValueError("invalid frozen robust guidance parameters")
        self.reset()

    @property
    def ema_by_phase(self) -> dict[str, float | None]:
        return dict(self._ema_by_phase)

    def reset(self) -> None:
        self._ema_by_phase = {phase: None for phase in PHASES}
        self._previous_guidance = self.base_guidance

    def _fallback(self, reason: str) -> GuidanceDecision:
        target = self.base_guidance
        final, limited = self._slew(target)
        return GuidanceDecision(
            stage_guidance=self.base_guidance,
            delta=None,
            ema=None,
            ratio=None,
            adaptive_multiplier=1.0,
            final_guidance=final,
            fallback_reason=reason,
            aggregation="median_z",
            target_guidance=target,
            slew_limited=limited,
            fallback_to_base=True,
        )

    def _slew(self, target: float) -> tuple[float, bool]:
        if not self.robust:
            self._previous_guidance = target
            return target, False
        lower = self._previous_guidance - self.slew_rate_cap
        upper = self._previous_guidance + self.slew_rate_cap
        actual = min(max(target, lower), upper)
        actual = min(max(actual, self.min_scale), self.max_scale)
        limited = abs(actual - target) > 1e-15
        self._previous_guidance = actual
        return actual, limited

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
        if residual_error:
            return self._fallback(residual_error)
        if not field_deltas:
            return self._fallback("invalid_or_empty_residual")
        time_bin = min(max(int(progress * self.time_bins), 0), self.time_bins - 1)
        z: dict[str, float] = {}
        for field, source_name in FIELD_ALIASES.items():
            value = field_deltas.get(source_name)
            if value is None or not math.isfinite(value) or value < 0:
                return self._fallback(f"{source_name}:invalid")
            mu, sigma = self._calibration[(phase, field, time_bin)]
            standardized = (math.log(value + self.log_eps) - mu) / sigma
            z[field] = min(max(standardized, -self.z_clip), self.z_clip)

        z_values = list(z.values())
        z_mean = sum(z_values) / len(z_values)
        z_median = median(z_values)
        field_std = pstdev(z_values)
        field_range = max(z_values) - min(z_values)
        previous_ema = self._ema_by_phase[phase]
        z_ema = (
            z_median
            if previous_ema is None
            else self.ema_beta * previous_ema + (1 - self.ema_beta) * z_median
        )
        self._ema_by_phase[phase] = z_ema

        if self.robust:
            signal_confidence = min(
                max((abs(z_ema) - self.signal_deadzone) / self.signal_scale, 0.0), 1.0
            )
            agreement_confidence = math.exp(-self.agreement_k * field_std)
            confidence = signal_confidence * agreement_confidence
            control_signal = z_ema
        else:
            signal_confidence = 1.0
            agreement_confidence = 1.0
            confidence = 1.0
            control_signal = z_median

        raw_delta = self.adaptation_amplitude * math.tanh(control_signal)
        target = self.base_guidance + confidence * raw_delta
        target = min(max(target, self.min_scale), self.max_scale)
        fallback_to_base = self.robust and signal_confidence == 0.0
        if fallback_to_base:
            target = self.base_guidance
        final, limited = self._slew(target)
        return GuidanceDecision(
            stage_guidance=self.base_guidance,
            delta=z_median,
            ema=z_ema,
            ratio=None,
            adaptive_multiplier=final / self.base_guidance,
            final_guidance=final,
            fallback_reason="low_confidence" if fallback_to_base else None,
            aggregation="median_z",
            z_atomic=z["atomic"],
            z_position=z["position"],
            z_cell=z["cell"],
            z_mean=z_mean,
            z_median=z_median,
            z_ema=z_ema,
            field_std=field_std,
            field_range=field_range,
            signal_confidence=signal_confidence,
            agreement_confidence=agreement_confidence,
            confidence=confidence,
            target_guidance=target,
            slew_limited=limited,
            fallback_to_base=fallback_to_base,
        )
