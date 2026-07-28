# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import csv
import json
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
from mattergen.diffusion.sampling.corrector_gating import (
    GATING_FIELDS,
    ConvergenceAwareCorrectorGate,
    CorrectorGateDecision,
    CorrectorGatingAccounting,
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

CORRECTOR_TRACE_FIELD_NAMES = (
    "seed",
    "sampling_step",
    "progress",
    "t",
    "decision",
    "corrector_executed",
    "corrector_skipped",
    "forced_calibration",
    "fallback",
    "fallback_reason",
    "residual_cell",
    "residual_pos",
    "residual_atomic",
    "corrector_residual_cell",
    "corrector_residual_pos",
    "corrector_residual_atomic",
    "predictor_residual_change_cell",
    "predictor_residual_change_pos",
    "predictor_residual_change_atomic",
    "corrector_residual_change_cell",
    "corrector_residual_change_pos",
    "corrector_residual_change_atomic",
    "predictor_update_cell",
    "predictor_update_pos",
    "predictor_update_atomic",
    "corrector_update_cell",
    "corrector_update_pos",
    "corrector_update_atomic",
    "cell_converged",
    "pos_converged",
    "atomic_converged",
    "global_converged",
    "stable_count",
    "consecutive_skip_count",
    "predictor_forward_count",
    "corrector_forward_count",
    "corrector_skipped_count",
    "corrector_calibration_count",
    "corrector_fallback_count",
    "corrector_rescue_count",
    "physical_forward_count",
    "joint_batch_forward_count",
    "conditional_only_forward_count",
    "logical_conditional_nfe",
    "logical_unconditional_nfe",
    "rescue",
    "rescue_reason",
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
        cfg_trace_mode: str = "auto",
        cfg_summary_path: str | None = None,
        corrector_gating_enabled: bool = False,
        corrector_warmup_frac: float = 0.15,
        corrector_min_progress: float = 0.15,
        corrector_max_progress: float = 0.95,
        corrector_convergence_threshold: float = 0.05,
        corrector_consecutive_stable_steps: int = 3,
        corrector_calibration_interval: int = 10,
        corrector_max_consecutive_skips: int = 8,
        corrector_fallback_threshold: float = 0.20,
        corrector_rescue_enabled: bool = True,
        corrector_trace_path: str | None = None,
        corrector_summary_path: str | None = None,
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
        if cfg_trace_mode not in ("auto", "off", "memory", "disk"):
            raise ValueError("cfg_trace_mode must be auto, off, memory, or disk")
        self._cfg_trace_mode = (
            ("disk" if save_trace or selected_trace_path is not None else "off")
            if cfg_trace_mode == "auto"
            else cfg_trace_mode
        )
        self._trace_enabled = self._cfg_trace_mode != "off"
        self._trace_to_disk = self._cfg_trace_mode == "disk"
        if self._trace_to_disk and selected_trace_path is None:
            raise ValueError(
                "Guidance trace is enabled but no guidance_trace_path or GUIDANCE_TRACE_PATH was set"
            )
        self._guidance_trace_path = (
            Path(selected_trace_path).expanduser() if selected_trace_path is not None else None
        )
        if self._guidance_trace_path is not None and not self._guidance_trace_path.is_absolute():
            raise ValueError("guidance_trace_path must be an absolute path")
        self._cfg_summary_path = (
            Path(cfg_summary_path).expanduser() if cfg_summary_path is not None else None
        )
        if self._cfg_summary_path is not None and not self._cfg_summary_path.is_absolute():
            raise ValueError("cfg_summary_path must be an absolute path")
        self._guidance_trace_rows: list[dict[str, Any]] = []
        self._trace_build_cpu_seconds = 0.0
        self._trace_write_cpu_seconds = 0.0
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
        self._corrector_gating_enabled = bool(corrector_gating_enabled)
        if self._corrector_gating_enabled and self._cfg_acceleration_enabled:
            raise ValueError(
                "corrector gating cannot be combined with the archived "
                "unconditional-reuse acceleration"
            )
        self._corrector_gate = ConvergenceAwareCorrectorGate(
            warmup_frac=corrector_warmup_frac,
            min_progress=corrector_min_progress,
            max_progress=corrector_max_progress,
            convergence_threshold=corrector_convergence_threshold,
            consecutive_stable_steps=corrector_consecutive_stable_steps,
            calibration_interval=corrector_calibration_interval,
            max_consecutive_skips=corrector_max_consecutive_skips,
            fallback_threshold=corrector_fallback_threshold,
            rescue_enabled=corrector_rescue_enabled,
        )
        self._corrector_trace_path = (
            Path(corrector_trace_path).expanduser()
            if corrector_trace_path is not None
            else None
        )
        self._corrector_summary_path = (
            Path(corrector_summary_path).expanduser()
            if corrector_summary_path is not None
            else None
        )
        for label, selected_path in (
            ("corrector_trace_path", self._corrector_trace_path),
            ("corrector_summary_path", self._corrector_summary_path),
        ):
            if selected_path is not None and not selected_path.is_absolute():
                raise ValueError(f"{label} must be an absolute path")
        self._corrector_trace_rows: list[dict[str, Any]] = []
        self._corrector_accounting = CorrectorGatingAccounting()
        self._corrector_current_decision = CorrectorGateDecision(
            True, "full_corrector", "initial"
        )
        self._corrector_rescue_performed = False
        self._corrector_rescue_reason = ""

    @property
    def guidance_schedule(self) -> str:
        return self._guidance_controller.schedule

    @property
    def guidance_ema_by_phase(self) -> Mapping[str, float | None]:
        return self._guidance_controller.ema_by_phase

    @property
    def cfg_nfe_summary(self) -> Mapping[str, int]:
        return self._cfg_nfe.as_dict()

    @property
    def corrector_gating_summary(self) -> Mapping[str, int]:
        return self._corrector_accounting.as_dict()

    def _corrector_gating_active(self) -> bool:
        return bool(getattr(self, "_corrector_gating_enabled", False))

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

    def _reset_corrector_gating_state(self) -> None:
        self._corrector_gate.reset()
        self._corrector_accounting = CorrectorGatingAccounting()
        self._corrector_trace_rows = []
        self._corrector_current_decision = CorrectorGateDecision(
            True, "full_corrector", "reset"
        )
        self._corrector_rescue_performed = False
        self._corrector_rescue_reason = ""

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self._guidance_controller.reset()
        self._reset_cfg_acceleration_state()
        self._guidance_trace_rows = []
        self._trace_build_cpu_seconds = 0.0
        self._trace_write_cpu_seconds = 0.0
        if hasattr(self, "_corrector_gate"):
            self._reset_corrector_gating_state()
        for label, selected_path in (
            ("corrector trace", getattr(self, "_corrector_trace_path", None)),
            ("corrector summary", getattr(self, "_corrector_summary_path", None)),
        ):
            if selected_path is not None and selected_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing {label}: {selected_path}"
                )
        if self._trace_to_disk:
            assert self._guidance_trace_path is not None
            if self._guidance_trace_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing guidance trace: {self._guidance_trace_path}"
                )
        if self._cfg_summary_path is not None and self._cfg_summary_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing CFG summary: {self._cfg_summary_path}"
            )

    def _on_sampling_end(self, error: BaseException | None) -> None:
        corrector_trace_path = getattr(self, "_corrector_trace_path", None)
        corrector_summary_path = getattr(self, "_corrector_summary_path", None)
        if corrector_trace_path is not None:
            self._atomic_write_csv(
                corrector_trace_path,
                CORRECTOR_TRACE_FIELD_NAMES,
                self._corrector_trace_rows,
            )
        if corrector_summary_path is not None:
            accounting = self._corrector_accounting.as_dict()
            baseline_physical = self.N * (1 + self._n_steps_corrector)
            if not self._corrector_gating_enabled:
                cfg_nfe = self._cfg_nfe.as_dict()
                accounting.update(
                    predictor_forward_count=self.N,
                    corrector_forward_count=(
                        self.N * self._n_steps_corrector
                    ),
                    physical_model_forward_count=cfg_nfe[
                        "physical_model_forward_count"
                    ],
                    joint_batch_forward_count=cfg_nfe[
                        "joint_batch_forward_count"
                    ],
                    conditional_only_forward_count=cfg_nfe[
                        "conditional_only_forward_count"
                    ],
                    logical_conditional_nfe=cfg_nfe[
                        "conditional_logical_nfe"
                    ],
                    logical_unconditional_nfe=cfg_nfe[
                        "unconditional_logical_nfe"
                    ],
                )
            payload = {
                "success": error is None,
                "error_type": None if error is None else type(error).__name__,
                "enabled": self._corrector_gating_enabled,
                "seed": self._sample_seed,
                "sampling_steps": self.N,
                "n_steps_corrector": self._n_steps_corrector,
                **accounting,
                "corrector_skip_rate": (
                    accounting["corrector_skipped_count"] / max(self.N, 1)
                ),
                "physical_forward_reduction": (
                    1.0
                    - accounting["physical_model_forward_count"]
                    / max(baseline_physical, 1)
                ),
                "final_gate_state": self._corrector_gate.snapshot(),
            }
            self._atomic_write_json(corrector_summary_path, payload)
        if self._trace_to_disk:
            started = time.perf_counter()
            assert self._guidance_trace_path is not None
            self._guidance_trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._guidance_trace_path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=TRACE_FIELD_NAMES)
                writer.writeheader()
                writer.writerows(self._guidance_trace_rows)
                stream.flush()
                os.fsync(stream.fileno())
            self._trace_write_cpu_seconds = time.perf_counter() - started
        if self._cfg_summary_path is not None:
            self._cfg_summary_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "success": error is None,
                "error_type": None if error is None else type(error).__name__,
                "trace_mode": self._cfg_trace_mode,
                "trace_rows": len(self._guidance_trace_rows),
                "trace_build_cpu_seconds": self._trace_build_cpu_seconds,
                "trace_write_cpu_seconds": self._trace_write_cpu_seconds,
                "nfe": self._cfg_nfe.as_dict(),
                "mode_counts": {
                    "full_cfg": self._cfg_nfe.full_cfg_steps,
                    "reuse": self._cfg_nfe.reuse_steps,
                    "extrapolate": self._cfg_nfe.extrapolation_steps,
                    "periodic_calibration": self._cfg_nfe.calibration_steps,
                    "fallback_full_cfg": self._cfg_nfe.fallback_steps,
                },
            }
            temporary = self._cfg_summary_path.with_name(
                f".{self._cfg_summary_path.name}.tmp.{os.getpid()}"
            )
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._cfg_summary_path)

    @staticmethod
    def _atomic_write_csv(
        path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _normalized_update_rms(
        before: Diffusable, after: Diffusable
    ) -> dict[str, float | None]:
        norms: dict[str, float | None] = {}
        for field in GATING_FIELDS:
            try:
                old = before[field]
                new = after[field]
                if (
                    not isinstance(old, torch.Tensor)
                    or not isinstance(new, torch.Tensor)
                    or old.shape != new.shape
                    or old.numel() == 0
                ):
                    raise ValueError("invalid tensors")
                delta_rms = torch.sqrt(
                    torch.mean((new.float() - old.float()).square())
                )
                scale = torch.sqrt(torch.mean(old.float().square()))
                value = delta_rms / (scale + 1e-8)
                norms[field] = (
                    float(value.item())
                    if bool(torch.isfinite(value).item())
                    else None
                )
            except (KeyError, TypeError, ValueError):
                norms[field] = None
        return norms

    def _corrector_gate_should_execute(
        self, *, batch: Diffusable, t: torch.Tensor, sampling_step: int
    ) -> bool:
        del batch, t
        self._corrector_rescue_performed = False
        self._corrector_rescue_reason = ""
        decision = self._corrector_gate.decide(
            sampling_step=sampling_step, num_steps=self.N
        )
        self._corrector_current_decision = decision
        self._corrector_gate.record_scheduled_decision(decision)
        if decision.forced_calibration:
            self._corrector_accounting.record_calibration()
        if decision.fallback:
            self._corrector_accounting.record_fallback()
        return decision.execute_corrector

    def _on_corrector_skipped(
        self, *, batch: Diffusable, t: torch.Tensor, sampling_step: int
    ) -> None:
        del batch, t, sampling_step
        self._corrector_accounting.record_skip()

    def _on_phase_model_forward(
        self, *, phase: str, sampling_step: int, rescue: bool
    ) -> None:
        del sampling_step
        if phase == "predictor":
            self._corrector_accounting.record_predictor_forward()
        elif phase == "corrector":
            self._corrector_accounting.record_corrector_forward(
                rescue=rescue
            )
        else:
            raise ValueError(f"unknown phase: {phase}")

    def _on_phase_update(
        self,
        *,
        phase: str,
        before: Diffusable,
        after: Diffusable,
        sampling_step: int,
        rescue: bool,
    ) -> None:
        del sampling_step, rescue
        self._corrector_gate.observe_update(
            phase=phase,
            update_norms=self._normalized_update_rms(before, after),
        )

    def _corrector_rescue_required(
        self, *, batch: Diffusable, t: torch.Tensor, sampling_step: int
    ) -> bool:
        del batch, t, sampling_step
        rescue, reason = self._corrector_gate.should_rescue_after_predictor()
        self._corrector_rescue_performed = rescue
        self._corrector_rescue_reason = reason
        if rescue:
            self._corrector_gate.record_rescue()
        return rescue

    def _on_timestep_end(
        self,
        *,
        batch: Diffusable,
        t: torch.Tensor,
        sampling_step: int,
        elapsed_ms: float,
    ) -> None:
        del batch
        self._corrector_gate.finalize_step()
        if self._corrector_trace_path is None:
            return
        snapshot = self._corrector_gate.snapshot()
        predictor_residual = snapshot["residuals"]["predictor"]
        corrector_residual = snapshot["residuals"]["corrector"]
        predictor_residual_change = snapshot["residual_relative_change"][
            "predictor"
        ]
        corrector_residual_change = snapshot["residual_relative_change"][
            "corrector"
        ]
        predictor_update = snapshot["update_norms"]["predictor"]
        corrector_update = snapshot["update_norms"]["corrector"]
        converged = snapshot["field_converged"]
        accounting = self._corrector_accounting.as_dict()
        decision = self._corrector_current_decision
        rescue = self._corrector_rescue_performed
        trace_decision = "rescue_corrector" if rescue else decision.mode
        self._corrector_trace_rows.append(
            {
                "seed": self._sample_seed,
                "sampling_step": sampling_step,
                "progress": sampling_step / max(self.N - 1, 1),
                "t": float(t.detach().reshape(-1)[0].cpu().item()),
                "decision": trace_decision,
                "corrector_executed": decision.execute_corrector or rescue,
                "corrector_skipped": not decision.execute_corrector,
                "forced_calibration": decision.forced_calibration,
                "fallback": decision.fallback or rescue,
                "fallback_reason": (
                    self._corrector_rescue_reason
                    if rescue
                    else decision.reason
                ),
                "residual_cell": predictor_residual["cell"],
                "residual_pos": predictor_residual["pos"],
                "residual_atomic": predictor_residual["atomic_numbers"],
                "corrector_residual_cell": corrector_residual["cell"],
                "corrector_residual_pos": corrector_residual["pos"],
                "corrector_residual_atomic": corrector_residual[
                    "atomic_numbers"
                ],
                "predictor_residual_change_cell": (
                    predictor_residual_change["cell"]
                ),
                "predictor_residual_change_pos": (
                    predictor_residual_change["pos"]
                ),
                "predictor_residual_change_atomic": (
                    predictor_residual_change["atomic_numbers"]
                ),
                "corrector_residual_change_cell": (
                    corrector_residual_change["cell"]
                ),
                "corrector_residual_change_pos": (
                    corrector_residual_change["pos"]
                ),
                "corrector_residual_change_atomic": (
                    corrector_residual_change["atomic_numbers"]
                ),
                "predictor_update_cell": predictor_update["cell"],
                "predictor_update_pos": predictor_update["pos"],
                "predictor_update_atomic": predictor_update[
                    "atomic_numbers"
                ],
                "corrector_update_cell": corrector_update["cell"],
                "corrector_update_pos": corrector_update["pos"],
                "corrector_update_atomic": corrector_update[
                    "atomic_numbers"
                ],
                "cell_converged": converged["cell"],
                "pos_converged": converged["pos"],
                "atomic_converged": converged["atomic_numbers"],
                "global_converged": snapshot["global_converged"],
                "stable_count": snapshot["stable_count"],
                "consecutive_skip_count": snapshot[
                    "consecutive_skip_count"
                ],
                "predictor_forward_count": accounting[
                    "predictor_forward_count"
                ],
                "corrector_forward_count": accounting[
                    "corrector_forward_count"
                ],
                "corrector_skipped_count": accounting[
                    "corrector_skipped_count"
                ],
                "corrector_calibration_count": accounting[
                    "corrector_calibration_count"
                ],
                "corrector_fallback_count": accounting[
                    "corrector_fallback_count"
                ],
                "corrector_rescue_count": accounting[
                    "corrector_rescue_count"
                ],
                "physical_forward_count": accounting[
                    "physical_model_forward_count"
                ],
                "joint_batch_forward_count": accounting[
                    "joint_batch_forward_count"
                ],
                "conditional_only_forward_count": accounting[
                    "conditional_only_forward_count"
                ],
                "logical_conditional_nfe": accounting[
                    "logical_conditional_nfe"
                ],
                "logical_unconditional_nfe": accounting[
                    "logical_unconditional_nfe"
                ],
                "rescue": rescue,
                "rescue_reason": self._corrector_rescue_reason,
                "elapsed_ms": elapsed_ms,
            }
        )

    def _trace_decision(
        self,
        *,
        t: torch.Tensor,
        field_deltas: Mapping[str, float | None],
        decision: Mapping[str, float | str | None],
    ) -> None:
        if not self._trace_enabled:
            return
        trace_started = time.perf_counter()
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
        self._trace_build_cpu_seconds += time.perf_counter() - trace_started

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
        trace_started = time.perf_counter()
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
        self._trace_build_cpu_seconds += time.perf_counter() - trace_started

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
            result = self._score_fn_unaccelerated(x, t)
            if self._cfg_summary_path is not None:
                self._cfg_nfe.record_full("full_cfg")
            return result
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
        if (
            self.guidance_schedule == "constant"
            and not self._trace_enabled
            and not self._corrector_gating_active()
        ):
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
        if self._corrector_gating_active():
            self._corrector_gate.observe_residual(
                phase=phase, residuals=field_deltas
            )
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
