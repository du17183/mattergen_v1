"""Shared-prefix sampler for the preregistered Phase-B feature study."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from ase import Atoms

from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.sampling.field_decoupled_cfg import (
    FIELDS,
    FieldDecoupledPolicySampler,
    _tensor_digest,
)
from mattergen.diffusion.sampling.pc_sampler import _mask_replace


EPS = 1.0e-12


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.detach().float().square())).cpu().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    denominator = float((torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)).cpu().item())
    if not math.isfinite(denominator) or denominator <= EPS:
        return 0.0
    return float(np.clip(float(torch.dot(left_flat, right_flat).cpu().item()) / denominator, -1.0, 1.0))


class PhaseBFeatureSampler(FieldDecoupledPolicySampler):
    """Capture leakage-free prefix features and fork the complete pulse bank."""

    def __init__(
        self,
        *,
        sample_seed: int,
        policy_specs: Sequence[Mapping[str, Any]],
        branch_point: int = 400,
        feature_ema_beta: float = 0.95,
        feature_windows: Sequence[int] = (25, 50, 100),
        **kwargs: Any,
    ) -> None:
        super().__init__(sample_seed=sample_seed, policy_specs=policy_specs, implementation_check=False, **kwargs)
        self.branch_point = int(branch_point)
        self.feature_ema_beta = float(feature_ema_beta)
        self.feature_windows = tuple(int(value) for value in feature_windows)
        if self.branch_point != 400:
            raise ValueError("Phase-B branch point is frozen to 400")
        if abs(self.feature_ema_beta - 0.95) > 1e-12:
            raise ValueError("Phase-B feature EMA beta is frozen to 0.95")
        if self.feature_windows != (25, 50, 100):
            raise ValueError("Phase-B temporal windows are frozen to (25, 50, 100)")
        expected = {"C0", "GPulse", "APulse", "PPulse", "CPulse"}
        observed = {str(policy["policy_id"]) for policy in self._policy_specs}
        if observed != expected or len(self._policy_specs) != len(expected):
            raise ValueError(f"Phase-B policy bank changed: {observed}")
        for policy in self._policy_specs:
            if policy["policy_id"] == "C0":
                continue
            if policy["kind"] != "pulse" or int(policy["start"]) != 400 or int(policy["duration"]) != 100:
                raise ValueError(f"non-branch-compatible Phase-B policy: {policy}")
        self.prebranch_trace: list[dict[str, Any]] = []
        self.branch_features: dict[str, Any] = {}
        self.prefix_checkpoint: dict[str, Any] | None = None
        self.conditioning_metadata: dict[str, Any] = {}
        self._phase_field_ema: dict[tuple[str, str], float] = {}
        self._latest_predictor_x0: Atoms | None = None

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self.prebranch_trace = []
        self.branch_features = {}
        self.prefix_checkpoint = None
        self.conditioning_metadata = {}
        self._phase_field_ema = {}
        self._latest_predictor_x0 = None

    def _on_before_sample_prior(self, conditioning_data: Any) -> None:
        metadata: dict[str, Any] = {"seed": self._risk_seed, "batch_size": int(conditioning_data.get_batch_size())}
        for key, value in conditioning_data.items():
            if isinstance(value, torch.Tensor):
                metadata[str(key)] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "finite": bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True,
                }
            elif isinstance(value, list):
                metadata[str(key)] = {"type": "list", "length": len(value)}
            else:
                metadata[str(key)] = {"type": type(value).__name__}
        self.conditioning_metadata = metadata

    def _score_at_field_scales(self, x: Any, t: torch.Tensor, scales: Mapping[str, float]) -> Any:
        unconditional, conditional = self._components(x, t)
        guided = unconditional.replace(
            **{
                field: torch.lerp(unconditional[field], conditional[field], float(scales[field]))
                for field in self._multi_corruption.corrupted_fields
            }
        )
        context = self.sampling_context
        step = int(context.get("sampling_step", -1))
        phase = str(context.get("phase", "unknown"))
        is_base = all(abs(float(scales[field]) - 2.0) <= 1e-12 for field in FIELDS)
        if is_base and 0 <= step < self.branch_point:
            row: dict[str, Any] = {
                "seed": self._risk_seed,
                "sampling_step": step,
                "phase": phase,
                "t_norm": float(t.detach().reshape(-1)[0].cpu().item()),
                "score_call_index": int(context.get("score_call_index", -1)),
            }
            residuals: list[float] = []
            for field in FIELDS:
                residual = conditional[field] - unconditional[field]
                delta = _rms(residual)
                norm_cond = _rms(conditional[field])
                norm_uncond = _rms(unconditional[field])
                previous_ema = self._phase_field_ema.get((phase, field))
                relative = 1.0 if previous_ema is None else delta / max(previous_ema, EPS)
                updated_ema = delta if previous_ema is None else self.feature_ema_beta * previous_ema + (1.0 - self.feature_ema_beta) * delta
                self._phase_field_ema[(phase, field)] = updated_ema
                row.update(
                    {
                        f"residual_rms_{field}": delta,
                        f"norm_cond_{field}": norm_cond,
                        f"norm_uncond_{field}": norm_uncond,
                        f"residual_to_uncond_{field}": delta / max(norm_uncond, EPS),
                        f"alignment_{field}": _cosine(unconditional[field], residual),
                        f"residual_ema_previous_{field}": delta if previous_ema is None else previous_ema,
                        f"residual_over_ema_{field}": relative,
                        f"residual_ema_updated_{field}": updated_ema,
                    }
                )
                residuals.append(delta)
            residual_array = np.asarray(residuals, dtype=float)
            row.update(
                residual_max_over_min=float(residual_array.max() / max(residual_array.min(), EPS)),
                residual_std=float(residual_array.std()),
                residual_cv=float(residual_array.std() / max(residual_array.mean(), EPS)),
                residual_atomic_over_pos=float(residuals[0] / max(residuals[1], EPS)),
                residual_atomic_over_cell=float(residuals[0] / max(residuals[2], EPS)),
                residual_pos_over_cell=float(residuals[1] / max(residuals[2], EPS)),
            )
            self.prebranch_trace.append(row)
            if phase == "predictor" and step == self.branch_point - 1:
                self._latest_predictor_x0 = self._clean_atoms(x=x, score=guided, t=t)
        return guided

    @staticmethod
    def _window_summary(values: np.ndarray) -> dict[str, float]:
        slope = float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0]) if len(values) >= 2 else 0.0
        return {
            "last": float(values[-1]),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "slope": slope,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    def _summarize_prefix(self, branch_batch: Any, prefix_digest: str) -> dict[str, Any]:
        features: dict[str, Any] = {
            "seed": self._risk_seed,
            "branch_point": self.branch_point,
            "prefix_state_sha256": prefix_digest,
            "target_dft_mag_density": 0.1,
            "trace_rows": len(self.prebranch_trace),
        }
        expected_rows = self.branch_point * (1 + (self._n_steps_corrector if self._correctors else 0))
        if len(self.prebranch_trace) != expected_rows:
            raise RuntimeError(f"prebranch trace row count changed: {len(self.prebranch_trace)} != {expected_rows}")
        for phase in ("corrector", "predictor"):
            phase_rows = [row for row in self.prebranch_trace if row["phase"] == phase]
            if len(phase_rows) != self.branch_point:
                raise RuntimeError(f"incomplete {phase} prefix trace")
            latest = phase_rows[-1]
            for field in FIELDS:
                safe_field = "atomic" if field == "atomic_numbers" else field
                for source in (
                    "residual_rms", "norm_cond", "norm_uncond", "residual_to_uncond",
                    "alignment", "residual_over_ema",
                ):
                    features[f"{phase}_{safe_field}_{source}_last"] = float(latest[f"{source}_{field}"])
                for window in self.feature_windows:
                    values = np.asarray([float(row[f"residual_rms_{field}"]) for row in phase_rows[-window:]], dtype=float)
                    for statistic, value in self._window_summary(values).items():
                        features[f"{phase}_{safe_field}_residual_w{window}_{statistic}"] = value
            for name in (
                "residual_max_over_min", "residual_std", "residual_cv",
                "residual_atomic_over_pos", "residual_atomic_over_cell", "residual_pos_over_cell",
            ):
                features[f"{phase}_{name}_last"] = float(latest[name])
        state_atoms = self._final_atoms(branch_batch)
        for name, value in self._structural_features(state_atoms).items():
            features[f"branch_state_{name.removeprefix('x0_')}"] = float(value)
        if self._latest_predictor_x0 is None:
            raise RuntimeError("latest predictor x0 feature was not captured")
        for name, value in self._structural_features(self._latest_predictor_x0).items():
            features[f"latest_predictor_{name}"] = float(value)
        if not all(isinstance(value, (str, int, float)) and (not isinstance(value, float) or math.isfinite(value)) for value in features.values()):
            raise RuntimeError("non-finite or non-scalar branch feature")
        return features

    @torch.no_grad()
    def _denoise(self, batch: Any, mask: dict[str, torch.Tensor], record: bool = False) -> tuple[Any, Any, None]:
        if record:
            raise ValueError("Phase-B sampler writes explicit study outputs")
        for key in self._predictors:
            mask.setdefault(key, None)
        for key in self._correctors:
            mask.setdefault(key, None)
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1), device=self._device)
        base_scales = {field: 2.0 for field in FIELDS}
        base = batch.clone()
        base_mean = batch.clone()
        for index in range(self.branch_point):
            base, base_mean = self._advance_field(
                base, mask=mask, timesteps=timesteps, dt=dt,
                start=index, stop=index + 1, scales=base_scales,
            )
        prefix_rng = self._rng_snapshot()
        prefix_digest = _tensor_digest(base)
        self.branch_features = self._summarize_prefix(base, prefix_digest)
        self.prefix_checkpoint = {
            "schema_version": 1,
            "seed": self._risk_seed,
            "branch_point": self.branch_point,
            "prefix_state_sha256": prefix_digest,
            "batch": base.clone(),
            "rng_state": prefix_rng,
        }
        pulse_policies = [policy for policy in self._policy_specs if policy["policy_id"] != "C0"]
        for policy in pulse_policies:
            self._restore_rng(prefix_rng)
            branch = base.clone()
            pulse_stop = self.branch_point + int(policy["duration"])
            branch, branch_mean = self._advance_field(
                branch, mask=mask, timesteps=timesteps, dt=dt,
                start=self.branch_point, stop=pulse_stop, scales=policy["scales"],
            )
            branch, branch_mean = self._advance_field(
                branch, mask=mask, timesteps=timesteps, dt=dt,
                start=pulse_stop, stop=self.N, scales=base_scales,
            )
            self._record(policy, self._final_atoms(branch_mean), prefix_digest)
        self._restore_rng(prefix_rng)
        c0_batch, c0_mean = self._advance_field(
            base.clone(), mask=mask, timesteps=timesteps, dt=dt,
            start=self.branch_point, stop=self.N, scales=base_scales,
        )
        c0 = next(policy for policy in self._policy_specs if policy["policy_id"] == "C0")
        self._record(c0, self._final_atoms(c0_mean), prefix_digest)
        if len(self.policy_structures) != 5:
            raise RuntimeError("Phase-B did not produce all five frozen outcomes")
        return c0_batch, c0_mean, None
