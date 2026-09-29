"""New study wrapper around the frozen field-decoupled sampler; old C1 code is untouched."""

from __future__ import annotations

from typing import Any

import torch

from mattergen.diffusion.sampling.field_decoupled_cfg import FieldDecoupledPolicySampler, _tensor_digest


class SearchSampler(FieldDecoupledPolicySampler):
    """Capture a self-describing step-400 checkpoint during the ordinary C0 path."""

    def __init__(self, *, sample_seed: int, policy_specs: Any, implementation_check: bool = False, **kwargs: Any) -> None:
        super().__init__(sample_seed=sample_seed, policy_specs=policy_specs, implementation_check=implementation_check, **kwargs)
        self.prefix_checkpoint: dict[str, Any] | None = None

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self.prefix_checkpoint = None

    def _advance_field(self, batch: Any, *, mask: Any, timesteps: torch.Tensor, dt: torch.Tensor, start: int, stop: int, scales: Any) -> tuple[Any, Any]:
        outcome = super()._advance_field(batch, mask=mask, timesteps=timesteps, dt=dt, start=start, stop=stop, scales=scales)
        if start == 399 and stop == 400 and self.prefix_checkpoint is None:
            state = outcome[0].clone()
            condition = state["dft_mag_density"] if "dft_mag_density" in state else None
            if condition is None:
                raise RuntimeError("target condition missing from full diffusion state")
            self.prefix_checkpoint = {
                "schema_version": 2,
                "seed": self._risk_seed,
                "branch_point": 400,
                "timestep_value": float(timesteps[400].detach().cpu()),
                "dt_value": float(dt.detach().cpu()),
                "num_steps": self.N,
                "condition_dft_mag_density": condition.clone() if isinstance(condition, torch.Tensor) else condition,
                "prefix_state_sha256": _tensor_digest(state),
                "batch": state,
                "rng_state": self._rng_snapshot(),
            }
        return outcome
