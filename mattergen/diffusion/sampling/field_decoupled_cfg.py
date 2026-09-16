"""Training-free paired sampler with independent CFG scales per corrupted field."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from ase import Atoms

from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.sampling.pc_sampler import _mask_replace
from mattergen.diffusion.sampling.risk_calibrated_cfg import (
    FullOutcomeCounterfactualSampler,
    _tensor_digest,
)


FIELDS = ("atomic_numbers", "pos", "cell")
ALIASES = {"atomic": "atomic_numbers", "position": "pos", "pos": "pos", "cell": "cell"}


def normalize_scales(scales: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_name, value in scales.items():
        name = ALIASES.get(str(raw_name), str(raw_name))
        if name not in FIELDS:
            raise ValueError(f"unknown guided field: {raw_name}")
        result[name] = float(value)
    if set(result) != set(FIELDS):
        raise ValueError(f"field scales must cover exactly {FIELDS}: {result}")
    if any(not 1.5 <= value <= 2.5 for value in result.values()):
        raise ValueError(f"field scale outside frozen safe range: {result}")
    return result


def normalize_policies(policies: Sequence[Mapping[str, Any]], n_steps: int = 1000) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for raw in policies:
        policy = dict(raw)
        identifier = str(policy["policy_id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate policy_id: {identifier}")
        kind = str(policy["kind"])
        scales = normalize_scales(policy["scales"])
        if kind == "constant":
            start, duration = None, None
        elif kind == "pulse":
            start, duration = int(policy["start"]), int(policy["duration"])
            if start < 0 or duration <= 0 or start + duration > n_steps:
                raise ValueError(f"invalid pulse window: {policy}")
        else:
            raise ValueError(f"unknown policy kind: {kind}")
        result.append({"policy_id": identifier, "kind": kind, "scales": scales, "start": start, "duration": duration})
        identifiers.add(identifier)
    if "C0" not in identifiers:
        raise ValueError("policies must include C0")
    c0 = next(policy for policy in result if policy["policy_id"] == "C0")
    if c0["kind"] != "constant" or any(value != 2.0 for value in c0["scales"].values()):
        raise ValueError("C0 must be constant (2,2,2)")
    return tuple(result)


class FieldDecoupledPolicySampler(FullOutcomeCounterfactualSampler):
    """Generate paired final outcomes for a frozen set of field-scale policies."""

    def __init__(
        self,
        *,
        sample_seed: int,
        policy_specs: Sequence[Mapping[str, Any]],
        implementation_check: bool = False,
        **kwargs: Any,
    ) -> None:
        self._policy_specs = normalize_policies(policy_specs)
        starts = tuple(sorted({int(policy["start"]) for policy in self._policy_specs if policy["kind"] == "pulse"})) or (400,)
        super().__init__(
            sample_seed=sample_seed,
            decision_indices=starts,
            candidate_scales=(1.75, 2.0, 2.25),
            base_scale=2.0,
            temporal_ema_beta=0.9,
            recent_window=5,
            **kwargs,
        )
        self._implementation_check = bool(implementation_check)
        self.policy_structures: list[tuple[dict[str, Any], Atoms]] = []
        self.policy_rows: list[dict[str, Any]] = []
        self.implementation_check_result: dict[str, Any] | None = None

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self.policy_structures = []
        self.policy_rows = []
        self.implementation_check_result = None

    def _score_at_field_scales(self, x: Any, t: torch.Tensor, scales: Mapping[str, float]) -> Any:
        unconditional, conditional = self._components(x, t)
        return unconditional.replace(
            **{
                field: torch.lerp(unconditional[field], conditional[field], float(scales[field]))
                for field in self._multi_corruption.corrupted_fields
            }
        )

    def _advance_field(
        self,
        batch: Any,
        *,
        mask: dict[str, torch.Tensor | None],
        timesteps: torch.Tensor,
        dt: torch.Tensor,
        start: int,
        stop: int,
        scales: Mapping[str, float],
    ) -> tuple[Any, Any]:
        mean_batch = batch.clone()
        for index in range(start, stop):
            t = torch.full((batch.get_batch_size(),), timesteps[index], device=self._device)
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    self._set_sampling_context(sampling_step=index, phase="corrector")
                    score = self._score_at_field_scales(batch, t, scales)
                    samples_means = apply(
                        fns={key: value.step_given_score for key, value in self._correctors.items()},
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    batch, mean_batch = _mask_replace(samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask)
            self._set_sampling_context(sampling_step=index, phase="predictor")
            score = self._score_at_field_scales(batch, t, scales)
            samples_means = apply(
                fns={key: value.update_given_score for key, value in self._predictors.items()},
                x=batch,
                score=score,
                broadcast={"t": t, "batch": batch, "dt": dt},
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )
            batch, mean_batch = _mask_replace(samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask)
        return batch, mean_batch

    def _record(self, policy: Mapping[str, Any], atoms: Atoms, prefix_digest: str) -> None:
        metadata = {
            "seed": self._risk_seed,
            "policy_id": str(policy["policy_id"]),
            "policy_kind": str(policy["kind"]),
            "g_atomic": float(policy["scales"]["atomic_numbers"]),
            "g_pos": float(policy["scales"]["pos"]),
            "g_cell": float(policy["scales"]["cell"]),
            "start": -1 if policy["start"] is None else int(policy["start"]),
            "duration": self.N if policy["duration"] is None else int(policy["duration"]),
            "prefix_state_sha256": prefix_digest,
        }
        self.policy_structures.append((metadata, atoms))
        self.policy_rows.append(dict(metadata))

    @staticmethod
    def _exact_atoms_comparison(left: Atoms, right: Atoms) -> dict[str, Any]:
        return {
            "atomic_numbers_identical": bool(np.array_equal(left.numbers, right.numbers)),
            "positions_max_abs_error": float(np.max(np.abs(left.get_scaled_positions(wrap=False) - right.get_scaled_positions(wrap=False)))),
            "cell_max_abs_error": float(np.max(np.abs(left.cell.array - right.cell.array))),
        }

    @torch.no_grad()
    def _denoise(self, batch: Any, mask: dict[str, torch.Tensor], record: bool = False) -> tuple[Any, Any, None]:
        if record:
            raise ValueError("field policy sampler writes explicit policy outputs")
        for key in self._predictors:
            mask.setdefault(key, None)
        for key in self._correctors:
            mask.setdefault(key, None)
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1), device=self._device)
        initial_rng = self._rng_snapshot()
        initial_digest = _tensor_digest(batch)
        shared_c0_atoms: Atoms | None = None
        if self._implementation_check:
            self._restore_rng(initial_rng)
            _, shared_mean = self._advance(batch.clone(), mask=mask, timesteps=timesteps, dt=dt, start=0, stop=self.N, scale=2.0)
            shared_c0_atoms = self._final_atoms(shared_mean)

        for policy in self._policy_specs:
            if policy["kind"] != "constant" or policy["policy_id"] == "C0":
                continue
            self._restore_rng(initial_rng)
            _, mean = self._advance_field(batch.clone(), mask=mask, timesteps=timesteps, dt=dt, start=0, stop=self.N, scales=policy["scales"])
            self._record(policy, self._final_atoms(mean), initial_digest)

        self._restore_rng(initial_rng)
        base = batch.clone()
        base_mean = batch.clone()
        base_scales = {field: 2.0 for field in FIELDS}
        by_start: dict[int, list[Mapping[str, Any]]] = {}
        for policy in self._policy_specs:
            if policy["kind"] == "pulse":
                by_start.setdefault(int(policy["start"]), []).append(policy)
        for index in range(self.N):
            if index in by_start:
                prefix_rng = self._rng_snapshot()
                prefix_digest = _tensor_digest(base)
                for policy in by_start[index]:
                    self._restore_rng(prefix_rng)
                    branch = base.clone()
                    pulse_stop = index + int(policy["duration"])
                    branch, branch_mean = self._advance_field(branch, mask=mask, timesteps=timesteps, dt=dt, start=index, stop=pulse_stop, scales=policy["scales"])
                    if pulse_stop < self.N:
                        branch, branch_mean = self._advance_field(branch, mask=mask, timesteps=timesteps, dt=dt, start=pulse_stop, stop=self.N, scales=base_scales)
                    self._record(policy, self._final_atoms(branch_mean), prefix_digest)
                self._restore_rng(prefix_rng)
            base, base_mean = self._advance_field(base, mask=mask, timesteps=timesteps, dt=dt, start=index, stop=index + 1, scales=base_scales)
        c0 = next(policy for policy in self._policy_specs if policy["policy_id"] == "C0")
        c0_atoms = self._final_atoms(base_mean)
        self._record(c0, c0_atoms, initial_digest)
        if shared_c0_atoms is not None:
            comparison = self._exact_atoms_comparison(shared_c0_atoms, c0_atoms)
            comparison["success"] = bool(comparison["atomic_numbers_identical"] and comparison["positions_max_abs_error"] == 0.0 and comparison["cell_max_abs_error"] == 0.0)
            self.implementation_check_result = comparison
            if not comparison["success"]:
                raise RuntimeError(f"STOP_IMPLEMENTATION_ERROR: {comparison}")
        if len(self.policy_structures) != len(self._policy_specs):
            raise RuntimeError("not every frozen field policy produced an outcome")
        return base_mean, base_mean, None
