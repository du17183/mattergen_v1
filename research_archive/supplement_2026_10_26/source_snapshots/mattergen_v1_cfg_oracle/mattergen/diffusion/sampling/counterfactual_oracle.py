"""Deterministic counterfactual CFG branching for the Adaptive-CFG oracle audit.

This sampler is diagnostic-only.  It follows a fixed-CFG=2 trajectory and, at
pre-registered reverse timesteps, clones the exact state and random streams for
three local CFG actions.  Counterfactual work is discarded before the base
trajectory continues, so diagnostic branches cannot perturb later base states.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from ase import Atoms

from mattergen.common.data.collate import collate
from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector, _mask_replace


ACTION_BY_SCALE = {1.75: "DOWN", 2.0: "KEEP", 2.25: "UP"}


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.detach().float().square())).cpu().item())


def _cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    denominator = float((torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)).cpu().item())
    if not math.isfinite(denominator) or denominator <= eps:
        return 0.0
    value = float(torch.dot(left_flat, right_flat).cpu().item()) / denominator
    return float(max(-1.0, min(1.0, value)))


def _tensor_digest(batch: Any) -> str:
    digest = hashlib.sha256()
    for field in ("atomic_numbers", "pos", "cell"):
        value = batch[field].detach().contiguous().cpu()
        digest.update(field.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class CounterfactualCFGOracleSampler(GuidedPredictorCorrector):
    """Run paired local CFG branches from states on a fixed-CFG trajectory."""

    def __init__(
        self,
        *,
        sample_seed: int,
        first_oracle_seed: int,
        decision_indices: Sequence[int],
        candidate_scales: Sequence[float] = (1.75, 2.0, 2.25),
        short_horizon_steps: int = 15,
        base_scale: float = 2.0,
        temporal_ema_beta: float = 0.9,
        **kwargs: Any,
    ) -> None:
        kwargs.update(
            guidance_scale=float(base_scale),
            guidance_schedule="constant",
            sample_seed=int(sample_seed),
        )
        super().__init__(**kwargs)
        self._oracle_seed = int(sample_seed)
        self._first_oracle_seed = int(first_oracle_seed)
        self._decision_indices = tuple(int(value) for value in decision_indices)
        self._candidate_scales = tuple(float(value) for value in candidate_scales)
        self._short_horizon = int(short_horizon_steps)
        self._base_scale = float(base_scale)
        self._temporal_beta = float(temporal_ema_beta)
        if tuple(sorted(set(self._decision_indices))) != self._decision_indices:
            raise ValueError("decision_indices must be unique and increasing")
        if not self._decision_indices or self._decision_indices[0] < 0 or self._decision_indices[-1] >= self.N:
            raise ValueError("decision index outside sampler range")
        if self._candidate_scales != (1.75, 2.0, 2.25):
            raise ValueError("oracle candidate set is frozen to (1.75, 2.0, 2.25)")
        if self._short_horizon != 15 or abs(self._base_scale - 2.0) > 1e-12:
            raise ValueError("oracle horizon/base CFG differs from frozen protocol")
        if not 0.0 <= self._temporal_beta < 1.0:
            raise ValueError("invalid temporal EMA beta")
        self.short_structures: list[tuple[dict[str, Any], Atoms]] = []
        self.full_structures: list[tuple[dict[str, Any], Atoms]] = []
        self.feature_rows: list[dict[str, Any]] = []
        self.branch_rows: list[dict[str, Any]] = []
        self._previous_log_residual: float | None = None
        self._ema_log_residual: float | None = None

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self.short_structures = []
        self.full_structures = []
        self.feature_rows = []
        self.branch_rows = []
        self._previous_log_residual = None
        self._ema_log_residual = None

    def _components(self, x: Any, t: torch.Tensor) -> tuple[Any, Any]:
        batch_no_condition = self._remove_conditioning_fn(x)
        batch_with_condition = self._keep_conditioning_fn(x)
        joint_batch = collate([batch_no_condition, batch_with_condition])
        for attr, value in batch_no_condition.items():
            if isinstance(value, list):
                joint_batch[attr] = batch_no_condition[attr] + batch_with_condition[attr]
        self._exact_score_call_count += 1
        combined = PredictorCorrector._score_fn(
            self, x=joint_batch, t=torch.cat([t, t], dim=0)
        )
        return combined[0], combined[1]

    def _score_at_scale(
        self, x: Any, t: torch.Tensor, scale: float, *, return_features: bool = False
    ) -> Any:
        unconditional, conditional = self._components(x, t)
        fields = tuple(self._multi_corruption.corrupted_fields)
        guided = unconditional.replace(
            **{
                field: torch.lerp(unconditional[field], conditional[field], float(scale))
                for field in fields
            }
        )
        if return_features:
            return guided, self._instantaneous_features(unconditional, conditional)
        return guided

    def _instantaneous_features(self, unconditional: Any, conditional: Any) -> dict[str, float]:
        row: dict[str, float] = {}
        residual_values: list[float] = []
        log_values: list[float] = []
        for field in ("atomic_numbers", "pos", "cell"):
            residual = conditional[field] - unconditional[field]
            residual_rms = _rms(residual)
            unconditional_rms = _rms(unconditional[field])
            row[f"r_{field}"] = residual_rms
            row[f"score_ratio_{field}"] = residual_rms / max(unconditional_rms, 1e-12)
            row[f"score_alignment_{field}"] = _cosine(unconditional[field], residual)
            residual_values.append(residual_rms)
            log_values.append(math.log(residual_rms + 1e-12))
        row["residual_std"] = float(np.std(residual_values))
        row["residual_range"] = float(max(residual_values) - min(residual_values))
        row["residual_median"] = float(np.median(residual_values))
        row["log_residual_std"] = float(np.std(log_values))
        row["log_residual_range"] = float(max(log_values) - min(log_values))
        row["log_residual_median"] = float(np.median(log_values))
        return row

    def _feature_row(
        self, *, sampling_step: int, t: torch.Tensor, features: Mapping[str, float]
    ) -> dict[str, Any]:
        current = float(features["log_residual_median"])
        previous = self._previous_log_residual
        ema_before = self._ema_log_residual
        return {
            "seed": self._oracle_seed,
            "sampling_step": int(sampling_step),
            "t_norm": float(t.detach().reshape(-1)[0].cpu().item()),
            **dict(features),
            "previous_log_residual": current if previous is None else previous,
            "ema_log_residual": current if ema_before is None else ema_before,
            "ema_slope": 0.0 if ema_before is None else current - ema_before,
            "previous_cfg": self._base_scale,
        }

    def _update_temporal(self, features: Mapping[str, float]) -> None:
        current = float(features["log_residual_median"])
        if self._ema_log_residual is None:
            self._ema_log_residual = current
        else:
            self._ema_log_residual = (
                self._temporal_beta * self._ema_log_residual
                + (1.0 - self._temporal_beta) * current
            )
        self._previous_log_residual = current

    def _advance(
        self,
        batch: Any,
        *,
        mask: dict[str, torch.Tensor | None],
        timesteps: torch.Tensor,
        dt: torch.Tensor,
        start: int,
        stop: int,
        scale: float,
        track_base_temporal: bool = False,
    ) -> tuple[Any, Any]:
        mean_batch = batch.clone()
        for index in range(start, stop):
            t = torch.full((batch.get_batch_size(),), timesteps[index], device=self._device)
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    self._set_sampling_context(sampling_step=index, phase="corrector")
                    score = self._score_at_scale(batch, t, scale)
                    samples_means = apply(
                        fns={key: value.step_given_score for key, value in self._correctors.items()},
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means,
                        batch=batch,
                        mean_batch=mean_batch,
                        mask=mask,
                    )
            self._set_sampling_context(sampling_step=index, phase="predictor")
            if track_base_temporal:
                score, features = self._score_at_scale(batch, t, scale, return_features=True)
                self._update_temporal(features)
            else:
                score = self._score_at_scale(batch, t, scale)
            samples_means = apply(
                fns={key: value.update_given_score for key, value in self._predictors.items()},
                x=batch,
                score=score,
                broadcast={"t": t, "batch": batch, "dt": dt},
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )
            batch, mean_batch = _mask_replace(
                samples_means=samples_means,
                batch=batch,
                mean_batch=mean_batch,
                mask=mask,
            )
        return batch, mean_batch

    def _rng_snapshot(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(self._device),
        }

    def _restore_rng(self, snapshot: Mapping[str, Any]) -> None:
        random.setstate(snapshot["python"])
        np.random.set_state(snapshot["numpy"])
        torch.set_rng_state(snapshot["torch_cpu"])
        torch.cuda.set_rng_state(snapshot["torch_cuda"], self._device)

    def _clean_atoms(self, *, x: Any, score: Any, t: torch.Tensor) -> Atoms:
        atom_corruption = self._multi_corruption.discrete_corruptions["atomic_numbers"]
        clean_atomic = torch.argmax(score["atomic_numbers"].float(), dim=-1) + int(atom_corruption.offset)
        pos_sde = self._multi_corruption.sdes["pos"]
        _, pos_std = pos_sde.marginal_prob(
            x=x["pos"], t=t, batch_idx=x.get_batch_idx("pos"), batch=x
        )
        clean_pos = pos_sde.wrap(x["pos"] + pos_std.square() * score["pos"])
        cell_sde = self._multi_corruption.sdes["cell"]
        alpha, cell_std = cell_sde.mean_coeff_and_std(x["cell"], t=t, batch=x)
        limit_mean = cell_sde.get_limit_mean(x=x["cell"], batch=x)
        clean_cell = (
            x["cell"] + cell_std.square() * score["cell"]
            - (1.0 - alpha) * limit_mean
        ) / alpha
        return Atoms(
            numbers=clean_atomic.detach().cpu().numpy().astype(int),
            scaled_positions=clean_pos.detach().cpu().double().numpy(),
            cell=clean_cell[0].detach().cpu().double().numpy(),
            pbc=True,
        )

    @staticmethod
    def _final_atoms(batch: Any) -> Atoms:
        return Atoms(
            numbers=batch["atomic_numbers"].detach().cpu().numpy().astype(int),
            scaled_positions=batch["pos"].detach().cpu().double().numpy(),
            cell=batch["cell"][0].detach().cpu().double().numpy(),
            pbc=True,
        )

    @torch.no_grad()
    def _denoise(
        self, batch: Any, mask: dict[str, torch.Tensor], record: bool = False
    ) -> tuple[Any, Any, None]:
        if record:
            raise ValueError("counterfactual oracle sampler uses explicit diagnostic outputs")
        for key in self._predictors:
            mask.setdefault(key, None)
        for key in self._correctors:
            mask.setdefault(key, None)
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1), device=self._device)
        decision_set = set(self._decision_indices)
        mean_batch = batch.clone()
        index = 0
        while index < self.N:
            if index in decision_set:
                ordinal = self._decision_indices.index(index)
                t = torch.full((batch.get_batch_size(),), timesteps[index], device=self._device)
                _, instantaneous = self._score_at_scale(
                    batch, t, self._base_scale, return_features=True
                )
                state_digest = _tensor_digest(batch)
                feature_row = self._feature_row(
                    sampling_step=index, t=t, features=instantaneous
                )
                feature_row.update(
                    decision_ordinal=ordinal,
                    state_sha256=state_digest,
                    full_rollout_selected=(
                        (self._oracle_seed - self._first_oracle_seed) % 4 == ordinal % 4
                    ),
                )
                self.feature_rows.append(feature_row)
                snapshot = self._rng_snapshot()
                stop = min(index + self._short_horizon, self.N)
                full_selected = bool(feature_row["full_rollout_selected"])
                for scale in self._candidate_scales:
                    self._restore_rng(snapshot)
                    branch_batch, branch_mean = self._advance(
                        batch.clone(),
                        mask=mask,
                        timesteps=timesteps,
                        dt=dt,
                        start=index,
                        stop=stop,
                        scale=scale,
                    )
                    if stop < self.N:
                        short_t = torch.full(
                            (branch_batch.get_batch_size(),),
                            timesteps[stop],
                            device=self._device,
                        )
                        short_score = self._score_at_scale(branch_batch, short_t, scale)
                        short_atoms = self._clean_atoms(
                            x=branch_batch, score=short_score, t=short_t
                        )
                        short_t_value = float(short_t[0].detach().cpu().item())
                    else:
                        short_atoms = self._final_atoms(branch_mean)
                        short_t_value = float(self._eps_t)
                    metadata = {
                        "seed": self._oracle_seed,
                        "decision_ordinal": ordinal,
                        "sampling_step": index,
                        "decision_t_norm": float(t[0].detach().cpu().item()),
                        "short_endpoint_t_norm": short_t_value,
                        "candidate_cfg": scale,
                        "candidate_action": ACTION_BY_SCALE[scale],
                        "branch_horizon_steps": stop - index,
                        "state_sha256": state_digest,
                        "full_rollout_selected": full_selected,
                    }
                    self.short_structures.append((dict(metadata), short_atoms))
                    if full_selected:
                        _, full_mean = self._advance(
                            branch_batch,
                            mask=mask,
                            timesteps=timesteps,
                            dt=dt,
                            start=stop,
                            stop=self.N,
                            scale=scale,
                        )
                        self.full_structures.append((dict(metadata), self._final_atoms(full_mean)))
                    self.branch_rows.append(dict(metadata))
                self._restore_rng(snapshot)
            next_index = min(
                [value for value in self._decision_indices if value > index] + [self.N]
            )
            start = index
            stop = next_index if index not in decision_set else min(index + 1, next_index)
            batch, mean_batch = self._advance(
                batch,
                mask=mask,
                timesteps=timesteps,
                dt=dt,
                start=start,
                stop=stop,
                scale=self._base_scale,
                track_base_temporal=True,
            )
            index = stop
        return batch, mean_batch, None
