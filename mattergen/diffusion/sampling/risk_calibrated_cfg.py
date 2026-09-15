"""Full-outcome counterfactual data and one-shot risk-calibrated CFG sampling.

The data sampler follows a fixed CFG=2 trajectory.  At each pre-registered
decision index it clones the exact state and all random streams, then holds one
of CFG {1.75, 2.0, 2.25} to the endpoint.  Diagnostic branches are discarded
before the base trajectory continues.  The module intentionally contains no
short-horizon proxy and no repeated online controller decisions.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_masses

from mattergen.common.data.collate import collate
from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector, _mask_replace


ACTION_BY_SCALE = {1.75: "DOWN", 2.0: "KEEP", 2.25: "UP"}
FIELDS = ("atomic_numbers", "pos", "cell")


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.detach().float().square())).cpu().item())


def _cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    denominator = float(
        (torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat))
        .cpu()
        .item()
    )
    if not math.isfinite(denominator) or denominator <= eps:
        return 0.0
    result = float(torch.dot(left_flat, right_flat).cpu().item()) / denominator
    return float(np.clip(result, -1.0, 1.0))


def _tensor_digest(batch: Any) -> str:
    digest = hashlib.sha256()
    for field in FIELDS:
        value = batch[field].detach().contiguous().cpu()
        digest.update(field.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class FullOutcomeCounterfactualSampler(GuidedPredictorCorrector):
    """Collect full paired outcomes at a small frozen set of base states."""

    def __init__(
        self,
        *,
        sample_seed: int,
        decision_indices: Sequence[int],
        candidate_scales: Sequence[float] = (1.75, 2.0, 2.25),
        base_scale: float = 2.0,
        temporal_ema_beta: float = 0.9,
        recent_window: int = 5,
        **kwargs: Any,
    ) -> None:
        kwargs.update(
            guidance_scale=float(base_scale),
            guidance_schedule="constant",
            sample_seed=int(sample_seed),
        )
        super().__init__(**kwargs)
        self._risk_seed = int(sample_seed)
        self._decision_indices = tuple(int(value) for value in decision_indices)
        self._candidate_scales = tuple(float(value) for value in candidate_scales)
        self._base_scale = float(base_scale)
        self._temporal_beta = float(temporal_ema_beta)
        self._recent_window = int(recent_window)
        if tuple(sorted(set(self._decision_indices))) != self._decision_indices:
            raise ValueError("decision_indices must be unique and increasing")
        if not 1 <= len(self._decision_indices) <= 4:
            raise ValueError("between one deployment timestep and four data timesteps are permitted")
        if self._decision_indices[0] < 0 or self._decision_indices[-1] >= self.N:
            raise ValueError("decision index outside sampler range")
        if self._candidate_scales != (1.75, 2.0, 2.25):
            raise ValueError("candidate CFG values are frozen to (1.75, 2.0, 2.25)")
        if abs(self._base_scale - 2.0) > 1e-12:
            raise ValueError("base CFG is frozen to 2.0")
        if not 0.0 <= self._temporal_beta < 1.0 or self._recent_window < 2:
            raise ValueError("invalid temporal feature configuration")
        self.full_structures: list[tuple[dict[str, Any], Atoms]] = []
        self.feature_rows: list[dict[str, Any]] = []
        self.branch_rows: list[dict[str, Any]] = []
        self._residual_history: list[float] = []
        self._ema_log_residual: float | None = None

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self.full_structures = []
        self.feature_rows = []
        self.branch_rows = []
        self._residual_history = []
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
        guided = unconditional.replace(
            **{
                field: torch.lerp(unconditional[field], conditional[field], float(scale))
                for field in self._multi_corruption.corrupted_fields
            }
        )
        if return_features:
            return guided, self._instantaneous_features(unconditional, conditional)
        return guided

    @staticmethod
    def _instantaneous_features(unconditional: Any, conditional: Any) -> dict[str, float]:
        row: dict[str, float] = {}
        residual_values: list[float] = []
        ratios: list[float] = []
        alignments: list[float] = []
        for field in FIELDS:
            residual = conditional[field] - unconditional[field]
            residual_rms = _rms(residual)
            unconditional_rms = _rms(unconditional[field])
            ratio = residual_rms / max(unconditional_rms, 1e-12)
            alignment = _cosine(unconditional[field], residual)
            row[f"r_{field}"] = residual_rms
            row[f"score_ratio_{field}"] = ratio
            row[f"score_alignment_{field}"] = alignment
            residual_values.append(residual_rms)
            ratios.append(ratio)
            alignments.append(alignment)
        for prefix, values in (
            ("residual", residual_values),
            ("ratio", ratios),
            ("alignment", alignments),
        ):
            row[f"{prefix}_mean"] = float(np.mean(values))
            row[f"{prefix}_median"] = float(np.median(values))
            row[f"{prefix}_std"] = float(np.std(values))
            row[f"{prefix}_range"] = float(max(values) - min(values))
        logs = np.log(np.asarray(residual_values, dtype=float) + 1e-12)
        row["log_residual_mean"] = float(logs.mean())
        row["log_residual_median"] = float(np.median(logs))
        row["log_residual_std"] = float(logs.std())
        row["log_residual_range"] = float(logs.max() - logs.min())
        return row

    def _clean_atoms(self, *, x: Any, score: Any, t: torch.Tensor) -> Atoms:
        atom_corruption = self._multi_corruption.discrete_corruptions["atomic_numbers"]
        clean_atomic = torch.argmax(score["atomic_numbers"].float(), dim=-1) + int(
            atom_corruption.offset
        )
        pos_sde = self._multi_corruption.sdes["pos"]
        _, pos_std = pos_sde.marginal_prob(
            x=x["pos"], t=t, batch_idx=x.get_batch_idx("pos"), batch=x
        )
        clean_pos = pos_sde.wrap(x["pos"] + pos_std.square() * score["pos"])
        cell_sde = self._multi_corruption.sdes["cell"]
        alpha, cell_std = cell_sde.mean_coeff_and_std(x["cell"], t=t, batch=x)
        limit_mean = cell_sde.get_limit_mean(x=x["cell"], batch=x)
        clean_cell = (
            x["cell"] + cell_std.square() * score["cell"] - (1.0 - alpha) * limit_mean
        ) / alpha
        return Atoms(
            numbers=clean_atomic.detach().cpu().numpy().astype(int),
            scaled_positions=clean_pos.detach().cpu().double().numpy(),
            cell=clean_cell[0].detach().cpu().double().numpy(),
            pbc=True,
        )

    @staticmethod
    def _structural_features(atoms: Atoms) -> dict[str, float]:
        numbers = np.asarray(atoms.numbers, dtype=int)
        try:
            volume = abs(float(atoms.get_volume()))
        except (ValueError, np.linalg.LinAlgError):
            volume = 0.0
        mass = float(atomic_masses[numbers].sum()) if len(numbers) else 0.0
        density = mass * 1.66053906660 / max(volume, 1e-12)
        structural_valid = float(np.isfinite(volume) and volume > 0.1)
        if len(numbers) > 1:
            try:
                distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
                triangular = distances[np.triu_indices(len(numbers), k=1)]
                finite = triangular[np.isfinite(triangular)]
                minimum_distance = float(finite.min()) if finite.size else 0.0
                mean_distance = float(finite.mean()) if finite.size else 0.0
            except (ValueError, np.linalg.LinAlgError):
                minimum_distance = mean_distance = 0.0
                structural_valid = 0.0
        else:
            minimum_distance = mean_distance = 0.0
        counts = np.bincount(numbers, minlength=119)[1:]
        fractions = counts[counts > 0].astype(float) / max(len(numbers), 1)
        entropy = float(-(fractions * np.log(fractions + 1e-12)).sum())
        values = {
            "x0_structural_valid": structural_valid,
            "x0_cell_volume": volume,
            "x0_density_g_cm3": density,
            "x0_num_atoms": float(len(numbers)),
            "x0_num_elements": float(np.count_nonzero(counts)),
            "x0_mean_atomic_number": float(numbers.mean()) if len(numbers) else 0.0,
            "x0_std_atomic_number": float(numbers.std()) if len(numbers) else 0.0,
            "x0_max_element_fraction": float(fractions.max()) if fractions.size else 0.0,
            "x0_element_entropy": entropy,
            "x0_min_periodic_distance": minimum_distance,
            "x0_mean_periodic_distance": mean_distance,
        }
        return {
            key: float(value) if np.isfinite(value) else 0.0
            for key, value in values.items()
        }

    def _feature_row(
        self,
        *,
        sampling_step: int,
        t: torch.Tensor,
        features: Mapping[str, float],
        x0: Atoms,
    ) -> dict[str, Any]:
        current = float(features["log_residual_median"])
        previous = self._residual_history[-1] if self._residual_history else current
        recent = self._residual_history[-self._recent_window :]
        recent_trend = (
            float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
            if len(recent) >= 2
            else 0.0
        )
        ema = current if self._ema_log_residual is None else self._ema_log_residual
        return {
            "seed": self._risk_seed,
            "sampling_step": int(sampling_step),
            "t_norm": float(t.detach().reshape(-1)[0].cpu().item()),
            **dict(features),
            "previous_log_residual": previous,
            "ema_log_residual": ema,
            "ema_slope": current - ema,
            "recent_residual_trend": recent_trend,
            "previous_cfg": self._base_scale,
            **self._structural_features(x0),
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
        self._residual_history.append(current)

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
            raise ValueError("full-outcome sampler writes explicit diagnostic outputs")
        for key in self._predictors:
            mask.setdefault(key, None)
        for key in self._correctors:
            mask.setdefault(key, None)
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1), device=self._device)
        decision_set = set(self._decision_indices)
        mean_batch = batch.clone()
        for index in range(self.N):
            if index in decision_set:
                ordinal = self._decision_indices.index(index)
                t = torch.full((batch.get_batch_size(),), timesteps[index], device=self._device)
                guided, instantaneous = self._score_at_scale(
                    batch, t, self._base_scale, return_features=True
                )
                state_digest = _tensor_digest(batch)
                x0 = self._clean_atoms(x=batch, score=guided, t=t)
                feature_row = self._feature_row(
                    sampling_step=index, t=t, features=instantaneous, x0=x0
                )
                feature_row.update(decision_ordinal=ordinal, state_sha256=state_digest)
                self.feature_rows.append(feature_row)
                snapshot = self._rng_snapshot()
                for scale in self._candidate_scales:
                    self._restore_rng(snapshot)
                    _, branch_mean = self._advance(
                        batch.clone(),
                        mask=mask,
                        timesteps=timesteps,
                        dt=dt,
                        start=index,
                        stop=self.N,
                        scale=scale,
                    )
                    metadata = {
                        "seed": self._risk_seed,
                        "decision_ordinal": ordinal,
                        "sampling_step": index,
                        "decision_t_norm": float(t[0].detach().cpu().item()),
                        "candidate_cfg": scale,
                        "candidate_action": ACTION_BY_SCALE[scale],
                        "state_sha256": state_digest,
                        "branch_horizon_steps": self.N - index,
                    }
                    self.full_structures.append((dict(metadata), self._final_atoms(branch_mean)))
                    self.branch_rows.append(dict(metadata))
                self._restore_rng(snapshot)
            batch, mean_batch = self._advance(
                batch,
                mask=mask,
                timesteps=timesteps,
                dt=dt,
                start=index,
                stop=index + 1,
                scale=self._base_scale,
                track_base_temporal=True,
            )
        return batch, mean_batch, None


class OneShotRiskCalibratedSampler(FullOutcomeCounterfactualSampler):
    """Deploy one frozen selective decision and emit paired C0/T0/A0 finals."""

    def __init__(
        self,
        *,
        sample_seed: int,
        decision_index: int,
        ensemble_path: str,
        metadata_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_seed=sample_seed,
            decision_indices=(decision_index,),
            candidate_scales=(1.75, 2.0, 2.25),
            base_scale=2.0,
            temporal_ema_beta=0.9,
            recent_window=5,
            **kwargs,
        )
        import joblib

        self._ensemble = joblib.load(ensemble_path)
        self._controller_metadata = __import__("json").loads(
            __import__("pathlib").Path(metadata_path).read_text()
        )
        if int(self._controller_metadata["sampling_step"]) != int(decision_index):
            raise ValueError("controller metadata timestep mismatch")
        if not bool(self._controller_metadata["one_shot"]):
            raise ValueError("controller metadata is not one-shot")
        self.policy_structures: list[tuple[dict[str, Any], Atoms]] = []
        self.controller_trace: list[dict[str, Any]] = []

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self.policy_structures = []
        self.controller_trace = []

    def _normalized_controller_features(self, row: Mapping[str, Any]):
        import pandas as pd

        values = dict(row)
        step = str(int(row["sampling_step"]))
        normalizers = self._controller_metadata["normalization_parameters"][step]
        for source, destination in (
            ("r_atomic_numbers", "z_atomic"), ("r_pos", "z_pos"), ("r_cell", "z_cell")
        ):
            parameters = normalizers[source.replace("r_atomic_numbers", "r_atomic")]
            values[destination] = (
                math.log(float(row[source]) + 1e-12) - float(parameters["log_mean"])
            ) / float(parameters["log_std"])
        rename = {
            "r_atomic_numbers": "r_atomic",
            "score_ratio_atomic_numbers": "score_ratio_atomic",
            "score_alignment_atomic_numbers": "score_alignment_atomic",
        }
        values.update({destination: values[source] for source, destination in rename.items()})
        columns = list(self._controller_metadata["features"])
        frame = pd.DataFrame([{name: float(values[name]) for name in columns}], columns=columns)
        if not np.isfinite(frame.to_numpy()).all():
            raise RuntimeError("non-finite deployment feature")
        return frame

    @staticmethod
    def _harm_probability(model: Any, frame: Any) -> float:
        probabilities = model.predict_proba(frame)[0]
        classes = list(map(int, model.classes_))
        return float(probabilities[classes.index(1)]) if 1 in classes else 0.0

    def _controller_decision(self, row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        frame = self._normalized_controller_features(row)
        operating = self._controller_metadata["operating_point"]
        candidates: list[tuple[float, int, str]] = []
        trace: dict[str, Any] = {}
        for action in ("DOWN", "UP"):
            effects = np.asarray(
                [member["regression"][action].predict(frame)[0] for member in self._ensemble["members"]],
                dtype=float,
            )
            harms = np.asarray(
                [self._harm_probability(member["harm"][action], frame) for member in self._ensemble["members"]],
                dtype=float,
            )
            mean = float(effects.mean())
            std = float(effects.std(ddof=1))
            harm = float(harms.mean())
            conservative = mean + std
            eligible = bool(
                mean < -float(operating["delta_min"])
                and conservative < -float(operating["delta_safe"])
                and harm < float(operating["eta"])
                and std <= float(operating["uncertainty_max"])
            )
            trace.update(
                {
                    f"predicted_delta_{action}": mean,
                    f"uncertainty_{action}": std,
                    f"conservative_upper_{action}": conservative,
                    f"predicted_harm_{action}": harm,
                    f"eligible_{action}": eligible,
                }
            )
            if eligible:
                candidates.append((mean, 1 if action == "DOWN" else 2, action))
        selected = min(candidates)[2] if candidates else "KEEP"
        return selected, trace

    @torch.no_grad()
    def _denoise(
        self, batch: Any, mask: dict[str, torch.Tensor], record: bool = False
    ) -> tuple[Any, Any, None]:
        if record:
            raise ValueError("one-shot sampler writes explicit policy outputs")
        for key in self._predictors:
            mask.setdefault(key, None)
        for key in self._correctors:
            mask.setdefault(key, None)
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1), device=self._device)
        decision_index = self._decision_indices[0]
        mean_batch = batch.clone()
        for index in range(decision_index):
            batch, mean_batch = self._advance(
                batch, mask=mask, timesteps=timesteps, dt=dt, start=index, stop=index + 1,
                scale=self._base_scale, track_base_temporal=True,
            )
        t = torch.full((batch.get_batch_size(),), timesteps[decision_index], device=self._device)
        guided, instantaneous = self._score_at_scale(batch, t, self._base_scale, return_features=True)
        state_digest = _tensor_digest(batch)
        feature_row = self._feature_row(
            sampling_step=decision_index,
            t=t,
            features=instantaneous,
            x0=self._clean_atoms(x=batch, score=guided, t=t),
        )
        selected_action, prediction_trace = self._controller_decision(feature_row)
        t0_action = str(self._controller_metadata["t0_action"])
        policies = {"C0": "KEEP", "T0": t0_action, "A0": selected_action}
        snapshot = self._rng_snapshot()
        a0_mean = None
        for policy, action in policies.items():
            self._restore_rng(snapshot)
            scale = {"DOWN": 1.75, "KEEP": 2.0, "UP": 2.25}[action]
            _, branch_mean = self._advance(
                batch.clone(), mask=mask, timesteps=timesteps, dt=dt,
                start=decision_index, stop=self.N, scale=scale,
            )
            metadata = {
                "seed": self._risk_seed,
                "policy": policy,
                "selected_action": action,
                "selected_cfg": scale,
                "sampling_step": decision_index,
                "decision_t_norm": float(t[0].detach().cpu().item()),
                "state_sha256": state_digest,
            }
            self.policy_structures.append((metadata, self._final_atoms(branch_mean)))
            if policy == "A0":
                a0_mean = branch_mean
        self._restore_rng(snapshot)
        self.controller_trace.append(
            {
                "seed": self._risk_seed,
                "sampling_step": decision_index,
                "state_sha256": state_digest,
                "selected_action": selected_action,
                "selected_cfg": {"DOWN": 1.75, "KEEP": 2.0, "UP": 2.25}[selected_action],
                "t0_action": t0_action,
                **prediction_trace,
                **feature_row,
            }
        )
        if a0_mean is None:
            raise RuntimeError("A0 branch was not produced")
        return a0_mean, a0_mean, None
