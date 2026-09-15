"""One-shot deployment sampler for the frozen calibrated safe-selection controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch
import yaml

from mattergen.diffusion.sampling.pc_sampler import _mask_replace
from mattergen.diffusion.sampling.risk_calibrated_cfg import (
    FullOutcomeCounterfactualSampler,
    _tensor_digest,
)


SCALE_BY_ACTION = {"DOWN": 1.75, "KEEP": 2.0, "UP": 2.25}


def _positive_probability(model: Any, values: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(values)
    classes = list(map(int, model.classes_))
    return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(values))


class OneShotSafeSelectionSampler(FullOutcomeCounterfactualSampler):
    """Emit paired C0/T0/S0 outcomes from one exact step-250 decision state."""

    def __init__(
        self,
        *,
        sample_seed: int,
        controller_bundle_path: str,
        controller_config_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sample_seed=sample_seed,
            decision_indices=(250,),
            candidate_scales=(1.75, 2.0, 2.25),
            base_scale=2.0,
            temporal_ema_beta=0.9,
            recent_window=5,
            **kwargs,
        )
        self._bundle = joblib.load(controller_bundle_path)
        self._controller = yaml.safe_load(Path(controller_config_path).read_text())
        if int(self._controller["decision_index"]) != 250 or not self._controller["one_shot"]:
            raise ValueError("controller is not the frozen one-shot step-250 controller")
        self.policy_structures: list[tuple[dict[str, Any], Any]] = []
        self.controller_trace: list[dict[str, Any]] = []

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self.policy_structures = []
        self.controller_trace = []

    def _feature_vector(self, row: Mapping[str, Any]) -> np.ndarray:
        values = dict(row)
        values.update(
            r_atomic=values["r_atomic_numbers"],
            score_ratio_atomic=values["score_ratio_atomic_numbers"],
            score_alignment_atomic=values["score_alignment_atomic_numbers"],
        )
        for source, destination in (
            ("r_atomic", "z_atomic"), ("r_pos", "z_pos"), ("r_cell", "z_cell")
        ):
            normalizer = self._controller["normalization_parameters"][source]
            values[destination] = (
                np.log(float(values[source]) + 1e-12) - float(normalizer["log_mean"])
            ) / float(normalizer["log_std"])
        columns = list(self._controller["feature_columns"])
        vector = np.asarray([[float(values[name]) for name in columns]], dtype=float)
        if not np.isfinite(vector).all():
            raise RuntimeError("non-finite deployment feature")
        return vector

    def _harm_probability(self, action: str, scaled: np.ndarray) -> float:
        harm_model = self._bundle["harm"][action]
        if hasattr(harm_model, "decision_function"):
            score = np.asarray(harm_model.decision_function(scaled), dtype=float).reshape(-1, 1)
        else:
            probability = np.clip(_positive_probability(harm_model, scaled), 1e-8, 1 - 1e-8)
            score = np.log(probability / (1 - probability)).reshape(-1, 1)
        calibrator = self._bundle["platt"][action]
        if isinstance(calibrator, dict):
            return float(calibrator["probability"])
        return float(_positive_probability(calibrator, score)[0])

    def _controller_decision(self, row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        scaled = self._bundle["scaler"].transform(self._feature_vector(row))
        eligible: list[tuple[float, int, str]] = []
        trace: dict[str, Any] = {}
        for tie, action in enumerate(("DOWN", "UP")):
            mean = float(self._bundle["ridge"][action].predict(scaled)[0])
            q = float(self._controller["conformal_q"][action])
            upper = mean + q
            harm = self._harm_probability(action, scaled)
            passes = bool(
                upper < -float(self._controller["delta_safe"])
                and harm < float(self._controller["eta"])
            )
            trace.update(
                {
                    f"predicted_delta_{action}": mean,
                    f"conformal_q_{action}": q,
                    f"conformal_upper_{action}": upper,
                    f"calibrated_harm_{action}": harm,
                    f"eligible_{action}": passes,
                }
            )
            if passes:
                eligible.append((upper, tie, action))
        selected = min(eligible)[2] if eligible else "KEEP"
        return selected, trace

    @torch.no_grad()
    def _denoise(
        self, batch: Any, mask: dict[str, torch.Tensor], record: bool = False
    ) -> tuple[Any, Any, None]:
        if record:
            raise ValueError("safe-selection sampler writes explicit paired outputs")
        for key in self._predictors:
            mask.setdefault(key, None)
        for key in self._correctors:
            mask.setdefault(key, None)
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1), device=self._device)
        mean_batch = batch.clone()
        for index in range(250):
            batch, mean_batch = self._advance(
                batch,
                mask=mask,
                timesteps=timesteps,
                dt=dt,
                start=index,
                stop=index + 1,
                scale=2.0,
                track_base_temporal=True,
            )
        t = torch.full((batch.get_batch_size(),), timesteps[250], device=self._device)
        guided, instantaneous = self._score_at_scale(batch, t, 2.0, return_features=True)
        state_digest = _tensor_digest(batch)
        feature_row = self._feature_row(
            sampling_step=250,
            t=t,
            features=instantaneous,
            x0=self._clean_atoms(x=batch, score=guided, t=t),
        )
        selected_action, prediction_trace = self._controller_decision(feature_row)
        policies = {
            "C0": "KEEP",
            "T0": str(self._controller["t0_action"]),
            "S0": selected_action,
        }
        snapshot = self._rng_snapshot()
        s0_mean = None
        for policy, action in policies.items():
            self._restore_rng(snapshot)
            _, branch_mean = self._advance(
                batch.clone(),
                mask=mask,
                timesteps=timesteps,
                dt=dt,
                start=250,
                stop=self.N,
                scale=SCALE_BY_ACTION[action],
            )
            metadata = {
                "seed": self._risk_seed,
                "policy": policy,
                "selected_action": action,
                "selected_cfg": SCALE_BY_ACTION[action],
                "sampling_step": 250,
                "decision_t_norm": float(t[0].detach().cpu().item()),
                "state_sha256": state_digest,
            }
            self.policy_structures.append((metadata, self._final_atoms(branch_mean)))
            if policy == "S0":
                s0_mean = branch_mean
        self._restore_rng(snapshot)
        self.controller_trace.append(
            {
                "seed": self._risk_seed,
                "sampling_step": 250,
                "state_sha256": state_digest,
                "selected_action": selected_action,
                "selected_cfg": SCALE_BY_ACTION[selected_action],
                "t0_action": self._controller["t0_action"],
                **prediction_trace,
                **feature_row,
            }
        )
        if s0_mean is None:
            raise RuntimeError("S0 branch was not produced")
        return s0_mean, s0_mean, None
