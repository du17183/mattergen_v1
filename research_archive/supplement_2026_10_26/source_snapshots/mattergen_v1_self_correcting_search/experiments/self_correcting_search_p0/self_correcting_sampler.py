"""Single-proposal probabilistic self-correction without gradients or steering."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector
from mattergen.diffusion.sampling.pc_sampler import (
    Diffusable,
    SampleAndMeanAndMaybeRecords,
    _mask_replace,
)


TRACE_FIELDS = (
    "seed", "timestep", "current_pred", "proposal_pred", "current_distance",
    "proposal_distance", "score_delta", "accept_probability", "accepted",
)


def acceptance_probability(score_delta: float, temperature: float = 0.01) -> float:
    """Public Metropolis form for a point-target energy difference."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return 1.0 if score_delta <= 0 else math.exp(-score_delta / temperature)


class LinearProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(513, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TinyMLPProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(513, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SelfCorrectingGuidedPredictorCorrector(GuidedPredictorCorrector):
    """Verify each ordinary corrector proposal in a frozen validation-selected window."""

    def __init__(
        self,
        *,
        probe_checkpoint_path: str,
        search_trace_path: str,
        search_seed: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        checkpoint = torch.load(probe_checkpoint_path, map_location="cpu", weights_only=False)
        if not checkpoint.get("probe_go", False):
            raise RuntimeError("self-correcting search is gated off by failed probe")
        if checkpoint.get("method") != "Hidden-State Probe P0 (no gradients, no steering)":
            raise RuntimeError("probe checkpoint method mismatch")
        architecture = checkpoint["architecture"]
        if architecture not in ("linear", "tiny_mlp"):
            raise RuntimeError(f"unknown probe architecture: {architecture}")
        self._probe = LinearProbe() if architecture == "linear" else TinyMLPProbe()
        self._probe.load_state_dict(checkpoint["state_dict"], strict=True)
        self._probe.eval()
        for parameter in self._probe.parameters():
            parameter.requires_grad_(False)
        self._probe_block = int(checkpoint["block"])
        self._feature_mean = checkpoint["feature_mean"].float()
        self._feature_std = checkpoint["feature_std"].float()
        self._reliable_max_t = float(checkpoint["reliable_max_t"])
        self._target = float(checkpoint["target"])
        self._temperature = float(checkpoint["acceptance_temperature"])
        if self._temperature != 0.01 or self._target != 0.2:
            raise RuntimeError("frozen search target/temperature changed")
        self._search_seed = int(search_seed)
        self._search_trace_path = Path(search_trace_path)
        if not self._search_trace_path.is_absolute():
            raise ValueError("search trace path must be absolute")
        self._search_rows: list[dict[str, Any]] = []
        self._accept_rng = np.random.default_rng(self._search_seed ^ 0x5C5C5C5C)
        self._capture_kind: str | None = None
        self._captured_hidden: torch.Tensor | None = None
        self._extra_probe_forward_calls = 0
        self._probe_seconds = 0.0
        gemnet = self.diffusion_module.model.gemnet
        if len(gemnet.int_blocks) != 4:
            raise RuntimeError("expected four frozen GemNet-T interaction blocks")
        self._hook_handle = gemnet.int_blocks[self._probe_block].register_forward_hook(
            self._hidden_hook
        )

    def _hidden_hook(self, _module, _inputs, output) -> None:
        if self._capture_kind is None:
            return
        if self._captured_hidden is not None:
            raise RuntimeError("multiple selected-block activations in one probe model call")
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 2 or hidden.shape[1] != 512:
            raise RuntimeError(f"unexpected selected-block hidden shape: {tuple(hidden.shape)}")
        if self._capture_kind == "current_cfg_joint":
            if hidden.shape[0] % 2:
                raise RuntimeError("CFG joint hidden cannot be split")
            hidden = hidden[hidden.shape[0] // 2 :]
        elif self._capture_kind != "proposal_conditional":
            raise RuntimeError(f"unknown capture kind: {self._capture_kind}")
        self._captured_hidden = hidden.float().mean(dim=0).detach()

    def _predict_hidden(self, hidden: torch.Tensor, t_value: float) -> float:
        device = hidden.device
        self._probe.to(device)
        mean = self._feature_mean.to(device)
        std = self._feature_std.to(device)
        features = torch.cat([(hidden - mean) / std, hidden.new_tensor([t_value])])
        with torch.no_grad():
            value = self._probe(features.unsqueeze(0))[0]
        return float(value.cpu().item())

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        if self._search_trace_path.exists():
            raise FileExistsError(f"refusing to overwrite search trace: {self._search_trace_path}")
        self._search_rows = []
        self._accept_rng = np.random.default_rng(self._search_seed ^ 0x5C5C5C5C)
        self._extra_probe_forward_calls = 0
        self._probe_seconds = 0.0

    def _on_sampling_end(self, error: BaseException | None) -> None:
        try:
            super()._on_sampling_end(error)
            if error is None:
                self._search_trace_path.parent.mkdir(parents=True, exist_ok=True)
                with self._search_trace_path.open("x", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(self._search_rows)
        finally:
            self._hook_handle.remove()

    @property
    def sampling_metrics(self) -> Mapping[str, float | int]:
        values = dict(super().sampling_metrics)
        values.update({
            "probe_scored_proposals": len(self._search_rows),
            "probe_extra_conditional_forward_calls": self._extra_probe_forward_calls,
            "total_logical_model_forward_calls": int(values["mattergen_score_calls"])
            + self._extra_probe_forward_calls,
            "accepted_proposals": sum(bool(row["accepted"]) for row in self._search_rows),
            "acceptance_rate": (
                sum(bool(row["accepted"]) for row in self._search_rows) / len(self._search_rows)
                if self._search_rows else 0.0
            ),
            "probe_verification_seconds": self._probe_seconds,
        })
        return values

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        recorded_samples = [] if record else None
        for key in self._predictors:
            mask.setdefault(key, None)
        for key in self._correctors:
            mask.setdefault(key, None)
        mean_batch = batch.clone()
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        from tqdm.auto import tqdm
        for index in tqdm(range(self.N), miniters=50, mininterval=5):
            t = torch.full((batch.get_batch_size(),), timesteps[index], device=self._device)
            t_value = float(t[0].cpu().item())
            score_this_step = t_value <= self._reliable_max_t + 1e-8
            x_before_last_corrector = None
            score_before_last_corrector = None
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    self._set_sampling_context(sampling_step=index, phase="corrector")
                    x_before_last_corrector = batch
                    self._capture_kind = "current_cfg_joint" if score_this_step else None
                    self._captured_hidden = None
                    score = self._evaluate_exact_score(batch, t)
                    current_hidden = self._captured_hidden
                    self._capture_kind = None
                    score_before_last_corrector = score
                    functions = {
                        key: corrector.step_given_score for key, corrector in self._correctors.items()
                    }
                    samples_means = apply(
                        fns=functions,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        assert recorded_samples is not None
                        recorded_samples.append(batch.clone().to("cpu"))
                    candidate_batch, candidate_mean = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                    )
                    if score_this_step:
                        probe_started = time.perf_counter()
                        if current_hidden is None:
                            raise RuntimeError("current hidden was not captured")
                        self._capture_kind = "proposal_conditional"
                        self._captured_hidden = None
                        _ = self._diffusion_module.score_fn(
                            self._keep_conditioning_fn(candidate_batch), t
                        )
                        proposal_hidden = self._captured_hidden
                        self._capture_kind = None
                        self._extra_probe_forward_calls += 1
                        if proposal_hidden is None:
                            raise RuntimeError("proposal hidden was not captured")
                        current_pred = self._predict_hidden(current_hidden, t_value)
                        proposal_pred = self._predict_hidden(proposal_hidden, t_value)
                        current_distance = abs(current_pred - self._target)
                        proposal_distance = abs(proposal_pred - self._target)
                        delta = proposal_distance - current_distance
                        probability = acceptance_probability(delta, self._temperature)
                        accepted = bool(self._accept_rng.random() < probability)
                        self._search_rows.append({
                            "seed": self._search_seed,
                            "timestep": t_value,
                            "current_pred": current_pred,
                            "proposal_pred": proposal_pred,
                            "current_distance": current_distance,
                            "proposal_distance": proposal_distance,
                            "score_delta": delta,
                            "accept_probability": probability,
                            "accepted": accepted,
                        })
                        self._probe_seconds += time.perf_counter() - probe_started
                        if accepted:
                            batch, mean_batch = candidate_batch, candidate_mean
                    else:
                        batch, mean_batch = candidate_batch, candidate_mean

            self._set_sampling_context(sampling_step=index, phase="predictor")
            if (
                self._residual_controller is not None
                and x_before_last_corrector is not None
                and score_before_last_corrector is not None
            ):
                score = self._residual_controller.score_after_corrector(
                    exact_score_fn=self._evaluate_exact_score,
                    x_before=x_before_last_corrector,
                    score_before=score_before_last_corrector,
                    x_after=batch,
                    t=t,
                    sampling_step=index,
                    progress=index / max(self.N - 1, 1),
                )
            else:
                score = self._evaluate_exact_score(batch, t)
            predictor_functions = {
                key: predictor.update_given_score for key, predictor in self._predictors.items()
            }
            samples_means = apply(
                fns=predictor_functions,
                x=batch,
                score=score,
                broadcast=dict(t=t, batch=batch, dt=dt),
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )
            if record:
                assert recorded_samples is not None
                recorded_samples.append(batch.clone().to("cpu"))
            batch, mean_batch = _mask_replace(
                samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
            )
        return batch, mean_batch, recorded_samples
