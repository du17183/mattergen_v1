# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any, Generic, Mapping, Tuple, TypeVar

import torch
from tqdm.auto import tqdm

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.diffusion.sampling.pc_partials import CorrectorPartial, PredictorPartial
from mattergen.diffusion.sampling.residual_distillation import (
    ResidualDistillationController,
    build_residual_controller,
)

Diffusable = TypeVar(
    "Diffusable", bound=BatchedData
)  # Don't use 'T' because it clashes with the 'T' for time
SampleAndMean = Tuple[Diffusable, Diffusable]
SampleAndMeanAndMaybeRecords = Tuple[Diffusable, Diffusable, list[Diffusable] | None]
SampleAndMeanAndRecords = Tuple[Diffusable, Diffusable, list[Diffusable]]


class PredictorCorrector(Generic[Diffusable]):
    """Generates samples using predictor-corrector sampling."""

    def __init__(
        self,
        *,
        diffusion_module: DiffusionModule,
        predictor_partials: dict[str, PredictorPartial] | None = None,
        corrector_partials: dict[str, CorrectorPartial] | None = None,
        device: torch.device,
        n_steps_corrector: int,
        N: int,
        eps_t: float = 1e-3,
        max_t: float | None = None,
        corrector_residual_adapter: Mapping[str, Any] | None = None,
    ):
        """
        Args:
            diffusion_module: diffusion module
            predictor_partials: partials for constructing predictors. Keys are the names of the corruptions.
            corrector_partials: partials for constructing correctors. Keys are the names of the corruptions.
            device: device to run on
            n_steps_corrector: number of corrector steps
            N: number of noise levels
            eps_t: diffusion time to stop denoising at
            max_t: diffusion time to start denoising at. If None, defaults to the maximum diffusion time. You may want to start at T-0.01, say, for numerical stability.
        """
        self._diffusion_module = diffusion_module
        self.N = N

        if max_t is None:
            max_t = self._multi_corruption.T
        assert max_t <= self._multi_corruption.T, "Denoising cannot start from beyond T"

        self._max_t = max_t
        assert (
            corrector_partials or predictor_partials
        ), "Must specify at least one predictor or corrector"
        corrector_partials = corrector_partials or {}
        predictor_partials = predictor_partials or {}
        if self._multi_corruption.discrete_corruptions:
            # These all have property 'N' because they are D3PM type
            assert set(c.N for c in self._multi_corruption.discrete_corruptions.values()) == {N}  # type: ignore

        self._predictors = {
            k: v(corruption=self._multi_corruption.corruptions[k], score_fn=None)
            for k, v in predictor_partials.items()
        }

        self._correctors = {
            k: v(
                corruption=self._multi_corruption.corruptions[k],
                n_steps=n_steps_corrector,
                score_fn=None,
            )
            for k, v in corrector_partials.items()
        }
        self._eps_t = eps_t
        self._n_steps_corrector = n_steps_corrector
        self._device = device
        self._sampling_context: dict[str, Any] = {}
        self._score_call_index = 0
        self._exact_score_call_count = 0
        self._residual_controller: ResidualDistillationController | None = (
            build_residual_controller(corrector_residual_adapter, device=device)
        )

    @property
    def diffusion_module(self) -> DiffusionModule:
        return self._diffusion_module

    @property
    def _multi_corruption(self) -> MultiCorruption:
        return self._diffusion_module.corruption

    def _score_fn(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        return self._diffusion_module.score_fn(x, t)

    def _evaluate_exact_score(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        """Evaluate the expensive score model and count logical GemNet calls."""

        self._exact_score_call_count += 1
        return self._score_fn(x, t)

    @property
    def sampling_context(self) -> Mapping[str, Any]:
        """Context for the score call currently being evaluated."""

        return self._sampling_context

    @property
    def sampling_metrics(self) -> Mapping[str, float | int]:
        """Counters for the most recent sampled batch."""

        metrics: dict[str, float | int] = {
            "mattergen_score_calls": self._exact_score_call_count,
            "theoretical_baseline_score_calls": self.N * (self._n_steps_corrector + 1),
        }
        if self._residual_controller is None:
            metrics.update(
                {
                    "score_opportunities": self.N if self._correctors else 0,
                    "adapter_calls": 0,
                    "adapter_accept_calls": 0,
                    "adapter_acceptance_overall": 0.0,
                    "adapter_acceptance_eligible": 0.0,
                    "fallback_calls": 0,
                    "saved_score_calls": 0,
                    "adapter_coverage": 0.0,
                    "second_forward_avoidance": 0.0,
                    "adapter_seconds": 0.0,
                    "exact_fallback_seconds": 0.0,
                }
            )
        else:
            metrics.update(self._residual_controller.metrics)
        # This also covers explicit Corrector-removal baselines, which save
        # exact score calls without going through the residual controller.
        metrics["saved_score_calls"] = max(
            int(metrics["theoretical_baseline_score_calls"])
            - self._exact_score_call_count,
            0,
        )
        metrics["forward_reduction"] = metrics["saved_score_calls"] / max(
            int(metrics["theoretical_baseline_score_calls"]), 1
        )
        return metrics

    def _on_sampling_start(self) -> None:
        """Reset batch-local sampling context before drawing a new prior."""

        self._sampling_context = {}
        self._score_call_index = 0
        self._exact_score_call_count = 0
        if self._residual_controller is not None:
            self._residual_controller.reset()

    def _on_before_sample_prior(self, conditioning_data: Diffusable) -> None:
        """Hook for read-only audit tooling immediately before prior sampling."""

    def _on_after_sample_prior(self, batch: Diffusable) -> None:
        """Hook for read-only audit tooling immediately after prior sampling."""

    def _on_sampling_end(self, error: BaseException | None) -> None:
        """Hook for run-local cleanup and trace persistence."""

    def _set_sampling_context(self, *, sampling_step: int, phase: str) -> None:
        self._sampling_context = {
            "sampling_step": sampling_step,
            "num_steps": self.N,
            "progress": sampling_step / max(self.N - 1, 1),
            "phase": phase,
            "score_call_index": self._score_call_index,
        }
        self._score_call_index += 1

    @classmethod
    def from_pl_module(cls, pl_module: DiffusionLightningModule, **kwargs) -> PredictorCorrector:
        return cls(diffusion_module=pl_module.diffusion_module, device=pl_module.device, **kwargs)

    @torch.no_grad()
    def sample(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ) -> SampleAndMean:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch). The difference between these is that `mean_batch` has no noise added at the final denoising step.

        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=False)[:2]

    @torch.no_grad()
    def sample_with_record(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ) -> SampleAndMeanAndRecords:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch). The difference between these is that `mean_batch` has no noise added at the final denoising step.

        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=True)

    @torch.no_grad()
    def _sample_maybe_record(
        self,
        conditioning_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch, recorded_samples, recorded_predictions).
           The difference between the former two is that `mean_batch` has no noise added at the final denoising step.
           The latter two are only returned if `record` is True, and contain the samples and predictions from each step of the diffusion process.

        """
        self._on_sampling_start()
        error: BaseException | None = None
        try:
            if isinstance(self._diffusion_module, torch.nn.Module):
                self._diffusion_module.eval()
            mask = mask or {}
            conditioning_data = conditioning_data.to(self._device)
            mask = {k: v.to(self._device) for k, v in mask.items()}
            self._on_before_sample_prior(conditioning_data)
            batch = _sample_prior(self._multi_corruption, conditioning_data, mask=mask)
            self._on_after_sample_prior(batch)
            return self._denoise(batch=batch, mask=mask, record=record)
        except BaseException as caught:
            error = caught
            raise
        finally:
            self._on_sampling_end(error)
            if self._residual_controller is not None:
                self._residual_controller.close(error=error)

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from a prior sample to a t=eps_t sample."""
        recorded_samples = None
        if record:
            recorded_samples = []
        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)

            # Corrector updates.
            x_before_last_corrector = None
            score_before_last_corrector = None
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    self._set_sampling_context(sampling_step=i, phase="corrector")
                    x_before_last_corrector = batch
                    score = self._evaluate_exact_score(batch, t)
                    score_before_last_corrector = score
                    fns = {
                        k: corrector.step_given_score for k, corrector in self._correctors.items()
                    }
                    samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                        fns=fns,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                    )

            # Predictor updates
            self._set_sampling_context(sampling_step=i, phase="predictor")
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
                    sampling_step=i,
                    progress=i / max(self.N - 1, 1),
                )
            else:
                score = self._evaluate_exact_score(batch, t)
            predictor_fns = {
                k: predictor.update_given_score for k, predictor in self._predictors.items()
            }
            samples_means = apply(
                fns=predictor_fns,
                x=batch,
                score=score,
                broadcast=dict(t=t, batch=batch, dt=dt),
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )
            if record:
                recorded_samples.append(batch.clone().to("cpu"))
            batch, mean_batch = _mask_replace(
                samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
            )

        return batch, mean_batch, recorded_samples


def _mask_replace(
    samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]],
    batch: BatchedData,
    mean_batch: BatchedData,
    mask: dict[str, torch.Tensor | None],
) -> SampleAndMean:
    # Apply masks
    samples_means = apply(
        fns={k: _mask_both for k in samples_means},
        broadcast={},
        sample_and_mean=samples_means,
        mask=mask,
        old_x=batch,
    )

    # Put the updated values in `batch` and `mean_batch`
    batch = batch.replace(**{k: v[0] for k, v in samples_means.items()})
    mean_batch = mean_batch.replace(**{k: v[1] for k, v in samples_means.items()})
    return batch, mean_batch


def _mask_both(
    *, sample_and_mean: Tuple[torch.Tensor, torch.Tensor], old_x: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return tuple(_mask(old_x=old_x, new_x=x, mask=mask) for x in sample_and_mean)  # type: ignore


def _mask(*, old_x: torch.Tensor, new_x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Replace new_x with old_x where mask is 1."""
    if mask is None:
        return new_x
    else:
        return new_x.lerp(old_x, mask)


def _sample_prior(
    multi_corruption: MultiCorruption,
    conditioning_data: BatchedData,
    mask: Mapping[str, torch.Tensor] | None,
) -> BatchedData:
    samples = {
        k: multi_corruption.corruptions[k]
        .prior_sampling(
            shape=conditioning_data[k].shape,
            conditioning_data=conditioning_data,
            batch_idx=conditioning_data.get_batch_idx(field_name=k),
        )
        .to(conditioning_data[k].device)
        for k in multi_corruption.corruptions
    }
    mask = mask or {}
    for k, msk in mask.items():
        if k in multi_corruption.corrupted_fields:
            samples[k].lerp_(conditioning_data[k], msk)
    return conditioning_data.replace(**samples)
