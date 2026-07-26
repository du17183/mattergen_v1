"""Lossless multi-trajectory A0 sampling with isolated per-seed RNG/state.

Only the expensive score-model evaluation is batched.  Prior draws, corrector
noise, predictor noise, and adaptive-guidance EMA state remain independent for
every trajectory.  This makes batch-size comparisons meaningful without
changing the A0 predictor/corrector algorithm.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Iterator, Mapping, Sequence

import torch
from tqdm.auto import tqdm

from mattergen.common.data.collate import collate
from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.sampling.classifier_free_guidance import (
    GuidedPredictorCorrector,
    score_residual_rms,
)
from mattergen.diffusion.sampling.pc_sampler import (
    PredictorCorrector,
    _mask_replace,
    _sample_prior,
)


def _clone_controller(controller):
    cloned = deepcopy(controller)
    cloned.reset()
    return cloned


class IndependentTrajectoryGuidedPredictorCorrector(GuidedPredictorCorrector):
    """Guided sampler whose stochastic and adaptive states are per trajectory."""

    def __init__(self, *, trajectory_seeds: Sequence[int], **kwargs) -> None:
        seeds = tuple(int(seed) for seed in trajectory_seeds)
        if not seeds:
            raise ValueError("trajectory_seeds must not be empty")
        if len(set(seeds)) != len(seeds):
            raise ValueError("trajectory_seeds must be unique within a batch")
        if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
            raise ValueError("trajectory seeds must fit the NumPy-compatible uint32 range")
        super().__init__(**kwargs)
        self.trajectory_seeds = seeds
        self._trajectory_controllers = [
            _clone_controller(self._guidance_controller) for _ in self.trajectory_seeds
        ]
        self._cpu_rng_states: list[torch.Tensor] = []
        self._device_rng_states: list[torch.Tensor | None] = []
        self.physical_model_forward_count = 0
        self.model_graphs_evaluated = 0

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        for controller in self._trajectory_controllers:
            controller.reset()
        self._cpu_rng_states = []
        self._device_rng_states = []
        for seed in self.trajectory_seeds:
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(seed)
            self._cpu_rng_states.append(cpu_generator.get_state())
            if self._device.type == "cuda":
                device_generator = torch.Generator(device=self._device)
                device_generator.manual_seed(seed)
                self._device_rng_states.append(device_generator.get_state())
            else:
                self._device_rng_states.append(None)
        self.physical_model_forward_count = 0
        self.model_graphs_evaluated = 0

    @contextmanager
    def _trajectory_rng(self, index: int) -> Iterator[None]:
        devices: list[int] = []
        device_index: int | None = None
        if self._device.type == "cuda":
            device_index = (
                self._device.index
                if self._device.index is not None
                else torch.cuda.current_device()
            )
            devices = [device_index]
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.set_rng_state(self._cpu_rng_states[index])
            if device_index is not None:
                device_state = self._device_rng_states[index]
                assert device_state is not None
                torch.cuda.set_rng_state(device_state, device=device_index)
            yield
            self._cpu_rng_states[index] = torch.get_rng_state()
            if device_index is not None:
                self._device_rng_states[index] = torch.cuda.get_rng_state(device=device_index)

    def _model_score(self, x: BatchedData, t: torch.Tensor) -> BatchedData:
        self.physical_model_forward_count += 1
        self.model_graphs_evaluated += x.get_batch_size()
        return PredictorCorrector._score_fn(self, x=x, t=t)

    def _score_fn(self, x: BatchedData, t: torch.Tensor) -> BatchedData:
        batch_size = x.get_batch_size()
        if batch_size != len(self.trajectory_seeds):
            raise ValueError(
                f"batch has {batch_size} trajectories but {len(self.trajectory_seeds)} seeds"
            )

        unconditional = self._remove_conditioning_fn(x)
        conditional = self._keep_conditioning_fn(x)
        if self.guidance_schedule == "constant" and abs(self._guidance_scale - 1) < 1e-15:
            return self._model_score(conditional, t)
        if self.guidance_schedule == "constant" and abs(self._guidance_scale) < 1e-15:
            return self._model_score(unconditional, t)

        unconditional_items = unconditional.to_data_list()
        conditional_items = conditional.to_data_list()
        joint = collate([*unconditional_items, *conditional_items])
        combined = self._model_score(joint, torch.cat([t, t], dim=0))
        combined_items = combined.to_data_list()
        if len(combined_items) != 2 * batch_size:
            raise RuntimeError("joint CFG score did not preserve the 2B graph partition")

        progress, phase = self._current_progress_and_phase()
        fields = tuple(self._multi_corruption.corrupted_fields)
        guided_items = []
        for index in range(batch_size):
            uncond_score = combined_items[index]
            cond_score = combined_items[index + batch_size]
            if self.guidance_schedule == "constant":
                guidance = self._guidance_scale
            else:
                field_deltas, residual_error = score_residual_rms(
                    unconditional_score=uncond_score,
                    conditional_score=cond_score,
                    fields=fields,
                )
                decision = self._trajectory_controllers[index].evaluate(
                    progress=progress,
                    phase=phase,
                    field_deltas=field_deltas,
                    residual_error=residual_error,
                )
                guidance = decision.final_guidance
            guided_items.append(
                uncond_score.replace(
                    **{
                        field: torch.lerp(
                            uncond_score[field],
                            cond_score[field],
                            guidance,
                        )
                        for field in fields
                    }
                )
            )
        return collate(guided_items)

    @torch.no_grad()
    def _sample_maybe_record(
        self,
        conditioning_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
        record: bool = False,
    ):
        self._on_sampling_start()
        error: BaseException | None = None
        try:
            if mask:
                raise NotImplementedError(
                    "Independent batched sampling currently supports generation only, not inpainting"
                )
            if isinstance(self._diffusion_module, torch.nn.Module):
                self._diffusion_module.eval()
            conditioning_data = conditioning_data.to(self._device)
            if conditioning_data.get_batch_size() != len(self.trajectory_seeds):
                raise ValueError("conditioning batch and trajectory seed counts differ")
            self._on_before_sample_prior(conditioning_data)
            prior_items = []
            for index, item in enumerate(conditioning_data.to_data_list()):
                singleton = collate([item]).to(self._device)
                with self._trajectory_rng(index):
                    sampled = _sample_prior(self._multi_corruption, singleton, mask={})
                prior_items.append(sampled.to_data_list()[0])
            batch = collate(prior_items)
            self._on_after_sample_prior(batch)
            return self._denoise(batch=batch, mask={}, record=record)
        except BaseException as caught:
            error = caught
            raise
        finally:
            self._on_sampling_end(error)

    def _independent_updates(
        self,
        *,
        batch: BatchedData,
        score: BatchedData,
        t: torch.Tensor,
        dt: torch.Tensor,
        predictor: bool,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        batch_items = batch.to_data_list()
        score_items = score.to_data_list()
        samplers = self._predictors if predictor else self._correctors
        per_field: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
            field: [] for field in samplers
        }
        for index, (batch_item, score_item) in enumerate(zip(batch_items, score_items, strict=True)):
            singleton = collate([batch_item]).to(self._device)
            singleton_score = collate([score_item]).to(self._device)
            batch_indices = self._multi_corruption._get_batch_indices(singleton)
            with self._trajectory_rng(index):
                for field, sampler in samplers.items():
                    common = {
                        "x": singleton[field],
                        "score": singleton_score[field],
                        "t": t[index : index + 1],
                        "dt": dt,
                        "batch_idx": batch_indices[field],
                    }
                    if predictor:
                        value = sampler.update_given_score(**common, batch=singleton)
                    else:
                        value = sampler.step_given_score(**common)
                    per_field[field].append(value)
        return {
            field: (
                torch.cat([value[0] for value in values], dim=0),
                torch.cat([value[1] for value in values], dim=0),
            )
            for field, values in per_field.items()
        }

    @torch.no_grad()
    def _denoise(
        self,
        batch: BatchedData,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ):
        recorded_samples = [] if record else None
        for field in self._predictors:
            mask.setdefault(field, None)
        for field in self._correctors:
            mask.setdefault(field, None)
        mean_batch = batch.clone()
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1), device=self._device)

        for step in tqdm(range(self.N), miniters=50, mininterval=5):
            t = torch.full(
                (batch.get_batch_size(),),
                timesteps[step],
                device=self._device,
            )
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    self._set_sampling_context(sampling_step=step, phase="corrector")
                    score = self._score_fn(batch, t)
                    samples_means = self._independent_updates(
                        batch=batch,
                        score=score,
                        t=t,
                        dt=dt,
                        predictor=False,
                    )
                    if record:
                        assert recorded_samples is not None
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means,
                        batch=batch,
                        mean_batch=mean_batch,
                        mask=mask,
                    )

            self._set_sampling_context(sampling_step=step, phase="predictor")
            score = self._score_fn(batch, t)
            samples_means = self._independent_updates(
                batch=batch,
                score=score,
                t=t,
                dt=dt,
                predictor=True,
            )
            if record:
                assert recorded_samples is not None
                recorded_samples.append(batch.clone().to("cpu"))
            batch, mean_batch = _mask_replace(
                samples_means=samples_means,
                batch=batch,
                mean_batch=mean_batch,
                mask=mask,
            )

        return batch, mean_batch, recorded_samples


__all__ = ["IndependentTrajectoryGuidedPredictorCorrector"]
