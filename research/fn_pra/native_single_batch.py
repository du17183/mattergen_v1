"""Instrumented native A0 sampler for the batch_size=1 reference."""

from __future__ import annotations

from typing import Sequence

import torch

from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector


class NativeSingleTrajectoryGuidedPredictorCorrector(GuidedPredictorCorrector):
    """Keep the official single-trajectory path while exposing benchmark counters."""

    def __init__(self, *, trajectory_seeds: Sequence[int], **kwargs) -> None:
        seeds = tuple(int(seed) for seed in trajectory_seeds)
        if len(seeds) != 1:
            raise ValueError("native single-trajectory reference requires exactly one seed")
        super().__init__(**kwargs)
        self.trajectory_seeds = seeds
        self.physical_model_forward_count = 0
        self.model_graphs_evaluated = 0

    def _on_sampling_start(self) -> None:
        seed = self.trajectory_seeds[0]
        torch.manual_seed(seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.physical_model_forward_count = 0
        self.model_graphs_evaluated = 0
        super()._on_sampling_start()

    def _score_fn(self, x, t):
        self.physical_model_forward_count += 1
        if self.guidance_schedule == "constant" and (
            abs(self._guidance_scale) < 1e-15
            or abs(self._guidance_scale - 1) < 1e-15
        ):
            self.model_graphs_evaluated += x.get_batch_size()
        else:
            self.model_graphs_evaluated += 2 * x.get_batch_size()
        return super()._score_fn(x, t)


__all__ = ["NativeSingleTrajectoryGuidedPredictorCorrector"]
