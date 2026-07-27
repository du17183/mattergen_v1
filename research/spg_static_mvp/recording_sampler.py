from __future__ import annotations

from typing import Any

import torch

from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector


class ShapeRecordingGuidedPredictorCorrector(GuidedPredictorCorrector):
    """Official C0 sampler with read-only pre-score state recording."""

    def __init__(self, *, trajectory_seed: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.trajectory_seed = int(trajectory_seed)
        self.recorded_states: list[dict[str, Any]] = []

    def _on_sampling_start(self) -> None:
        torch.manual_seed(self.trajectory_seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed(self.trajectory_seed)
            torch.cuda.manual_seed_all(self.trajectory_seed)
        self.recorded_states = []
        super()._on_sampling_start()

    def _score_fn(self, x, t):
        context = dict(self.sampling_context)
        self.recorded_states.append(
            {
                "seed": self.trajectory_seed,
                "state_index": len(self.recorded_states),
                "sampling_step": int(context["sampling_step"]),
                "phase": str(context["phase"]),
                "progress": float(context["progress"]),
                "t": float(t.detach().reshape(-1)[0].cpu()),
                "pos": x["pos"].detach().cpu().clone(),
                "cell": x["cell"].detach().cpu().clone(),
                "atomic_numbers": x["atomic_numbers"].detach().cpu().clone(),
                "num_atoms": x["num_atoms"].detach().cpu().clone(),
            }
        )
        return super()._score_fn(x, t)


__all__ = ["ShapeRecordingGuidedPredictorCorrector"]
