"""Read-only per-site clean-position diagnostic for the official sampler."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector


class SitePositionDiagnosticSampler(GuidedPredictorCorrector):
    """Record clean position and exact periodic displacement for every atom."""

    def __init__(self, *, site_trace_path: str, diagnostic_seed: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._trace_path = Path(site_trace_path)
        if not self._trace_path.is_absolute():
            raise ValueError("site trace path must be absolute")
        self._diagnostic_seed = int(diagnostic_seed)
        self._subset = "calibration" if self._diagnostic_seed <= 99031 else "validation"
        self._observation_indices = tuple(range(self.N))
        self._steps: list[int] = []
        self._model_t: list[float] = []
        self._clean_positions: list[np.ndarray] = []
        self._clean_cells: list[np.ndarray] = []

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        if self._trace_path.exists():
            raise FileExistsError(self._trace_path)
        self._steps = []
        self._model_t = []
        self._clean_positions = []
        self._clean_cells = []

    def _on_sampling_end(self, error: BaseException | None) -> None:
        super()._on_sampling_end(error)
        if error is not None:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        if len(self._steps) != self.N:
            raise RuntimeError(f"expected {self.N} site observations, got {len(self._steps)}")
        with self._trace_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                seed=np.asarray(self._diagnostic_seed, dtype=np.int64),
                subset=np.asarray(self._subset),
                sampling_step=np.asarray(self._steps, dtype=np.int32),
                model_t=np.asarray(self._model_t, dtype=np.float32),
                clean_frac_position=np.stack(self._clean_positions).astype(np.float32),
                clean_cell=np.stack(self._clean_cells).astype(np.float32),
            )

    @property
    def sampling_metrics(self) -> Mapping[str, float | int]:
        values = dict(super().sampling_metrics)
        values["site_diagnostic_observation_count"] = len(self._observation_indices)
        values["site_diagnostic_row_count"] = (
            len(self._steps) * len(self._clean_positions[0]) if self._clean_positions else 0
        )
        return values

    def _evaluate_exact_score(self, x, t):
        score = super()._evaluate_exact_score(x, t)
        context = self.sampling_context
        step = int(context.get("sampling_step", -1))
        if context.get("phase") == "predictor" and step in self._observation_indices:
            self._observe(x=x, score=score, t=t, step=step)
        return score

    def _observe(self, *, x, score, t: torch.Tensor, step: int) -> None:
        if x.get_batch_size() != 1:
            raise RuntimeError("site diagnostic requires batch_size=1")
        pos_sde = self._multi_corruption.sdes["pos"]
        _, pos_std = pos_sde.marginal_prob(
            x=x["pos"], t=t, batch_idx=x.get_batch_idx("pos"), batch=x
        )
        clean_position = pos_sde.wrap(
            x["pos"] + pos_std.square() * score["pos"]
        ).detach().cpu().double().numpy()

        cell_sde = self._multi_corruption.sdes["cell"]
        alpha, cell_std = cell_sde.mean_coeff_and_std(x["cell"], t=t, batch=x)
        limit_mean = cell_sde.get_limit_mean(x=x["cell"], batch=x)
        clean_cell = (
            (x["cell"] + cell_std.square() * score["cell"]
             - (1.0 - alpha) * limit_mean) / alpha
        )[0].detach().cpu().double().numpy()

        model_t = float(t[0].detach().cpu().item())
        if not np.isfinite(clean_position).all() or not np.isfinite(clean_cell).all():
            raise FloatingPointError(f"non-finite clean estimate at seed={self._diagnostic_seed}, step={step}")
        self._steps.append(step)
        self._model_t.append(model_t)
        self._clean_positions.append(clean_position.copy())
        self._clean_cells.append(clean_cell.copy())
