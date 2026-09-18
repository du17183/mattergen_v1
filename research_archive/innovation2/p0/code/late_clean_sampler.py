"""Read-only complete clean-x0 capture at frozen late timesteps."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from ase import Atoms
from ase.io import write

from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector


REQUESTED_T = (0.10, 0.05, 0.02, 0.01)


class LateCleanEstimateSampler(GuidedPredictorCorrector):
    """Official sampler plus a read-only hook saving complete predicted x0."""

    def __init__(self, *, output_dir: str, diagnostic_seed: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._output_dir = Path(output_dir)
        if not self._output_dir.is_absolute():
            raise ValueError("output_dir must be absolute")
        self._seed = int(diagnostic_seed)
        grid = torch.linspace(self._max_t, self._eps_t, self.N)
        self._step_to_requested = {
            int(torch.argmin(torch.abs(grid - requested)).item()): requested
            for requested in REQUESTED_T
        }
        if len(self._step_to_requested) != len(REQUESTED_T):
            raise RuntimeError("requested diagnostic times collapsed to duplicate steps")
        self._records: list[dict[str, object]] = []
        self._atoms: dict[float, Atoms] = {}

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        if self._output_dir.exists():
            raise FileExistsError(self._output_dir)
        self._records = []
        self._atoms = {}

    def _on_sampling_end(self, error: BaseException | None) -> None:
        super()._on_sampling_end(error)
        if error is not None:
            return
        if set(self._atoms) != set(REQUESTED_T):
            raise RuntimeError(f"missing requested clean estimates: {set(REQUESTED_T) - set(self._atoms)}")
        self._output_dir.mkdir(parents=True, exist_ok=False)
        for requested in REQUESTED_T:
            write(self._output_dir / f"x0_hat_t_{requested:.2f}.extxyz", self._atoms[requested])
        with (self._output_dir / "clean_estimate_manifest.csv").open(
            "x", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=tuple(self._records[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(self._records)

    @property
    def sampling_metrics(self) -> Mapping[str, float | int]:
        values = dict(super().sampling_metrics)
        values["complete_clean_estimate_count"] = len(self._records)
        return values

    def _evaluate_exact_score(self, x, t):
        score = super()._evaluate_exact_score(x, t)
        context = self.sampling_context
        step = int(context.get("sampling_step", -1))
        if context.get("phase") == "predictor" and step in self._step_to_requested:
            self._capture(x=x, score=score, t=t, step=step)
        return score

    def _capture(self, *, x, score, t: torch.Tensor, step: int) -> None:
        if x.get_batch_size() != 1:
            raise RuntimeError("diagnostic requires batch_size=1")
        requested = self._step_to_requested[step]
        atomic_corruption = self._multi_corruption.discrete_corruptions["atomic_numbers"]
        logits = score["atomic_numbers"].float()
        probability = torch.softmax(logits, dim=-1)
        clean_atomic = torch.argmax(probability, dim=-1) + int(atomic_corruption.offset)

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

        numbers = clean_atomic.detach().cpu().numpy().astype(int)
        frac = clean_pos.detach().cpu().double().numpy()
        cell = clean_cell[0].detach().cpu().double().numpy()
        if not (np.isfinite(frac).all() and np.isfinite(cell).all()):
            raise FloatingPointError(f"non-finite clean estimate seed={self._seed} t={requested}")
        atoms = Atoms(numbers=numbers, scaled_positions=frac, cell=cell, pbc=True)
        self._atoms[requested] = atoms
        current_atomic = x["atomic_numbers"].detach().cpu().numpy()
        self._records.append({
            "seed": self._seed,
            "requested_t_norm": requested,
            "actual_model_t": float(t[0].detach().cpu().item()),
            "sampling_step": step,
            "num_atoms": len(numbers),
            "atomic_logits_shape": "x".join(str(dim) for dim in logits.shape),
            "atomic_logits_num_classes": int(logits.shape[-1]),
            "atomic_corruption_offset": int(atomic_corruption.offset),
            "clean_atomic_number_min": int(numbers.min()),
            "clean_atomic_number_max": int(numbers.max()),
            "clean_atomic_mean_confidence": float(probability.max(dim=-1).values.mean().item()),
            "clean_atomic_current_state_agreement": float(np.mean(numbers == current_atomic)),
            "clean_cell_signed_volume": float(np.linalg.det(cell)),
            "clean_estimate_finite": True,
        })
