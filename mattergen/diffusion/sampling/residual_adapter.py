# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Lightweight, field-aware residual adapter for Predictor-Corrector sampling.

The module predicts the change in the score caused by one Corrector update.  It
never flattens a crystal into a fixed-size vector.  Position and cell outputs
are scalar combinations of covariant input tensors; the atomic-number output
is a small per-atom low-rank logit head driven by invariant summaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn

from mattergen.diffusion.data.batched_data import BatchedData


CONTEXT_FEATURE_NAMES = (
    "t",
    "progress",
    "log_num_atoms",
    "log_pos_displacement_rms",
    "log_cell_displacement_rms",
    "log_pos_score_rms",
    "log_cell_score_rms",
    "log_atomic_score_rms",
    "log_pos_displacement_max",
    "log_cell_displacement_max",
)
RISK_FEATURE_NAMES = CONTEXT_FEATURE_NAMES + (
    "log_predicted_pos_residual_rms",
    "log_predicted_cell_residual_rms",
    "log_predicted_atomic_residual_rms",
)
MASKED_LOGIT_ABS_THRESHOLD = 1.0e8


@dataclass(frozen=True)
class ResidualAdapterArchitecture:
    """Serializable architecture parameters stored beside every checkpoint."""

    atomic_score_dim: int = 101
    hidden_dim: int = 32
    context_dim: int = 16
    atomic_rank: int = 8

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterPrediction:
    """Adapter output and invariant features used by uncertainty calibration."""

    score: BatchedData
    residuals: Mapping[str, torch.Tensor]
    context_features: torch.Tensor
    risk_features: torch.Tensor


def minimum_image_displacement(after: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
    """Return the wrapped fractional displacement in ``[-0.5, 0.5)``."""

    delta = after - before
    return torch.remainder(delta + 0.5, 1.0) - 0.5


def _batch_index(
    batch: BatchedData, field: str, *, batch_size: int, device: torch.device
) -> torch.LongTensor:
    index = batch.get_batch_idx(field)
    if index is None:
        return torch.arange(batch_size, device=device)
    return index.to(device=device, dtype=torch.long)


def _per_sample_rms(
    values: torch.Tensor,
    batch_index: torch.LongTensor,
    batch_size: int,
    *,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    values_float = values.float()
    squared = values_float.square().reshape(values.shape[0], -1)
    if valid is not None:
        valid_flat = valid.reshape(values.shape[0], -1).to(dtype=squared.dtype)
        squared = squared * valid_flat
        counts_per_row = valid_flat.sum(dim=1)
    else:
        counts_per_row = torch.full(
            (values.shape[0],),
            squared.shape[1],
            dtype=squared.dtype,
            device=squared.device,
        )
    sums = torch.zeros(batch_size, dtype=squared.dtype, device=squared.device)
    counts = torch.zeros_like(sums)
    sums.index_add_(0, batch_index, squared.sum(dim=1))
    counts.index_add_(0, batch_index, counts_per_row)
    return torch.sqrt(sums / counts.clamp_min(1.0))


def _per_sample_max(
    values: torch.Tensor, batch_index: torch.LongTensor, batch_size: int
) -> torch.Tensor:
    per_row = values.float().abs().reshape(values.shape[0], -1).amax(dim=1)
    out = torch.zeros(batch_size, dtype=per_row.dtype, device=per_row.device)
    if hasattr(out, "scatter_reduce_"):
        out.scatter_reduce_(0, batch_index, per_row, reduce="amax", include_self=True)
        return out
    for sample_index in range(batch_size):  # pragma: no cover - old PyTorch fallback
        selected = per_row[batch_index == sample_index]
        if selected.numel():
            out[sample_index] = selected.max()
    return out


def _log_feature(value: torch.Tensor, multiplier: float = 1.0) -> torch.Tensor:
    return torch.log1p(value.float().clamp(min=0.0, max=1.0e12) * multiplier)


class FieldwiseResidualAdapter(nn.Module):
    """A compact residual model with independent position/cell/atomic heads.

    The position prediction is ``a * delta_x + b * score_before``.  The cell
    head has the analogous form.  Consequently these two heads cannot invent
    a preferred Cartesian axis.  The atomic head operates independently per
    atom and emits a low-rank correction in element-logit space.
    """

    def __init__(
        self,
        architecture: ResidualAdapterArchitecture | None = None,
        *,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.architecture = architecture or ResidualAdapterArchitecture()
        input_dim = len(CONTEXT_FEATURE_NAMES)
        hidden_dim = self.architecture.hidden_dim
        context_dim = self.architecture.context_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
            nn.SiLU(),
        )
        self.pos_head = nn.Linear(context_dim, 2)
        self.cell_head = nn.Linear(context_dim, 2)
        self.atomic_encoder = nn.Sequential(
            nn.Linear(context_dim + 2, hidden_dim),
            nn.SiLU(),
        )
        self.atomic_head = nn.Linear(hidden_dim, self.architecture.atomic_rank + 1)
        self.atomic_basis = nn.Parameter(
            torch.empty(self.architecture.atomic_rank, self.architecture.atomic_score_dim)
        )
        nn.init.normal_(self.atomic_basis, mean=0.0, std=0.01)

        # Zero initialization makes the untrained module exactly the Reuse
        # baseline and avoids a random perturbation at the start of training.
        nn.init.zeros_(self.pos_head.weight)
        nn.init.zeros_(self.pos_head.bias)
        nn.init.zeros_(self.cell_head.weight)
        nn.init.zeros_(self.cell_head.bias)
        nn.init.zeros_(self.atomic_head.weight)
        nn.init.zeros_(self.atomic_head.bias)

        mean = torch.zeros(input_dim) if feature_mean is None else feature_mean.float()
        std = torch.ones(input_dim) if feature_std is None else feature_std.float()
        if mean.shape != (input_dim,) or std.shape != (input_dim,):
            raise ValueError(f"feature statistics must both have shape ({input_dim},)")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std.clamp_min(1.0e-6))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def set_feature_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != self.feature_mean.shape or std.shape != self.feature_std.shape:
            raise ValueError("feature statistics have the wrong shape")
        self.feature_mean.copy_(mean.float())
        self.feature_std.copy_(std.float().clamp_min(1.0e-6))

    def _context_features(
        self,
        *,
        x_before: BatchedData,
        x_after: BatchedData,
        score_before: BatchedData,
        t: torch.Tensor,
        progress: float | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = x_after.get_batch_size()
        device = x_after["pos"].device
        pos_index = _batch_index(x_after, "pos", batch_size=batch_size, device=device)
        cell_index = _batch_index(x_after, "cell", batch_size=batch_size, device=device)
        atomic_index = _batch_index(
            x_after, "atomic_numbers", batch_size=batch_size, device=device
        )

        pos_delta = minimum_image_displacement(x_after["pos"], x_before["pos"])
        cell_delta = x_after["cell"] - x_before["cell"]
        atomic_score = score_before["atomic_numbers"]
        if atomic_score.shape[-1] != self.architecture.atomic_score_dim:
            raise ValueError(
                "atomic score dimension mismatch: "
                f"expected {self.architecture.atomic_score_dim}, got {atomic_score.shape[-1]}"
            )
        active_atomic_logits = atomic_score.abs() < MASKED_LOGIT_ABS_THRESHOLD

        try:
            num_atoms = x_after["num_atoms"].reshape(-1).float()
        except (KeyError, TypeError):
            num_atoms = torch.bincount(pos_index, minlength=batch_size).float()
        if num_atoms.numel() != batch_size:
            num_atoms = torch.bincount(pos_index, minlength=batch_size).float()

        pos_displacement_rms = _per_sample_rms(pos_delta, pos_index, batch_size)
        cell_displacement_rms = _per_sample_rms(cell_delta, cell_index, batch_size)
        pos_score_rms = _per_sample_rms(score_before["pos"], pos_index, batch_size)
        cell_score_rms = _per_sample_rms(score_before["cell"], cell_index, batch_size)
        atomic_score_rms = _per_sample_rms(
            atomic_score,
            atomic_index,
            batch_size,
            valid=active_atomic_logits,
        )
        pos_displacement_max = _per_sample_max(pos_delta, pos_index, batch_size)
        cell_displacement_max = _per_sample_max(cell_delta, cell_index, batch_size)
        timestep = t.float().reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(batch_size)
        if timestep.numel() != batch_size:
            raise ValueError("timestep tensor does not match batch size")
        if isinstance(progress, torch.Tensor):
            progress_tensor = progress.to(device=device, dtype=timestep.dtype).reshape(-1)
            if progress_tensor.numel() == 1:
                progress_tensor = progress_tensor.expand(batch_size)
            if progress_tensor.numel() != batch_size:
                raise ValueError("progress tensor does not match batch size")
        else:
            progress_tensor = torch.full_like(timestep, float(progress))

        features = torch.stack(
            (
                timestep,
                progress_tensor,
                _log_feature(num_atoms),
                _log_feature(pos_displacement_rms, 100.0),
                _log_feature(cell_displacement_rms, 10.0),
                _log_feature(pos_score_rms),
                _log_feature(cell_score_rms),
                _log_feature(atomic_score_rms),
                _log_feature(pos_displacement_max, 100.0),
                _log_feature(cell_displacement_max, 10.0),
            ),
            dim=1,
        )
        intermediates = {
            "pos_index": pos_index,
            "cell_index": cell_index,
            "atomic_index": atomic_index,
            "pos_delta": pos_delta,
            "cell_delta": cell_delta,
            "active_atomic_logits": active_atomic_logits,
        }
        return features, intermediates

    def forward(
        self,
        *,
        x_before: BatchedData,
        x_after: BatchedData,
        score_before: BatchedData,
        t: torch.Tensor,
        progress: float | torch.Tensor,
    ) -> AdapterPrediction:
        features, values = self._context_features(
            x_before=x_before,
            x_after=x_after,
            score_before=score_before,
            t=t,
            progress=progress,
        )
        normalized = (features - self.feature_mean) / self.feature_std
        context = self.context_encoder(normalized)

        pos_coefficients = self.pos_head(context)[values["pos_index"]]
        pos_residual = (
            pos_coefficients[:, :1] * values["pos_delta"]
            + pos_coefficients[:, 1:] * score_before["pos"]
        )

        cell_coefficients = self.cell_head(context)[values["cell_index"]]
        cell_residual = (
            cell_coefficients[:, :1, None] * values["cell_delta"]
            + cell_coefficients[:, 1:, None] * score_before["cell"]
        )

        atomic_score = score_before["atomic_numbers"]
        atom_score_rms = _log_feature(
            torch.sqrt(
                torch.where(
                    values["active_atomic_logits"], atomic_score.float().square(), 0.0
                ).sum(dim=1)
                / values["active_atomic_logits"].sum(dim=1).clamp_min(1)
            )
        )
        atom_displacement = _log_feature(values["pos_delta"].float().norm(dim=1), 100.0)
        atomic_context = torch.cat(
            (
                context[values["atomic_index"]],
                atom_score_rms[:, None],
                atom_displacement[:, None],
            ),
            dim=1,
        )
        atomic_coefficients = self.atomic_head(self.atomic_encoder(atomic_context))
        active_score = torch.where(values["active_atomic_logits"], atomic_score, 0.0)
        atomic_residual = (
            atomic_coefficients[:, :1] * active_score
            + atomic_coefficients[:, 1:] @ self.atomic_basis
        )
        atomic_residual = torch.where(
            values["active_atomic_logits"], atomic_residual, torch.zeros_like(atomic_residual)
        )

        residuals = {
            "pos": pos_residual.to(dtype=score_before["pos"].dtype),
            "cell": cell_residual.to(dtype=score_before["cell"].dtype),
            "atomic_numbers": atomic_residual.to(dtype=atomic_score.dtype),
        }
        predicted_score = score_before.replace(
            **{field: score_before[field] + residual for field, residual in residuals.items()}
        )
        batch_size = x_after.get_batch_size()
        predicted_norms = torch.stack(
            (
                _log_feature(
                    _per_sample_rms(residuals["pos"], values["pos_index"], batch_size)
                ),
                _log_feature(
                    _per_sample_rms(residuals["cell"], values["cell_index"], batch_size)
                ),
                _log_feature(
                    _per_sample_rms(
                        residuals["atomic_numbers"],
                        values["atomic_index"],
                        batch_size,
                        valid=values["active_atomic_logits"],
                    )
                ),
            ),
            dim=1,
        )
        return AdapterPrediction(
            score=predicted_score,
            residuals=residuals,
            context_features=features,
            risk_features=torch.cat((features, predicted_norms), dim=1),
        )


def all_prediction_tensors_finite(prediction: AdapterPrediction) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in prediction.residuals.values())
