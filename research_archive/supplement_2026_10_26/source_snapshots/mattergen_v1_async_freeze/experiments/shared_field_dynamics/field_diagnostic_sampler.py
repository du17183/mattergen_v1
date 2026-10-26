"""Read-only field-dynamics instrumentation for the official guided sampler."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mattergen.diffusion.discrete_time import to_discrete_time
from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector


REQUESTED_TIMESTEPS = (1.0, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.08,
                       0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00)
TRACE_FIELDS = (
    "seed", "subset", "observation_ordinal", "sampling_step", "model_t",
    "reverse_progress", "d3pm_discrete_t", "num_atoms",
    "atomic_entropy", "atomic_argmax_change_rate", "atomic_mean_max_probability",
    "atomic_mask_fraction", "atomic_clean_argmax_state_agreement",
    "position_rms_cart_displacement", "position_max_cart_displacement",
    "position_raw_to_clean_rms_cart",
    "cell_metric_ltl_relative_change", "cell_metric_llt_relative_change",
    "cell_volume_relative_change", "cell_length_rms_relative_change",
    "cell_length_max_relative_change", "cell_angle_rms_change_deg",
    "cell_angle_max_change_deg", "cell_clean_alpha", "cell_clean_inverse_alpha",
    "cell_clean_signed_determinant", "cell_clean_volume", "cell_clean_condition_number",
    "cell_clean_min_singular_value", "cell_clean_asymmetry_relative",
    "cell_raw_to_clean_metric_relative", "finite_clean_estimates",
)


def observation_indices(n_steps: int, max_t: float, eps_t: float) -> tuple[int, ...]:
    """About 50 points: every 20 steps plus explicitly requested late times."""
    timesteps = torch.linspace(max_t, eps_t, n_steps)
    indices = set(range(0, n_steps, 20)) | {n_steps - 1}
    for requested in REQUESTED_TIMESTEPS:
        indices.add(int(torch.argmin(torch.abs(timesteps - requested)).item()))
    return tuple(sorted(indices))


def _relative_change(current: np.ndarray, previous: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(current - previous) / max(np.linalg.norm(previous), eps))


def _cell_geometry(cell: np.ndarray) -> dict[str, Any]:
    singular = np.linalg.svd(cell, compute_uv=False)
    lengths = np.linalg.norm(cell, axis=1)
    angles = []
    for j, k in ((1, 2), (2, 0), (0, 1)):
        denominator = max(lengths[j] * lengths[k], 1e-12)
        cosine = float(np.clip(np.dot(cell[j], cell[k]) / denominator, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cosine))))
    determinant = float(np.linalg.det(cell))
    return {
        "ltl": cell.T @ cell,
        "llt": cell @ cell.T,
        "lengths": lengths,
        "angles": np.asarray(angles),
        "signed_determinant": determinant,
        "volume": abs(determinant),
        "condition_number": float(singular.max() / max(singular.min(), 1e-12)),
        "min_singular_value": float(singular.min()),
        "asymmetry_relative": float(
            np.linalg.norm(cell - cell.T) / max(np.linalg.norm(cell), 1e-12)
        ),
    }


def _position_change(
    current: np.ndarray, previous: np.ndarray, reference_cell: np.ndarray
) -> tuple[float, float]:
    # This is the preregistered fractional-coordinate periodic wrap.  The fixed
    # previous clean-cell reference isolates positional from cell deformation.
    delta_fractional = np.remainder(current - previous + 0.5, 1.0) - 0.5
    delta_cartesian = delta_fractional @ reference_cell
    norms = np.linalg.norm(delta_cartesian, axis=1)
    return float(np.sqrt(np.mean(norms**2))), float(np.max(norms))


class FieldDynamicsDiagnosticSampler(GuidedPredictorCorrector):
    """Official sampler with a read-only predictor-score observation hook."""

    def __init__(self, *, diagnostic_trace_path: str, diagnostic_seed: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._trace_path = Path(diagnostic_trace_path)
        if not self._trace_path.is_absolute():
            raise ValueError("diagnostic trace path must be absolute")
        self._diagnostic_seed = int(diagnostic_seed)
        self._subset = "calibration" if self._diagnostic_seed <= 99031 else "validation"
        self._observation_indices = observation_indices(self.N, self._max_t, self._eps_t)
        if len(self._observation_indices) != 55:
            raise RuntimeError(f"observation grid changed: {len(self._observation_indices)}")
        self._rows: list[dict[str, Any]] = []
        self._previous_atomic_argmax: np.ndarray | None = None
        self._previous_clean_position: np.ndarray | None = None
        self._previous_clean_cell: np.ndarray | None = None

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        if self._trace_path.exists():
            raise FileExistsError(f"refusing to overwrite trace: {self._trace_path}")
        self._rows = []
        self._previous_atomic_argmax = None
        self._previous_clean_position = None
        self._previous_clean_cell = None

    def _on_sampling_end(self, error: BaseException | None) -> None:
        super()._on_sampling_end(error)
        if error is not None:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self._rows)

    @property
    def sampling_metrics(self) -> Mapping[str, float | int]:
        values = dict(super().sampling_metrics)
        values["diagnostic_observation_count"] = len(self._rows)
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
            raise RuntimeError("diagnostic requires batch_size=1 for trajectory-level metrics")
        model_t = float(t[0].detach().cpu().item())
        atom_corruption = self._multi_corruption.discrete_corruptions["atomic_numbers"]
        logits = score["atomic_numbers"].float()
        log_probability = torch.log_softmax(logits, dim=-1)
        probability = log_probability.exp()
        atomic_argmax = (
            torch.argmax(probability, dim=-1) + int(atom_corruption.offset)
        ).detach().cpu().numpy()
        atomic_entropy = float(
            (-(probability * log_probability).sum(dim=-1).mean()).detach().cpu().item()
        )
        atomic_confidence = float(probability.max(dim=-1).values.mean().detach().cpu().item())
        current_atomic = x["atomic_numbers"].detach().cpu().numpy()
        atomic_change = (
            float(np.mean(atomic_argmax != self._previous_atomic_argmax))
            if self._previous_atomic_argmax is not None else float("nan")
        )

        pos_corruption = self._multi_corruption.sdes["pos"]
        pos_batch = x.get_batch_idx("pos")
        _, pos_std = pos_corruption.marginal_prob(
            x=x["pos"], t=t, batch_idx=pos_batch, batch=x
        )
        clean_position = pos_corruption.wrap(
            x["pos"] + pos_std.square() * score["pos"]
        ).detach().cpu().double().numpy()

        cell_corruption = self._multi_corruption.sdes["cell"]
        alpha, cell_std = cell_corruption.mean_coeff_and_std(x["cell"], t=t, batch=x)
        limit_mean = cell_corruption.get_limit_mean(x=x["cell"], batch=x)
        clean_cell_tensor = (
            x["cell"] + cell_std.square() * score["cell"]
            - (1.0 - alpha) * limit_mean
        ) / alpha
        clean_cell = clean_cell_tensor[0].detach().cpu().double().numpy()
        raw_cell = x["cell"][0].detach().cpu().double().numpy()
        clean_geometry = _cell_geometry(clean_cell)
        raw_geometry = _cell_geometry(raw_cell)

        if self._previous_clean_position is None or self._previous_clean_cell is None:
            position_rms = position_max = float("nan")
            cell_changes = {name: float("nan") for name in (
                "ltl", "llt", "volume", "length_rms", "length_max", "angle_rms", "angle_max"
            )}
        else:
            position_rms, position_max = _position_change(
                clean_position, self._previous_clean_position, self._previous_clean_cell
            )
            previous_geometry = _cell_geometry(self._previous_clean_cell)
            length_relative = np.abs(
                clean_geometry["lengths"] - previous_geometry["lengths"]
            ) / np.maximum(previous_geometry["lengths"], 1e-12)
            angle_change = np.abs(clean_geometry["angles"] - previous_geometry["angles"])
            cell_changes = {
                "ltl": _relative_change(clean_geometry["ltl"], previous_geometry["ltl"]),
                "llt": _relative_change(clean_geometry["llt"], previous_geometry["llt"]),
                "volume": float(abs(clean_geometry["volume"] - previous_geometry["volume"])
                                / max(previous_geometry["volume"], 1e-12)),
                "length_rms": float(np.sqrt(np.mean(length_relative**2))),
                "length_max": float(np.max(length_relative)),
                "angle_rms": float(np.sqrt(np.mean(angle_change**2))),
                "angle_max": float(np.max(angle_change)),
            }
        raw_position = x["pos"].detach().cpu().double().numpy()
        raw_to_clean_rms, _ = _position_change(clean_position, raw_position, raw_cell)
        finite = bool(
            np.isfinite(clean_position).all()
            and np.isfinite(clean_cell).all()
            and all(np.isfinite(value) for value in clean_geometry.values()
                    if not isinstance(value, np.ndarray))
        )
        ordinal = self._observation_indices.index(step)
        self._rows.append({
            "seed": self._diagnostic_seed,
            "subset": self._subset,
            "observation_ordinal": ordinal,
            "sampling_step": step,
            "model_t": model_t,
            "reverse_progress": 1.0 - model_t,
            "d3pm_discrete_t": int(to_discrete_time(
                t, N=atom_corruption.N, T=atom_corruption.T
            )[0].detach().cpu().item()),
            "num_atoms": int(x["num_atoms"][0].detach().cpu().item()),
            "atomic_entropy": atomic_entropy,
            "atomic_argmax_change_rate": atomic_change,
            "atomic_mean_max_probability": atomic_confidence,
            "atomic_mask_fraction": float(np.mean(current_atomic == 101)),
            "atomic_clean_argmax_state_agreement": float(np.mean(atomic_argmax == current_atomic)),
            "position_rms_cart_displacement": position_rms,
            "position_max_cart_displacement": position_max,
            "position_raw_to_clean_rms_cart": raw_to_clean_rms,
            "cell_metric_ltl_relative_change": cell_changes["ltl"],
            "cell_metric_llt_relative_change": cell_changes["llt"],
            "cell_volume_relative_change": cell_changes["volume"],
            "cell_length_rms_relative_change": cell_changes["length_rms"],
            "cell_length_max_relative_change": cell_changes["length_max"],
            "cell_angle_rms_change_deg": cell_changes["angle_rms"],
            "cell_angle_max_change_deg": cell_changes["angle_max"],
            "cell_clean_alpha": float(alpha.reshape(-1)[0].detach().cpu().item()),
            "cell_clean_inverse_alpha": float((1.0 / alpha.reshape(-1)[0]).detach().cpu().item()),
            "cell_clean_signed_determinant": clean_geometry["signed_determinant"],
            "cell_clean_volume": clean_geometry["volume"],
            "cell_clean_condition_number": clean_geometry["condition_number"],
            "cell_clean_min_singular_value": clean_geometry["min_singular_value"],
            "cell_clean_asymmetry_relative": clean_geometry["asymmetry_relative"],
            "cell_raw_to_clean_metric_relative": _relative_change(
                clean_geometry["ltl"], raw_geometry["ltl"]
            ),
            "finite_clean_estimates": finite,
        })
        self._previous_atomic_argmax = atomic_argmax.copy()
        self._previous_clean_position = clean_position.copy()
        self._previous_clean_cell = clean_cell.copy()
