"""Conservative late-stage MatterSim position-force score guidance."""
from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from ase import Atoms
from mattersim.forcefield.potential import MatterSimCalculator, Potential

from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector


TRACE_FIELDS = (
    "seed", "sampling_step", "t_norm", "phase", "energy_ev", "mean_force_ev_a",
    "max_force_ev_a", "force_reference_ev_a", "nominal_cart_step_a",
    "hard_cart_cap_a", "max_cart_correction_a", "score_coeff_min",
    "min_distance_before_a", "min_distance_after_a", "accepted", "fallback",
    "fallback_reason", "physics_seconds",
)


def _minimum_distance(atoms: Atoms) -> float:
    if len(atoms) < 2:
        return float("inf")
    distance = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def bounded_cartesian_force_correction(
    force_ev_a: np.ndarray, *, force_reference_ev_a: float,
    nominal_cart_step_a: float, hard_cart_cap_a: float,
) -> np.ndarray:
    """Turn Cartesian force into a translation-free bounded displacement."""
    centered = np.asarray(force_ev_a, dtype=float) - np.asarray(
        force_ev_a, dtype=float
    ).mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    scale = nominal_cart_step_a / max(float(norms.max()), force_reference_ev_a)
    correction = centered * scale
    correction_norms = np.linalg.norm(correction, axis=1)
    if float(correction_norms.max()) > hard_cart_cap_a:
        correction *= hard_cart_cap_a / float(correction_norms.max())
    return correction


def cartesian_to_fractional_correction(
    correction_cart: np.ndarray, clean_cell: np.ndarray
) -> np.ndarray:
    """Map row-vector Cartesian displacement to fractional: dr = df @ cell."""
    return np.asarray(correction_cart, dtype=float) @ np.linalg.inv(clean_cell)


def position_correction_is_safe(
    atoms: Atoms, correction_fractional: np.ndarray, min_distance_floor_a: float
) -> tuple[bool, float]:
    candidate = atoms.copy()
    candidate.set_scaled_positions(
        np.remainder(candidate.get_scaled_positions() + correction_fractional, 1.0)
    )
    min_distance = _minimum_distance(candidate)
    return bool(np.isfinite(min_distance) and min_distance >= min_distance_floor_a), min_distance


def _load_potential(path: str, device: str) -> Potential:
    original = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = compatible_load
    try:
        return Potential.from_checkpoint(
            load_path=path, device=device, load_training_state=False
        )
    finally:
        torch.load = original


class LateMatterSimForceGuidedSampler(GuidedPredictorCorrector):
    """Guide only the position predictor at frozen late timesteps.

    MatterSim is evaluated on the model clean estimate. The desired Cartesian
    energy-descent displacement is mapped to fractional coordinates with the
    clean lattice and divided by the ancestral predictor's exact score
    coefficient. Thus ``score_coeff * delta_score`` is the requested bounded
    fractional predictor contribution. Cell and atomic scores are untouched.
    """

    def __init__(
        self, *, trace_path: str, sample_seed: int, mattersim_checkpoint: str,
        guidance_t_max: float = 0.02, force_reference_ev_a: float = 0.07795149218357911,
        nominal_cart_step_a: float = 0.005, hard_cart_cap_a: float = 0.01,
        min_distance_floor_a: float = 0.5, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._trace_path = Path(trace_path)
        if not self._trace_path.is_absolute():
            raise ValueError("trace_path must be absolute")
        self._seed = int(sample_seed)
        self._t_max = float(guidance_t_max)
        self._force_reference = float(force_reference_ev_a)
        self._nominal_step = float(nominal_cart_step_a)
        self._hard_cap = float(hard_cart_cap_a)
        self._min_distance_floor = float(min_distance_floor_a)
        if not 0 < self._nominal_step <= self._hard_cap <= 0.02:
            raise ValueError("invalid Cartesian trust region")
        if self._hard_cap > 0.01 + 1e-12:
            raise ValueError("P0 freezes a stricter 0.01 A hard cap")
        potential = _load_potential(mattersim_checkpoint, str(self._device))
        self._calculator = MatterSimCalculator(
            potential=potential, compute_stress=False, device=str(self._device)
        )
        self._rows: list[dict[str, Any]] = []
        self._eligible = 0
        self._accepted = 0
        self._fallbacks = 0
        self._physics_seconds = 0.0

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        if self._trace_path.exists():
            raise FileExistsError(self._trace_path)
        self._rows = []
        self._eligible = self._accepted = self._fallbacks = 0
        self._physics_seconds = 0.0

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
        values.update(
            mattersim_guidance_eligible=self._eligible,
            mattersim_guidance_accepted=self._accepted,
            mattersim_guidance_fallbacks=self._fallbacks,
            mattersim_guidance_seconds=self._physics_seconds,
            mattersim_guidance_acceptance=(self._accepted / max(self._eligible, 1)),
        )
        return values

    def _evaluate_exact_score(self, x, t):
        score = super()._evaluate_exact_score(x, t)
        context = self.sampling_context
        model_t = float(t[0].detach().cpu().item())
        if context.get("phase") != "predictor" or model_t > self._t_max + 1e-8:
            return score
        self._eligible += 1
        started = time.perf_counter()
        row: dict[str, Any] = {
            "seed": self._seed,
            "sampling_step": int(context.get("sampling_step", -1)),
            "t_norm": model_t,
            "phase": "predictor",
            "energy_ev": "", "mean_force_ev_a": "", "max_force_ev_a": "",
            "force_reference_ev_a": self._force_reference,
            "nominal_cart_step_a": self._nominal_step,
            "hard_cart_cap_a": self._hard_cap,
            "max_cart_correction_a": 0.0,
            "score_coeff_min": "", "min_distance_before_a": "",
            "min_distance_after_a": "", "accepted": False, "fallback": True,
            "fallback_reason": "",
        }
        try:
            if x.get_batch_size() != 1:
                raise RuntimeError("P0 guidance requires batch_size=1")
            atoms, clean_cell = self._clean_atoms(x=x, score=score, t=t)
            row["min_distance_before_a"] = _minimum_distance(atoms)
            if not np.isfinite(clean_cell).all() or np.linalg.det(clean_cell) <= 0:
                raise ValueError("nonpositive_or_nonfinite_clean_cell")
            with torch.enable_grad():
                atoms.calc = self._calculator
                energy = float(atoms.get_potential_energy())
                force = np.asarray(atoms.get_forces(), dtype=float)
            norms = np.linalg.norm(force, axis=1)
            if not np.isfinite(energy) or not np.isfinite(force).all():
                raise FloatingPointError("nonfinite_mattersim_output")
            correction_cart = bounded_cartesian_force_correction(
                force, force_reference_ev_a=self._force_reference,
                nominal_cart_step_a=self._nominal_step,
                hard_cart_cap_a=self._hard_cap,
            )
            correction_norms = np.linalg.norm(correction_cart, axis=1)
            correction_frac = cartesian_to_fractional_correction(
                correction_cart, clean_cell
            )
            safe, min_after = position_correction_is_safe(
                atoms, correction_frac, self._min_distance_floor
            )
            row.update(
                energy_ev=energy, mean_force_ev_a=float(norms.mean()),
                max_force_ev_a=float(norms.max()),
                max_cart_correction_a=float(correction_norms.max()),
                min_distance_after_a=min_after,
            )
            if not safe:
                raise ValueError("min_distance_below_0.5A")
            predictor = self._predictors["pos"]
            dt = torch.full_like(t, -1.0 / self.N)
            _, score_coeff, _ = predictor._get_coeffs(
                x=x["pos"], t=t, dt=dt, batch_idx=x.get_batch_idx("pos"), batch=x
            )
            row["score_coeff_min"] = float(score_coeff.abs().min().detach().cpu().item())
            if not bool(torch.isfinite(score_coeff).all().item()) or bool((score_coeff.abs() < 1e-12).any().item()):
                raise FloatingPointError("invalid_predictor_score_coefficient")
            delta_fractional = torch.as_tensor(
                correction_frac, device=score["pos"].device, dtype=score["pos"].dtype
            )
            delta_score = delta_fractional / score_coeff
            if not bool(torch.isfinite(delta_score).all().item()):
                raise FloatingPointError("nonfinite_fractional_score_correction")
            score = score.replace(pos=score["pos"] + delta_score)
            row.update(accepted=True, fallback=False, fallback_reason="")
            self._accepted += 1
        except BaseException as error:
            row["fallback_reason"] = f"{type(error).__name__}: {str(error)[:240]}"
            self._fallbacks += 1
        elapsed = time.perf_counter() - started
        row["physics_seconds"] = elapsed
        self._physics_seconds += elapsed
        self._rows.append(row)
        return score

    def _clean_atoms(self, *, x, score, t: torch.Tensor) -> tuple[Atoms, np.ndarray]:
        atom_corruption = self._multi_corruption.discrete_corruptions["atomic_numbers"]
        clean_atomic = torch.argmax(score["atomic_numbers"].float(), dim=-1) + int(atom_corruption.offset)
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
        cell = clean_cell[0].detach().cpu().double().numpy()
        atoms = Atoms(
            numbers=clean_atomic.detach().cpu().numpy().astype(int),
            scaled_positions=clean_pos.detach().cpu().double().numpy(),
            cell=cell, pbc=True,
        )
        return atoms, cell
