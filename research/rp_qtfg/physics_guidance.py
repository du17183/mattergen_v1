from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from pymatgen.core import Lattice, Structure


CHGNET_SITE_PACKAGES = Path(
    "/data/dxl/envs/fn_pra_teacher/lib/python3.10/site-packages"
)


@dataclass(frozen=True)
class RPQTFGConfig:
    enabled: bool = False
    guidance_fields: str = "position"
    start_progress: float = 0.75
    position_eta: float = 0.01
    position_radius_angstrom: float = 0.02
    cell_eta_per_gpa: float = 0.00025
    cell_strain_radius: float = 0.003
    backtrack_max: int = 3
    conflict_threshold: float = -0.20
    score_ratio_max: float = 0.25
    minimum_distance_angstrom: float = 0.5
    maximum_cell_condition: float = 100.0
    force_loss_weight: float = 0.01
    stress_loss_weight: float = 1e-5
    short_bond_loss_weight: float = 10.0
    score_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.guidance_fields not in {"position", "position_cell"}:
            raise ValueError(
                "guidance_fields must be 'position' or 'position_cell'"
            )
        if not 0.0 <= self.start_progress <= 1.0:
            raise ValueError("start_progress must be in [0, 1]")
        if self.position_eta < 0 or self.position_radius_angstrom <= 0:
            raise ValueError("position guidance parameters must be positive")
        if self.cell_eta_per_gpa < 0 or self.cell_strain_radius <= 0:
            raise ValueError("cell guidance parameters must be positive")
        if not 1 <= self.backtrack_max <= 3:
            raise ValueError("backtrack_max must be in [1, 3]")
        if not -1.0 <= self.conflict_threshold <= 1.0:
            raise ValueError("conflict_threshold must be in [-1, 1]")
        if self.score_ratio_max <= 0:
            raise ValueError("score_ratio_max must be positive")
        if self.minimum_distance_angstrom <= 0:
            raise ValueError("minimum_distance_angstrom must be positive")

    @property
    def uses_cell(self) -> bool:
        return self.guidance_fields == "position_cell"


def periodic_delta(delta: torch.Tensor) -> torch.Tensor:
    return delta - torch.round(delta)


def cosine_similarity(
    left: torch.Tensor,
    right: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    left_flat = left.float().reshape(-1)
    right_flat = right.float().reshape(-1)
    denominator = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(
        right_flat
    )
    if not bool(torch.isfinite(denominator).item()) or float(denominator) <= eps:
        return 1.0
    value = torch.dot(left_flat, right_flat) / (denominator + eps)
    return float(torch.clamp(value, -1.0, 1.0).item())


def clean_estimate(
    *,
    noisy: torch.Tensor,
    score: torch.Tensor,
    corruption: Any,
    t: torch.Tensor,
    batch_idx: torch.LongTensor | None,
    batch: Any,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean_coeff, std = corruption.mean_coeff_and_std(
        noisy,
        t,
        batch_idx,
        batch,
    )
    clean = (noisy + std.square() * score) / torch.clamp(
        mean_coeff,
        min=eps,
    )
    if hasattr(corruption, "wrap"):
        clean = corruption.wrap(clean)
    return clean, mean_coeff, std


def score_correction_from_clean_delta(
    *,
    clean_delta: torch.Tensor,
    mean_coeff: torch.Tensor,
    std: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return mean_coeff * clean_delta / torch.clamp(std.square(), min=eps)


def clip_score_correction(
    correction: torch.Tensor,
    guided_score: torch.Tensor,
    residual: torch.Tensor,
    ratio_max: float,
    eps: float,
) -> tuple[torch.Tensor, bool]:
    correction_rms = torch.sqrt(torch.mean(correction.float().square()))
    reference_rms = torch.maximum(
        torch.sqrt(torch.mean(guided_score.float().square())),
        torch.sqrt(torch.mean(residual.float().square())),
    )
    limit = ratio_max * torch.clamp(reference_rms, min=eps)
    if float(correction_rms) <= float(limit):
        return correction, False
    scale = limit / torch.clamp(correction_rms, min=eps)
    return correction * scale.to(correction.dtype), True


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def _safe_structure(
    cell: np.ndarray,
    fractional_positions: np.ndarray,
    atomic_numbers: np.ndarray,
    config: RPQTFGConfig,
) -> Structure | None:
    try:
        if (
            not np.isfinite(cell).all()
            or not np.isfinite(fractional_positions).all()
            or not np.isfinite(atomic_numbers).all()
        ):
            return None
        volume = abs(float(np.linalg.det(cell)))
        condition = float(np.linalg.cond(cell))
        if (
            volume <= 0.1
            or not math.isfinite(condition)
            or condition > config.maximum_cell_condition
            or np.any(atomic_numbers <= 0)
        ):
            return None
        structure = Structure(
            lattice=Lattice(cell),
            species=atomic_numbers.astype(int),
            coords=np.mod(fractional_positions, 1.0),
            coords_are_cartesian=False,
            to_unit_cell=True,
        )
        if len(structure) > 1:
            distances = np.asarray(structure.distance_matrix, dtype=float)
            np.fill_diagonal(distances, np.inf)
            if float(np.min(distances)) < config.minimum_distance_angstrom:
                return None
        return structure
    except Exception:
        return None


def _objective(
    prediction: Mapping[str, Any],
    structure: Structure,
    config: RPQTFGConfig,
) -> float:
    energy = float(np.asarray(prediction["e"]).reshape(-1)[0])
    forces = np.asarray(prediction["f"], dtype=float)
    stress = np.asarray(prediction["s"], dtype=float)
    force_term = float(np.mean(forces**2))
    stress_term = float(np.mean(stress**2))
    short_bond_term = 0.0
    if len(structure) > 1:
        distances = np.asarray(structure.distance_matrix, dtype=float)
        np.fill_diagonal(distances, np.inf)
        violation = max(
            0.0,
            config.minimum_distance_angstrom - float(np.min(distances)),
        )
        short_bond_term = violation**2
    return (
        energy
        + config.force_loss_weight * force_term
        + config.stress_loss_weight * stress_term
        + config.short_bond_loss_weight * short_bond_term
    )


class RPQTFGEngine:
    def __init__(
        self,
        *,
        config: RPQTFGConfig,
        multi_corruption: Any,
        trace_path: str | None,
        sample_seed: int | None,
    ) -> None:
        self.config = config
        self.multi_corruption = multi_corruption
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path is not None and not self.trace_path.is_absolute():
            raise ValueError("rp_qtfg_trace_path must be absolute")
        self.sample_seed = sample_seed
        self._model: Any | None = None
        self.reset()

    def reset(self) -> None:
        self.stats: dict[str, Any] = {
            "sample_seed": self.sample_seed,
            "config": asdict(self.config),
            "eligible_calls": 0,
            "guided_calls": 0,
            "model_forward_batches": 0,
            "chgnet_forward_count": 0,
            "chgnet_backward_count": 0,
            "backtracking_count": 0,
            "fallback_count": 0,
            "conflict_count": 0,
            "clipping_count": 0,
            "position_accept_count": 0,
            "cell_accept_count": 0,
            "exception_count": 0,
            "elapsed_seconds": 0.0,
            "model_load_seconds": 0.0,
            "atomic_numbers_modified": False,
            "trace": [],
        }

    def _load_model(self, device: torch.device) -> Any:
        if self._model is not None:
            return self._model
        started = time.monotonic()
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            try:
                from chgnet.model.model import CHGNet
            except ModuleNotFoundError:
                sys.path.append(str(CHGNET_SITE_PACKAGES))
                from chgnet.model.model import CHGNet
            self._model = CHGNet.load(
                model_name="0.3.0",
                verbose=False,
                use_device="cuda" if device.type == "cuda" else "cpu",
            )
            self._model.eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        self.stats["model_load_seconds"] += time.monotonic() - started
        return self._model

    def _predict(
        self,
        model: Any,
        structures: list[Structure],
    ) -> list[dict[str, Any]]:
        with torch.enable_grad():
            prediction = model.predict_structure(
                structures,
                task="efs",
                batch_size=max(1, len(structures)),
            )
        predictions = prediction if isinstance(prediction, list) else [prediction]
        self.stats["model_forward_batches"] += 1
        self.stats["chgnet_forward_count"] += len(structures)
        self.stats["chgnet_backward_count"] += len(structures)
        return predictions

    def _propose(
        self,
        *,
        structure: Structure,
        prediction: Mapping[str, Any],
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        cell = np.asarray(structure.lattice.matrix, dtype=float)
        fractional = np.asarray(structure.frac_coords, dtype=float)
        forces = np.asarray(prediction["f"], dtype=float)
        displacement_cart = self.config.position_eta * scale * forces
        norms = np.linalg.norm(displacement_cart, axis=1)
        cap = self.config.position_radius_angstrom * scale
        clipped = bool(np.any(norms > cap))
        displacement_cart *= np.minimum(
            1.0,
            cap / np.maximum(norms, 1e-12),
        )[:, None]
        fractional_delta = displacement_cart @ np.linalg.inv(cell)
        proposed_fractional = np.mod(fractional + fractional_delta, 1.0)
        proposed_cell = cell.copy()
        if self.config.uses_cell:
            stress = np.asarray(prediction["s"], dtype=float)
            stress = 0.5 * (stress + stress.T)
            strain = -self.config.cell_eta_per_gpa * scale * stress
            strain_norm = float(np.linalg.norm(strain))
            strain_cap = self.config.cell_strain_radius * scale
            if strain_norm > strain_cap:
                strain *= strain_cap / strain_norm
                clipped = True
            proposed_cell = cell @ (np.eye(3) + strain).T
        return proposed_cell, proposed_fractional, clipped

    def _build_clean_structures(
        self,
        *,
        clean_cell: torch.Tensor,
        clean_pos: torch.Tensor,
        atomic_logits: torch.Tensor,
        pos_batch_idx: torch.LongTensor | None,
    ) -> tuple[list[Structure], list[np.ndarray]]:
        batch_size = clean_cell.shape[0]
        if pos_batch_idx is None:
            if batch_size != 1:
                raise RuntimeError("dense positions are only supported for batch_size=1")
            pos_batch_idx = torch.zeros(
                clean_pos.shape[0],
                dtype=torch.long,
                device=clean_pos.device,
            )
        atomic_numbers = torch.argmax(atomic_logits, dim=-1)
        structures = []
        masks = []
        for batch_index in range(batch_size):
            mask = (pos_batch_idx == batch_index).detach().cpu().numpy()
            structure = _safe_structure(
                clean_cell[batch_index].detach().cpu().double().numpy(),
                clean_pos[batch_index == pos_batch_idx]
                .detach()
                .cpu()
                .double()
                .numpy(),
                atomic_numbers[batch_index == pos_batch_idx]
                .detach()
                .cpu()
                .numpy(),
                self.config,
            )
            if structure is None:
                raise RuntimeError(
                    f"invalid predicted clean structure at batch index {batch_index}"
                )
            structures.append(structure)
            masks.append(mask)
        return structures, masks

    def apply(
        self,
        *,
        x: Any,
        t: torch.Tensor,
        guided_score: Any,
        conditional_score: Any,
        unconditional_score: Any,
        context: Mapping[str, Any],
    ) -> Any:
        progress = float(context.get("progress", 0.0))
        if not self.config.enabled or progress < self.config.start_progress:
            return guided_score
        self.stats["eligible_calls"] += 1
        started = time.monotonic()
        trace: dict[str, Any] = {
            "sampling_step": context.get("sampling_step"),
            "phase": context.get("phase"),
            "progress": progress,
            "accepted": False,
            "fallback_reason": None,
        }
        try:
            pos_corruption = self.multi_corruption.corruptions["pos"]
            cell_corruption = self.multi_corruption.corruptions["cell"]
            pos_batch_idx = x.get_batch_idx("pos")
            cell_batch_idx = x.get_batch_idx("cell")
            clean_pos, pos_mean, pos_std = clean_estimate(
                noisy=x["pos"],
                score=guided_score["pos"],
                corruption=pos_corruption,
                t=t,
                batch_idx=pos_batch_idx,
                batch=x,
                eps=self.config.score_eps,
            )
            clean_cell, cell_mean, cell_std = clean_estimate(
                noisy=x["cell"],
                score=guided_score["cell"],
                corruption=cell_corruption,
                t=t,
                batch_idx=cell_batch_idx,
                batch=x,
                eps=self.config.score_eps,
            )
            structures, masks = self._build_clean_structures(
                clean_cell=clean_cell,
                clean_pos=clean_pos,
                atomic_logits=guided_score["atomic_numbers"],
                pos_batch_idx=pos_batch_idx,
            )
            model = self._load_model(x["pos"].device)
            old_predictions = self._predict(model, structures)
            unresolved = list(range(len(structures)))
            accepted: dict[int, tuple[Structure, int, bool]] = {}
            for backtrack in range(self.config.backtrack_max):
                if not unresolved:
                    break
                scale = 0.5**backtrack
                proposed: list[Structure] = []
                proposed_meta: list[tuple[int, bool]] = []
                for index in unresolved:
                    cell, fractional, clipped = self._propose(
                        structure=structures[index],
                        prediction=old_predictions[index],
                        scale=scale,
                    )
                    candidate = _safe_structure(
                        cell,
                        fractional,
                        np.asarray(structures[index].atomic_numbers),
                        self.config,
                    )
                    if candidate is not None:
                        proposed.append(candidate)
                        proposed_meta.append((index, clipped))
                if not proposed:
                    continue
                new_predictions = self._predict(model, proposed)
                remaining = []
                predicted_by_index = {
                    index: (candidate, prediction, clipped)
                    for (index, clipped), candidate, prediction in zip(
                        proposed_meta,
                        proposed,
                        new_predictions,
                        strict=True,
                    )
                }
                for index in unresolved:
                    proposal = predicted_by_index.get(index)
                    if proposal is None:
                        remaining.append(index)
                        continue
                    candidate, prediction, clipped = proposal
                    old_objective = _objective(
                        old_predictions[index],
                        structures[index],
                        self.config,
                    )
                    new_objective = _objective(
                        prediction,
                        candidate,
                        self.config,
                    )
                    if (
                        math.isfinite(new_objective)
                        and new_objective <= old_objective + 1e-7
                    ):
                        accepted[index] = (candidate, backtrack, clipped)
                    else:
                        remaining.append(index)
                unresolved = remaining

            if not accepted:
                self.stats["fallback_count"] += 1
                trace["fallback_reason"] = "backtracking_failed"
                return guided_score

            target_pos = clean_pos.detach().clone()
            target_cell = clean_cell.detach().clone()
            for batch_index, (candidate, backtrack, clipped) in accepted.items():
                mask_tensor = (
                    pos_batch_idx == batch_index
                    if pos_batch_idx is not None
                    else torch.ones(
                        clean_pos.shape[0],
                        dtype=torch.bool,
                        device=clean_pos.device,
                    )
                )
                target_pos[mask_tensor] = torch.as_tensor(
                    np.asarray(candidate.frac_coords),
                    device=clean_pos.device,
                    dtype=clean_pos.dtype,
                )
                target_cell[batch_index] = torch.as_tensor(
                    np.asarray(candidate.lattice.matrix).copy(),
                    device=clean_cell.device,
                    dtype=clean_cell.dtype,
                )
                self.stats["backtracking_count"] += backtrack
                self.stats["clipping_count"] += int(clipped)

            pos_clean_delta = periodic_delta(target_pos - clean_pos)
            pos_correction = score_correction_from_clean_delta(
                clean_delta=pos_clean_delta,
                mean_coeff=pos_mean,
                std=pos_std,
                eps=self.config.score_eps,
            )
            pos_residual = (
                conditional_score["pos"] - unconditional_score["pos"]
            )
            pos_cosine = cosine_similarity(
                pos_correction,
                pos_residual,
                self.config.score_eps,
            )
            trace["position_residual_cosine"] = pos_cosine
            replacements = {}
            if pos_cosine < self.config.conflict_threshold:
                self.stats["conflict_count"] += 1
                trace["position_rejected_for_conflict"] = True
            else:
                pos_correction, clipped = clip_score_correction(
                    pos_correction,
                    guided_score["pos"],
                    pos_residual,
                    self.config.score_ratio_max,
                    self.config.score_eps,
                )
                self.stats["clipping_count"] += int(clipped)
                replacements["pos"] = guided_score["pos"] + pos_correction
                self.stats["position_accept_count"] += 1

            if self.config.uses_cell:
                cell_clean_delta = target_cell - clean_cell
                cell_correction = score_correction_from_clean_delta(
                    clean_delta=cell_clean_delta,
                    mean_coeff=cell_mean,
                    std=cell_std,
                    eps=self.config.score_eps,
                )
                cell_residual = (
                    conditional_score["cell"] - unconditional_score["cell"]
                )
                cell_cosine = cosine_similarity(
                    cell_correction,
                    cell_residual,
                    self.config.score_eps,
                )
                trace["cell_residual_cosine"] = cell_cosine
                if cell_cosine < self.config.conflict_threshold:
                    self.stats["conflict_count"] += 1
                    trace["cell_rejected_for_conflict"] = True
                else:
                    cell_correction, clipped = clip_score_correction(
                        cell_correction,
                        guided_score["cell"],
                        cell_residual,
                        self.config.score_ratio_max,
                        self.config.score_eps,
                    )
                    self.stats["clipping_count"] += int(clipped)
                    replacements["cell"] = (
                        guided_score["cell"] + cell_correction
                    )
                    self.stats["cell_accept_count"] += 1

            if not replacements:
                self.stats["fallback_count"] += 1
                trace["fallback_reason"] = "all_fields_conflicted"
                return guided_score
            self.stats["guided_calls"] += 1
            trace["accepted"] = True
            return guided_score.replace(**replacements)
        except Exception as error:
            self.stats["exception_count"] += 1
            self.stats["fallback_count"] += 1
            trace["fallback_reason"] = f"{type(error).__name__}:{error}"
            return guided_score
        finally:
            elapsed = time.monotonic() - started
            self.stats["elapsed_seconds"] += elapsed
            trace["elapsed_seconds"] = elapsed
            if len(self.stats["trace"]) < 5000:
                self.stats["trace"].append(trace)

    def finish(self, error: BaseException | None) -> None:
        if self.trace_path is None:
            return
        eligible = max(int(self.stats["eligible_calls"]), 1)
        output = dict(self.stats)
        output.update(
            {
                "sampling_error": (
                    f"{type(error).__name__}:{error}" if error else None
                ),
                "fallback_rate": self.stats["fallback_count"] / eligible,
                "conflict_rate": self.stats["conflict_count"] / eligible,
                "clipping_rate": self.stats["clipping_count"] / eligible,
                "guided_call_rate": self.stats["guided_calls"] / eligible,
            }
        )
        _atomic_json(self.trace_path, output)
