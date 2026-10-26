"""Read-only FP-PC proposal instrumentation for the official sampler."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from tqdm.auto import tqdm

from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector
from mattergen.diffusion.sampling.pc_sampler import _mask_replace


TRACE_FIELDS = (
    "seed", "subset", "sampling_step", "timestep", "reverse_progress", "stage",
    "num_atoms", "min_distance_before", "min_distance_candidate",
    "VN_before", "VN_candidate", "signed_volume_before", "signed_volume_candidate",
    "before_finite", "candidate_finite", "before_distance_violation",
    "before_volume_violation", "before_any_violation", "candidate_distance_violation",
    "candidate_volume_violation", "candidate_any_violation",
    "proposal_entered_distance_violation", "proposal_entered_volume_violation",
    "proposal_entered_any_violation", "proposal_worsened_distance",
    "proposal_worsened_volume", "accepted_unmodified",
)


class ProposalViolationDiagnosticSampler(GuidedPredictorCorrector):
    """Exact official PC path plus side-effect-free before/candidate measurements."""

    def __init__(
        self, *, diagnostic_trace_path: str, diagnostic_seed: int,
        feasible_region_path: str, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._trace_path = Path(diagnostic_trace_path)
        self._diagnostic_seed = int(diagnostic_seed)
        region = json.loads(Path(feasible_region_path).read_text(encoding="utf-8"))
        if not region.get("frozen") or not region.get("frozen_before_diagnostic_outcomes"):
            raise RuntimeError("feasible region was not preregistered/frozen")
        self._d_floor = float(region["d_min_floor_angstrom"])
        self._vn_low = float(region["VN_low_angstrom3_per_atom"])
        self._vn_high = float(region["VN_high_angstrom3_per_atom"])
        self._rows: list[dict[str, Any]] = []
        self._shifts: torch.Tensor | None = None

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        if self._trace_path.exists():
            raise FileExistsError(f"refusing to overwrite trace: {self._trace_path}")
        self._rows = []
        self._shifts = torch.tensor(
            [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            device=self._device, dtype=torch.float64,
        )

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
        metrics = dict(super().sampling_metrics)
        metrics["proposal_observation_count"] = len(self._rows)
        metrics["proposal_entry_count"] = sum(
            bool(row["proposal_entered_any_violation"]) for row in self._rows
        )
        return metrics

    def _measure(self, state) -> dict[str, Any]:
        if state.get_batch_size() != 1:
            raise RuntimeError("B0 instrumentation requires batch_size=1")
        assert self._shifts is not None
        pos = torch.remainder(state["pos"].double(), 1.0)
        cell = state["cell"][0].double()
        n = int(state["num_atoms"][0].item())
        finite = bool(torch.isfinite(pos).all().item() and torch.isfinite(cell).all().item())
        if finite:
            delta = pos[:, None, None, :] - pos[None, :, None, :] + self._shifts[None, None, :, :]
            cart = torch.matmul(delta, cell)
            distances = torch.linalg.vector_norm(cart, dim=-1)
            diagonal = torch.arange(n, device=pos.device)
            zero_shift = 13
            distances[diagonal, diagonal, zero_shift] = torch.inf
            d_min = float(distances.min().item())
            signed_volume = float(torch.linalg.det(cell).item())
            vn = abs(signed_volume) / n
        else:
            d_min = vn = signed_volume = float("nan")
        finite = finite and all(math.isfinite(x) for x in (d_min, vn, signed_volume))
        distance_violation = (not finite) or d_min < self._d_floor
        volume_violation = (not finite) or vn < self._vn_low or vn > self._vn_high
        return {
            "min_distance": d_min,
            "VN": vn,
            "signed_volume": signed_volume,
            "finite": finite,
            "distance_violation": distance_violation,
            "volume_violation": volume_violation,
            "any_violation": distance_violation or volume_violation,
        }

    def _record(self, *, before, candidate, stage: str, step: int, t: torch.Tensor) -> None:
        b = self._measure(before)
        c = self._measure(candidate)
        before_feasible = not b["any_violation"]
        entry_d = before_feasible and c["distance_violation"]
        entry_v = before_feasible and c["volume_violation"]
        self._rows.append({
            "seed": self._diagnostic_seed,
            "subset": "calibration" if self._diagnostic_seed <= 99031 else "validation",
            "sampling_step": step,
            "timestep": float(t[0].detach().cpu().item()),
            "reverse_progress": step / max(self.N - 1, 1),
            "stage": stage,
            "num_atoms": int(before["num_atoms"][0].item()),
            "min_distance_before": b["min_distance"],
            "min_distance_candidate": c["min_distance"],
            "VN_before": b["VN"],
            "VN_candidate": c["VN"],
            "signed_volume_before": b["signed_volume"],
            "signed_volume_candidate": c["signed_volume"],
            "before_finite": b["finite"],
            "candidate_finite": c["finite"],
            "before_distance_violation": b["distance_violation"],
            "before_volume_violation": b["volume_violation"],
            "before_any_violation": b["any_violation"],
            "candidate_distance_violation": c["distance_violation"],
            "candidate_volume_violation": c["volume_violation"],
            "candidate_any_violation": c["any_violation"],
            "proposal_entered_distance_violation": entry_d,
            "proposal_entered_volume_violation": entry_v,
            "proposal_entered_any_violation": entry_d or entry_v,
            "proposal_worsened_distance": c["min_distance"] < b["min_distance"],
            "proposal_worsened_volume": (
                c["VN"] < b["VN"] if b["VN"] < self._vn_low else
                c["VN"] > b["VN"] if b["VN"] > self._vn_high else False
            ),
            # Diagnostic never modifies a proposal; this flag audits exact official behavior.
            "accepted_unmodified": True,
        })

    @torch.no_grad()
    def _denoise(self, batch, mask: dict[str, torch.Tensor], record: bool = False):
        recorded_samples = [] if record else None
        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)
            x_before_last_corrector = score_before_last_corrector = None
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    self._set_sampling_context(sampling_step=i, phase="corrector")
                    before = batch
                    x_before_last_corrector = batch
                    score = self._evaluate_exact_score(batch, t)
                    score_before_last_corrector = score
                    samples_means = apply(
                        fns={k: corrector.step_given_score for k, corrector in self._correctors.items()},
                        broadcast={"t": t, "dt": dt}, x=batch, score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                    )
                    self._record(before=before, candidate=batch, stage="corrector", step=i, t=t)

            self._set_sampling_context(sampling_step=i, phase="predictor")
            before = batch
            if (
                self._residual_controller is not None
                and x_before_last_corrector is not None
                and score_before_last_corrector is not None
            ):
                score = self._residual_controller.score_after_corrector(
                    exact_score_fn=self._evaluate_exact_score,
                    x_before=x_before_last_corrector, score_before=score_before_last_corrector,
                    x_after=batch, t=t, sampling_step=i,
                    progress=i / max(self.N - 1, 1),
                )
            else:
                score = self._evaluate_exact_score(batch, t)
            samples_means = apply(
                fns={k: predictor.update_given_score for k, predictor in self._predictors.items()},
                x=batch, score=score, broadcast={"t": t, "batch": batch, "dt": dt},
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )
            if record:
                recorded_samples.append(batch.clone().to("cpu"))
            batch, mean_batch = _mask_replace(
                samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
            )
            self._record(before=before, candidate=batch, stage="predictor", step=i, t=t)
        return batch, mean_batch, recorded_samples
