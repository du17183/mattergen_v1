"""Terminal feature capture and CG-TDR integration for MatterGen sampling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from tqdm.auto import tqdm

from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector
from mattergen.diffusion.sampling.pc_sampler import (
    PredictorCorrector,
    SampleAndMeanAndMaybeRecords,
    _mask_replace,
)

from .model import CGTDRConfig, CGTDRRefiner


def _rms_per_graph(
    value: torch.Tensor,
    batch_index: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor:
    flattened = value.float().reshape(value.shape[0], -1)
    squared = flattened.square().mean(dim=-1)
    if batch_index is None:
        return squared.sqrt()
    output = squared.new_zeros(batch_size)
    output.index_add_(0, batch_index, squared)
    counts = torch.bincount(batch_index, minlength=batch_size).to(squared.dtype).clamp_min_(1)
    return (output / counts).sqrt()


class CGTDRGuidedPredictorCorrector(GuidedPredictorCorrector):
    """A0 sampler with a single deterministic refiner call after all PC steps."""

    def __init__(
        self,
        *,
        cg_tdr_enabled: bool = False,
        cg_tdr_checkpoint: str | None = None,
        cg_tdr_teacher_dump_dir: str | None = None,
        cg_tdr_enable_cell: bool = True,
        cg_tdr_config: Mapping[str, Any] | None = None,
        cg_tdr_metrics_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._cg_tdr_enabled = bool(cg_tdr_enabled)
        self._cg_tdr_checkpoint = cg_tdr_checkpoint
        self._cg_tdr_teacher_dump_dir = (
            Path(cg_tdr_teacher_dump_dir).expanduser()
            if cg_tdr_teacher_dump_dir is not None
            else None
        )
        self._cg_tdr_enable_cell = bool(cg_tdr_enable_cell)
        self._cg_tdr_config = CGTDRConfig(**dict(cg_tdr_config or {}))
        self._cg_tdr_metrics_path = Path(cg_tdr_metrics_path).expanduser() if cg_tdr_metrics_path else None
        self._cg_tdr_model: CGTDRRefiner | None = None
        self._terminal_history: list[Any] = []
        self._last_residual: dict[str, float | None] = {}
        self._last_score_rms: dict[str, torch.Tensor] = {}
        self._cg_tdr_metrics: dict[str, Any] = {}
        if self._cg_tdr_enabled and not self._cg_tdr_checkpoint:
            raise ValueError("cg_tdr_checkpoint is required when cg_tdr_enabled=True")

    @property
    def cg_tdr_metrics(self) -> Mapping[str, Any]:
        return self._cg_tdr_metrics

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self._terminal_history = []
        self._last_residual = {}
        self._last_score_rms = {}
        self._cg_tdr_metrics = {}
        if self._cg_tdr_metrics_path is not None and self._cg_tdr_metrics_path.exists():
            raise FileExistsError(f"Refusing to overwrite CG-TDR metrics: {self._cg_tdr_metrics_path}")

    def _on_sampling_end(self, error: BaseException | None) -> None:
        super()._on_sampling_end(error)
        if self._cg_tdr_metrics_path is None:
            return
        self._cg_tdr_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "enabled": self._cg_tdr_enabled,
            "sample_seed": self._sample_seed,
            "error": None if error is None else repr(error),
            **self._cg_tdr_metrics,
        }
        temporary = self._cg_tdr_metrics_path.with_suffix(self._cg_tdr_metrics_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self._cg_tdr_metrics_path)

    def _trace_decision(
        self,
        *,
        t: torch.Tensor,
        field_deltas: Mapping[str, float | None],
        decision: Mapping[str, float | str | None],
    ) -> None:
        self._last_residual = dict(field_deltas)
        super()._trace_decision(t=t, field_deltas=field_deltas, decision=decision)

    def _score_fn(self, x: Any, t: torch.Tensor) -> Any:
        score = super()._score_fn(x=x, t=t)
        batch_size = x.get_batch_size()
        self._last_score_rms = {
            field: _rms_per_graph(score[field], x.get_batch_idx(field), batch_size)
            for field in ("pos", "cell")
            if field in score
        }
        return score

    def _remember_terminal_state(self, mean_batch: Any) -> None:
        self._terminal_history.append(
            mean_batch.replace(
                pos=mean_batch["pos"].detach().clone(),
                cell=mean_batch["cell"].detach().clone(),
                atomic_numbers=mean_batch["atomic_numbers"].detach().clone(),
            )
        )
        del self._terminal_history[:-3]

    @torch.no_grad()
    def _denoise(
        self,
        batch: Any,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Mirror the official PC loop and retain only three terminal means."""

        recorded_samples = [] if record else None
        for key in self._predictors:
            mask.setdefault(key, None)
        for key in self._correctors:
            mask.setdefault(key, None)
        mean_batch = batch.clone()
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        for step in tqdm(range(self.N), miniters=50, mininterval=5):
            t = torch.full((batch.get_batch_size(),), timesteps[step], device=self._device)
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    self._set_sampling_context(sampling_step=step, phase="corrector")
                    score = self._score_fn(batch, t)
                    functions = {
                        key: corrector.step_given_score
                        for key, corrector in self._correctors.items()
                    }
                    samples_means = apply(
                        fns=functions,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
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
            predictor_functions = {
                key: predictor.update_given_score for key, predictor in self._predictors.items()
            }
            samples_means = apply(
                fns=predictor_functions,
                x=batch,
                score=score,
                broadcast={"t": t, "batch": batch, "dt": dt},
                batch_idx=self._multi_corruption._get_batch_indices(batch),
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
            self._remember_terminal_state(mean_batch)

        mean_batch = self._terminal_refine_or_dump(mean_batch)
        return batch, mean_batch, recorded_samples

    def _state_change(self, newer: Any, older: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = newer.get_batch_size()
        batch_index = newer.get_batch_idx("pos")
        assert batch_index is not None
        frac_delta = newer["pos"] - older["pos"]
        frac_delta = frac_delta - torch.round(frac_delta)
        cart_delta = torch.bmm(
            frac_delta[:, None, :], newer["cell"][batch_index]
        ).squeeze(1)
        position_change = _rms_per_graph(cart_delta, batch_index, batch_size)
        cell_delta = newer["cell"] - older["cell"]
        cell_change = torch.linalg.matrix_norm(
            cell_delta, ord="fro", dim=(-2, -1)
        ) / torch.linalg.matrix_norm(older["cell"], ord="fro", dim=(-2, -1)).clamp_min(1.0e-8)
        return position_change, cell_change

    def _convergence_features(self, mean_batch: Any) -> torch.Tensor:
        batch_size = mean_batch.get_batch_size()
        zeros = mean_batch["cell"].new_zeros(batch_size)
        if len(self._terminal_history) >= 2:
            pos_last, cell_last = self._state_change(
                self._terminal_history[-1], self._terminal_history[-2]
            )
        else:
            pos_last, cell_last = zeros, zeros
        if len(self._terminal_history) >= 3:
            pos_previous, cell_previous = self._state_change(
                self._terminal_history[-2], self._terminal_history[-3]
            )
        else:
            pos_previous, cell_previous = zeros, zeros
        residual_pos = zeros + float(self._last_residual.get("pos") or 0.0)
        residual_cell = zeros + float(self._last_residual.get("cell") or 0.0)
        score_pos = self._last_score_rms.get("pos", zeros)
        score_cell = self._last_score_rms.get("cell", zeros)
        return torch.stack(
            [
                pos_last,
                pos_previous,
                cell_last,
                cell_previous,
                residual_pos,
                residual_cell,
                score_pos,
                score_cell,
            ],
            dim=-1,
        )

    def _capture_terminal_features(self, mean_batch: Any) -> tuple[torch.Tensor, Any]:
        denoiser = self._diffusion_module.model
        gemnet = getattr(denoiser, "gemnet", None)
        if gemnet is None:
            raise TypeError("CG-TDR requires a denoiser with a GemNet 'gemnet' module")
        captured: list[torch.Tensor] = []

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            captured.append(output.node_embeddings.detach())

        hook = gemnet.register_forward_hook(capture)
        try:
            terminal_t = torch.full(
                (mean_batch.get_batch_size(),),
                self._eps_t,
                device=self._device,
            )
            conditional_score = PredictorCorrector._score_fn(
                self, self._keep_conditioning_fn(mean_batch), terminal_t
            )
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError(f"Expected one GemNet feature capture, got {len(captured)}")
        return captured[0], conditional_score

    def _load_refiner(self, device: torch.device) -> CGTDRRefiner:
        if self._cg_tdr_model is not None:
            return self._cg_tdr_model
        assert self._cg_tdr_checkpoint is not None
        payload = torch.load(self._cg_tdr_checkpoint, map_location="cpu", weights_only=False)
        config = CGTDRConfig(**payload.get("config", self._cg_tdr_config.as_dict()))
        model = CGTDRRefiner(config)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        self._cg_tdr_model = model.to(device)
        return self._cg_tdr_model

    def _dump_teacher_record(
        self,
        *,
        mean_batch: Any,
        node_features: torch.Tensor,
        convergence: torch.Tensor,
        conditional_score: Any,
    ) -> None:
        if self._cg_tdr_teacher_dump_dir is None:
            return
        self._cg_tdr_teacher_dump_dir.mkdir(parents=True, exist_ok=True)
        seed = int(self._sample_seed) if self._sample_seed is not None else -1
        final_path = self._cg_tdr_teacher_dump_dir / f"seed_{seed}.pt"
        if final_path.exists():
            raise FileExistsError(f"Refusing to overwrite teacher record: {final_path}")
        payload = {
            "schema_version": 1,
            "seed": seed,
            "pos": mean_batch["pos"].detach().cpu(),
            "cell": mean_batch["cell"].detach().cpu(),
            "atomic_numbers": mean_batch["atomic_numbers"].detach().cpu(),
            "num_atoms": mean_batch["num_atoms"].detach().cpu(),
            "batch_index": mean_batch.get_batch_idx("pos").detach().cpu(),
            "node_features": node_features.detach().cpu(),
            "convergence": convergence.detach().cpu(),
            "conditional_score_pos": conditional_score["pos"].detach().cpu(),
            "conditional_score_cell": conditional_score["cell"].detach().cpu(),
            "adaptive_residual": dict(self._last_residual),
        }
        temporary_path = final_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, final_path)

    def _terminal_refine_or_dump(self, mean_batch: Any) -> Any:
        if not self._cg_tdr_enabled and self._cg_tdr_teacher_dump_dir is None:
            return mean_batch
        node_features, conditional_score = self._capture_terminal_features(mean_batch)
        convergence = self._convergence_features(mean_batch)
        self._dump_teacher_record(
            mean_batch=mean_batch,
            node_features=node_features,
            convergence=convergence,
            conditional_score=conditional_score,
        )
        if not self._cg_tdr_enabled:
            return mean_batch

        refiner = self._load_refiner(mean_batch["cell"].device)
        output = refiner(
            node_features=node_features,
            frac_pos=mean_batch["pos"],
            cell=mean_batch["cell"],
            batch_idx=mean_batch.get_batch_idx("pos"),
            convergence=convergence,
            enable_cell=self._cg_tdr_enable_cell,
        )
        self._cg_tdr_metrics = {
            "position_gate_mean": float(output.position_gate.mean().item()),
            "cell_gate_mean": float(output.cell_gate.mean().item()),
            "position_clipping_rate": float(output.position_clipped.float().mean().item()),
            "cell_fallback_rate": float(output.cell_fallback.float().mean().item()),
            "position_residual_rms": float(
                output.position_residual_cart.float().square().mean().sqrt().item()
            ),
        }
        return mean_batch.replace(pos=output.frac_pos, cell=output.cell)
