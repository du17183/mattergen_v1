from __future__ import annotations

import torch

from mattergen.common.data.collate import collate
from mattergen.diffusion.sampling.classifier_free_guidance import (
    GuidedPredictorCorrector,
    score_residual_rms,
)
from mattergen.diffusion.sampling.pc_sampler import Diffusable, PredictorCorrector
from research.rp_qtfg.physics_guidance import RPQTFGConfig, RPQTFGEngine


class RPQTFGGuidedPredictorCorrector(GuidedPredictorCorrector):
    """A0 Adaptive CFG with residual-preserving, trust-region CHGNet guidance."""

    def __init__(
        self,
        *,
        rp_qtfg_enabled: bool = False,
        rp_qtfg_guidance_fields: str = "position",
        rp_qtfg_start_progress: float = 0.75,
        rp_qtfg_position_eta: float = 0.01,
        rp_qtfg_position_radius_angstrom: float = 0.02,
        rp_qtfg_cell_eta_per_gpa: float = 0.00025,
        rp_qtfg_cell_strain_radius: float = 0.003,
        rp_qtfg_backtrack_max: int = 3,
        rp_qtfg_conflict_threshold: float = -0.20,
        rp_qtfg_score_ratio_max: float = 0.25,
        rp_qtfg_trace_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        config = RPQTFGConfig(
            enabled=bool(rp_qtfg_enabled),
            guidance_fields=rp_qtfg_guidance_fields,
            start_progress=float(rp_qtfg_start_progress),
            position_eta=float(rp_qtfg_position_eta),
            position_radius_angstrom=float(
                rp_qtfg_position_radius_angstrom
            ),
            cell_eta_per_gpa=float(rp_qtfg_cell_eta_per_gpa),
            cell_strain_radius=float(rp_qtfg_cell_strain_radius),
            backtrack_max=int(rp_qtfg_backtrack_max),
            conflict_threshold=float(rp_qtfg_conflict_threshold),
            score_ratio_max=float(rp_qtfg_score_ratio_max),
        )
        self._rp_qtfg_engine = RPQTFGEngine(
            config=config,
            multi_corruption=self._multi_corruption,
            trace_path=rp_qtfg_trace_path,
            sample_seed=self._sample_seed,
        )

    def _on_sampling_start(self) -> None:
        super()._on_sampling_start()
        self._rp_qtfg_engine.reset()

    def _on_sampling_end(self, error: BaseException | None) -> None:
        self._rp_qtfg_engine.finish(error)
        super()._on_sampling_end(error)

    def _score_fn(
        self,
        x: Diffusable,
        t: torch.Tensor,
    ) -> Diffusable:
        if not self._rp_qtfg_engine.config.enabled:
            return super()._score_fn(x, t)

        batch_no_condition = self._remove_conditioning_fn(x)
        batch_with_condition = self._keep_conditioning_fn(x)
        joint_batch = collate([batch_no_condition, batch_with_condition])
        for attr, value in batch_no_condition.items():
            if isinstance(value, list):
                joint_batch[attr] = (
                    batch_no_condition[attr] + batch_with_condition[attr]
                )
        combined_score = PredictorCorrector._score_fn(
            self,
            x=joint_batch,
            t=torch.cat([t, t], dim=0),
        )
        unconditional_score = combined_score[0]
        conditional_score = combined_score[1]
        fields = tuple(self._multi_corruption.corrupted_fields)
        field_deltas, residual_error = score_residual_rms(
            unconditional_score=unconditional_score,
            conditional_score=conditional_score,
            fields=fields,
        )
        progress, phase = self._current_progress_and_phase()
        decision = self._guidance_controller.evaluate(
            progress=progress,
            phase=phase,
            field_deltas=field_deltas,
            residual_error=residual_error,
        )
        decision_dict = decision.as_dict()
        self._trace_decision(
            t=t,
            field_deltas=field_deltas,
            decision=decision_dict,
        )
        guided_score = unconditional_score.replace(
            **{
                field: torch.lerp(
                    unconditional_score[field],
                    conditional_score[field],
                    decision.final_guidance,
                )
                for field in fields
            }
        )
        return self._rp_qtfg_engine.apply(
            x=x,
            t=t,
            guided_score=guided_score,
            conditional_score=conditional_score,
            unconditional_score=unconditional_score,
            context=self.sampling_context,
        )
