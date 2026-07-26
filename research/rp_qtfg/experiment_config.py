from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    config_id: str
    enabled: bool
    guidance_fields: str = "position"
    start_progress: float = 0.75
    position_eta: float = 0.001
    position_radius_angstrom: float = 0.005
    cell_eta_per_gpa: float = 0.000025
    cell_strain_radius: float = 0.0005
    score_ratio_max: float = 0.10
    backtrack_max: int = 3
    conflict_threshold: float = -0.20

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sampling_overrides(self, trace_path: str) -> list[str]:
        if not self.enabled:
            return ["sampler_partial.N=1000"]
        return [
            "sampler_partial._target_=research.rp_qtfg.sampler."
            "RPQTFGGuidedPredictorCorrector.from_pl_module",
            "sampler_partial.N=1000",
            "+sampler_partial.rp_qtfg_enabled=true",
            f"+sampler_partial.rp_qtfg_guidance_fields={self.guidance_fields}",
            f"+sampler_partial.rp_qtfg_start_progress={self.start_progress}",
            f"+sampler_partial.rp_qtfg_position_eta={self.position_eta}",
            "+sampler_partial.rp_qtfg_position_radius_angstrom="
            f"{self.position_radius_angstrom}",
            f"+sampler_partial.rp_qtfg_cell_eta_per_gpa={self.cell_eta_per_gpa}",
            "+sampler_partial.rp_qtfg_cell_strain_radius="
            f"{self.cell_strain_radius}",
            f"+sampler_partial.rp_qtfg_backtrack_max={self.backtrack_max}",
            "+sampler_partial.rp_qtfg_conflict_threshold="
            f"{self.conflict_threshold}",
            f"+sampler_partial.rp_qtfg_score_ratio_max={self.score_ratio_max}",
            f"+sampler_partial.rp_qtfg_trace_path={trace_path}",
        ]


CONFIGS = {
    "A0": ExperimentConfig(config_id="A0", enabled=False),
    "G1_P75_S": ExperimentConfig(
        config_id="G1_P75_S",
        enabled=True,
        guidance_fields="position",
        start_progress=0.75,
        position_eta=0.001,
        position_radius_angstrom=0.005,
        score_ratio_max=0.10,
    ),
    "G1_P60_M": ExperimentConfig(
        config_id="G1_P60_M",
        enabled=True,
        guidance_fields="position",
        start_progress=0.60,
        position_eta=0.0025,
        position_radius_angstrom=0.010,
        score_ratio_max=0.15,
    ),
    "G2_P75_S": ExperimentConfig(
        config_id="G2_P75_S",
        enabled=True,
        guidance_fields="position_cell",
        start_progress=0.75,
        position_eta=0.001,
        position_radius_angstrom=0.005,
        cell_eta_per_gpa=0.000025,
        cell_strain_radius=0.0005,
        score_ratio_max=0.10,
    ),
    "G2_P60_M": ExperimentConfig(
        config_id="G2_P60_M",
        enabled=True,
        guidance_fields="position_cell",
        start_progress=0.60,
        position_eta=0.0025,
        position_radius_angstrom=0.010,
        cell_eta_per_gpa=0.000050,
        cell_strain_radius=0.0010,
        score_ratio_max=0.15,
    ),
}

EIGHT_SEED_CONFIGS = tuple(CONFIGS)
EIGHT_SEEDS = tuple(range(22000, 22008))
THIRTY_TWO_SEEDS = tuple(range(22000, 22032))
