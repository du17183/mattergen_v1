# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Literal

import fire
from pymatgen.core.structure import Structure

from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_classes import PRETRAINED_MODEL_NAME, MatterGenCheckpointInfo, ProgressCallback
from mattergen.generator import CrystalGenerator


def main(
    output_path: str,
    pretrained_name: PRETRAINED_MODEL_NAME | None = None,
    model_path: str | None = None,
    batch_size: int = 64,
    num_batches: int = 1,
    config_overrides: list[str] | None = None,
    checkpoint_epoch: Literal["best", "last"] | int = "last",
    properties_to_condition_on: TargetProperty | None = None,
    sampling_config_path: str | None = None,
    sampling_config_name: str = "default",
    sampling_config_overrides: list[str] | None = None,
    record_trajectories: bool = True,
    diffusion_guidance_factor: float | None = None,
    strict_checkpoint_loading: bool = True,
    target_compositions: list[dict[str, int]] | None = None,
    progress_callback: ProgressCallback | None = None,
    seed: int | None = None,
    deterministic: bool = False,
    guidance_schedule: str = "constant",
    guidance_warmup_frac: float = 0.1,
    guidance_decay_frac: float = 0.1,
    guidance_min_scale: float = 0.0,
    guidance_max_scale: float = 5.0,
    guidance_adaptive_alpha: float = 0.5,
    guidance_adaptive_ema: float = 0.95,
    guidance_adaptive_eps: float = 1e-6,
    guidance_trace_path: str | None = None,
    guidance_run_id: str | None = None,
    cfg_acceleration_enabled: bool = False,
    cfg_warmup_frac: float = 0.15,
    cfg_convergence_threshold: float = 0.05,
    cfg_consecutive_stable_steps: int = 3,
    cfg_calibration_interval: int = 10,
    cfg_max_reuse_steps: int = 8,
    cfg_extrapolation_enabled: bool = False,
    cfg_extrapolation_order: int = 1,
    cfg_fallback_threshold: float = 0.20,
    cfg_min_progress: float = 0.0,
    cfg_max_progress: float = 1.0,
    cfg_trace_path: str | None = None,
    cfg_trace_mode: str = "auto",
    cfg_summary_path: str | None = None,
    corrector_gating_enabled: bool = False,
    corrector_warmup_frac: float = 0.15,
    corrector_min_progress: float = 0.15,
    corrector_max_progress: float = 0.95,
    corrector_convergence_threshold: float = 0.05,
    corrector_consecutive_stable_steps: int = 3,
    corrector_calibration_interval: int = 10,
    corrector_max_consecutive_skips: int = 8,
    corrector_fallback_threshold: float = 0.20,
    corrector_rescue_enabled: bool = True,
    corrector_budget_aware_enabled: bool = False,
    corrector_max_skip_ratio: float = 1.0,
    corrector_atomic_veto_enabled: bool = False,
    corrector_atomic_stability_threshold: float = 0.05,
    corrector_atomic_min_stable_steps: int = 1,
    corrector_adaptive_calibration_enabled: bool = False,
    corrector_calibration_interval_min: int = 4,
    corrector_calibration_interval_max: int = 16,
    corrector_field_aggregation: str = "all_fields",
    corrector_trace_path: str | None = None,
    corrector_summary_path: str | None = None,
) -> list[Structure]:
    """
    Evaluate diffusion model against molecular metrics.

    Args:
        model_path: Path to DiffusionLightningModule checkpoint directory.
        output_path: Path to output directory.
        config_overrides: Overrides for the model config, e.g., `model.num_layers=3 model.hidden_dim=128`.
        properties_to_condition_on: Property value to draw conditional sampling with respect to. When this value is an empty dictionary (default), unconditional samples are drawn.
        sampling_config_path: Path to the sampling config file. (default: None, in which case we use `DEFAULT_SAMPLING_CONFIG_PATH` from explorers.common.utils.utils.py)
        sampling_config_name: Name of the sampling config (corresponds to `{sampling_config_path}/{sampling_config_name}.yaml` on disk). (default: default)
        sampling_config_overrides: Overrides for the sampling config, e.g., `condition_loader_partial.batch_size=32`.
        load_epoch: Epoch to load from the checkpoint. If None, the best epoch is loaded. (default: None)
        record: Whether to record the trajectories of the generated structures. (default: True)
        strict_checkpoint_loading: Whether to raise an exception when not all parameters from the checkpoint can be matched to the model.
        target_compositions: List of dictionaries with target compositions to condition on. Each dictionary should have the form `{element: number_of_atoms}`. If None, the target compositions are not conditioned on.
           Only supported for models trained for crystal structure prediction (CSP) (default: None)
        progress_callback: Optional callback function that takes in a single float argument representing the progress of the generation process (between 0 and 1).
        seed: Optional seed applied once after model loading and before condition/prior sampling.
        deterministic: Enable strict deterministic PyTorch/cuBLAS/cuDNN execution. (default: False)
        guidance_schedule: constant, piecewise, adaptive, or stage_adaptive. (default: constant)
        guidance_trace_path: Optional absolute CSV path. Existing files are never overwritten.
    NOTE: When specifying dictionary values via the CLI, make sure there is no whitespace between the key and value, e.g., `--properties_to_condition_on={key1:value1}`.
    """
    assert (
        pretrained_name is not None or model_path is not None
    ), "Either pretrained_name or model_path must be provided."
    assert (
        pretrained_name is None or model_path is None
    ), "Only one of pretrained_name or model_path can be provided."

    if deterministic:
        # Set before checkpoint loading or any model CUDA work. CrystalGenerator
        # applies the corresponding PyTorch/cuDNN flags.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sampling_config_overrides = sampling_config_overrides or []
    config_overrides = config_overrides or []
    # Disable generating element types which are not supported or not in the desired chemical
    # system (if provided).
    config_overrides += [
        "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
    ]
    properties_to_condition_on = properties_to_condition_on or {}
    target_compositions = target_compositions or []

    if pretrained_name is not None:
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(
            pretrained_name, config_overrides=config_overrides
        )
    else:
        checkpoint_info = MatterGenCheckpointInfo(
            model_path=Path(model_path).resolve(),
            load_epoch=checkpoint_epoch,
            config_overrides=config_overrides,
            strict_checkpoint_loading=strict_checkpoint_loading,
        )
    _sampling_config_path = Path(sampling_config_path) if sampling_config_path is not None else None
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        properties_to_condition_on=properties_to_condition_on,
        batch_size=batch_size,
        num_batches=num_batches,
        sampling_config_name=sampling_config_name,
        sampling_config_path=_sampling_config_path,
        sampling_config_overrides=sampling_config_overrides,
        record_trajectories=record_trajectories,
        diffusion_guidance_factor=(
            diffusion_guidance_factor if diffusion_guidance_factor is not None else 0.0
        ),
        target_compositions_dict=target_compositions,
        progress_callback=progress_callback,
        seed=seed,
        deterministic=deterministic,
        guidance_schedule=guidance_schedule,
        guidance_warmup_frac=guidance_warmup_frac,
        guidance_decay_frac=guidance_decay_frac,
        guidance_min_scale=guidance_min_scale,
        guidance_max_scale=guidance_max_scale,
        guidance_adaptive_alpha=guidance_adaptive_alpha,
        guidance_adaptive_ema=guidance_adaptive_ema,
        guidance_adaptive_eps=guidance_adaptive_eps,
        guidance_trace_path=guidance_trace_path,
        guidance_run_id=guidance_run_id,
        cfg_acceleration_enabled=cfg_acceleration_enabled,
        cfg_warmup_frac=cfg_warmup_frac,
        cfg_convergence_threshold=cfg_convergence_threshold,
        cfg_consecutive_stable_steps=cfg_consecutive_stable_steps,
        cfg_calibration_interval=cfg_calibration_interval,
        cfg_max_reuse_steps=cfg_max_reuse_steps,
        cfg_extrapolation_enabled=cfg_extrapolation_enabled,
        cfg_extrapolation_order=cfg_extrapolation_order,
        cfg_fallback_threshold=cfg_fallback_threshold,
        cfg_min_progress=cfg_min_progress,
        cfg_max_progress=cfg_max_progress,
        cfg_trace_path=cfg_trace_path,
        cfg_trace_mode=cfg_trace_mode,
        cfg_summary_path=cfg_summary_path,
        corrector_gating_enabled=corrector_gating_enabled,
        corrector_warmup_frac=corrector_warmup_frac,
        corrector_min_progress=corrector_min_progress,
        corrector_max_progress=corrector_max_progress,
        corrector_convergence_threshold=corrector_convergence_threshold,
        corrector_consecutive_stable_steps=corrector_consecutive_stable_steps,
        corrector_calibration_interval=corrector_calibration_interval,
        corrector_max_consecutive_skips=corrector_max_consecutive_skips,
        corrector_fallback_threshold=corrector_fallback_threshold,
        corrector_rescue_enabled=corrector_rescue_enabled,
        corrector_budget_aware_enabled=corrector_budget_aware_enabled,
        corrector_max_skip_ratio=corrector_max_skip_ratio,
        corrector_atomic_veto_enabled=corrector_atomic_veto_enabled,
        corrector_atomic_stability_threshold=(
            corrector_atomic_stability_threshold
        ),
        corrector_atomic_min_stable_steps=(
            corrector_atomic_min_stable_steps
        ),
        corrector_adaptive_calibration_enabled=(
            corrector_adaptive_calibration_enabled
        ),
        corrector_calibration_interval_min=(
            corrector_calibration_interval_min
        ),
        corrector_calibration_interval_max=(
            corrector_calibration_interval_max
        ),
        corrector_field_aggregation=corrector_field_aggregation,
        corrector_trace_path=corrector_trace_path,
        corrector_summary_path=corrector_summary_path,
    )
    return generator.generate(output_dir=Path(output_path))


def _main():
    # use fire instead of argparse to allow for the specification of dictionary values via the CLI
    fire.Fire(main)


if __name__ == "__main__":
    _main()
