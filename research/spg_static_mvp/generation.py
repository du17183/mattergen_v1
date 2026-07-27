from __future__ import annotations

import os

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import open_dict

from mattergen.common.data.collate import collate
from mattergen.common.data.condition_factory import NumAtomsCrystalDataset
from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from mattergen.common.data.transform import SetProperty
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator
from research.spg_static_mvp.common import (
    CHECKPOINT_ROOT,
    CHECKPOINT_SHA256,
    sha256_file,
)


def configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")


def make_condition(seed: int):
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        dataset = NumAtomsCrystalDataset.from_num_atoms_distribution(
            num_atoms_distribution=NUM_ATOMS_DISTRIBUTIONS["ALEX_MP_20"],
            num_samples=1,
            transforms=[SetProperty("dft_mag_density", 0.10)],
        )
        return dataset[0]
    finally:
        np.random.set_state(numpy_state)


def build_c0_generator(*, sampling_steps: int = 1000) -> CrystalGenerator:
    checkpoint = CHECKPOINT_ROOT / "checkpoints/last.ckpt"
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("official dft_mag_density checkpoint SHA256 mismatch")
    overrides = [
        "++lightning_module.diffusion_module.model.element_mask_func="
        "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
    ]
    if sampling_steps != 1000:
        overrides.append(
            "lightning_module.diffusion_module.corruption.discrete_corruptions."
            f"atomic_numbers.d3pm.schedule.num_steps={sampling_steps}"
        )
    checkpoint_info = MatterGenCheckpointInfo(
        model_path=CHECKPOINT_ROOT,
        load_epoch="last",
        config_overrides=overrides,
    )
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=1,
        num_batches=1,
        num_atoms_distribution="ALEX_MP_20",
        diffusion_guidance_factor=2.0,
        properties_to_condition_on={"dft_mag_density": 0.10},
        deterministic=True,
        guidance_schedule="constant",
        sampling_config_overrides=[f"sampler_partial.N={sampling_steps}"],
        record_trajectories=False,
    )
    generator.prepare()
    return generator


def build_recording_sampler(generator: CrystalGenerator, seed: int):
    config = generator.load_sampling_config(batch_size=1, num_batches=1)
    with open_dict(config.sampler_partial):
        config.sampler_partial._target_ = (
            "research.spg_static_mvp.recording_sampler."
            "ShapeRecordingGuidedPredictorCorrector.from_pl_module"
        )
        config.sampler_partial.trajectory_seed = int(seed)
        config.sampler_partial.guidance_schedule = "constant"
    return instantiate(config.sampler_partial)(pl_module=generator.model)


def singleton_condition(seed: int):
    return collate([make_condition(seed)])


def find_gemnet(model):
    for module in model.modules():
        if module.__class__.__name__ in {"GemNetT", "GemNetTCtrl"}:
            return module
    raise RuntimeError("GemNet score module not found")
