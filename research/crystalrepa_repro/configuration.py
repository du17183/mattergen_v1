from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import open_dict

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo

CHECKPOINT_ROOT = Path("/data/dxl/checkpoints/official/hf_mattergen/checkpoints/mp_20_base")
CHECKPOINT = CHECKPOINT_ROOT / "checkpoints/last.ckpt"
CHECKPOINT_SHA256 = "ffb80e4425a6f99f479a67b8cd111885d45117234e8947ff77eb3a55df420b9a"
PROJECTION_KEY = "diffusion_module.model.student_projection."


def crystalrepa_lightning_config(base_config, learning_rate: float = 1e-4):
    config = deepcopy(base_config.lightning_module)
    with open_dict(config.diffusion_module):
        config.diffusion_module._target_ = "mattergen.crystalrepa.diffusion.RepaDiffusionModule"
    with open_dict(config.diffusion_module.model):
        model = config.diffusion_module.model
        model._target_ = "mattergen.crystalrepa.model.CrystalRepaDenoiser"
        model.alignment_block = 2
        model.teacher_feature_dim = 64
        model.alignment_weight = 1.0
        model.alignment_temperature = 0.1
        model.alignment_enabled = True
        model.inference_only = False
    with open_dict(config.optimizer_partial):
        config.optimizer_partial.lr = learning_rate
    # torch>=2.7 removed the deprecated ReduceLROnPlateau(verbose=...) argument.
    for scheduler_config in config.scheduler_partials:
        with open_dict(scheduler_config.scheduler):
            scheduler_config.scheduler.pop("verbose", None)
    return config


def initialize_training_model(learning_rate: float = 1e-4):
    info = MatterGenCheckpointInfo(str(CHECKPOINT_ROOT))
    if Path(info.checkpoint_path).resolve() != CHECKPOINT.resolve():
        raise RuntimeError(f"Unexpected MP-20 checkpoint: {info.checkpoint_path}")
    model = instantiate(crystalrepa_lightning_config(info.config, learning_rate))
    checkpoint = torch.load(info.checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    expected = {
        f"{PROJECTION_KEY}norm.weight",
        f"{PROJECTION_KEY}norm.bias",
        f"{PROJECTION_KEY}residual.weight",
        f"{PROJECTION_KEY}residual.bias",
        f"{PROJECTION_KEY}output.weight",
        f"{PROJECTION_KEY}output.bias",
    }
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected U0→R1 incompatibility: {incompatible}")
    return model, info, checkpoint, incompatible


def load_r1_as_inference_model(checkpoint_path: str | Path, device: torch.device):
    """Load a trained R1 backbone into the unmodified official inference architecture."""
    info = MatterGenCheckpointInfo(str(CHECKPOINT_ROOT))
    model = instantiate(info.config.lightning_module)
    trained = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {
        key: value
        for key, value in trained["state_dict"].items()
        if not key.startswith(PROJECTION_KEY)
    }
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"R1 inference checkpoint mismatch: {incompatible}")
    forbidden_modules = [
        name
        for name, module in model.named_modules()
        if "student_projection" in name.lower()
        or "teacher" in name.lower()
        or type(module).__module__.startswith("mattergen.crystalrepa")
    ]
    if forbidden_modules:
        raise RuntimeError("Training-only Teacher/projection module present at inference")
    return model.to(device).eval()
