from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import open_dict

from mattergen.common.data.collate import collate
from mattergen.common.data.condition_factory import NumAtomsCrystalDataset
from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from mattergen.common.data.transform import SetProperty
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator
from research.gemnet_fused_fastgate.common import (
    CHECKPOINT,
    CHECKPOINT_ROOT,
    CHECKPOINT_SHA256,
    configure_environment,
    sha256_file,
)


SOURCE_RESULTS = Path("/data/dxl/results/spg_static_mvp")


def configure_determinism() -> None:
    configure_environment()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


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
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
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


def build_sampler(generator: CrystalGenerator):
    config = generator.load_sampling_config(batch_size=1, num_batches=1)
    with open_dict(config.sampler_partial):
        config.sampler_partial.guidance_schedule = "constant"
    return instantiate(config.sampler_partial)(pl_module=generator.model)


def find_gemnet(model):
    for module in model.modules():
        if module.__class__.__name__ in {"GemNetT", "GemNetTCtrl"}:
            return module
    raise RuntimeError("GemNet score module not found")


def joint_batch(sampler, batch):
    unconditional = sampler._remove_conditioning_fn(batch)
    conditional = sampler._keep_conditioning_fn(batch)
    joint = collate([unconditional, conditional])
    for attr, value in unconditional.items():
        if isinstance(value, list):
            joint[attr] = unconditional[attr] + conditional[attr]
    return joint


def select_state_records(count: int) -> list[dict]:
    paths = sorted((SOURCE_RESULTS / "shape_states").glob("seed_*/shape_statistics.csv"))
    if not paths:
        raise FileNotFoundError("saved C0 real-state statistics are unavailable")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.sort_values(
        ["num_atoms", "triplet_count", "seed", "state_index"]
    ).reset_index(drop=True)
    indices = np.linspace(0, len(frame) - 1, count, dtype=np.int64)
    return frame.iloc[indices].to_dict("records")


def load_states(count: int) -> list[dict]:
    records = select_state_records(count)
    cache: dict[int, list[dict]] = {}
    states = []
    for record in records:
        seed = int(record["seed"])
        if seed not in cache:
            cache[seed] = torch.load(
                SOURCE_RESULTS / f"shape_states/seed_{seed}/states.pt",
                map_location="cpu",
                weights_only=False,
            )
        state = dict(cache[seed][int(record["state_index"])])
        state["edge_count"] = int(record["edge_count"])
        state["triplet_count"] = int(record["triplet_count"])
        states.append(state)
    return states


def prepare_joint_states(states: list[dict], sampler, device: torch.device) -> list[dict]:
    prepared = []
    for state in states:
        seed = int(state["seed"])
        batch = collate([make_condition(seed)]).replace(
            pos=state["pos"],
            cell=state["cell"],
            atomic_numbers=state["atomic_numbers"],
            num_atoms=state["num_atoms"],
        ).to(device)
        timestep = torch.tensor([state["t"]], dtype=torch.float32, device=device)
        prepared.append(
            {
                "seed": seed,
                "state_index": int(state["state_index"]),
                "sampling_step": int(state["sampling_step"]),
                "phase": str(state["phase"]),
                "num_atoms": int(state["num_atoms"].sum()),
                "edge_count": int(state["edge_count"]),
                "triplet_count": int(state["triplet_count"]),
                "joint_batch": joint_batch(sampler, batch),
                "joint_timestep": torch.cat([timestep, timestep], dim=0),
            }
        )
    return prepared


def run_joint_score(diffusion_module, sampler, state: dict):
    combined = diffusion_module.score_fn(
        state["joint_batch"], state["joint_timestep"]
    )
    unconditional = combined[0]
    conditional = combined[1]
    return unconditional.replace(
        **{
            field: torch.lerp(
                unconditional[field],
                conditional[field],
                sampler._guidance_scale,
            )
            for field in sampler._multi_corruption.corrupted_fields
        }
    )
