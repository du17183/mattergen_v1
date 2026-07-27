from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate

from mattergen.common.data.collate import collate
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.diffusion.sampling.pc_sampler import _sample_prior
from mattergen.generator import CrystalGenerator
from research.mps_fastgate.common import RESULTS, atomic_json, configure_environment, now


CHECKPOINT = Path("/data/dxl/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")


def digest(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    result = hashlib.sha256()
    result.update(str(cpu.dtype).encode())
    result.update(json.dumps(list(cpu.shape)).encode())
    result.update(cpu.numpy().tobytes())
    return result.hexdigest()


def main() -> None:
    configure_environment()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    seed = 27000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    checkpoint = MatterGenCheckpointInfo(
        model_path=CHECKPOINT,
        load_epoch="last",
        strict_checkpoint_loading=True,
    )
    generator = CrystalGenerator(
        checkpoint_info=checkpoint,
        properties_to_condition_on={"dft_mag_density": 0.10},
        batch_size=1,
        num_batches=1,
        diffusion_guidance_factor=2.0,
        seed=seed,
        deterministic=True,
        guidance_schedule="constant",
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator._seed_sampling_rngs()
    sampling = generator.load_sampling_config(batch_size=1, num_batches=1)
    condition_loader = generator.get_condition_loader(sampling)
    sampler = instantiate(sampling.sampler_partial)(pl_module=generator.model)
    conditioning, mask = next(iter(condition_loader))
    conditioning = conditioning.to(generator.model.device)
    mask = {key: value.to(generator.model.device) for key, value in (mask or {}).items()}
    batch = _sample_prior(sampler._multi_corruption, conditioning, mask=mask)
    timestep = torch.full((batch.get_batch_size(),), 0.50, device=generator.model.device)
    unconditional = sampler._remove_conditioning_fn(batch)
    conditional = sampler._keep_conditioning_fn(batch)
    joint = collate([unconditional, conditional])
    for attribute, value in unconditional.items():
        if isinstance(value, list):
            joint[attribute] = unconditional[attribute] + conditional[attribute]
    joint_timestep = torch.cat([timestep, timestep], dim=0)
    with torch.inference_mode():
        score = sampler.diffusion_module.score_fn(joint, joint_timestep)
    torch.cuda.synchronize()
    fields = {
        name: digest(score[name])
        for name in ("atomic_numbers", "pos", "cell")
    }
    atomic_json(
        RESULTS / "smoke/mattergen_forward.json",
        {
            "completed_at": now(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_cuda": torch.version.cuda,
            "active_thread_percentage": 50,
            "score_hashes": fields,
            "finite": all(torch.isfinite(score[name]).all().item() for name in fields),
        },
    )


if __name__ == "__main__":
    main()
