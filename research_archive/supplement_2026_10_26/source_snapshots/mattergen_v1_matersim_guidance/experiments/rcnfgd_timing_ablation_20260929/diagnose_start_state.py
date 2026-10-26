"""Seed-740000-only, one-forward diagnostic; never performs a denoising trajectory."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from hydra.utils import instantiate

if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator
from mattergen.diffusion.sampling.pc_sampler import _sample_prior


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")
ELEMENT_MASK_OVERRIDE = (
    "++lightning_module.diffusion_module.model.element_mask_func="
    "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
)
FIELDS = ("atomic_numbers", "pos", "cell", "num_atoms", "dft_mag_density")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_summary(batch) -> dict:
    result = {}
    for name in FIELDS:
        try:
            value = batch[name]
        except (KeyError, AttributeError):
            continue
        if not isinstance(value, torch.Tensor):
            continue
        host = value.detach().contiguous().cpu()
        result[name] = {
            "shape": list(host.shape), "dtype": str(host.dtype),
            "sha256": digest_bytes(host.numpy().tobytes()),
        }
    return result


def rng_summary() -> dict:
    return {
        "python": digest_bytes(repr(random.getstate()).encode()),
        "numpy": digest_bytes(repr(np.random.get_state()).encode()),
        "torch_cpu": digest_bytes(torch.get_rng_state().numpy().tobytes()),
        "torch_cuda_visible_0": digest_bytes(torch.cuda.get_rng_state(0).cpu().numpy().tobytes()),
    }


def model_state_summary(model) -> dict:
    digest = hashlib.sha256()
    state = model.state_dict()
    for name, value in sorted(state.items()):
        host = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(host.dtype).encode())
        digest.update(str(tuple(host.shape)).encode())
        digest.update(host.numpy().tobytes())
    return {"tensor_count": len(state), "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", choices=("historical", "replay"), required=True)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    if args.entry == "historical":
        from experiments.innovation2_finalization import generate_frozen_pair as entry_module
        factory_function = entry_module.frozen.instantiate
    else:
        from experiments.rcnfgd_timing_ablation_20260929 import replay_740000 as entry_module
        factory_function = instantiate

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(MODEL_ROOT), config_overrides=[ELEMENT_MASK_OVERRIDE]),
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={"dft_mag_density": 0.2},
        diffusion_guidance_factor=2.0,
        guidance_schedule="constant",
        seed=740000,
        deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    model_state = model_state_summary(generator.model)
    config = generator.load_sampling_config(batch_size=1, num_batches=1)
    sampler = factory_function(config.sampler_partial)(pl_module=generator.model)
    generator.seed = 740000
    generator._seed_sampling_rngs()
    output = {"entry": args.entry, "seed": 740000,
              "model_state": model_state,
              "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
              "omp_threads_env": os.environ.get("OMP_NUM_THREADS"),
              "torch_version": torch.__version__,
              "rng_after_reseed": rng_summary()}
    condition, mask = next(iter(generator.get_condition_loader(config)))
    output["condition"] = tensor_summary(condition)
    output["rng_after_condition"] = rng_summary()
    with torch.no_grad():
        sampler._diffusion_module.eval()
        sampler._on_sampling_start()
        condition = condition.to(sampler._device)
        sampler._on_before_sample_prior(condition)
        prior = _sample_prior(sampler._multi_corruption, condition, mask=mask or {})
        output["prior"] = tensor_summary(prior)
        output["rng_after_prior"] = rng_summary()
        t = torch.full((prior.get_batch_size(),), 1.0, device=sampler._device)
        score = sampler._score_fn(prior, t)
        output["first_score"] = tensor_summary(score)
        output["rng_after_first_score"] = rng_summary()
    dest = ROOT / "diagnostics"
    dest.mkdir(exist_ok=True)
    label = args.label or args.entry
    path = dest / f"{label}_start_state.json"
    score_path = dest / f"{label}_first_score.npz"
    if path.exists() or score_path.exists():
        raise FileExistsError(f"Diagnostic output already exists: {path} or {score_path}")
    np.savez_compressed(score_path, **{
        name: score[name].detach().contiguous().cpu().numpy()
        for name in ("atomic_numbers", "pos", "cell")
    })
    output["first_score_npz"] = str(score_path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"entry": args.entry, "output": str(path),
                      "condition": output["condition"], "prior": output["prior"],
                      "first_score": output["first_score"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
