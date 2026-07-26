"""Generate one A0 or P1 structure for the paired FN-PRA Phase-1 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path

import ase.io
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import open_dict
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.globals import get_device
from mattergen.generator import CrystalGenerator
from research.fn_pra.phase1_common import atomic_json, now, sha256_file
from research.fn_pra.validate_v1_integration import repa_config


CHECKPOINT_ROOT = Path(
    "/data/dxl/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
)
OFFICIAL_CHECKPOINT = CHECKPOINT_ROOT / "checkpoints/last.ckpt"
OFFICIAL_SHA = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
P1_CHECKPOINT = Path(
    "/data/dxl/results/fn_pra/phase1/training/v1_decision/checkpoints/"
    "best-step=003000-loss_val=0.297921.ckpt"
)
P1_SHA = "68be996761ad079917b4e7c63e76b2acc7587de1f1fc28f8708325fbe76f17d7"
FIELDS = ("atomic_numbers", "cell", "pos")


def hash_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def hash_graph(batch) -> dict[str, str]:
    fields = {field: hash_tensor(batch[field]) for field in FIELDS}
    fields["combined"] = hashlib.sha256(
        json.dumps(fields, sort_keys=True).encode()
    ).hexdigest()
    return fields


def hash_rng_state() -> str:
    payload = {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        "torch_cpu": hash_tensor(torch.get_rng_state()),
        "torch_cuda": [hash_tensor(state) for state in torch.cuda.get_rng_state_all()],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_p1_model(checkpoint_info: MatterGenCheckpointInfo):
    config = repa_config(checkpoint_info.config)
    with open_dict(config.diffusion_module.model):
        config.diffusion_module.model.inference_only = True
    model = instantiate(config)
    checkpoint = torch.load(P1_CHECKPOINT, map_location="cpu")
    inference_state = {
        key: value
        for key, value in checkpoint["state_dict"].items()
        if "student_projection" not in key and "teacher_projection" not in key
    }
    incompatible = model.load_state_dict(inference_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"P1 inference checkpoint mismatch: {incompatible}")
    model = model.to(get_device()).eval()
    adapter = model.diffusion_module.model
    if adapter.student_projection is not None or adapter.teacher_projection is not None:
        raise RuntimeError("training-only projection heads were allocated during P1 inference")
    teacher_modules = [
        name
        for name, module in model.named_modules()
        if "teacher" in name.lower() or "chgnet" in type(module).__name__.lower()
    ]
    if teacher_modules:
        raise RuntimeError(f"Teacher-related inference modules found: {teacher_modules}")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("A0", "P1"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--repeat-index", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    overrides = [
        "++lightning_module.diffusion_module.model.element_mask_func="
        "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
    ]
    if args.sampling_steps != 1000:
        overrides.append(
            "lightning_module.diffusion_module.corruption.discrete_corruptions."
            f"atomic_numbers.d3pm.schedule.num_steps={args.sampling_steps}"
        )
    checkpoint_info = MatterGenCheckpointInfo(
        model_path=CHECKPOINT_ROOT,
        load_epoch="last",
        config_overrides=overrides,
    )
    config = {
        "created_at": now(),
        "method": args.method,
        "seed": args.seed,
        "repeat_index": args.repeat_index,
        "physical_gpu": args.physical_gpu,
        "sampling_steps": args.sampling_steps,
        "target": {"dft_mag_density": 0.10},
        "base_guidance": 2.0,
        "guidance_schedule": "adaptive",
        "adaptive_alpha": 0.50,
        "adaptive_ema": 0.95,
        "adaptive_epsilon": 1e-6,
        "guidance_min_scale": 0.0,
        "guidance_max_scale": 5.0,
        "official_checkpoint_sha256": sha256_file(OFFICIAL_CHECKPOINT),
        "p1_checkpoint": str(P1_CHECKPOINT) if args.method == "P1" else None,
        "p1_checkpoint_sha256": sha256_file(P1_CHECKPOINT) if args.method == "P1" else None,
        "teacher_used_at_inference": False,
        "projection_heads_loaded_at_inference": False,
    }
    atomic_json(output / "run_config.json", config)
    if config["official_checkpoint_sha256"] != OFFICIAL_SHA:
        raise RuntimeError("official checkpoint SHA mismatch")
    if args.method == "P1" and config["p1_checkpoint_sha256"] != P1_SHA:
        raise RuntimeError("P1 checkpoint SHA mismatch")

    success = False
    error = None
    initial = {}
    rng_before_prior = None
    model_load_seconds = None
    generation_seconds = None
    cpu_seconds = None
    try:
        generator = CrystalGenerator(
            checkpoint_info=checkpoint_info,
            batch_size=1,
            num_batches=1,
            num_atoms_distribution="ALEX_MP_20",
            diffusion_guidance_factor=2.0,
            properties_to_condition_on={"dft_mag_density": 0.10},
            seed=args.seed,
            deterministic=True,
            guidance_schedule="adaptive",
            guidance_min_scale=0.0,
            guidance_max_scale=5.0,
            guidance_adaptive_alpha=0.50,
            guidance_adaptive_ema=0.95,
            guidance_adaptive_eps=1e-6,
            sampling_config_overrides=[f"sampler_partial.N={args.sampling_steps}"],
            record_trajectories=False,
        )
        load_started = time.monotonic()
        if args.method == "P1":
            generator._model = load_p1_model(checkpoint_info)
        else:
            generator.prepare()
        model_load_seconds = time.monotonic() - load_started

        from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector

        original_before = PredictorCorrector._on_before_sample_prior
        original_after = PredictorCorrector._on_after_sample_prior

        def before_prior(self, conditioning_data):
            nonlocal rng_before_prior
            original_before(self, conditioning_data)
            rng_before_prior = hash_rng_state()

        def after_prior(self, batch):
            original_after(self, batch)
            initial.update(hash_graph(batch))

        PredictorCorrector._on_before_sample_prior = before_prior
        PredictorCorrector._on_after_sample_prior = after_prior
        try:
            torch.cuda.reset_peak_memory_stats()
            cpu_started = time.process_time()
            generation_started = time.monotonic()
            structures = generator.generate(output_dir=str(output))
            torch.cuda.synchronize()
            generation_seconds = time.monotonic() - generation_started
            cpu_seconds = time.process_time() - cpu_started
        finally:
            PredictorCorrector._on_before_sample_prior = original_before
            PredictorCorrector._on_after_sample_prior = original_after

        frames = ase.io.read(output / "generated_crystals.extxyz", ":")
        if not isinstance(frames, list):
            frames = [frames]
        if len(frames) != 1 or len(structures) != 1:
            raise RuntimeError("expected exactly one generated structure")
        atoms = frames[0]
        arrays = (
            np.asarray(atoms.numbers),
            np.asarray(atoms.positions),
            np.asarray(atoms.cell.array),
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise RuntimeError("generated structure contains NaN/Inf")
        final_digest = hashlib.sha256()
        for array in arrays:
            final_digest.update(np.ascontiguousarray(array).tobytes())
        from mattergen.evaluation.metrics.structure import is_smact_valid, structure_validity

        structure = AseAtomsAdaptor.get_structure(atoms)
        summary = {
            "success": True,
            "method": args.method,
            "seed": args.seed,
            "physical_gpu": args.physical_gpu,
            "model_load_seconds": model_load_seconds,
            "generation_seconds": generation_seconds,
            "cpu_seconds": cpu_seconds,
            "cpu_percent_one_core": 100.0 * cpu_seconds / generation_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "rng_before_prior_hash": rng_before_prior,
            "initial_hashes": initial,
            "final_structure_hash": final_digest.hexdigest(),
            "formula": atoms.get_chemical_formula(),
            "num_atoms": len(atoms),
            "structure_valid": bool(structure_validity(structure)),
            "composition_valid": bool(is_smact_valid(structure)),
            "teacher_used_at_inference": False,
            "projection_heads_loaded_at_inference": False,
        }
        atomic_json(output / "summary.json", summary)
        success = True
    except BaseException:
        error = traceback.format_exc()
        (output / "error.txt").write_text(error, encoding="utf-8")
    atomic_json(
        output / "status.json",
        {"success": success, "error": error, "finished_at": now()},
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
