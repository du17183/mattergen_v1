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
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.globals import get_device
from mattergen.generator import CrystalGenerator
from research.crystalrepa_repro.common import atomic_json, now, sha256_file
from research.crystalrepa_repro.configuration import CHECKPOINT, CHECKPOINT_ROOT, CHECKPOINT_SHA256, load_r1_as_inference_model

INFERENCE = Path("/data/dxl/results/crystalrepa_repro/inference/r1_inference.ckpt")
FIELDS = ("atomic_numbers", "cell", "pos")


def safe_composition_validity(structure, checker=None) -> tuple[bool, str | None]:
    if checker is None:
        from mattergen.evaluation.metrics.structure import is_smact_valid

        checker = is_smact_valid
    try:
        return bool(checker(structure)), None
    except TypeError as error:
        # Missing SMACT oxidation-state data makes the composition invalid for
        # this metric, but does not invalidate an otherwise complete inference.
        return False, f"{type(error).__name__}: {error}"


def hash_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def hash_graph(batch) -> dict[str, str]:
    fields = {field: hash_tensor(batch[field]) for field in FIELDS}
    fields["combined"] = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return fields


def hash_rng_state() -> str:
    payload = {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        "torch_cpu": hash_tensor(torch.get_rng_state()),
        "torch_cuda": [hash_tensor(state) for state in torch.cuda.get_rng_state_all()],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("U0", "R1"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--repeat-index", type=int, default=1)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "status.json"
    if existing.exists() and json.loads(existing.read_text()).get("success") is True:
        return 0
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    overrides = ["++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"]
    checkpoint_info = MatterGenCheckpointInfo(model_path=CHECKPOINT_ROOT, load_epoch="last", config_overrides=overrides)
    config = {
        "created_at": now(), "method": args.method, "seed": args.seed,
        "repeat_index": args.repeat_index, "physical_gpu": args.physical_gpu,
        "sampling_steps": 1000, "condition_fields": [], "cfg": False,
        "adaptive_cfg": False, "predictor_corrector": "official full flow",
        "official_checkpoint_sha256": sha256_file(CHECKPOINT),
        "r1_checkpoint": str(INFERENCE) if args.method == "R1" else None,
        "r1_checkpoint_sha256": sha256_file(INFERENCE) if args.method == "R1" else None,
        "teacher_used_at_inference": False, "projection_loaded_at_inference": False,
    }
    atomic_json(output / "run_config.json", config)
    if config["official_checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("Official checkpoint SHA mismatch")
    success, error = False, None
    initial: dict[str, str] = {}
    rng_before_prior = None
    try:
        generator = CrystalGenerator(
            checkpoint_info=checkpoint_info, batch_size=1, num_batches=1,
            num_atoms_distribution="ALEX_MP_20", diffusion_guidance_factor=0.0,
            properties_to_condition_on=None, seed=args.seed, deterministic=True,
            guidance_schedule="constant", record_trajectories=False,
        )
        sampling = generator.load_sampling_config(batch_size=1, num_batches=1)
        sampler_config = sampling.sampler_partial
        if int(sampler_config.N) != 1000:
            raise RuntimeError(f"Expected official 1000 sampling steps, got {sampler_config.N}")
        if int(sampler_config.n_steps_corrector) != 1:
            raise RuntimeError("Official full corrector flow requires one corrector step")
        if float(sampler_config.guidance_scale) != 0.0:
            raise RuntimeError("Unconditional reproduction requires guidance_scale=0")
        if str(sampler_config.guidance_schedule) != "constant":
            raise RuntimeError("Adaptive CFG must be disabled")
        config["actual_sampling_config"] = {
            "N": int(sampler_config.N),
            "n_steps_corrector": int(sampler_config.n_steps_corrector),
            "guidance_scale": float(sampler_config.guidance_scale),
            "guidance_schedule": str(sampler_config.guidance_schedule),
            "predictor_fields": sorted(sampler_config.predictor_partials.keys()),
            "corrector_fields": sorted(sampler_config.corrector_partials.keys()),
        }
        atomic_json(output / "run_config.json", config)
        load_started = time.monotonic()
        if args.method == "R1":
            generator._model = load_r1_as_inference_model(INFERENCE, get_device())
            generator._cfg = checkpoint_info.config
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
            started = time.monotonic()
            structures = generator.generate(output_dir=str(output))
            torch.cuda.synchronize()
            generation_seconds = time.monotonic() - started
        finally:
            PredictorCorrector._on_before_sample_prior = original_before
            PredictorCorrector._on_after_sample_prior = original_after
        frames = ase.io.read(output / "generated_crystals.extxyz", ":")
        frames = frames if isinstance(frames, list) else [frames]
        if len(frames) != 1 or len(structures) != 1:
            raise RuntimeError("Expected exactly one generated structure")
        atoms = frames[0]
        arrays = (np.asarray(atoms.numbers), np.asarray(atoms.positions), np.asarray(atoms.cell.array))
        if not all(np.isfinite(value).all() for value in arrays):
            raise RuntimeError("Generated structure contains NaN/Inf")
        final = hashlib.sha256()
        for value in arrays:
            final.update(np.ascontiguousarray(value).tobytes())
        from mattergen.evaluation.metrics.structure import structure_validity
        structure = AseAtomsAdaptor.get_structure(atoms)
        composition_valid, composition_validity_error = safe_composition_validity(
            structure
        )
        summary = {
            "success": True, "method": args.method, "seed": args.seed,
            "physical_gpu": args.physical_gpu, "model_load_seconds": model_load_seconds,
            "generation_seconds": generation_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "rng_before_prior_hash": rng_before_prior, "initial_hashes": initial,
            "final_structure_hash": final.hexdigest(), "extxyz_sha256": sha256_file(output / "generated_crystals.extxyz"),
            "formula": atoms.get_chemical_formula(), "num_atoms": len(atoms),
            "structure_valid": bool(structure_validity(structure)),
            "composition_valid": composition_valid,
            "composition_validity_error": composition_validity_error,
            "teacher_used_at_inference": False, "projection_loaded_at_inference": False,
        }
        atomic_json(output / "summary.json", summary)
        success = True
    except BaseException:
        error = traceback.format_exc()
        (output / "error.txt").write_text(error, encoding="utf-8")
    atomic_json(output / "status.json", {"success": success, "error": error, "finished_at": now()})
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
