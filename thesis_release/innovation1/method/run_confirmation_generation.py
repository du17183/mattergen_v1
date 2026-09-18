"""Generate frozen deployment-mode paired confirmatory outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from ase.io import write
from hydra.utils import instantiate
import numpy as np
import torch

if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.diffusion.sampling.field_decoupled_cfg import normalize_policies
from mattergen.generator import CrystalGenerator
from experiments.frozen_linear_k2_confirmation.confirmatory_sampler import FrozenLinearK2ConfirmSampler


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/frozen_linear_k2_confirmation"
PROTOCOL_ROOT = ROOT / "protocol"
MODEL_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")
MODEL_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
ELEMENT_MASK_OVERRIDE = "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
SAMPLER_OVERRIDE = "sampler_partial._target_=experiments.frozen_linear_k2_confirmation.confirmatory_sampler.FrozenLinearK2ConfirmSampler.from_pl_module"
TARGET = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policies() -> tuple[dict, ...]:
    return normalize_policies([
        {"policy_id": "C0", "kind": "constant", "scales": {"atomic": 2.0, "pos": 2.0, "cell": 2.0}},
        {"policy_id": "GPulse", "kind": "pulse", "start": 400, "duration": 100, "scales": {"atomic": 1.9, "pos": 1.9, "cell": 1.9}},
        {"policy_id": "APulse", "kind": "pulse", "start": 400, "duration": 100, "scales": {"atomic": 1.9, "pos": 2.0, "cell": 2.0}},
        {"policy_id": "PPulse", "kind": "pulse", "start": 400, "duration": 100, "scales": {"atomic": 2.0, "pos": 1.9, "cell": 2.0}},
        {"policy_id": "CPulse", "kind": "pulse", "start": 400, "duration": 100, "scales": {"atomic": 2.0, "pos": 2.0, "cell": 1.9}},
    ])


def seed_manifest() -> dict[str, Any]:
    return json.loads((PROTOCOL_ROOT / "seed_manifest_256.json").read_text())


def phase_seeds(phase: str) -> tuple[int, ...]:
    key = "c1_128" if phase == "c1_128" else "c2_128"
    return tuple(map(int, seed_manifest()[key]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def configure_runtime() -> None:
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> CrystalGenerator:
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(MODEL_ROOT), config_overrides=[ELEMENT_MASK_OVERRIDE]),
        batch_size=1, num_batches=1, properties_to_condition_on={"dft_mag_density": TARGET},
        diffusion_guidance_factor=2.0, guidance_schedule="constant",
        sampling_config_overrides=[SAMPLER_OVERRIDE], seed=seed, deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode(); generator.prepare(); generator.model.eval()
    for parameter in generator.model.parameters(): parameter.requires_grad_(False)
    return generator


def sample_once(generator: CrystalGenerator, seed: int) -> tuple[FrozenLinearK2ConfirmSampler, Any, float]:
    manifest = json.loads((PROTOCOL_ROOT / "frozen_method_manifest.json").read_text())
    seeds = seed_manifest()
    generator.seed = seed; generator._seed_sampling_rngs()
    sampling = generator.load_sampling_config(batch_size=1, num_batches=1)
    if int(sampling.sampler_partial.N) != 1000 or int(sampling.sampler_partial.n_steps_corrector) != 1:
        raise RuntimeError("Predictor/Corrector configuration changed")
    artifact = PROTOCOL_ROOT / "frozen_linear_k2_pipeline.joblib"
    parameters = PROTOCOL_ROOT / "frozen_linear_k2_parameters.json"
    if sha256(parameters) != manifest["allocator"]["parameters_sha256"]:
        raise RuntimeError("frozen allocator parameter hash mismatch")
    sampler = instantiate(sampling.sampler_partial)(
        pl_module=generator.model, sample_seed=seed, policy_specs=policies(), branch_point=400,
        feature_ema_beta=0.95, feature_windows=(25, 50, 100),
        artifact_path=str(artifact), parameters_path=str(parameters),
        expected_artifact_sha256=manifest["allocator"]["artifact_sha256"],
        random_policies=seeds["random_k2_assignments"][str(seed)],
    )
    if not isinstance(sampler, FrozenLinearK2ConfirmSampler): raise TypeError(type(sampler))
    batches = list(generator.get_condition_loader(sampling))
    if len(batches) != 1: raise RuntimeError("expected one condition batch")
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
    _, returned_mean = sampler.sample(*batches[0]); torch.cuda.synchronize()
    return sampler, returned_mean, time.perf_counter() - started


def run(phase: str, seeds: tuple[int, ...]) -> None:
    registered = set(phase_seeds(phase))
    if not seeds or not set(seeds).issubset(registered): raise ValueError("worker received unregistered phase seed")
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA256: raise RuntimeError("MatterGen checkpoint hash mismatch")
    configure_runtime(); generator = make_generator(seeds[0]); phase_root = ROOT / phase
    for seed in seeds:
        destination = phase_root / "generation" / str(seed); partial = destination.with_name(destination.name + ".partial")
        if (destination / "run_summary.json").exists(): print(json.dumps({"seed": seed, "event": "already_complete"}), flush=True); continue
        if destination.exists() or partial.exists(): raise FileExistsError(f"retained incomplete output: {destination} or {partial}")
        partial.mkdir(parents=True)
        sampler, returned_mean, elapsed = sample_once(generator, seed)
        if len(sampler.prebranch_trace) != 800:
            raise RuntimeError(f"frozen prefix trace row count changed: {len(sampler.prebranch_trace)}")
        feature_columns = json.loads((PROTOCOL_ROOT / "frozen_linear_k2_parameters.json").read_text())["feature_columns"]
        if len(feature_columns) != 179 or any(name not in sampler.branch_features for name in feature_columns):
            raise RuntimeError("frozen 179-feature vector is incomplete")
        if len(sampler.allocator_predictions) != 4:
            raise RuntimeError("allocator must score exactly four frozen candidates")
        if any(len(allocation) != 2 for allocation in sampler.method_allocations.values()):
            raise RuntimeError("every paired method must allocate exactly two candidates")
        if int(sampler.sampling_metrics["mattergen_score_calls"]) != 11200:
            raise RuntimeError(f"acquisition score-call count changed: {sampler.sampling_metrics}")
        if int(sampler.sampling_metrics["theoretical_baseline_score_calls"]) != 2000:
            raise RuntimeError(f"baseline score-call count changed: {sampler.sampling_metrics}")
        atoms_list = []
        for metadata, atoms in sampler.deployment_structures:
            atoms.info.update(metadata); atoms_list.append(atoms)
        c0 = next(atoms for metadata, atoms in sampler.deployment_structures if metadata["method"] == "C0")
        returned_check = sampler._exact_atoms_comparison(c0, sampler._final_atoms(returned_mean))
        if not returned_check["atomic_numbers_identical"] or returned_check["positions_max_abs_error"] != 0 or returned_check["cell_max_abs_error"] != 0: raise RuntimeError("returned shared C0 mismatch")
        write(partial / "deployment_final.extxyz", atoms_list)
        assert sampler.standalone_c0_atoms is not None
        standalone = sampler.standalone_c0_atoms.copy(); standalone.info.update(seed=seed, method="B0_Standalone_C0")
        write(partial / "standalone_c0.extxyz", standalone)
        write_csv(partial / "deployment_manifest.csv", sampler.deployment_rows)
        write_csv(partial / "allocator_predictions.csv", sampler.allocator_predictions)
        write_csv(partial / "prebranch_trace.csv", sampler.prebranch_trace)
        (partial / "branch_features.json").write_text(json.dumps(sampler.branch_features, indent=2) + "\n")
        (partial / "method_allocations.json").write_text(json.dumps(sampler.method_allocations, indent=2) + "\n")
        torch.save(sampler.prefix_checkpoint, partial / "prefix_checkpoint.pt")
        summary = {
            "success": True, "phase": phase, "seed": seed, "deployment_structure_count": len(atoms_list),
            "elapsed_seconds": elapsed, "physical_gpu": os.environ.get("LINEAR_K2_PHYSICAL_GPU", "unknown"),
            "mattergen_score_calls_acquisition": int(sampler.sampling_metrics["mattergen_score_calls"]),
            "baseline_score_calls": int(sampler.sampling_metrics["theoretical_baseline_score_calls"]),
            "acquisition_compute_multiplier": float(sampler.sampling_metrics["mattergen_score_calls"]) / float(sampler.sampling_metrics["theoretical_baseline_score_calls"]),
            "per_method_score_calls": 4400, "per_method_compute_multiplier": 2.2,
            "standalone_c0_score_calls": 2000, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "reference_reproduction": sampler.reference_comparison, "returned_c0_check": returned_check,
            "artifact_sha256": sha256(PROTOCOL_ROOT / "frozen_linear_k2_pipeline.joblib"),
            "seed_manifest_sha256": sha256(PROTOCOL_ROOT / "seed_manifest_256.json"),
            "checkpoint_sha256": MODEL_SHA256, "surrogate_only": True, "dft_verified": False,
        }
        (partial / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, destination); print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("c1_128", "c2_128"), required=True); parser.add_argument("--seeds", required=True); args = parser.parse_args()
    run(args.phase, tuple(int(value) for value in args.seeds.split(",") if value))
