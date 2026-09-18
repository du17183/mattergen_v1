"""Generate the 64-seed shared-prefix Phase-B supervision cohort."""

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
import yaml

if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.diffusion.sampling.field_decoupled_cfg import normalize_policies
from mattergen.generator import CrystalGenerator
from experiments.reference_preserved_budgeted_cfg.phase_b_sampler import PhaseBFeatureSampler


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/reference_preserved_budgeted_cfg"
FEATURE_ROOT = ROOT / "feature_study"
MODEL_ROOT = PROJECT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
MODEL_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
ELEMENT_MASK_OVERRIDE = "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
SAMPLER_OVERRIDE = "sampler_partial._target_=experiments.reference_preserved_budgeted_cfg.phase_b_sampler.PhaseBFeatureSampler.from_pl_module"
TARGET = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "feature_study_protocol.yaml").read_text())


def policies() -> tuple[dict, ...]:
    return normalize_policies(protocol()["sampling"]["policies"])


def registered_seeds() -> tuple[int, ...]:
    payload = json.loads((FEATURE_ROOT / "feature_study_seed_manifest.json").read_text())
    if not payload["preregistered_before_generation"] or payload["historical_overlap"]:
        raise RuntimeError("fresh-seed registration audit is not valid")
    return tuple(map(int, payload["seeds"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure_runtime() -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> CrystalGenerator:
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(MODEL_ROOT), config_overrides=[ELEMENT_MASK_OVERRIDE]),
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={"dft_mag_density": TARGET},
        diffusion_guidance_factor=2.0,
        guidance_schedule="constant",
        sampling_config_overrides=[SAMPLER_OVERRIDE],
        seed=seed,
        deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    return generator


def sample_once(generator: CrystalGenerator, seed: int) -> tuple[PhaseBFeatureSampler, Any, float]:
    frozen = protocol()
    generator.seed = seed
    generator._seed_sampling_rngs()
    sampling = generator.load_sampling_config(batch_size=1, num_batches=1)
    if int(sampling.sampler_partial.N) != 1000 or int(sampling.sampler_partial.n_steps_corrector) != 1:
        raise RuntimeError("frozen Predictor/Corrector configuration changed")
    sampler = instantiate(sampling.sampler_partial)(
        pl_module=generator.model,
        sample_seed=seed,
        policy_specs=policies(),
        branch_point=int(frozen["sampling"]["branch_point"]),
        feature_ema_beta=float(frozen["prebranch_capture"]["ema_beta"]),
        feature_windows=tuple(frozen["prebranch_capture"]["windows"]),
    )
    if not isinstance(sampler, PhaseBFeatureSampler):
        raise TypeError(type(sampler))
    batches = list(generator.get_condition_loader(sampling))
    if len(batches) != 1:
        raise RuntimeError("expected one condition batch")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    _, returned_mean = sampler.sample(*batches[0])
    torch.cuda.synchronize()
    return sampler, returned_mean, time.perf_counter() - started


def run(seeds: tuple[int, ...]) -> None:
    registered = set(registered_seeds())
    if not seeds or not set(seeds).issubset(registered):
        raise ValueError("worker received an unregistered Phase-B seed")
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA256:
        raise RuntimeError("MatterGen checkpoint hash mismatch")
    configure_runtime()
    generator = make_generator(seeds[0])
    split_assignment = json.loads((FEATURE_ROOT / "feature_study_seed_manifest.json").read_text())["split_assignment"]
    for seed in seeds:
        destination = FEATURE_ROOT / "generation" / str(seed)
        partial = destination.with_name(destination.name + ".partial")
        if (destination / "run_summary.json").exists():
            print(json.dumps({"seed": seed, "event": "already_complete"}), flush=True)
            continue
        if destination.exists() or partial.exists():
            raise FileExistsError(f"retained incomplete output: {destination} or {partial}")
        partial.mkdir(parents=True)
        sampler, returned_mean, elapsed = sample_once(generator, seed)
        atoms_list = []
        for metadata, atoms in sampler.policy_structures:
            atoms.info.update(metadata)
            atoms_list.append(atoms)
        c0 = next(atoms for metadata, atoms in sampler.policy_structures if metadata["policy_id"] == "C0")
        comparison = sampler._exact_atoms_comparison(c0, sampler._final_atoms(returned_mean))
        if not comparison["atomic_numbers_identical"] or comparison["positions_max_abs_error"] != 0.0 or comparison["cell_max_abs_error"] != 0.0:
            raise RuntimeError(f"returned C0 mismatch: {comparison}")
        write(partial / "policy_final.extxyz", atoms_list)
        write_csv(partial / "policy_manifest.csv", sampler.policy_rows)
        write_csv(partial / "prebranch_trace.csv", sampler.prebranch_trace)
        (partial / "branch_features.json").write_text(json.dumps(sampler.branch_features, indent=2) + "\n")
        conditioning = dict(sampler.conditioning_metadata)
        conditioning.update(target_dft_mag_density=TARGET, deterministic=True, split=split_assignment[str(seed)])
        (partial / "conditioning_metadata.json").write_text(json.dumps(conditioning, indent=2) + "\n")
        if sampler.prefix_checkpoint is None:
            raise RuntimeError("missing prefix checkpoint")
        torch.save(sampler.prefix_checkpoint, partial / "prefix_checkpoint.pt")
        summary = {
            "success": True,
            "phase": "feature_study",
            "seed": seed,
            "split": split_assignment[str(seed)],
            "policy_count": len(sampler.policy_structures),
            "trace_rows": len(sampler.prebranch_trace),
            "feature_count": len(sampler.branch_features),
            "branch_point": sampler.branch_point,
            "prefix_state_sha256": sampler.branch_features["prefix_state_sha256"],
            "elapsed_seconds": elapsed,
            "physical_gpu": os.environ.get("RPB_CFG_PHYSICAL_GPU", "unknown"),
            "mattergen_score_calls": sampler.sampling_metrics["mattergen_score_calls"],
            "theoretical_compute_multiplier": float(sampler.sampling_metrics["mattergen_score_calls"]) / float(sampler.sampling_metrics["theoretical_baseline_score_calls"]),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "c0_return_pairing_check": comparison,
            "protocol_sha256": sha256(ROOT / "feature_study_protocol.yaml"),
            "seed_manifest_sha256": sha256(FEATURE_ROOT / "feature_study_seed_manifest.json"),
            "checkpoint_sha256": MODEL_SHA256,
            "surrogate_property_eval": True,
            "dft_verified": False,
        }
        (partial / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, destination)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True)
    arguments = parser.parse_args()
    run(tuple(int(value) for value in arguments.seeds.split(",") if value))
