"""Generate paired field-decoupled CFG outcomes for all gated cohorts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from ase.io import write
from hydra.utils import instantiate
import yaml

if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.diffusion.sampling.field_decoupled_cfg import FieldDecoupledPolicySampler, normalize_policies
from mattergen.generator import CrystalGenerator


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/field_decoupled_adaptive_cfg"
MODEL_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")
MODEL_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
ELEMENT_MASK_OVERRIDE = "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
SAMPLER_OVERRIDE = "sampler_partial._target_=mattergen.diffusion.sampling.field_decoupled_cfg.FieldDecoupledPolicySampler.from_pl_module"
TARGET = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen() -> dict:
    manifest = json.loads((ROOT / "frozen_manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        if sha256(PROJECT / relative) != expected:
            raise RuntimeError(f"frozen file changed: {relative}")
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA256:
        raise RuntimeError("MatterGen checkpoint hash mismatch")
    return manifest


def calibration_policies() -> tuple[dict, ...]:
    payload = yaml.safe_load((ROOT / "calibration/policies.yaml").read_text())
    result = normalize_policies(payload["policies"])
    if len(result) != 12:
        raise RuntimeError("calibration policy count changed")
    return result


def phase_policies(phase: str) -> tuple[dict, ...]:
    if phase == "calibration":
        return calibration_policies()
    if phase not in ("p0", "formal256"):
        raise ValueError(phase)
    frozen_path = ROOT / "implementation/frozen_field_cfg.yaml"
    manifest = json.loads((ROOT / "implementation/frozen_field_cfg_manifest.json").read_text())
    if sha256(frozen_path) != manifest["frozen_field_cfg_sha256"]:
        raise RuntimeError("frozen field policy changed after calibration")
    frozen = yaml.safe_load(frozen_path.read_text())
    lookup = {policy["policy_id"]: policy for policy in calibration_policies()}
    identifiers = ("C0", "G19", "GPulse", str(frozen["policy_id"]))
    if len(set(identifiers)) != 4:
        raise RuntimeError("S0 must be field-decoupled and distinct from baselines")
    return normalize_policies([lookup[identifier] for identifier in identifiers])


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
    generator._configure_deterministic_mode(); generator.prepare(); generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    return generator


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty rows")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def configure_runtime() -> None:
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def sample_once(generator: CrystalGenerator, seed: int, policies: tuple[dict, ...], implementation_check: bool = False) -> tuple[FieldDecoupledPolicySampler, Any, float]:
    generator.seed = seed; generator._seed_sampling_rngs()
    sampling = generator.load_sampling_config(batch_size=1, num_batches=1)
    if int(sampling.sampler_partial.N) != 1000 or int(sampling.sampler_partial.n_steps_corrector) != 1:
        raise RuntimeError("frozen Predictor/Corrector configuration changed")
    sampler = instantiate(sampling.sampler_partial)(pl_module=generator.model, sample_seed=seed, policy_specs=policies, implementation_check=implementation_check)
    if not isinstance(sampler, FieldDecoupledPolicySampler):
        raise TypeError(type(sampler))
    batches = list(generator.get_condition_loader(sampling))
    if len(batches) != 1:
        raise RuntimeError("expected one condition batch")
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
    _, returned_mean = sampler.sample(*batches[0])
    torch.cuda.synchronize()
    return sampler, returned_mean, time.perf_counter() - started


def implementation_smoke() -> None:
    destination = ROOT / "implementation/implementation_check.json"
    if destination.exists():
        raise FileExistsError(destination)
    configure_runtime()
    policies = normalize_policies([{"policy_id": "C0", "kind": "constant", "scales": {"atomic": 2.0, "pos": 2.0, "cell": 2.0}}])
    sampler, _, elapsed = sample_once(make_generator(42), 42, policies, implementation_check=True)
    result = dict(sampler.implementation_check_result or {})
    result.update(seed=42, historical_seed_only=True, elapsed_seconds=elapsed, mattergen_score_calls=sampler.sampling_metrics["mattergen_score_calls"], physical_gpu=os.environ.get("FIELD_CFG_PHYSICAL_GPU", "unknown"), status="PASS" if result.get("success") else "STOP_IMPLEMENTATION_ERROR")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    if not result.get("success"):
        raise RuntimeError("STOP_IMPLEMENTATION_ERROR")
    print(json.dumps(result))


def run_worker(phase: str, seeds: tuple[int, ...]) -> None:
    manifest = assert_frozen(); phase_root = ROOT / phase
    registered = tuple(map(int, json.loads((phase_root / "seeds.json").read_text())["seeds"]))
    if not seeds or not set(seeds).issubset(set(registered)):
        raise ValueError("worker received unregistered seed")
    policies = phase_policies(phase); configure_runtime(); generator = make_generator(seeds[0])
    for seed in seeds:
        destination = phase_root / "generation" / str(seed); partial = destination.with_name(destination.name + ".partial")
        if (destination / "run_summary.json").exists():
            print(json.dumps({"phase": phase, "seed": seed, "event": "already_complete"}), flush=True); continue
        if destination.exists() or partial.exists():
            raise FileExistsError(f"retained incomplete output: {destination} or {partial}")
        partial.mkdir(parents=True)
        sampler, returned_mean, elapsed = sample_once(generator, seed, policies)
        if len(sampler.policy_structures) != len(policies):
            raise RuntimeError("policy outcome count mismatch")
        atoms_list = []
        for metadata, atoms in sampler.policy_structures:
            atoms.info.update(metadata); atoms_list.append(atoms)
        c0 = next(atoms for metadata, atoms in sampler.policy_structures if metadata["policy_id"] == "C0")
        comparison = sampler._exact_atoms_comparison(c0, sampler._final_atoms(returned_mean))
        if not comparison["atomic_numbers_identical"] or comparison["positions_max_abs_error"] != 0 or comparison["cell_max_abs_error"] != 0:
            raise RuntimeError(f"returned C0 mismatch: {comparison}")
        write(partial / "policy_final.extxyz", atoms_list); write_csv(partial / "policy_manifest.csv", sampler.policy_rows)
        summary = {"success": True, "phase": phase, "seed": seed, "policy_count": len(policies), "elapsed_seconds": elapsed, "physical_gpu": os.environ.get("FIELD_CFG_PHYSICAL_GPU", "unknown"), "mattergen_score_calls": sampler.sampling_metrics["mattergen_score_calls"], "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "c0_return_pairing_check": comparison, "frozen_manifest_sha256": sha256(ROOT / "frozen_manifest.json"), "manifest_git_base": manifest["git_base_commit"], "checkpoint_sha256": MODEL_SHA256, "surrogate_property_eval": True, "dft_verified": False}
        (partial / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n"); os.replace(partial, destination); print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("calibration", "p0", "formal256")); parser.add_argument("--seeds"); parser.add_argument("--implementation-smoke", action="store_true"); args = parser.parse_args()
    if args.implementation_smoke:
        implementation_smoke()
    else:
        if not args.phase or not args.seeds: parser.error("--phase and --seeds are required outside smoke mode")
        run_worker(args.phase, tuple(int(value) for value in args.seeds.split(",") if value))
