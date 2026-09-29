"""Fresh, preregistered trajectory-search cohorts; no historical output is written."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from ase.io import write
from hydra.utils import instantiate
import numpy as np
import torch

if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.diffusion.sampling.field_decoupled_cfg import normalize_policies
from mattergen.generator import CrystalGenerator
from experiments.trajectory_search.search_sampler import SearchSampler


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/trajectory_search"
MODEL_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")
MODEL_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
ELEMENT_OVERRIDE = "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
SAMPLER_OVERRIDE = "sampler_partial._target_=experiments.trajectory_search.search_sampler.SearchSampler.from_pl_module"
TARGET = 0.1
ORDER = ("GPulse", "PPulse", "APulse", "CPulse")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policies(cohort: str) -> tuple[dict, ...]:
    rows = [
        {"policy_id": "C0", "kind": "constant", "scales": {"atomic": 2.0, "pos": 2.0, "cell": 2.0}},
        {"policy_id": "GPulse", "kind": "pulse", "start": 400, "duration": 100, "scales": {"atomic": 1.9, "pos": 1.9, "cell": 1.9}},
        {"policy_id": "PPulse", "kind": "pulse", "start": 400, "duration": 100, "scales": {"atomic": 2.0, "pos": 1.9, "cell": 2.0}},
        {"policy_id": "APulse", "kind": "pulse", "start": 400, "duration": 100, "scales": {"atomic": 1.9, "pos": 2.0, "cell": 2.0}},
        {"policy_id": "CPulse", "kind": "pulse", "start": 400, "duration": 100, "scales": {"atomic": 2.0, "pos": 2.0, "cell": 1.9}},
    ]
    return normalize_policies(rows[:3] if cohort == "search_baseline" else rows)


def registry(cohort: str) -> tuple[tuple[int, ...], dict[int, int]]:
    spec = json.loads((ROOT / "protocol/seed_manifest.json").read_text())
    key = "baseline" if cohort == "search_baseline" else "budget_scaling"
    row = spec[key]
    primary = tuple(int(row["primary_start"]) + i for i in range(int(row["n"])))
    children = {seed: int(row["independent_second_start"]) + i for i, seed in enumerate(primary)} if cohort == "search_baseline" else {}
    if len(set(primary) | set(children.values())) != len(primary) + len(children):
        raise RuntimeError("new seed collision")
    return primary, children


def configure() -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> CrystalGenerator:
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(MODEL_ROOT), config_overrides=[ELEMENT_OVERRIDE]),
        batch_size=1, num_batches=1, properties_to_condition_on={"dft_mag_density": TARGET},
        diffusion_guidance_factor=2.0, guidance_schedule="constant",
        sampling_config_overrides=[SAMPLER_OVERRIDE], seed=seed, deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    return generator


def sample(generator: CrystalGenerator, seed: int, policy_bank: tuple[dict, ...], check: bool) -> tuple[SearchSampler, float, int]:
    generator.seed = seed
    generator._seed_sampling_rngs()
    config = generator.load_sampling_config(batch_size=1, num_batches=1)
    if int(config.sampler_partial.N) != 1000 or int(config.sampler_partial.n_steps_corrector) != 1:
        raise RuntimeError("frozen predictor/corrector schedule changed")
    sampler = instantiate(config.sampler_partial)(
        pl_module=generator.model, sample_seed=seed, policy_specs=policy_bank, implementation_check=check,
    )
    if not isinstance(sampler, SearchSampler):
        raise TypeError(type(sampler))
    batches = list(generator.get_condition_loader(config))
    if len(batches) != 1:
        raise RuntimeError("expected exactly one condition batch")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    sampler.sample(*batches[0])
    torch.cuda.synchronize()
    return sampler, time.perf_counter() - start, int(torch.cuda.max_memory_allocated())


def one_seed(generator: CrystalGenerator, cohort: str, seed: int, child: int | None) -> dict:
    destination = PROJECT / "results" / cohort / "generation" / str(seed)
    if (destination / "run_summary.json").exists():
        return {"seed": seed, "status": "already_complete"}
    if destination.exists():
        raise FileExistsError(f"incomplete output retained: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"incomplete output retained: {partial}")
    partial.mkdir(parents=True)
    sampler, elapsed, peak = sample(generator, seed, policies(cohort), True)
    if sampler.prefix_checkpoint is None or not sampler.implementation_check_result or not sampler.implementation_check_result["success"]:
        raise RuntimeError("missing saved state or failed exact C0 replay")
    expected = 6400 if cohort == "search_baseline" else 8800
    if int(sampler.sampling_metrics["mattergen_score_calls"]) != expected:
        raise RuntimeError(f"unexpected primary acquisition calls: {sampler.sampling_metrics}")
    atoms_list = []
    for metadata, atoms in sampler.policy_structures:
        atoms.info.update({**metadata, "seed": seed, "cohort": cohort, "method": "Trajectory", "branch_id": f"{seed}:{metadata['policy_id']}"})
        atoms_list.append(atoms)
    secondary_elapsed = 0.0
    secondary_calls = 0
    if child is not None:
        secondary, secondary_elapsed, secondary_peak = sample(generator, child, policies("search_baseline")[:1], False)
        if len(secondary.policy_structures) != 1 or secondary.policy_structures[0][0]["policy_id"] != "C0":
            raise RuntimeError("independent complete C0 draw missing")
        secondary_calls = int(secondary.sampling_metrics["mattergen_score_calls"])
        if secondary_calls != 2000:
            raise RuntimeError(f"independent draw calls changed: {secondary_calls}")
        peak = max(peak, secondary_peak)
        independent = secondary.policy_structures[0][1]
        independent.info.update(seed=seed, independent_seed=child, cohort=cohort, method="Independent", policy_id="Independent_C0", branch_id=f"{seed}:Independent_C0", allocation_rank=1)
        atoms_list.append(independent)
    if len(atoms_list) != (4 if child is not None else 5):
        raise RuntimeError("candidate count mismatch")
    write(partial / "candidates.extxyz", atoms_list)
    torch.save(sampler.prefix_checkpoint, partial / "prefix_checkpoint.pt")
    summary = {
        "success": True, "cohort": cohort, "seed": seed, "independent_seed": child,
        "primary_acquisition_score_calls": expected, "independent_score_calls": secondary_calls,
        "total_acquisition_score_calls": expected + secondary_calls,
        "primary_elapsed_seconds": elapsed, "independent_elapsed_seconds": secondary_elapsed,
        "peak_allocated_bytes": peak, "candidate_count": len(atoms_list),
        "reference_reproduction": sampler.implementation_check_result,
        "prefix_state_sha256": sampler.prefix_checkpoint["prefix_state_sha256"],
        "manifest_sha256": sha256(ROOT / "protocol/seed_manifest.json"),
        "sampler_sha256": sha256(ROOT / "search_sampler.py"),
        "generator_sha256": sha256(ROOT / "run_generation.py"),
        "model_sha256": MODEL_SHA256, "surrogate_only": True, "dft_verified": False,
        "physical_gpu": os.environ.get("SEARCH_PHYSICAL_GPU", "unknown"),
    }
    (partial / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(partial, destination)
    return summary


def main(cohort: str, shard_index: int, shard_count: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA256:
        raise RuntimeError("MatterGen checkpoint changed")
    if not (0 <= shard_index < shard_count):
        raise ValueError("invalid shard")
    seeds, children = registry(cohort)
    assigned = seeds[shard_index::shard_count]
    if not assigned:
        return
    configure()
    generator = make_generator(assigned[0])
    for seed in assigned:
        print(json.dumps(one_seed(generator, cohort, seed, children.get(seed))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("search_baseline", "budget_scaling"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    main(args.cohort, args.shard_index, args.shard_count)
