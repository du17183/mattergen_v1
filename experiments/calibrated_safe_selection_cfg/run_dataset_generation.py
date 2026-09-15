"""Generate the frozen 128-trajectory, step-250 full-outcome dataset."""

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

if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.diffusion.sampling.risk_calibrated_cfg import FullOutcomeCounterfactualSampler
from mattergen.generator import CrystalGenerator


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/calibrated_safe_selection_cfg"
DATASET = ROOT / "dataset"
FROZEN_MANIFEST = ROOT / "frozen_manifest.json"
MODEL_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/"
    "checkpoints/dft_mag_density"
)
MODEL_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
ELEMENT_MASK_OVERRIDE = (
    "++lightning_module.diffusion_module.model.element_mask_func="
    "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
)
SAMPLER_OVERRIDE = (
    "sampler_partial._target_="
    "mattergen.diffusion.sampling.risk_calibrated_cfg."
    "FullOutcomeCounterfactualSampler.from_pl_module"
)
TARGET = 0.1
DECISION_INDICES = (250,)
CANDIDATES = (1.75, 2.0, 2.25)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def registered_seeds() -> tuple[int, ...]:
    payload = json.loads((DATASET / "seeds.json").read_text())
    seeds = tuple(map(int, payload["seeds"]))
    if len(seeds) != 128 or len(set(seeds)) != 128 or payload["historical_overlap"] != 0:
        raise RuntimeError("dataset seed registration is invalid")
    return seeds


def assert_frozen() -> dict:
    manifest = json.loads(FROZEN_MANIFEST.read_text())
    if not manifest["frozen_before_generation"]:
        raise RuntimeError("protocol is not frozen")
    for relative, expected in manifest["sha256"].items():
        if sha256(PROJECT / relative) != expected:
            raise RuntimeError(f"frozen file changed: {relative}")
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA256:
        raise RuntimeError("MatterGen checkpoint hash mismatch")
    return manifest


def make_generator(seed: int) -> CrystalGenerator:
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(
            model_path=str(MODEL_ROOT), config_overrides=[ELEMENT_MASK_OVERRIDE]
        ),
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


def maximum_atoms_difference(left, right) -> dict[str, float | bool]:
    return {
        "atomic_numbers_equal": bool(np.array_equal(left.numbers, right.numbers)),
        "max_fractional_position_abs_error": float(
            np.max(
                np.abs(
                    left.get_scaled_positions(wrap=False)
                    - right.get_scaled_positions(wrap=False)
                )
            )
        ),
        "max_cell_abs_error": float(np.max(np.abs(left.cell.array - right.cell.array))),
    }


def run_worker(seeds: tuple[int, ...]) -> None:
    manifest = assert_frozen()
    allowed = registered_seeds()
    if not seeds or not set(seeds).issubset(set(allowed)):
        raise ValueError("worker received an unregistered dataset seed")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = make_generator(seeds[0])
    for seed in seeds:
        destination = DATASET / "generation" / str(seed)
        partial = destination.with_name(destination.name + ".partial")
        if (destination / "run_summary.json").exists():
            print(json.dumps({"seed": seed, "event": "already_complete"}), flush=True)
            continue
        if destination.exists() or partial.exists():
            raise FileExistsError(f"retained incomplete output: {destination} or {partial}")
        partial.mkdir(parents=True)
        generator.seed = seed
        generator._seed_sampling_rngs()
        sampling = generator.load_sampling_config(batch_size=1, num_batches=1)
        if int(sampling.sampler_partial.N) != 1000:
            raise RuntimeError("sampling step count changed")
        if int(sampling.sampler_partial.n_steps_corrector) != 1:
            raise RuntimeError("corrector step count changed")
        sampler = instantiate(sampling.sampler_partial)(
            pl_module=generator.model,
            sample_seed=seed,
            decision_indices=DECISION_INDICES,
            candidate_scales=CANDIDATES,
            base_scale=2.0,
            temporal_ema_beta=0.9,
            recent_window=5,
        )
        if not isinstance(sampler, FullOutcomeCounterfactualSampler):
            raise TypeError(type(sampler))
        batches = list(generator.get_condition_loader(sampling))
        if len(batches) != 1:
            raise RuntimeError(f"expected one condition batch, got {len(batches)}")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        _, base_mean = sampler.sample(*batches[0])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if len(sampler.full_structures) != 3:
            raise RuntimeError("full branch count mismatch")
        if len(sampler.feature_rows) != 1 or len(sampler.branch_rows) != 3:
            raise RuntimeError("metadata count mismatch")
        branch_atoms = []
        for metadata, atoms in sampler.full_structures:
            atoms.info.update(metadata)
            branch_atoms.append(atoms)
        base_atoms = sampler._final_atoms(base_mean)
        base_atoms.info.update(seed=seed, candidate_action="KEEP", candidate_cfg=2.0)
        keep_atoms = next(
            atoms
            for metadata, atoms in sampler.full_structures
            if metadata["candidate_action"] == "KEEP"
        )
        pairing_check = maximum_atoms_difference(base_atoms, keep_atoms)
        if (
            not pairing_check["atomic_numbers_equal"]
            or pairing_check["max_fractional_position_abs_error"] != 0.0
            or pairing_check["max_cell_abs_error"] != 0.0
        ):
            raise RuntimeError(f"CFG2 branch failed exact pairing: {pairing_check}")
        write(partial / "full_horizon_final.extxyz", branch_atoms)
        write(partial / "base_cfg2_final.extxyz", [base_atoms])
        write_csv(partial / "online_features.csv", sampler.feature_rows)
        write_csv(partial / "branch_manifest.csv", sampler.branch_rows)
        summary = {
            "success": True,
            "seed": seed,
            "full_branches": 3,
            "decision_states": 1,
            "elapsed_seconds": elapsed,
            "physical_gpu": os.environ.get("SAFE_CFG_PHYSICAL_GPU", "unknown"),
            "mattergen_score_calls": sampler.sampling_metrics["mattergen_score_calls"],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "pairing_check": pairing_check,
            "frozen_manifest_sha256": sha256(FROZEN_MANIFEST),
            "checkpoint_sha256": MODEL_SHA256,
            "manifest_git_parent": manifest["git_parent_commit"],
            "surrogate_property_eval": True,
            "dft_verified": False,
        }
        (partial / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, destination)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True, help="comma-separated registered seeds")
    arguments = parser.parse_args()
    run_worker(tuple(int(value) for value in arguments.seeds.split(",") if value))
