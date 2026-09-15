"""Generate paired C0/T0/S0 outputs for an authorized fresh cohort."""

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
from mattergen.diffusion.sampling.safe_selection_cfg import OneShotSafeSelectionSampler
from mattergen.generator import CrystalGenerator


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/calibrated_safe_selection_cfg"
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
    "mattergen.diffusion.sampling.safe_selection_cfg."
    "OneShotSafeSelectionSampler.from_pl_module"
)
TARGET = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_once(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def assert_authorized(cohort: str) -> tuple[tuple[int, ...], Path]:
    status = json.loads((ROOT / "final_status.json").read_text())
    reservation = json.loads((ROOT / "future_seed_reservations.json").read_text())
    if cohort == "p0":
        if status["OFFLINE_SAFE_SELECTION"] != "GO" or status["next"] != "RUN_FRESH_P0":
            raise RuntimeError("fresh P0 is not authorized by the offline gate")
        key, expected = "p0_seeds", 32
    elif cohort == "formal256":
        if status["SAFE_SELECTION_P0"] != "GO" or status["next"] != "RUN_FRESH_FORMAL256":
            raise RuntimeError("fresh Formal256 is not authorized by the P0 gate")
        key, expected = "formal256_seeds", 256
    else:
        raise ValueError(cohort)
    seeds = tuple(map(int, reservation[key]))
    if len(seeds) != expected or len(set(seeds)) != expected or reservation["historical_overlap"] != 0:
        raise RuntimeError(f"invalid {cohort} seed reservation")
    cohort_root = ROOT / cohort
    seed_path = cohort_root / "seeds.json"
    payload = {"seeds": seeds, "count": expected, "source": "future_seed_reservations.json"}
    if seed_path.exists():
        if tuple(json.loads(seed_path.read_text())["seeds"]) != seeds:
            raise RuntimeError(f"{cohort} seed file changed")
    else:
        seed_path.write_text(json.dumps(payload, indent=2) + "\n")
    hashes = json.loads((ROOT / "implementation/model_hashes.json").read_text())
    for relative, expected_hash in hashes.items():
        if sha256(ROOT / relative) != expected_hash:
            raise RuntimeError(f"frozen controller artifact changed: {relative}")
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA256:
        raise RuntimeError("MatterGen checkpoint hash mismatch")
    return seeds, cohort_root


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


def run_worker(cohort: str, requested: tuple[int, ...]) -> None:
    allowed, cohort_root = assert_authorized(cohort)
    if not requested or not set(requested).issubset(set(allowed)):
        raise ValueError(f"worker received an unregistered {cohort} seed")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = make_generator(requested[0])
    for seed in requested:
        destination = cohort_root / "generation" / str(seed)
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
        sampler = instantiate(sampling.sampler_partial)(
            pl_module=generator.model,
            sample_seed=seed,
            controller_bundle_path=str(ROOT / "implementation/controller_bundle.pkl"),
            controller_config_path=str(ROOT / "implementation/frozen_safe_controller.yaml"),
        )
        if not isinstance(sampler, OneShotSafeSelectionSampler):
            raise TypeError(type(sampler))
        batches = list(generator.get_condition_loader(sampling))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        sampler.sample(*batches[0])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if len(sampler.policy_structures) != 3 or len(sampler.controller_trace) != 1:
            raise RuntimeError("paired policy output count mismatch")
        atoms_list = []
        for metadata, atoms in sampler.policy_structures:
            atoms.info.update(metadata)
            atoms_list.append(atoms)
        for left_index, left in enumerate(atoms_list):
            for right in atoms_list[left_index + 1 :]:
                if left.info["selected_action"] != right.info["selected_action"]:
                    continue
                if not np.array_equal(left.numbers, right.numbers):
                    raise RuntimeError("identical-action paired branches diverged in elements")
                if np.max(np.abs(left.cell.array - right.cell.array)) != 0:
                    raise RuntimeError("identical-action paired branches diverged in cell")
                if np.max(np.abs(left.get_scaled_positions(wrap=False) - right.get_scaled_positions(wrap=False))) != 0:
                    raise RuntimeError("identical-action paired branches diverged in positions")
        write(partial / "policy_final.extxyz", atoms_list)
        write_csv_once(partial / "controller_trace.csv", sampler.controller_trace)
        summary = {
            "success": True, "cohort": cohort, "seed": seed, "elapsed_seconds": elapsed,
            "physical_gpu": os.environ.get("SAFE_CFG_PHYSICAL_GPU", "unknown"),
            "mattergen_score_calls": sampler.sampling_metrics["mattergen_score_calls"],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "controller_action": sampler.controller_trace[0]["selected_action"],
            "one_shot_decision_count": 1,
        }
        (partial / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, destination)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("p0", "formal256"), required=True)
    parser.add_argument("--seeds", required=True)
    arguments = parser.parse_args()
    run_worker(arguments.cohort, tuple(int(value) for value in arguments.seeds.split(",") if value))
