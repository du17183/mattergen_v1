"""Generate paired C0/T0/A0 fresh P0 outputs after an offline GO only."""

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
from mattergen.diffusion.sampling.risk_calibrated_cfg import OneShotRiskCalibratedSampler
from mattergen.generator import CrystalGenerator


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/risk_calibrated_adaptive_cfg"
P0 = ROOT / "p0"
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
    "OneShotRiskCalibratedSampler.from_pl_module"
)
TARGET = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def assert_authorized() -> tuple[tuple[int, ...], dict]:
    decision = json.loads((ROOT / "offline/decision_summary.json").read_text())
    if decision["OFFLINE_PREDICTABILITY"] != "GO" or decision["next"] != "RUN_FRESH_P0":
        raise RuntimeError("fresh P0 prohibited because offline predictability is not GO")
    reservation = json.loads((ROOT / "future_seed_reservations.json").read_text())
    seeds = tuple(map(int, reservation["p0_seeds"]))
    if len(seeds) != 32 or len(set(seeds)) != 32 or reservation["historical_overlap"] != 0:
        raise RuntimeError("P0 seed reservation invalid")
    seed_path = P0 / "seeds.json"
    if not seed_path.exists():
        seed_path.write_text(
            json.dumps({"seeds": seeds, "count": 32, "source": "future_seed_reservations.json"}, indent=2) + "\n"
        )
    elif tuple(json.loads(seed_path.read_text())["seeds"]) != seeds:
        raise RuntimeError("P0 seeds changed")
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA256:
        raise RuntimeError("MatterGen checkpoint hash mismatch")
    return seeds, decision


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
    generator._configure_deterministic_mode(); generator.prepare(); generator.model.eval()
    for parameter in generator.model.parameters(): parameter.requires_grad_(False)
    return generator


def run_worker(requested: tuple[int, ...]) -> None:
    allowed, decision = assert_authorized()
    if not requested or not set(requested).issubset(set(allowed)):
        raise ValueError("worker received an unregistered P0 seed")
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    generator = make_generator(requested[0])
    metadata_path = ROOT / "models/selected_model_metadata.json"
    ensemble_path = ROOT / "models/selected_ensemble.joblib"
    for seed in requested:
        destination = P0 / "generation" / str(seed)
        partial = destination.with_name(destination.name + ".partial")
        if (destination / "run_summary.json").exists():
            print(json.dumps({"seed": seed, "event": "already_complete"}), flush=True); continue
        if destination.exists() or partial.exists():
            raise FileExistsError(f"retained incomplete output: {destination} or {partial}")
        partial.mkdir(parents=True)
        generator.seed = seed; generator._seed_sampling_rngs()
        sampling = generator.load_sampling_config(batch_size=1, num_batches=1)
        sampler = instantiate(sampling.sampler_partial)(
            pl_module=generator.model,
            sample_seed=seed,
            decision_index=int(decision["selected_timestep"]),
            ensemble_path=str(ensemble_path),
            metadata_path=str(metadata_path),
        )
        batches = list(generator.get_condition_loader(sampling))
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
        sampler.sample(*batches[0]); torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        if len(sampler.policy_structures) != 3 or len(sampler.controller_trace) != 1:
            raise RuntimeError("P0 paired policy output count mismatch")
        atoms_list = []
        for metadata, atoms in sampler.policy_structures:
            atoms.info.update(metadata); atoms_list.append(atoms)
        for left_index, left in enumerate(atoms_list):
            for right in atoms_list[left_index + 1:]:
                if left.info["selected_action"] != right.info["selected_action"]: continue
                if not np.array_equal(left.numbers, right.numbers):
                    raise RuntimeError("identical-action paired branches diverged in elements")
                if np.max(np.abs(left.cell.array - right.cell.array)) != 0:
                    raise RuntimeError("identical-action paired branches diverged in cell")
                if np.max(np.abs(left.get_scaled_positions(wrap=False) - right.get_scaled_positions(wrap=False))) != 0:
                    raise RuntimeError("identical-action paired branches diverged in positions")
        write(partial / "policy_final.extxyz", atoms_list)
        write_csv(partial / "controller_trace.csv", sampler.controller_trace)
        summary = {
            "success": True, "seed": seed, "elapsed_seconds": elapsed,
            "physical_gpu": os.environ.get("RISK_CFG_PHYSICAL_GPU", "unknown"),
            "mattergen_score_calls": sampler.sampling_metrics["mattergen_score_calls"],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "selected_timestep": decision["selected_timestep"],
            "selected_model_family": decision["selected_model_family"],
            "controller_action": sampler.controller_trace[0]["selected_action"],
            "one_shot_decision_count": 1,
        }
        (partial / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, destination); print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--seeds", required=True)
    args = parser.parse_args()
    run_worker(tuple(int(value) for value in args.seeds.split(",") if value))
