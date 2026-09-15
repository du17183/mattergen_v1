"""Generate paired final structures for calibration, P0, or Formal256."""

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
from mattergen.diffusion.sampling.stage_bounded_cfg import StageScheduleSampler, normalized_specs
from mattergen.generator import CrystalGenerator


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/stage_calibrated_bounded_cfg"
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
    "mattergen.diffusion.sampling.stage_bounded_cfg.StageScheduleSampler.from_pl_module"
)
TARGET = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen() -> dict:
    manifest = json.loads((ROOT / "frozen_manifest.json").read_text())
    if not manifest["frozen_before_generation"]:
        raise RuntimeError("protocol is not frozen")
    for relative, expected in manifest["sha256"].items():
        if sha256(PROJECT / relative) != expected:
            raise RuntimeError(f"frozen file changed: {relative}")
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA256:
        raise RuntimeError("MatterGen checkpoint hash mismatch")
    return manifest


def phase_specs(phase: str) -> tuple[dict, ...]:
    if phase == "calibration":
        grid = yaml.safe_load((ROOT / "preregistered_schedule_grid.yaml").read_text())
        specs: list[dict] = [{"kind": "c0"}]
        specs.extend({"kind": "constant", "cfg": float(cfg)} for cfg in grid["constant_cfg_candidates"])
        specs.extend(
            {"kind": "permanent", "start": int(start), "delta_g": float(delta)}
            for start in grid["permanent_counterparts"]["starts"]
            for delta in grid["permanent_counterparts"]["delta_g"]
        )
        specs.extend(
            {"kind": "bounded", "start": int(start), "delta_g": float(delta), "duration": int(duration)}
            for start in grid["candidate_starts"]
            for delta in grid["candidate_delta_g"]
            for duration in grid["candidate_durations"]
        )
    elif phase in ("p0", "formal256"):
        frozen_path = ROOT / "implementation/frozen_stage_cfg.yaml"
        selection_manifest = json.loads((ROOT / "implementation/frozen_stage_cfg_manifest.json").read_text())
        if sha256(frozen_path) != selection_manifest["frozen_stage_cfg_sha256"]:
            raise RuntimeError("post-calibration frozen stage configuration changed")
        frozen = yaml.safe_load(frozen_path.read_text())
        specs = [
            {"kind": "c0"},
            {"kind": "constant", "cfg": float(frozen["b1_constant_cfg"])},
            {"kind": "permanent", "start": int(frozen["start"]), "delta_g": float(frozen["delta_g"])},
            {"kind": "bounded", "start": int(frozen["start"]), "delta_g": float(frozen["delta_g"]), "duration": int(frozen["duration"])},
        ]
    else:
        raise ValueError(phase)
    result = normalized_specs(specs)
    if phase == "calibration" and len(result) != 23:
        raise RuntimeError(f"frozen calibration schedule count changed: {len(result)}")
    if phase != "calibration" and len(result) != 4:
        raise RuntimeError("fresh comparison must contain exactly C0/B1/B2/S0")
    return result


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


def maximum_atoms_difference(left, right) -> dict[str, float | bool]:
    return {
        "atomic_numbers_equal": bool(np.array_equal(left.numbers, right.numbers)),
        "max_fractional_position_abs_error": float(np.max(np.abs(left.get_scaled_positions(wrap=False) - right.get_scaled_positions(wrap=False)))),
        "max_cell_abs_error": float(np.max(np.abs(left.cell.array - right.cell.array))),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty rows")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_worker(phase: str, seeds: tuple[int, ...]) -> None:
    manifest = assert_frozen()
    phase_root = ROOT / phase
    registered = tuple(map(int, json.loads((phase_root / "seeds.json").read_text())["seeds"]))
    if not seeds or not set(seeds).issubset(set(registered)):
        raise ValueError("worker received an unregistered seed")
    specs = phase_specs(phase)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = make_generator(seeds[0])
    for seed in seeds:
        destination = phase_root / "generation" / str(seed)
        partial = destination.with_name(destination.name + ".partial")
        if (destination / "run_summary.json").exists():
            print(json.dumps({"phase": phase, "seed": seed, "event": "already_complete"}), flush=True)
            continue
        if destination.exists() or partial.exists():
            raise FileExistsError(f"retained incomplete output: {destination} or {partial}")
        partial.mkdir(parents=True)
        generator.seed = seed
        generator._seed_sampling_rngs()
        sampling = generator.load_sampling_config(batch_size=1, num_batches=1)
        if int(sampling.sampler_partial.N) != 1000 or int(sampling.sampler_partial.n_steps_corrector) != 1:
            raise RuntimeError("frozen Predictor/Corrector configuration changed")
        sampler = instantiate(sampling.sampler_partial)(pl_module=generator.model, sample_seed=seed, schedule_specs=specs)
        if not isinstance(sampler, StageScheduleSampler):
            raise TypeError(type(sampler))
        batches = list(generator.get_condition_loader(sampling))
        if len(batches) != 1:
            raise RuntimeError("expected one condition batch")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        _, returned_mean = sampler.sample(*batches[0])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if len(sampler.schedule_structures) != len(specs) or len(sampler.schedule_rows) != len(specs):
            raise RuntimeError("schedule outcome count mismatch")
        atoms_list = []
        c0_atoms = None
        for metadata, atoms in sampler.schedule_structures:
            atoms.info.update(metadata)
            atoms_list.append(atoms)
            if metadata["schedule_id"] == "C0":
                c0_atoms = atoms
        if c0_atoms is None:
            raise RuntimeError("C0 output missing")
        pairing_check = maximum_atoms_difference(c0_atoms, sampler._final_atoms(returned_mean))
        if not pairing_check["atomic_numbers_equal"] or pairing_check["max_fractional_position_abs_error"] != 0 or pairing_check["max_cell_abs_error"] != 0:
            raise RuntimeError(f"returned C0 failed exact pairing: {pairing_check}")
        write(partial / "schedule_final.extxyz", atoms_list)
        write_csv(partial / "schedule_manifest.csv", sampler.schedule_rows)
        summary = {
            "success": True,
            "phase": phase,
            "seed": seed,
            "schedule_count": len(specs),
            "elapsed_seconds": elapsed,
            "physical_gpu": os.environ.get("STAGE_CFG_PHYSICAL_GPU", "unknown"),
            "mattergen_score_calls": sampler.sampling_metrics["mattergen_score_calls"],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "c0_return_pairing_check": pairing_check,
            "frozen_manifest_sha256": sha256(ROOT / "frozen_manifest.json"),
            "manifest_git_base": manifest["git_base_commit"],
            "checkpoint_sha256": MODEL_SHA256,
            "surrogate_property_eval": True,
            "dft_verified": False,
        }
        (partial / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, destination)
        print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "p0", "formal256"), required=True)
    parser.add_argument("--seeds", required=True)
    args = parser.parse_args()
    run_worker(args.phase, tuple(int(value) for value in args.seeds.split(",") if value))


if __name__ == "__main__":
    main()
