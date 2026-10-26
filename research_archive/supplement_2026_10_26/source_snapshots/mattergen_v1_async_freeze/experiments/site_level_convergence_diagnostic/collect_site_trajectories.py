"""Collect a deterministic shard of per-site diagnostics with one persistent model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from ase.io import read
from hydra.utils import instantiate

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler
from experiments.shared_field_dynamics.generate_diagnostic_one import (
    ELEMENT_MASK_OVERRIDE, EXPECTED_SHA256, MODEL_ROOT, sha256,
)


ROOT = Path(__file__).resolve().parent
EXISTING = Path(__file__).resolve().parents[1] / "shared_field_dynamics"


def _same_structure(seed: int, generated_path: Path) -> dict[str, float | bool]:
    old = read(EXISTING / "trajectories" / str(seed) / "generated_crystals.extxyz", index=":")[0]
    new = read(generated_path, index=":")[0]
    same_numbers = bool(np.array_equal(old.numbers, new.numbers))
    cell_diff = float(np.max(np.abs(old.cell.array - new.cell.array)))
    frac_delta = np.remainder(new.get_scaled_positions() - old.get_scaled_positions() + 0.5, 1.0) - 0.5
    cart_delta = frac_delta @ old.cell.array
    position_diff = float(np.max(np.linalg.norm(cart_delta, axis=1)))
    return {
        "same_atomic_numbers_as_existing": same_numbers,
        "max_cell_abs_diff_vs_existing": cell_diff,
        "max_periodic_position_diff_vs_existing_A": position_diff,
        "exact_reproduction": bool(same_numbers and cell_diff <= 1e-7 and position_diff <= 1e-7),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=("calibration", "validation"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    cohort = tuple(range(99000, 99032)) if args.subset == "calibration" else tuple(range(99032, 99064))
    seeds = cohort[args.shard_index::args.num_shards]
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != EXPECTED_SHA256:
        raise RuntimeError("official checkpoint hash mismatch")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(
            model_path=str(MODEL_ROOT), config_overrides=[ELEMENT_MASK_OVERRIDE]
        ),
        batch_size=1, num_batches=1,
        properties_to_condition_on={"dft_mag_density": 0.2},
        diffusion_guidance_factor=2.0, guidance_schedule="constant",
        seed=seeds[0], deterministic=True, record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)

    for ordinal, seed in enumerate(seeds, start=1):
        run_dir = ROOT / "rerun" / str(seed)
        partial = run_dir.with_name(run_dir.name + ".partial")
        if (run_dir / "run_summary.json").is_file():
            print(json.dumps({"event": "skip", "seed": seed}), flush=True)
            continue
        if run_dir.exists() or partial.exists():
            raise RuntimeError(f"incomplete output retained: {run_dir} or {partial}")
        partial.mkdir(parents=True)
        generator.seed = seed
        generator._seed_sampling_rngs()
        sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
        if (int(sampling_config.sampler_partial.N) != 1000
                or float(sampling_config.sampler_partial.guidance_scale) != 2.0
                or str(sampling_config.sampler_partial.guidance_schedule) != "constant"
                or int(sampling_config.sampler_partial.n_steps_corrector) != 1):
            raise RuntimeError("official sampling contract changed")
        sampling_config.sampler_partial._target_ = (
            "experiments.site_level_convergence_diagnostic.site_position_sampler."
            "SitePositionDiagnosticSampler.from_pl_module"
        )
        sampler = instantiate(sampling_config.sampler_partial)(
            pl_module=generator.model,
            site_trace_path=str((partial / "site_clean_estimates.npz").resolve()),
            diagnostic_seed=seed,
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=generator.get_condition_loader(sampling_config),
            properties_to_condition_on={"dft_mag_density": 0.2},
            output_path=partial, cfg=generator.cfg, record_trajectories=False,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = dict(sampler.sampling_metrics)
        if int(metrics["mattergen_score_calls"]) != 2000:
            raise RuntimeError("score-call contract changed")
        if int(metrics["site_diagnostic_observation_count"]) != 1000:
            raise RuntimeError("observation count changed")
        reproduction = _same_structure(seed, partial / "generated_crystals.extxyz")
        if not reproduction["exact_reproduction"]:
            raise RuntimeError(f"site rerun failed deterministic reproduction: {seed} {reproduction}")
        summary = {
            "success": True, "seed": seed, "subset": args.subset,
            "formula": structures[0].composition.reduced_formula,
            "num_atoms": len(structures[0]), "elapsed_seconds": elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "target_dft_mag_density": 0.2, "guidance_scale": 2.0,
            "guidance_schedule": "constant", "adaptive_cfg_used": False,
            "sampling_steps": 1000, "corrector_steps_per_timestep": 1,
            "checkpoint_sha256": EXPECTED_SHA256,
            "reused_existing_trajectories": False,
            "reused_existing_final_quality_and_forces": True,
            "surrogate_property_eval": True, "dft_verified": False,
            **reproduction, **metrics,
        }
        (partial / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial, run_dir)
        print(json.dumps({"event": "complete", "seed": seed, "ordinal": ordinal,
                          "total": len(seeds), "seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
