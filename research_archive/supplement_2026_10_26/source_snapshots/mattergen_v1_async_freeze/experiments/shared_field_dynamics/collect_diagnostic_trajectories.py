"""Collect a resumable shard of diagnostic trajectories with one persistent model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from hydra.utils import instantiate

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler
from experiments.shared_field_dynamics.generate_diagnostic_one import (
    ELEMENT_MASK_OVERRIDE,
    EXPECTED_SHA256,
    MODEL_ROOT,
    ROOT,
    sha256,
)


ALL_SEEDS = tuple(range(99000, 99064))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    seeds = ALL_SEEDS[args.shard_index::args.num_shards]
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
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={"dft_mag_density": 0.2},
        diffusion_guidance_factor=2.0,
        guidance_schedule="constant",
        seed=seeds[0],
        deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)

    for ordinal, seed in enumerate(seeds, start=1):
        run_dir = ROOT / "trajectories" / str(seed)
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
        if (
            int(sampling_config.sampler_partial.N) != 1000
            or float(sampling_config.sampler_partial.guidance_scale) != 2.0
            or str(sampling_config.sampler_partial.guidance_schedule) != "constant"
            or int(sampling_config.sampler_partial.n_steps_corrector) != 1
        ):
            raise RuntimeError("official sampling contract changed")
        sampling_config.sampler_partial._target_ = (
            "experiments.shared_field_dynamics.field_diagnostic_sampler."
            "FieldDynamicsDiagnosticSampler.from_pl_module"
        )
        sampler_partial = instantiate(sampling_config.sampler_partial)
        sampler = sampler_partial(
            pl_module=generator.model,
            diagnostic_trace_path=str((partial / "field_dynamics.csv").resolve()),
            diagnostic_seed=seed,
        )
        condition_loader = generator.get_condition_loader(sampling_config)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=condition_loader,
            properties_to_condition_on={"dft_mag_density": 0.2},
            output_path=partial,
            cfg=generator.cfg,
            record_trajectories=False,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = dict(sampler.sampling_metrics)
        if int(metrics["mattergen_score_calls"]) != 2000:
            raise RuntimeError("official score-call contract changed")
        if int(metrics["diagnostic_observation_count"]) != 55:
            raise RuntimeError("diagnostic observation count changed")
        summary = {
            "success": True,
            "seed": seed,
            "subset": "calibration" if seed <= 99031 else "validation",
            "formula": structures[0].composition.reduced_formula,
            "num_atoms": len(structures[0]),
            "target_dft_mag_density": 0.2,
            "guidance_scale": 2.0,
            "guidance_schedule": "constant",
            "adaptive_cfg_used": False,
            "sampling_steps": 1000,
            "corrector_steps_per_timestep": 1,
            "element_masking": "official_mask_disallowed_elements",
            "observation_phase": "predictor_pre_update_after_official_corrector",
            "clean_estimate_not_raw_sample": True,
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "checkpoint_sha256": EXPECTED_SHA256,
            "surrogate_property_eval": True,
            "dft_verified": False,
            **metrics,
        }
        (partial / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial, run_dir)
        print(json.dumps({"event": "complete", "seed": seed, "ordinal": ordinal,
                          "total": len(seeds), "seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
