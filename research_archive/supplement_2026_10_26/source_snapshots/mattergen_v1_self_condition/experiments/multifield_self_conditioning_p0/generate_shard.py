"""Generate a seed shard while loading the official/SC checkpoints only once."""
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

from generate_one import install_sampling_cache, load_sc_checkpoint, sha256


METHODS = ("B0", "S-off", "S-on")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--sc-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.method == "B0") != (args.sc_checkpoint is None):
        raise ValueError("B0 takes no SC checkpoint; S-off/S-on require one")
    if any(not 95000 <= seed <= 95031 for seed in args.seeds):
        raise ValueError("seed outside preregistered B cohort")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    checkpoint_root = args.checkpoint_root.resolve()
    checkpoint_path = checkpoint_root / "checkpoints/last.ckpt"
    checkpoint_hash = sha256(checkpoint_path)
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(checkpoint_root)),
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={"dft_mag_density": 0.2},
        diffusion_guidance_factor=2.0,
        guidance_schedule="constant",
        seed=args.seeds[0],
        deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    denoiser = generator.model.diffusion_module.model
    sc_hash = None
    if args.sc_checkpoint is not None:
        sc_hash = load_sc_checkpoint(
            generator.model,
            args.sc_checkpoint.resolve(),
            enabled=args.method == "S-on",
        )
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)

    for seed in args.seeds:
        run_dir = args.output_root.resolve() / "generation" / args.method / str(seed)
        if (run_dir / "run_summary.json").is_file():
            print(json.dumps({"event": "preexisting", "method": args.method, "seed": seed}), flush=True)
            continue
        if run_dir.exists():
            raise RuntimeError(f"incomplete run exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        generator.seed = seed
        generator._seed_sampling_rngs()
        sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
        if (
            int(sampling_config.sampler_partial.N) != 1000
            or float(sampling_config.sampler_partial.guidance_scale) != 2.0
            or str(sampling_config.sampler_partial.guidance_schedule) != "constant"
        ):
            raise RuntimeError("frozen sampling configuration changed")
        condition_loader = generator.get_condition_loader(sampling_config)
        sampler = instantiate(sampling_config.sampler_partial)(pl_module=generator.model)
        cache_state = {"updates": 0, "uses": 0}
        if args.method == "S-on":
            cache_state = install_sampling_cache(sampler, denoiser)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=condition_loader,
            properties_to_condition_on={"dft_mag_density": 0.2},
            output_path=run_dir,
            cfg=generator.cfg,
            record_trajectories=False,
        )
        torch.cuda.synchronize()
        metrics = dict(sampler.sampling_metrics)
        if int(metrics["mattergen_score_calls"]) != 2000:
            raise RuntimeError("sampling score-call contract changed")
        if args.method == "S-on" and cache_state != {"updates": 2000, "uses": 1999}:
            raise RuntimeError(f"self-conditioning cache contract failed: {cache_state}")
        summary = {
            "success": True,
            "method": args.method,
            "seed": seed,
            "target": {"dft_mag_density": 0.2},
            "guidance_scale": 2.0,
            "guidance_schedule": "constant",
            "adaptive_cfg_used": False,
            "sampling_steps": 1000,
            "corrector_steps_per_time": 1,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "sc_checkpoint": str(args.sc_checkpoint.resolve()) if args.sc_checkpoint else None,
            "sc_checkpoint_sha256": sc_hash,
            "self_conditioning_enabled": args.method == "S-on",
            "sc_cache_updates": cache_state["updates"],
            "sc_cache_uses": cache_state["uses"],
            "extra_sampling_forwards": 0,
            "formula": structures[0].composition.reduced_formula,
            "num_atoms": len(structures[0]),
            "surrogate_property_eval": True,
            "dft_verified": False,
            **metrics,
        }
        with (run_dir / "run_summary.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
