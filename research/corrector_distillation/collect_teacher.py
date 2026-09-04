"""Collect one exact MatterGen Predictor-Corrector teacher trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from hydra.utils import instantiate

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--guidance-schedule", choices=("constant", "adaptive"), default="constant")
    parser.add_argument("--target", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--storage-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    output_root = args.output_root.expanduser().resolve()
    run_dir = output_root / "teacher_runs" / args.split / str(args.seed)
    teacher_dir = output_root / "teacher_data" / args.split / str(args.seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoint_path = checkpoint_root / "checkpoints" / "last.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    residual_overrides = [
        "sampler_partial.corrector_residual_adapter.enabled=true",
        "sampler_partial.corrector_residual_adapter.mode=teacher",
        f"sampler_partial.corrector_residual_adapter.teacher_output_dir={teacher_dir}",
        f"sampler_partial.corrector_residual_adapter.teacher_split={args.split}",
        f"sampler_partial.corrector_residual_adapter.teacher_shard_size={args.shard_size}",
        f"sampler_partial.corrector_residual_adapter.teacher_storage_dtype={args.storage_dtype}",
        f"sampler_partial.corrector_residual_adapter.sample_seed={args.seed}",
    ]
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(checkpoint_root)),
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={"dft_mag_density": args.target},
        diffusion_guidance_factor=args.guidance_scale,
        guidance_schedule=args.guidance_schedule,
        guidance_adaptive_alpha=0.5,
        guidance_adaptive_ema=0.95,
        guidance_adaptive_eps=1.0e-6,
        guidance_min_scale=0.0,
        guidance_max_scale=5.0,
        sampling_config_overrides=residual_overrides,
        seed=args.seed,
        deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator._seed_sampling_rngs()
    sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
    condition_loader = generator.get_condition_loader(sampling_config)
    sampler = instantiate(sampling_config.sampler_partial)(pl_module=generator.model)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    structures = draw_samples_from_sampler(
        sampler=sampler,
        condition_loader=condition_loader,
        properties_to_condition_on={"dft_mag_density": args.target},
        output_path=run_dir,
        cfg=generator.cfg,
        record_trajectories=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = dict(sampler.sampling_metrics)
    summary = {
        "success": True,
        "seed": args.seed,
        "split": args.split,
        "guidance_schedule": args.guidance_schedule,
        "target": {"dft_mag_density": args.target},
        "guidance_scale": args.guidance_scale,
        "elapsed_seconds": elapsed,
        "samples_per_hour": 3600.0 / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "formula": structures[0].composition.reduced_formula,
        "num_atoms": len(structures[0]),
        **metrics,
    }
    with (run_dir / "run_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()

