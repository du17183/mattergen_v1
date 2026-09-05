"""Collect exact teacher labels on states visited by an Adapter rollout."""

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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generation-label")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("train", "validation", "smoke"), required=True)
    parser.add_argument("--coverage", type=float, default=0.5)
    parser.add_argument("--target", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--storage-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.coverage <= 1.0:
        raise ValueError("coverage must be in [0, 1]")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    output_root = args.output_root.expanduser().resolve()
    run_dir = (
        output_root / "generation" / args.generation_label / str(args.seed)
        if args.generation_label
        else output_root / "dagger_runs" / args.split / str(args.seed)
    )
    dagger_dir = output_root / "dagger_data" / args.split / str(args.seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoint_path = checkpoint_root / "checkpoints" / "last.ckpt"
    adapter_checkpoint = args.adapter_checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not adapter_checkpoint.is_file():
        raise FileNotFoundError(adapter_checkpoint)

    trace_path = run_dir / "residual_trace.csv"
    overrides = [
        "sampler_partial.corrector_residual_adapter.enabled=true",
        "sampler_partial.corrector_residual_adapter.mode=adapter",
        f"sampler_partial.corrector_residual_adapter.checkpoint_path={adapter_checkpoint}",
        f"sampler_partial.corrector_residual_adapter.coverage_target={args.coverage}",
        "sampler_partial.corrector_residual_adapter.risk_mode=global",
        f"sampler_partial.corrector_residual_adapter.trace_path={trace_path}",
        f"sampler_partial.corrector_residual_adapter.dagger_output_dir={dagger_dir}",
        f"sampler_partial.corrector_residual_adapter.dagger_split={args.split}",
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
        guidance_schedule="constant",
        sampling_config_overrides=overrides,
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
    policy_second_forward_calls = int(metrics["fallback_calls"])
    summary = {
        "success": True,
        "seed": args.seed,
        "split": args.split,
        "trajectory_kind": "dagger",
        "method": args.generation_label,
        "rollout_policy": "on-policy Adapter+Fallback",
        "coverage_target": args.coverage,
        "target": {"dft_mag_density": args.target},
        "guidance_scale": args.guidance_scale,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "adapter_checkpoint": str(adapter_checkpoint),
        "adapter_checkpoint_sha256": sha256(adapter_checkpoint),
        "dagger_manifest": str((dagger_dir / "manifest.json").resolve()),
        "policy_second_forward_calls": policy_second_forward_calls,
        "policy_logical_score_calls": 1000 + policy_second_forward_calls,
        "policy_saved_score_calls": 1000 - policy_second_forward_calls,
        "formula": structures[0].composition.reduced_formula,
        "num_atoms": len(structures[0]),
        **metrics,
    }
    with (run_dir / "run_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
