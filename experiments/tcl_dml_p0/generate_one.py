"""Generate one frozen C0, FT0, TCL, or DML P0 sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from hydra.utils import instantiate

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


METHODS = ("C0", "FT0", "TCL", "DML")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.method == "C0") != (args.finetuned_checkpoint is None):
        raise ValueError("C0 takes no finetuned checkpoint; trained methods require one")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    base_checkpoint = checkpoint_root / "checkpoints/last.ckpt"
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    run_dir = args.output_root.resolve() / "generation" / args.method / str(args.seed)
    run_dir.mkdir(parents=True, exist_ok=False)

    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(checkpoint_root)),
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={"dft_mag_density": args.target},
        diffusion_guidance_factor=args.guidance_scale,
        guidance_schedule="constant",
        seed=args.seed,
        deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()

    finetuned_path = None
    finetuned_hash = None
    if args.finetuned_checkpoint is not None:
        finetuned_path = args.finetuned_checkpoint.resolve()
        checkpoint = torch.load(finetuned_path, map_location="cpu", weights_only=False)
        found_method = checkpoint.get("method")
        if found_method != args.method:
            raise ValueError(
                f"checkpoint mismatch: requested {args.method}, found {found_method}"
            )
        generator.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        finetuned_hash = sha256(finetuned_path)
        del checkpoint
    gemnet = generator.model.diffusion_module.model.gemnet
    for attribute in ("global_adapter", "cross_field_adapter", "quality_adapter"):
        setattr(gemnet, attribute, None)
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    generator._seed_sampling_rngs()
    sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
    if int(sampling_config.sampler_partial.N) != 1000:
        raise RuntimeError("sampling config is not the frozen 1000-step setup")
    if int(sampling_config.sampler_partial.n_steps_corrector) != 1:
        raise RuntimeError("sampling config does not use the original Corrector")
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
    summary = {
        "success": True,
        "method": args.method,
        "seed": args.seed,
        "target": {"dft_mag_density": args.target},
        "guidance_scale": args.guidance_scale,
        "guidance_schedule": "constant",
        "sampling_steps": 1000,
        "corrector_steps_per_time": 1,
        "batch_size": 1,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "base_checkpoint_sha256": sha256(base_checkpoint),
        "finetuned_checkpoint": str(finetuned_path) if finetuned_path else None,
        "finetuned_checkpoint_sha256": finetuned_hash,
        "formula": structures[0].composition.reduced_formula,
        "num_atoms": len(structures[0]),
        "dft_verified": False,
        **dict(sampler.sampling_metrics),
    }
    with (run_dir / "run_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
