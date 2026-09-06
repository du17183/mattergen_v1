"""Generate one real 1000-step PC sample for C0, M1, or M2."""
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
from mattergen.quality_adapter import QualityAdapterStack


METHODS = ("C0", "M1", "M2", "M2-L1", "M2-L2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.method == "C0") != (args.adapter_checkpoint is None):
        raise ValueError("C0 takes no adapter checkpoint; M1/M2 require one")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoint_path = checkpoint_root / "checkpoints" / "last.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    run_dir = args.output_root.expanduser().resolve() / "generation" / args.method / str(args.seed)
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

    adapter_path = None
    adapter_hash = None
    adapter_parameter_l2 = None
    if args.adapter_checkpoint is not None:
        adapter_path = args.adapter_checkpoint.expanduser().resolve()
        checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)
        expected_checkpoint_method = "M2" if args.method.startswith("M2-L") else args.method
        if checkpoint.get("method") != expected_checkpoint_method:
            raise ValueError(
                f"adapter method mismatch: requested {args.method} (expected metadata {expected_checkpoint_method}), checkpoint has {checkpoint.get('method')}"
            )
        adapter = QualityAdapterStack(
            hidden_dim=512,
            bottleneck_dim=64,
            block_indices=(1, 2),
            stage_aware=False,
        )
        adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
        device = next(generator.model.parameters()).device
        adapter.to(device).eval()
        generator.model.diffusion_module.model.gemnet.quality_adapter = adapter
        for parameter in generator.model.parameters():
            parameter.requires_grad_(False)
        adapter_parameter_l2 = float(
            torch.sqrt(
                sum(parameter.detach().float().square().sum() for parameter in adapter.parameters())
            ).item()
        )
        adapter_hash = sha256(adapter_path)

    generator.model.eval()
    generator._seed_sampling_rngs()
    sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
    if int(sampling_config.sampler_partial.N) != 1000:
        raise RuntimeError(f"expected 1000 sampling steps, got {sampling_config.sampler_partial.N}")
    if int(sampling_config.sampler_partial.n_steps_corrector) != 1:
        raise RuntimeError("expected one corrector step")
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
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "adapter_checkpoint": str(adapter_path) if adapter_path else None,
        "adapter_checkpoint_sha256": adapter_hash,
        "adapter_parameter_l2": adapter_parameter_l2,
        "formula": structures[0].composition.reduced_formula,
        "num_atoms": len(structures[0]),
        **dict(sampler.sampling_metrics),
    }
    with (run_dir / "run_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
