"""Generate one frozen 1000-step baseline or Cell--Atom Fusion sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from hydra.utils import instantiate

from mattergen.cell_atom_fusion import CellAtomBidirectionalFusion
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


METHODS = ("B0", "C0")
EXPECTED_BASE_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target", type=float, default=0.2)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.method == "B0") != (args.fusion_checkpoint is None):
        raise ValueError("B0 takes no fusion; C0 requires the trained fusion checkpoint")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoint_path = checkpoint_root / "checkpoints/last.ckpt"
    if sha256(checkpoint_path) != EXPECTED_BASE_SHA256:
        raise RuntimeError("official base checkpoint hash mismatch")
    final_dir = (
        args.output_root.expanduser().resolve() / "generation" / args.method / str(args.seed)
    )
    partial_dir = final_dir.with_name(final_dir.name + ".partial")
    if final_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite {final_dir} or {partial_dir}")
    partial_dir.mkdir(parents=True)

    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(checkpoint_root)),
        batch_size=1, num_batches=1,
        properties_to_condition_on={"dft_mag_density": args.target},
        diffusion_guidance_factor=args.guidance_scale,
        guidance_schedule="constant", seed=args.seed,
        deterministic=True, record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()

    fusion_path = None
    fusion_hash = None
    fusion = None
    if args.fusion_checkpoint is not None:
        fusion_path = args.fusion_checkpoint.expanduser().resolve()
        checkpoint = torch.load(fusion_path, map_location="cpu", weights_only=False)
        if checkpoint.get("base_checkpoint_sha256") != EXPECTED_BASE_SHA256:
            raise RuntimeError("fusion checkpoint does not match the official base")
        architecture = checkpoint.get("architecture", {})
        if architecture.get("architecture") != "cell_atom_bidirectional_fusion_v1":
            raise RuntimeError(f"unexpected fusion architecture: {architecture}")
        device = next(generator.model.parameters()).device
        fusion = CellAtomBidirectionalFusion().to(device)
        fusion.load_state_dict(checkpoint["fusion_state_dict"], strict=True)
        fusion.eval()
        generator.model.diffusion_module.model.gemnet.cell_atom_fusion = fusion
        fusion_hash = sha256(fusion_path)

    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    generator._seed_sampling_rngs()
    sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
    if (
        int(sampling_config.sampler_partial.N) != 1000
        or int(sampling_config.sampler_partial.n_steps_corrector) != 1
        or float(sampling_config.sampler_partial.guidance_scale) != 2.0
        or str(sampling_config.sampler_partial.guidance_schedule) != "constant"
    ):
        raise RuntimeError("frozen sampling contract changed")
    sampler = instantiate(sampling_config.sampler_partial)(pl_module=generator.model)
    condition_loader = generator.get_condition_loader(sampling_config)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    structures = draw_samples_from_sampler(
        sampler=sampler, condition_loader=condition_loader,
        properties_to_condition_on={"dft_mag_density": args.target},
        output_path=partial_dir, cfg=generator.cfg, record_trajectories=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    sampling_metrics = dict(sampler.sampling_metrics)
    if int(sampling_metrics.get("mattergen_score_calls", -1)) != 2000:
        raise RuntimeError(f"score-call contract changed: {sampling_metrics}")
    summary = {
        "success": True, "method": args.method, "seed": args.seed,
        "target": {"dft_mag_density": args.target},
        "guidance_scale": args.guidance_scale, "guidance_schedule": "constant",
        "adaptive_cfg_used": False, "sampling_steps": 1000,
        "corrector_steps_per_timestep": 1,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "base_checkpoint_path": str(checkpoint_path),
        "base_checkpoint_sha256": EXPECTED_BASE_SHA256,
        "fusion_checkpoint": str(fusion_path) if fusion_path else None,
        "fusion_checkpoint_sha256": fusion_hash,
        "fusion_stats_final_forward": dict(fusion.last_stats) if fusion else None,
        "formula": structures[0].composition.reduced_formula,
        "num_atoms": len(structures[0]),
        "surrogate_property_eval": True, "dft_verified": False,
        **sampling_metrics,
    }
    (partial_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial_dir, final_dir)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
