"""Generate one frozen B0/A0 adaptive-neighborhood P0 sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from hydra.utils import instantiate

from mattergen.adaptive_neighborhood import TimestepRadialGate
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


METHODS = ("B0", "A0")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target", type=float, default=0.2)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.method == "B0") != (args.gate_checkpoint is None):
        raise ValueError("B0 takes no gate checkpoint; A0 requires one")
    if not 96000 <= args.seed <= 96031:
        raise ValueError("seed is outside the preregistered fresh A cohort")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoint_path = checkpoint_root / "checkpoints/last.ckpt"
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
    gate_hash = None
    gate_values = None
    if args.gate_checkpoint is not None:
        gate_path = args.gate_checkpoint.expanduser().resolve()
        checkpoint = torch.load(gate_path, map_location="cpu", weights_only=False)
        if checkpoint.get("method") != "Diffusion-Stage Adaptive Neighborhood GemNet-T":
            raise ValueError("radial gate checkpoint method mismatch")
        gate = TimestepRadialGate()
        gate.load_state_dict(checkpoint["gate_state_dict"], strict=True)
        device = next(generator.model.parameters()).device
        gate.to(device).eval()
        generator.model.diffusion_module.model.gemnet.adaptive_radial_gate = gate
        gate_hash = sha256(gate_path)
        with torch.no_grad():
            gate_values = gate.gate_values(
                torch.tensor([0.8, 0.5, 0.25, 0.1, 0.05, 0.02], device=device)
            ).cpu().tolist()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    generator._seed_sampling_rngs()
    sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
    if int(sampling_config.sampler_partial.N) != 1000:
        raise RuntimeError("sampling steps differ from 1000")
    if float(sampling_config.sampler_partial.guidance_scale) != 2.0:
        raise RuntimeError("CFG differs from 2.0")
    if str(sampling_config.sampler_partial.guidance_schedule) != "constant":
        raise RuntimeError("Adaptive CFG is prohibited")
    condition_loader = generator.get_condition_loader(sampling_config)
    sampler = instantiate(sampling_config.sampler_partial)(pl_module=generator.model)
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
    torch.cuda.synchronize()
    metrics = dict(sampler.sampling_metrics)
    if int(metrics["mattergen_score_calls"]) != 2000:
        raise RuntimeError("sampling score-call contract changed")
    summary = {
        "success": True,
        "method": args.method,
        "seed": args.seed,
        "target": {"dft_mag_density": args.target},
        "guidance_scale": args.guidance_scale,
        "guidance_schedule": "constant",
        "adaptive_cfg_used": False,
        "sampling_steps": 1000,
        "corrector_steps_per_time": 1,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "gate_checkpoint": str(args.gate_checkpoint.resolve()) if args.gate_checkpoint else None,
        "gate_checkpoint_sha256": gate_hash,
        "gate_values_t_0p8_0p5_0p25_0p1_0p05_0p02": gate_values,
        "graph_topology_modified": False,
        "cutoff_angstrom": 7.0,
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
