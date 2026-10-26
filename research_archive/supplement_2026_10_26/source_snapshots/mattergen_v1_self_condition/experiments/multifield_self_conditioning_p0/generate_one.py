"""Generate one frozen B0/S-off/S-on self-conditioning P0 sample."""
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
from mattergen.self_conditioning import MultiFieldSelfConditioner, clean_prediction


METHODS = ("B0", "S-off", "S-on")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sc_checkpoint(model, checkpoint_path: Path, enabled: bool) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "Multi-field Self-Conditioned MatterGen":
        raise ValueError("self-conditioning checkpoint method mismatch")
    denoiser = model.diffusion_module.model
    device = next(model.parameters()).device
    denoiser.self_conditioner = MultiFieldSelfConditioner().to(device)
    parameters = dict(model.named_parameters())
    state = checkpoint["trainable_state_dict"]
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise RuntimeError(f"checkpoint parameters absent from model: {missing[:3]}")
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            if parameter.shape != value.shape:
                raise RuntimeError(f"checkpoint shape mismatch: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    denoiser.self_conditioner.enabled = enabled
    denoiser.clear_self_condition()
    return sha256(checkpoint_path)


def install_sampling_cache(sampler, denoiser) -> dict[str, int]:
    cache_state = {"updates": 0, "uses": 0}
    previous = {"value": None}
    exact = sampler._evaluate_exact_score
    on_start = sampler._on_sampling_start
    on_end = sampler._on_sampling_end

    def reset_start():
        previous["value"] = None
        denoiser.clear_self_condition()
        return on_start()

    def clear_end(error):
        try:
            return on_end(error)
        finally:
            previous["value"] = None
            denoiser.clear_self_condition()

    def cached_exact(x, t):
        if previous["value"] is None:
            denoiser.clear_self_condition()
        else:
            denoiser.set_self_condition(previous["value"])
            cache_state["uses"] += 1
        score = exact(x, t)
        previous["value"] = clean_prediction(
            diffusion=sampler.diffusion_module,
            noisy_batch=x,
            t=t,
            output=score,
            output_is_score=True,
        )
        cache_state["updates"] += 1
        denoiser.clear_self_condition()
        return score

    sampler._on_sampling_start = reset_start
    sampler._on_sampling_end = clear_end
    sampler._evaluate_exact_score = cached_exact
    return cache_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--sc-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target", type=float, default=0.2)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.method == "B0") != (args.sc_checkpoint is None):
        raise ValueError("B0 takes no SC checkpoint; S-off/S-on require one")
    if not 95000 <= args.seed <= 95031:
        raise ValueError("seed is outside the preregistered fresh B cohort")
    torch.set_num_threads(args.cpu_threads)
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
    sc_hash = None
    denoiser = generator.model.diffusion_module.model
    if args.sc_checkpoint is not None:
        sc_path = args.sc_checkpoint.expanduser().resolve()
        sc_hash = load_sc_checkpoint(
            generator.model, sc_path, enabled=args.method == "S-on"
        )

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
    cache_state = {"updates": 0, "uses": 0}
    if args.method == "S-on":
        cache_state = install_sampling_cache(sampler, denoiser)

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
    if args.method == "S-on" and cache_state != {"updates": 2000, "uses": 1999}:
        raise RuntimeError(f"self-conditioning cache contract failed: {cache_state}")
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
