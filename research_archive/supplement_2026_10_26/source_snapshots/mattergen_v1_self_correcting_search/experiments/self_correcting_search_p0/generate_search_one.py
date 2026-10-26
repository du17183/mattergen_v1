"""Generate one paired held-out B0 or single-proposal S0 structure."""
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/self_correcting_search_p0"
MODEL_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")
PROBE = ROOT / "probe_checkpoint.pt"
EXPECTED_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
METHODS = ("B0", "S0")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 98200 <= args.seed <= 98207:
        raise ValueError("seed outside preregistered independent search cohort")
    checkpoint_path = MODEL_ROOT / "checkpoints/last.ckpt"
    if sha256(checkpoint_path) != EXPECTED_SHA256:
        raise RuntimeError("official checkpoint hash mismatch")
    if args.method == "S0":
        checkpoint = torch.load(PROBE, map_location="cpu", weights_only=False)
        if not checkpoint.get("probe_go", False):
            raise RuntimeError("S0 prohibited because Probe P0 did not pass")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run_dir = ROOT / "search_generation" / args.method / str(args.seed)
    partial = run_dir.with_name(run_dir.name + ".partial")
    if (run_dir / "run_summary.json").is_file():
        print(json.dumps({"event": "skip", "method": args.method, "seed": args.seed}))
        return
    if run_dir.exists() or partial.exists():
        raise RuntimeError(f"incomplete search output retained: {run_dir} or {partial}")
    partial.mkdir(parents=True)

    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(MODEL_ROOT)),
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={"dft_mag_density": 0.2},
        diffusion_guidance_factor=2.0,
        guidance_schedule="constant",
        seed=args.seed,
        deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    generator._seed_sampling_rngs()
    sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
    if int(sampling_config.sampler_partial.N) != 1000:
        raise RuntimeError("sampling steps changed")
    if float(sampling_config.sampler_partial.guidance_scale) != 2.0:
        raise RuntimeError("CFG changed")
    if str(sampling_config.sampler_partial.guidance_schedule) != "constant":
        raise RuntimeError("adaptive CFG is prohibited")
    if args.method == "S0":
        sampling_config.sampler_partial._target_ = (
            "experiments.self_correcting_search_p0.self_correcting_sampler."
            "SelfCorrectingGuidedPredictorCorrector.from_pl_module"
        )
    condition_loader = generator.get_condition_loader(sampling_config)
    sampler_partial = instantiate(sampling_config.sampler_partial)
    extra = {}
    if args.method == "S0":
        extra = {
            "probe_checkpoint_path": str(PROBE),
            "search_trace_path": str((partial / "search_trace.csv").resolve()),
            "search_seed": args.seed,
        }
    sampler = sampler_partial(pl_module=generator.model, **extra)
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
    summary = {
        "success": True,
        "method": args.method,
        "seed": args.seed,
        "target": {"dft_mag_density": 0.2},
        "guidance_scale": 2.0,
        "guidance_schedule": "constant",
        "adaptive_cfg_used": False,
        "sampling_steps": 1000,
        "corrector_steps_per_timestep": 1,
        "proposal_count": 1 if args.method == "S0" else 0,
        "probabilistic_accept_reject": args.method == "S0",
        "probe_gradient_used": False,
        "hidden_state_steering_used": False,
        "hard_bounds_used": False,
        "strict_mala_claim": False,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint_sha256": EXPECTED_SHA256,
        "probe_checkpoint_sha256": sha256(PROBE) if args.method == "S0" else None,
        "formula": structures[0].composition.reduced_formula,
        "num_atoms": len(structures[0]),
        "surrogate_property_eval": True,
        "dft_verified": False,
        **metrics,
    }
    (partial / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, run_dir)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
