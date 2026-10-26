"""Generate one official trajectory and save complete late clean estimates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from hydra.utils import instantiate

# The available CUDA runtime supplies NumPy 2.x, while the frozen MatterGen
# GemNet basis helper still references the removed `np.math` compatibility
# alias. Restoring that alias is semantics-preserving and avoids editing the
# frozen model implementation.
if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/mattersim_late_force_guidance_p0"
MODEL_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")
EXPECTED_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
ELEMENT_MASK_OVERRIDE = (
    "++lightning_module.diffusion_module.model.element_mask_func="
    "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if not 710000 <= args.seed <= 710031:
        raise ValueError("seed outside frozen A0 cohort")
    checkpoint = MODEL_ROOT / "checkpoints/last.ckpt"
    if sha256(checkpoint) != EXPECTED_SHA256:
        raise RuntimeError("official checkpoint hash mismatch")
    run_dir = ROOT / "diagnostic_trajectories" / str(args.seed)
    partial = run_dir.with_name(run_dir.name + ".partial")
    if (run_dir / "run_summary.json").is_file():
        print(json.dumps({"event": "skip", "seed": args.seed}), flush=True)
        return
    if run_dir.exists() or partial.exists():
        raise RuntimeError(f"incomplete output retained: {run_dir} or {partial}")
    partial.mkdir(parents=True)

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
        batch_size=1, num_batches=1,
        properties_to_condition_on={"dft_mag_density": 0.2},
        diffusion_guidance_factor=2.0, guidance_schedule="constant",
        seed=args.seed, deterministic=True, record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
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
        "experiments.mattersim_late_force_guidance_p0.late_clean_sampler."
        "LateCleanEstimateSampler.from_pl_module"
    )
    sampler = instantiate(sampling_config.sampler_partial)(
        pl_module=generator.model,
        output_dir=str((partial / "clean_estimates").resolve()),
        diagnostic_seed=args.seed,
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    structures = draw_samples_from_sampler(
        sampler=sampler,
        condition_loader=generator.get_condition_loader(sampling_config),
        properties_to_condition_on={"dft_mag_density": 0.2},
        output_path=partial, cfg=generator.cfg, record_trajectories=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = dict(sampler.sampling_metrics)
    if int(metrics["mattergen_score_calls"]) != 2000:
        raise RuntimeError("score-call contract changed")
    if int(metrics["complete_clean_estimate_count"]) != 4:
        raise RuntimeError("clean-estimate capture count changed")
    summary = {
        "success": True, "seed": args.seed,
        "subset": "calibration" if args.seed <= 710015 else "validation",
        "formula": structures[0].composition.reduced_formula,
        "num_atoms": len(structures[0]), "elapsed_seconds": elapsed,
        "physical_gpu": os.environ.get("A_PHYSICAL_GPU", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "target_dft_mag_density": 0.2, "guidance_scale": 2.0,
        "guidance_schedule": "constant", "adaptive_cfg_used": False,
        "sampling_steps": 1000, "corrector_steps_per_timestep": 1,
        "checkpoint_sha256": EXPECTED_SHA256,
        "clean_estimate_not_raw_xt": True,
        "surrogate_property_eval": True, "dft_verified": False,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        **metrics,
    }
    (partial / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, run_dir)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
