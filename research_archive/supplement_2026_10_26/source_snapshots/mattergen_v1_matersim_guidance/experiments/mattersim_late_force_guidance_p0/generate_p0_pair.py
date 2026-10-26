"""Generate one fresh B0/F0 pair under the frozen P0 protocol."""
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

if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/mattersim_late_force_guidance_p0"
MODEL_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")
MATTERSIM = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth")
MODEL_SHA = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
MATTERSIM_SHA = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
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
    if not 710032 <= args.seed <= 710047:
        raise ValueError("seed outside frozen A P0 cohort")
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA:
        raise RuntimeError("MatterGen checkpoint hash mismatch")
    if sha256(MATTERSIM) != MATTERSIM_SHA:
        raise RuntimeError("MatterSim checkpoint hash mismatch")
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

    for method in ("B0", "F0"):
        run_dir = ROOT / "p0_generation" / method / str(args.seed)
        partial = run_dir.with_name(run_dir.name + ".partial")
        if (run_dir / "run_summary.json").is_file():
            continue
        if run_dir.exists() or partial.exists():
            raise RuntimeError(f"incomplete output retained: {run_dir} or {partial}")
        partial.mkdir(parents=True)
        sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
        if (
            int(sampling_config.sampler_partial.N) != 1000
            or float(sampling_config.sampler_partial.guidance_scale) != 2.0
            or str(sampling_config.sampler_partial.guidance_schedule) != "constant"
            or int(sampling_config.sampler_partial.n_steps_corrector) != 1
        ):
            raise RuntimeError("official sampling contract changed")
        sampler_kwargs = {"pl_module": generator.model}
        potential_load_started = None
        potential_load_seconds = 0.0
        if method == "F0":
            sampling_config.sampler_partial._target_ = (
                "experiments.mattersim_late_force_guidance_p0.late_force_sampler."
                "LateMatterSimForceGuidedSampler.from_pl_module"
            )
            sampler_kwargs.update(
                trace_path=str((partial / "late_force_trace.csv").resolve()),
                sample_seed=args.seed, mattersim_checkpoint=str(MATTERSIM),
                guidance_t_max=0.02,
                force_reference_ev_a=0.07795149218357911,
                nominal_cart_step_a=0.005, hard_cart_cap_a=0.01,
                min_distance_floor_a=0.5,
            )
            potential_load_started = time.perf_counter()
        sampler = instantiate(sampling_config.sampler_partial)(**sampler_kwargs)
        if potential_load_started is not None:
            potential_load_seconds = time.perf_counter() - potential_load_started
        generator.seed = args.seed
        generator._seed_sampling_rngs()
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
        summary = {
            "success": True, "method": method, "seed": args.seed,
            "formula": structures[0].composition.reduced_formula,
            "num_atoms": len(structures[0]), "elapsed_seconds": elapsed,
            "potential_load_seconds": potential_load_seconds,
            "end_to_end_seconds": elapsed + potential_load_seconds,
            "physical_gpu": os.environ.get("A_PHYSICAL_GPU", "unknown"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "target_dft_mag_density": 0.2, "guidance_scale": 2.0,
            "guidance_schedule": "constant", "adaptive_cfg_used": False,
            "sampling_steps": 1000, "corrector_steps_per_timestep": 1,
            "checkpoint_sha256": MODEL_SHA,
            "mattersim_checkpoint_sha256": MATTERSIM_SHA if method == "F0" else None,
            "position_force_guidance": method == "F0",
            "stress_guidance": False, "cell_guidance": False,
            "atomic_guidance": False, "dft_verified": False,
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
