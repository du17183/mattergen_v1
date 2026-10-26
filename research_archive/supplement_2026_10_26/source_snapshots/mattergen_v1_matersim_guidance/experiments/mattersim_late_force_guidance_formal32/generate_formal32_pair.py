"""Generate one frozen C0/F0 Formal32 pair using the exact P0 guidance class."""
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
import yaml
from hydra.utils import instantiate

if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/mattersim_late_force_guidance_formal32"
P0 = PROJECT_ROOT / "experiments/mattersim_late_force_guidance_p0"
MODEL_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/"
    "checkpoints/dft_mag_density"
)
MATTERSIM = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/"
    "mattersim-v1.0.0-5M.pth"
)
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


def load_protocol() -> tuple[dict, tuple[int, ...]]:
    protocol = yaml.safe_load((ROOT / "formal32_config.yaml").read_text(encoding="utf-8"))
    seed_file = json.loads((ROOT / "formal32_seeds.json").read_text(encoding="utf-8"))
    seeds = tuple(int(item["seed"]) for item in seed_file["paired_seeds"])
    if seeds != tuple(range(730000, 730032)) or protocol["paired_seeds"] != list(seeds):
        raise RuntimeError("Formal32 seed lock mismatch")
    for name, expected in protocol["source"]["sha256"].items():
        path = Path(protocol["source"]["files"][name])
        if sha256(path) != expected:
            raise RuntimeError(f"P0 source changed after Formal32 freeze: {name}")
    return protocol, seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    protocol, seeds = load_protocol()
    if args.seed not in seeds:
        raise ValueError("seed outside frozen Formal32 cohort")
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA:
        raise RuntimeError("MatterGen checkpoint hash mismatch")
    if sha256(MATTERSIM) != MATTERSIM_SHA:
        raise RuntimeError("MatterSim checkpoint hash mismatch")

    common = protocol["common"]
    frozen = protocol["F0"]
    if frozen["field"] != "position_only" or frozen["guidance_lambda_sweep"] is not False:
        raise RuntimeError("Formal32 method is not the frozen position-only P0")
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
        batch_size=int(common["batch_size"]), num_batches=1,
        properties_to_condition_on={"dft_mag_density": float(common["target_dft_mag_density"])},
        diffusion_guidance_factor=float(common["cfg"]),
        guidance_schedule=str(common["guidance_schedule"]),
        seed=args.seed, deterministic=True, record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)

    for method in ("C0", "F0"):
        run_dir = ROOT / "generation" / method / str(args.seed)
        partial = run_dir.with_name(run_dir.name + ".partial")
        if (run_dir / "run_summary.json").is_file():
            continue
        if run_dir.exists() or partial.exists():
            raise RuntimeError(f"incomplete output retained: {run_dir} or {partial}")
        partial.mkdir(parents=True)
        sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
        if (
            int(sampling_config.sampler_partial.N) != int(common["diffusion_steps"])
            or float(sampling_config.sampler_partial.guidance_scale) != float(common["cfg"])
            or str(sampling_config.sampler_partial.guidance_schedule) != str(common["guidance_schedule"])
            or int(sampling_config.sampler_partial.n_steps_corrector)
            != int(common["corrector_steps_per_timestep"])
        ):
            raise RuntimeError("official frozen sampling contract changed")
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
                sample_seed=args.seed,
                mattersim_checkpoint=str(MATTERSIM),
                guidance_t_max=float(frozen["guidance_t_max"]),
                force_reference_ev_a=float(frozen["force_reference_ev_a"]),
                nominal_cart_step_a=float(frozen["nominal_cartesian_step_a"]),
                hard_cart_cap_a=float(frozen["hard_cartesian_cap_a"]),
                min_distance_floor_a=float(frozen["min_periodic_distance_floor_a"]),
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
            properties_to_condition_on={"dft_mag_density": float(common["target_dft_mag_density"])},
            output_path=partial, cfg=generator.cfg, record_trajectories=False,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = dict(sampler.sampling_metrics)
        if int(metrics["mattergen_score_calls"]) != 2000:
            raise RuntimeError("MatterGen score-call contract changed")
        if method == "F0" and int(metrics["mattersim_guidance_eligible"]) != int(
            protocol["expected_guidance_events_per_sample"]
        ):
            raise RuntimeError("Formal32 guidance trigger count differs from P0")
        summary = {
            "success": True, "method": method, "seed": args.seed,
            "formula": structures[0].composition.reduced_formula,
            "num_atoms": len(structures[0]), "elapsed_seconds": elapsed,
            "potential_load_seconds": potential_load_seconds,
            "end_to_end_seconds": elapsed + potential_load_seconds,
            "physical_gpu": os.environ.get("FORMAL32_PHYSICAL_GPU", "unknown"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "target_dft_mag_density": float(common["target_dft_mag_density"]),
            "guidance_scale": float(common["cfg"]),
            "guidance_schedule": str(common["guidance_schedule"]),
            "adaptive_cfg_used": False,
            "sampling_steps": int(common["diffusion_steps"]),
            "corrector_steps_per_timestep": int(common["corrector_steps_per_timestep"]),
            "checkpoint_sha256": MODEL_SHA,
            "mattersim_checkpoint_sha256": MATTERSIM_SHA if method == "F0" else None,
            "position_force_guidance": method == "F0",
            "stress_guidance": False, "cell_guidance": False,
            "atomic_guidance": False, "energy_guidance": False,
            "dft_verified": False,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "p0_guidance_source_sha256": protocol["source"]["sha256"]["guidance_implementation"],
            **metrics,
        }
        (partial / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial, run_dir)
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
