"""Isolated replay of the frozen Formal256 C0/F0 seed 740000 pair.

This writes only to this experiment's replay directory. It intentionally
reuses the frozen P0 sampler without changing its implementation or settings.
"""
from __future__ import annotations

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


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
FROZEN = PROJECT / "experiments/mattersim_late_force_guidance_formal256"
MODEL_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")
MATTERSIM = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth")
MODEL_SHA = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
MATTERSIM_SHA = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
SAMPLER_SHA = "7a1598338e305986f03f232b719cdbb90687ba3388b8827ace8aa191bb8d9523"
ELEMENT_MASK_OVERRIDE = (
    "++lightning_module.diffusion_module.model.element_mask_func="
    "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    frozen = yaml.safe_load((FROZEN / "formal256_config.yaml").read_text(encoding="utf-8"))
    common, force = frozen["common"], frozen["F0"]
    if 740000 not in frozen["seeds"] or force["guidance_t_max"] != 0.02:
        raise RuntimeError("Formal256 protocol differs from locked seed/method")
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA:
        raise RuntimeError("MatterGen checkpoint hash mismatch")
    if sha256(MATTERSIM) != MATTERSIM_SHA:
        raise RuntimeError("MatterSim checkpoint hash mismatch")
    if sha256(PROJECT / "experiments/mattersim_late_force_guidance_p0/late_force_sampler.py") != SAMPLER_SHA:
        raise RuntimeError("frozen force sampler source hash mismatch")
    implementation_lock = json.loads((FROZEN / "implementation_lock.json").read_text(encoding="utf-8"))
    for source, expected in implementation_lock["sha256"].items():
        if "/mattergen/diffusion/sampling/" not in source:
            continue
        if sha256(Path(source)) != expected:
            raise RuntimeError(f"frozen MatterGen sampler source hash mismatch: {source}")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(MODEL_ROOT), config_overrides=[ELEMENT_MASK_OVERRIDE]),
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={"dft_mag_density": float(common["target_dft_mag_density"])},
        diffusion_guidance_factor=float(common["cfg"]),
        guidance_schedule=str(common["guidance_schedule"]),
        seed=740000,
        deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)

    for method in ("C0", "F0"):
        run_dir = ROOT / "replay_740000" / method
        if run_dir.exists():
            raise FileExistsError(f"replay output already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        config = generator.load_sampling_config(batch_size=1, num_batches=1)
        if (
            int(config.sampler_partial.N) != int(common["diffusion_steps"])
            or int(config.sampler_partial.n_steps_corrector) != int(common["corrector_steps_per_timestep"])
            or float(config.sampler_partial.guidance_scale) != float(common["cfg"])
            or str(config.sampler_partial.guidance_schedule) != str(common["guidance_schedule"])
        ):
            raise RuntimeError("frozen sampling contract mismatch")
        kwargs = {"pl_module": generator.model}
        if method == "F0":
            config.sampler_partial._target_ = (
                "experiments.mattersim_late_force_guidance_p0.late_force_sampler."
                "LateMatterSimForceGuidedSampler.from_pl_module"
            )
            kwargs.update(
                trace_path=str((run_dir / "late_force_trace.csv").resolve()),
                sample_seed=740000,
                mattersim_checkpoint=str(MATTERSIM),
                guidance_t_max=float(force["guidance_t_max"]),
                force_reference_ev_a=float(force["force_reference_ev_a"]),
                nominal_cart_step_a=float(force["nominal_cartesian_step_a"]),
                hard_cart_cap_a=float(force["hard_cartesian_cap_a"]),
                min_distance_floor_a=float(force["min_periodic_distance_floor_a"]),
            )
        sampler = instantiate(config.sampler_partial)(**kwargs)
        generator.seed = 740000
        generator._seed_sampling_rngs()
        torch.cuda.synchronize()
        start = time.perf_counter()
        structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=generator.get_condition_loader(config),
            properties_to_condition_on={"dft_mag_density": float(common["target_dft_mag_density"])},
            output_path=run_dir,
            cfg=generator.cfg,
            record_trajectories=False,
        )
        torch.cuda.synchronize()
        metrics = dict(sampler.sampling_metrics)
        if metrics["mattergen_score_calls"] != 2000:
            raise RuntimeError("historical score-call contract mismatch")
        if method == "F0" and metrics["mattersim_guidance_eligible"] != 20:
            raise RuntimeError("historical force trigger count mismatch")
        summary = {
            "method": method,
            "seed": 740000,
            "formula": structures[0].composition.reduced_formula,
            "num_atoms": len(structures[0]),
            "elapsed_seconds": time.perf_counter() - start,
            "extxyz_sha256": sha256(run_dir / "generated_crystals.extxyz"),
            "model_sha256": MODEL_SHA,
            "mattersim_sha256": MATTERSIM_SHA,
            "sampler_sha256": SAMPLER_SHA,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            **metrics,
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
