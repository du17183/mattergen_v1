"""Frozen-parameter, four-arm timing ablation for one preregistered paired seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
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
OUT = ROOT / "timing_formal32"
HISTORY = PROJECT / "experiments/mattersim_late_force_guidance_formal256"
MODEL = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")
MATTERSIM = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth")
MODEL_SHA = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
MATTERSIM_SHA = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
FORCE_SHA = "7a1598338e305986f03f232b719cdbb90687ba3388b8827ace8aa191bb8d9523"
ELEMENT_MASK_OVERRIDE = (
    "++lightning_module.diffusion_module.model.element_mask_func="
    "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
)
ALM_PYTHON = Path("/mnt/datasets-livsyn/dxl/alm/.venv/bin/python")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(seed: int) -> tuple[dict, dict, dict]:
    protocol = json.loads((ROOT / "timing_protocol.json").read_text())
    if not protocol["seed_start"] <= seed < protocol["seed_stop_exclusive"]:
        raise ValueError("seed outside preregistered Timing cohort")
    if Path(sys.executable).resolve() != ALM_PYTHON.resolve():
        raise RuntimeError(f"ALM interpreter required: {sys.executable}")
    gate = json.loads((ROOT / "alm_replay_740000/gate.json").read_text())
    if gate["status"] != "PASS" or set(gate["methods"]) != {"C0", "F0"}:
        raise RuntimeError("historical seed 740000 reproducibility gate is not PASS")
    if sha256(MODEL / "checkpoints/last.ckpt") != MODEL_SHA:
        raise RuntimeError("MatterGen checkpoint hash changed")
    if sha256(MATTERSIM) != MATTERSIM_SHA:
        raise RuntimeError("MatterSim checkpoint hash changed")
    force_source = PROJECT / "experiments/mattersim_late_force_guidance_p0/late_force_sampler.py"
    if sha256(force_source) != FORCE_SHA:
        raise RuntimeError("frozen force sampler source changed")
    lock = json.loads((HISTORY / "implementation_lock.json").read_text())
    for source, expected in lock["sha256"].items():
        if "/mattergen/diffusion/sampling/" in source and sha256(Path(source)) != expected:
            raise RuntimeError(f"frozen MatterGen sampling code changed: {source}")
    frozen = yaml.safe_load((HISTORY / "formal256_config.yaml").read_text())
    common, force = frozen["common"], frozen["F0"]
    for key in ("force_reference_ev_a", "nominal_cartesian_step_a",
                "hard_cartesian_cap_a", "min_periodic_distance_floor_a"):
        if key not in force:
            raise RuntimeError(f"missing frozen A1 force parameter: {key}")
    if (float(common["target_dft_mag_density"]) != protocol["target_dft_mag_density"]
            or float(common["cfg"]) != protocol["cfg"]
            or int(common["diffusion_steps"]) != protocol["sampler_steps"]
            or int(common["corrector_steps_per_timestep"]) != protocol["corrector_steps_per_timestep"]
            or float(force["guidance_t_max"]) != protocol["guidance_t_max"]["Late"]):
        raise RuntimeError("Timing protocol diverges from frozen A1 setup")
    return protocol, common, force


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    protocol, common, force = preflight(args.seed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(MODEL), config_overrides=[ELEMENT_MASK_OVERRIDE]),
        batch_size=1, num_batches=1,
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
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", "model_loaded": True,
                          "seed": args.seed, "methods": protocol["methods"]}), flush=True)
        return

    for method in protocol["methods"]:
        run_dir = OUT / "generation" / method / str(args.seed)
        partial = run_dir.with_name(run_dir.name + ".partial")
        if (run_dir / "run_summary.json").is_file():
            print(json.dumps({"event": "skip_complete", "method": method, "seed": args.seed}), flush=True)
            continue
        if run_dir.exists() or partial.exists():
            raise RuntimeError(f"incomplete output retained; no overwrite: {run_dir} / {partial}")
        partial.mkdir(parents=True)
        config = generator.load_sampling_config(batch_size=1, num_batches=1)
        if (int(config.sampler_partial.N) != protocol["sampler_steps"]
                or int(config.sampler_partial.n_steps_corrector) != protocol["corrector_steps_per_timestep"]
                or float(config.sampler_partial.guidance_scale) != protocol["cfg"]
                or str(config.sampler_partial.guidance_schedule) != str(common["guidance_schedule"])):
            raise RuntimeError("frozen MatterGen sampling contract differs")
        kwargs = {"pl_module": generator.model}
        if method != "C0":
            config.sampler_partial._target_ = (
                "experiments.mattersim_late_force_guidance_p0.late_force_sampler."
                "LateMatterSimForceGuidedSampler.from_pl_module"
            )
            kwargs.update(
                trace_path=str((partial / "late_force_trace.csv").resolve()),
                sample_seed=args.seed,
                mattersim_checkpoint=str(MATTERSIM),
                guidance_t_max=float(protocol["guidance_t_max"][method]),
                force_reference_ev_a=float(force["force_reference_ev_a"]),
                nominal_cart_step_a=float(force["nominal_cartesian_step_a"]),
                hard_cart_cap_a=float(force["hard_cartesian_cap_a"]),
                min_distance_floor_a=float(force["min_periodic_distance_floor_a"]),
            )
        potential_started = time.perf_counter()
        sampler = instantiate(config.sampler_partial)(**kwargs)
        potential_load_seconds = time.perf_counter() - potential_started if method != "C0" else 0.0
        calls = {"actual_force_model_calls": 0, "failed_force_model_calls": 0}
        if method != "C0":
            original_calculate = sampler._calculator.calculate

            def counted_calculate(*args, **kwargs):
                calls["actual_force_model_calls"] += 1
                try:
                    return original_calculate(*args, **kwargs)
                except BaseException:
                    calls["failed_force_model_calls"] += 1
                    raise

            sampler._calculator.calculate = counted_calculate
        generator.seed = args.seed
        generator._seed_sampling_rngs()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        started = time.perf_counter()
        structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=generator.get_condition_loader(config),
            properties_to_condition_on={"dft_mag_density": float(common["target_dft_mag_density"])},
            output_path=partial, cfg=generator.cfg, record_trajectories=False,
        )
        event_end.record()
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        cuda_event_seconds = event_start.elapsed_time(event_end) / 1000.0
        metrics = dict(sampler.sampling_metrics)
        expected_eligible = int(protocol["expected_predictor_force_opportunities"][method])
        if int(metrics["mattergen_score_calls"]) != 2000:
            raise RuntimeError(f"{method}/{args.seed}: score-call contract changed")
        if method != "C0" and int(metrics["mattersim_guidance_eligible"]) != expected_eligible:
            raise RuntimeError(f"{method}/{args.seed}: injection opportunity count changed")
        if method != "C0":
            (partial / "force_call_accounting.json").write_text(
                json.dumps(calls, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        summary = {
            "success": True, "method": method, "seed": args.seed,
            "injection_t_max": protocol["guidance_t_max"][method],
            "formula": structures[0].composition.reduced_formula,
            "num_atoms": len(structures[0]),
            "elapsed_seconds": wall_seconds,
            "potential_load_seconds": potential_load_seconds,
            "end_to_end_seconds": wall_seconds + potential_load_seconds,
            "cuda_event_elapsed_seconds": cuda_event_seconds,
            "cuda_event_note": "Event interval includes CPU stalls; not active-SM utilization",
            "physical_gpu": os.environ.get("FORMAL32_PHYSICAL_GPU", "unknown"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "checkpoint_sha256": MODEL_SHA, "mattersim_checkpoint_sha256": MATTERSIM_SHA if method != "C0" else None,
            "p0_guidance_source_sha256": FORCE_SHA,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "actual_force_model_calls": calls["actual_force_model_calls"],
            "failed_force_model_calls": calls["failed_force_model_calls"],
            "target_dft_mag_density": float(common["target_dft_mag_density"]),
            "guidance_scale": float(common["cfg"]),
            "guidance_schedule": str(common["guidance_schedule"]),
            "sampling_steps": int(common["diffusion_steps"]),
            "corrector_steps_per_timestep": int(common["corrector_steps_per_timestep"]),
            "surrogate_property_eval": True, "dft_verified": False,
            **metrics,
        }
        (partial / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        partial.rename(run_dir)
        print(json.dumps({"event": "arm_complete", "method": method, "seed": args.seed,
                          "formula": summary["formula"], "wall_seconds": wall_seconds,
                          "actual_force_model_calls": calls["actual_force_model_calls"]}), flush=True)


if __name__ == "__main__":
    main()
