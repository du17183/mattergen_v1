"""Replay one frozen historical seed with read-only FP-PC instrumentation."""
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

# The CUDA environment carries NumPy 2.x, while the frozen GemNet basis
# implementation still refers to the removed ``np.math`` alias.  Restoring the
# compatibility alias is semantics-preserving and leaves model code/checkpoints
# untouched.
if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/fp_pc_p0"
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
    if not 99000 <= args.seed <= 99063:
        raise ValueError("diagnostic replay seed outside frozen historical cohort")
    checkpoint_path = MODEL_ROOT / "checkpoints/last.ckpt"
    if sha256(checkpoint_path) != EXPECTED_SHA256:
        raise RuntimeError("official checkpoint hash mismatch")
    run_dir = ROOT / "diagnostic_runs" / str(args.seed)
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
    if not (
        int(sampling_config.sampler_partial.N) == 1000
        and float(sampling_config.sampler_partial.guidance_scale) == 2.0
        and str(sampling_config.sampler_partial.guidance_schedule) == "constant"
        and int(sampling_config.sampler_partial.n_steps_corrector) == 1
    ):
        raise RuntimeError("official frozen sampling contract changed")
    sampling_config.sampler_partial._target_ = (
        "experiments.fp_pc_p0.proposal_diagnostic_sampler."
        "ProposalViolationDiagnosticSampler.from_pl_module"
    )
    condition_loader = generator.get_condition_loader(sampling_config)
    sampler = instantiate(sampling_config.sampler_partial)(
        pl_module=generator.model,
        diagnostic_trace_path=str((partial / "proposal_trace.csv").resolve()),
        diagnostic_seed=args.seed,
        feasible_region_path=str((ROOT / "feasible_region.json").resolve()),
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    structures = draw_samples_from_sampler(
        sampler=sampler, condition_loader=condition_loader,
        properties_to_condition_on={"dft_mag_density": 0.2},
        output_path=partial, cfg=generator.cfg, record_trajectories=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = dict(sampler.sampling_metrics)
    if int(metrics["mattergen_score_calls"]) != 2000:
        raise RuntimeError("official score-call contract changed")
    if int(metrics["proposal_observation_count"]) != 2000:
        raise RuntimeError("proposal observation count changed")
    summary = {
        "success": True, "seed": args.seed,
        "subset": "calibration" if args.seed <= 99031 else "validation",
        "formula": structures[0].composition.reduced_formula,
        "num_atoms": len(structures[0]), "target_dft_mag_density": 0.2,
        "guidance_scale": 2.0, "guidance_schedule": "constant",
        "adaptive_cfg_used": False, "sampling_steps": 1000,
        "corrector_steps_per_timestep": 1,
        "element_masking": "official_mask_disallowed_elements",
        "diagnostic_only_no_proposal_modified": True,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "physical_gpu_id": os.environ.get("FPPC_PHYSICAL_GPU_ID", "unspecified"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": EXPECTED_SHA256,
        "surrogate_property_eval": True, "dft_verified": False,
        **metrics,
    }
    (partial / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, run_dir)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
