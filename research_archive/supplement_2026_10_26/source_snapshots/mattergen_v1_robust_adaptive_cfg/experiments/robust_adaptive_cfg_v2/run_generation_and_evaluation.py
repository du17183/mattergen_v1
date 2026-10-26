"""Generation and shared evaluation stages for frozen Robust Adaptive CFG V2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from ase.io import read, write
from hydra.utils import instantiate

if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.diffusion.sampling.robust_guidance import RobustAdaptiveGuidanceController
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/robust_adaptive_cfg_v2"
P0 = ROOT / "p0"
FORMAL = ROOT / "formal256"
CALIBRATION_JSON = ROOT / "calibration/residual_calibration.json"
CALIBRATION_CSV = ROOT / "calibration/residual_calibration.csv"
FROZEN_CONFIG = ROOT / "implementation/robust_adaptive_cfg_v2_config.yaml"
FROZEN_MANIFEST = ROOT / "implementation/frozen_manifest.json"
MODEL_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/"
    "checkpoints/dft_mag_density"
)
MODEL_SHA = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
ELEMENT_MASK_OVERRIDE = (
    "++lightning_module.diffusion_module.model.element_mask_func="
    "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
)
TARGET = 0.1
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"
F0_WORKTREE = Path("/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance")
GPU_POOL = tuple(
    int(value)
    for value in os.environ.get("ROBUST_CFG_GPU_POOL", "0,1,3,4,5,6,7").split(",")
    if value
)
METHODS = {
    "p0": ("C0", "A_OLD", "A_NORM", "A_ROBUST"),
    "formal256": ("C0", "A_OLD", "A_ROBUST"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cohort_root(cohort: str) -> Path:
    return P0 if cohort == "p0" else FORMAL


def seeds_for(cohort: str) -> tuple[int, ...]:
    payload = json.loads((cohort_root(cohort) / "seeds.json").read_text())
    seeds = tuple(map(int, payload["seeds"]))
    expected = 32 if cohort == "p0" else 256
    if len(seeds) != expected or len(set(seeds)) != expected or payload["historical_overlap"] != 0:
        raise RuntimeError(f"{cohort} seed registration mismatch")
    return seeds


def assert_frozen() -> dict:
    manifest = json.loads(FROZEN_MANIFEST.read_text())
    if not manifest["frozen"] or sha256(FROZEN_CONFIG) != manifest["config_sha256"]:
        raise RuntimeError("frozen V2 configuration hash mismatch")
    if not CALIBRATION_JSON.exists() or not CALIBRATION_CSV.exists():
        raise RuntimeError("calibration is not complete")
    return manifest


def make_generator(seed: int) -> CrystalGenerator:
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(
            model_path=str(MODEL_ROOT), config_overrides=[ELEMENT_MASK_OVERRIDE]
        ),
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={"dft_mag_density": TARGET},
        diffusion_guidance_factor=2.0,
        guidance_schedule="constant",
        seed=seed,
        deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    return generator


def generation_worker(cohort: str, seeds: tuple[int, ...]) -> None:
    manifest = assert_frozen()
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA:
        raise RuntimeError("MatterGen checkpoint hash mismatch")
    registered = set(seeds_for(cohort))
    if not set(seeds).issubset(registered):
        raise ValueError("worker received an unregistered seed")
    methods = METHODS[cohort]
    output_root = cohort_root(cohort)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = make_generator(seeds[0])
    for seed in seeds:
        for method in methods:
            destination = output_root / "generation" / method / str(seed)
            partial = destination.with_name(destination.name + ".partial")
            if (destination / "run_summary.json").exists():
                continue
            if destination.exists() or partial.exists():
                raise FileExistsError(f"retained incomplete output: {destination} or {partial}")
            partial.mkdir(parents=True)
            generator.guidance_schedule = "constant" if method == "C0" else "adaptive"
            sampling = generator.load_sampling_config(batch_size=1, num_batches=1)
            if int(sampling.sampler_partial.N) != 1000 or int(sampling.sampler_partial.n_steps_corrector) != 1:
                raise RuntimeError("sampling protocol changed")
            trace_path = partial / "control_trace.csv"
            sampler = instantiate(sampling.sampler_partial)(
                pl_module=generator.model,
                sample_seed=seed,
                run_id=f"{cohort}_{method}_seed_{seed}",
                guidance_trace_path=str(trace_path.resolve()),
            )
            if method in ("A_NORM", "A_ROBUST"):
                sampler._guidance_controller = RobustAdaptiveGuidanceController(
                    calibration_json=CALIBRATION_JSON,
                    calibration_csv=CALIBRATION_CSV,
                    robust=method == "A_ROBUST",
                    base_guidance=2.0,
                    min_scale=1.5,
                    max_scale=2.5,
                    adaptation_amplitude=0.5,
                    slew_rate_cap=0.05,
                )
            generator.seed = seed
            generator._seed_sampling_rngs()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            structures = draw_samples_from_sampler(
                sampler=sampler,
                condition_loader=generator.get_condition_loader(sampling),
                properties_to_condition_on={"dft_mag_density": TARGET},
                output_path=partial,
                cfg=generator.cfg,
                record_trajectories=False,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            trace_rows = sum(1 for _ in trace_path.open()) - 1
            if trace_rows != 2000:
                raise RuntimeError(f"{method}/{seed}: trace count={trace_rows}")
            metrics = dict(sampler.sampling_metrics)
            if int(metrics["mattergen_score_calls"]) != 2000:
                raise RuntimeError("MatterGen score-call contract changed")
            summary = {
                "success": True,
                "cohort": cohort,
                "method": method,
                "seed": seed,
                "formula": structures[0].composition.reduced_formula,
                "num_atoms": len(structures[0]),
                "target_dft_mag_density": TARGET,
                "elapsed_seconds": elapsed,
                "end_to_end_seconds": elapsed,
                "physical_gpu": os.environ.get("ROBUST_CFG_PHYSICAL_GPU", "unknown"),
                "guidance_schedule": sampler.guidance_schedule,
                "trace_rows": trace_rows,
                "checkpoint_sha256": MODEL_SHA,
                "frozen_config_sha256": manifest["config_sha256"],
                "calibration_json_sha256": sha256(CALIBRATION_JSON),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "surrogate_property_eval": True,
                "dft_verified": False,
                **metrics,
            }
            (partial / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            os.replace(partial, destination)
            print(json.dumps(summary), flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(cohort: str) -> None:
    root = cohort_root(cohort)
    seeds = seeds_for(cohort)
    methods = METHODS[cohort]
    structures_dir = root / "structures"
    structures_dir.mkdir(exist_ok=True)
    p0_structures = root / "p0_structures"
    p0_structures.mkdir(exist_ok=True)
    manifest_rows = []
    hashes = {}
    for method in methods:
        atoms = []
        for seed in seeds:
            source = root / "generation" / method / str(seed)
            summary = json.loads((source / "run_summary.json").read_text())
            if not summary["success"]:
                raise RuntimeError(f"failed generation retained: {method}/{seed}")
            item = read(source / "generated_crystals.extxyz")
            item.info.update(sample_seed=seed, sample_index_within_seed=0, method=method)
            atoms.append(item)
            manifest_rows.append(summary)
            hashes[str(source / "generated_crystals.extxyz")] = sha256(
                source / "generated_crystals.extxyz"
            )
        output = structures_dir / f"{method}_generated.extxyz"
        write(output, atoms)
        mirrored = p0_structures / f"{method}_generated.extxyz"
        write(mirrored, atoms)
    write_csv(root / "generation_manifest.csv", manifest_rows)
    (structures_dir / "manifest.json").write_text(
        json.dumps({"cohort": cohort, "seeds": list(seeds), "methods": methods, "source_sha256": hashes}, indent=2)
        + "\n"
    )


def properties(cohort: str) -> None:
    sys.path.insert(0, str(F0_WORKTREE))
    from experiments.mattersim_late_force_guidance_p0 import evaluate_p0_properties as module

    root = cohort_root(cohort)
    module.ROOT = root
    module.SEEDS = seeds_for(cohort)
    module.METHODS = METHODS[cohort]
    module.TARGET = TARGET
    module.main()
    source = root / "p0_property_metrics.csv"
    (root / "property_metrics.csv").write_bytes(source.read_bytes())


def relax(cohort: str) -> None:
    sys.path.insert(0, str(F0_WORKTREE))
    from experiments.mattersim_late_force_guidance_p0 import run_p0_relaxation as module

    root = cohort_root(cohort)
    module.ROOT = root
    pending = [
        method
        for method in METHODS[cohort]
        if not (root / "p0_relaxation" / method / "relaxation_summary.json").exists()
    ]
    for start in range(0, len(pending), len(GPU_POOL)):
        jobs = tuple(zip(pending[start : start + len(GPU_POOL)], GPU_POOL))
        module.JOBS = jobs
        if jobs:
            module.main()


def quality(cohort: str) -> None:
    sys.path.insert(0, str(F0_WORKTREE))
    from experiments.mattersim_late_force_guidance_p0 import run_p0_quality as module

    root = cohort_root(cohort)
    module.ROOT = root
    module.METHODS = tuple(
        method for method in METHODS[cohort]
        if not (root / "p0_quality" / method / "official_detailed.json.gz").exists()
    )
    if module.METHODS:
        module.main()


def aggregate_traces(cohort: str) -> None:
    root = cohort_root(cohort)
    first = True
    destination = root / "control_trace.csv"
    if destination.exists():
        destination.unlink()
    with destination.open("x") as target:
        for method in METHODS[cohort]:
            for seed in seeds_for(cohort):
                source = root / "generation" / method / str(seed) / "control_trace.csv"
                with source.open() as stream:
                    if not first:
                        next(stream)
                    for line in stream:
                        target.write(line)
                first = False


def launch_generation(cohort: str) -> None:
    seeds = seeds_for(cohort)
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(PROJECT),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
    )

    def launch(gpu: int, shard: tuple[int, ...]) -> None:
        local = env | {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "ROBUST_CFG_PHYSICAL_GPU": str(gpu),
        }
        log = ROOT / "logs" / f"{cohort}_generation_gpu_{gpu}.log"
        with log.open("a") as stream:
            result = subprocess.run(
                [PYTHON, str(Path(__file__).resolve()), "--cohort", cohort, "--worker", ",".join(map(str, shard))],
                cwd=PROJECT,
                env=local,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{cohort} generation failed on GPU {gpu}; see {log}")

    with ThreadPoolExecutor(max_workers=len(GPU_POOL)) as pool:
        futures = [
            pool.submit(launch, gpu, seeds[index :: len(GPU_POOL)])
            for index, gpu in enumerate(GPU_POOL)
        ]
        for future in futures:
            future.result()


def pipeline(cohort: str) -> None:
    root = cohort_root(cohort)
    status = root / "pipeline_status.json"
    try:
        for stage, function in [
            ("generation", launch_generation),
            ("prepare", prepare),
            ("properties", properties),
            ("relaxation", relax),
            ("quality", quality),
            ("trace_aggregation", aggregate_traces),
        ]:
            status.write_text(json.dumps({"status": "RUNNING", "stage": stage}, indent=2) + "\n")
            function(cohort)
        status.write_text(json.dumps({"status": "EVALUATED", "stage": "analysis_pending"}, indent=2) + "\n")
    except BaseException as error:
        status.write_text(json.dumps({"status": "FAILED", "error": str(error)}, indent=2) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("p0", "formal256"), required=True)
    parser.add_argument("--worker")
    parser.add_argument("--stage", choices=("prepare", "properties", "relax", "quality", "traces"))
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    if args.worker:
        generation_worker(args.cohort, tuple(map(int, args.worker.split(","))))
    elif args.stage:
        {
            "prepare": prepare,
            "properties": properties,
            "relax": relax,
            "quality": quality,
            "traces": aggregate_traces,
        }[args.stage](args.cohort)
    elif args.pipeline:
        pipeline(args.cohort)
    else:
        parser.error("select --worker, --stage, or --pipeline")


if __name__ == "__main__":
    main()
