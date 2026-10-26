"""Deterministically replay an observed fixed-CFG cohort for calibration.

The generated structure for every replayed seed is compared with its archived
C0 structure.  These are calibration trajectories, never validation samples.
"""

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
import time

import numpy as np
import torch
from ase.io import read
from hydra.utils import instantiate

if not hasattr(np, "math"):
    np.math = math

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "experiments/robust_adaptive_cfg_v2"
OUT = ROOT / "calibration"
REPLAY = OUT / "replay_fixed_cfg2"
HISTORICAL = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/"
    "adaptive_cfg_force_guidance_compatibility/generation/C0"
)
MODEL_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/"
    "checkpoints/dft_mag_density"
)
MODEL_SHA = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
ELEMENT_MASK_OVERRIDE = (
    "++lightning_module.diffusion_module.model.element_mask_func="
    "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
)
SEEDS = tuple(range(744000, 744064))
GPUS = (0, 1, 3, 4, 5, 6, 7)
TARGET = 0.2
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_equal(left: Path, right: Path) -> tuple[bool, dict[str, float | bool]]:
    a = read(left)
    b = read(right)
    same_numbers = np.array_equal(a.numbers, b.numbers)
    position_error = float(np.max(np.abs(a.positions - b.positions))) if len(a) == len(b) else math.inf
    cell_error = float(np.max(np.abs(a.cell.array - b.cell.array)))
    equal = same_numbers and position_error <= 1e-6 and cell_error <= 1e-6
    return equal, {
        "same_atomic_numbers": bool(same_numbers),
        "max_position_abs_error": position_error,
        "max_cell_abs_error": cell_error,
    }


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


def worker(seeds: tuple[int, ...]) -> None:
    if sha256(MODEL_ROOT / "checkpoints/last.ckpt") != MODEL_SHA:
        raise RuntimeError("MatterGen checkpoint hash mismatch")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = make_generator(seeds[0])
    for seed in seeds:
        destination = REPLAY / str(seed)
        partial = destination.with_name(destination.name + ".partial")
        if (destination / "replay_summary.json").exists():
            continue
        if destination.exists() or partial.exists():
            raise FileExistsError(f"retained incomplete replay: {destination} or {partial}")
        partial.mkdir(parents=True)
        sampling = generator.load_sampling_config(batch_size=1, num_batches=1)
        assert int(sampling.sampler_partial.N) == 1000
        assert int(sampling.sampler_partial.n_steps_corrector) == 1
        sampler = instantiate(sampling.sampler_partial)(
            pl_module=generator.model,
            sample_seed=seed,
            run_id=f"calibration_fixed_cfg2_seed_{seed}",
            guidance_trace_path=str((partial / "baseline_cfg_trace.csv").resolve()),
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
        archived = HISTORICAL / str(seed) / "generated_crystals.extxyz"
        equal, comparison = structure_equal(partial / "generated_crystals.extxyz", archived)
        if not equal:
            raise RuntimeError(f"seed {seed} deterministic replay differs: {comparison}")
        trace_rows = sum(1 for _ in (partial / "baseline_cfg_trace.csv").open()) - 1
        if trace_rows != 2000:
            raise RuntimeError(f"seed {seed}: expected 2000 trace rows, got {trace_rows}")
        summary = {
            "seed": seed,
            "purpose": "historical fixed-CFG calibration replay; NOT validation",
            "target_dft_mag_density": TARGET,
            "guidance_scale": 2.0,
            "guidance_schedule": "constant",
            "sampling_steps": 1000,
            "corrector_steps": 1,
            "trace_rows": trace_rows,
            "elapsed_seconds": elapsed,
            "physical_gpu": os.environ.get("ROBUST_CFG_PHYSICAL_GPU", "unknown"),
            "checkpoint_sha256": MODEL_SHA,
            "archived_structure": str(archived),
            "archived_structure_sha256": sha256(archived),
            "replayed_structure_sha256": sha256(partial / "generated_crystals.extxyz"),
            "structure_numeric_match": comparison,
            "structure_match": True,
            "formula": structures[0].composition.reduced_formula,
        }
        (partial / "replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, destination)
        print(json.dumps(summary), flush=True)


def aggregate() -> None:
    summaries = []
    trace_paths = []
    for seed in SEEDS:
        path = REPLAY / str(seed)
        summaries.append(json.loads((path / "replay_summary.json").read_text()))
        trace_paths.append(path / "baseline_cfg_trace.csv")
    with (OUT / "baseline_replay_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    combined = OUT / "baseline_cfg_trace.csv"
    if combined.exists():
        combined.unlink()
    with combined.open("x", newline="") as target:
        for index, path in enumerate(trace_paths):
            with path.open(newline="") as source:
                if index:
                    next(source)
                for line in source:
                    target.write(line)
    rows = sum(1 for _ in combined.open()) - 1
    if rows != 128000:
        raise RuntimeError(f"expected 128000 aggregate rows, got {rows}")
    completion = {
        "status": "COMPLETE",
        "purpose": "historical fixed-CFG calibration replay; NOT validation",
        "seeds": list(SEEDS),
        "n": len(SEEDS),
        "trace_rows": rows,
        "all_archived_structures_matched": all(row["structure_match"] for row in summaries),
        "checkpoint_sha256": MODEL_SHA,
        "combined_trace_sha256": sha256(combined),
    }
    (OUT / "baseline_replay_completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2))


def pipeline() -> None:
    ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    status_path = OUT / "replay_pipeline_status.json"
    status_path.write_text(json.dumps({"status": "RUNNING", "stage": "replay"}, indent=2) + "\n")
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(PROJECT),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
    )

    def launch(gpu: int, seeds: tuple[int, ...]) -> None:
        if not seeds:
            return
        local = env | {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "ROBUST_CFG_PHYSICAL_GPU": str(gpu),
        }
        log = ROOT / "logs" / f"calibration_replay_gpu_{gpu}.log"
        with log.open("a") as stream:
            result = subprocess.run(
                [PYTHON, str(Path(__file__).resolve()), "--worker", ",".join(map(str, seeds))],
                cwd=PROJECT,
                env=local,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"GPU {gpu} replay failed; see {log}")

    try:
        with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
            jobs = [pool.submit(launch, gpu, SEEDS[index :: len(GPUS)]) for index, gpu in enumerate(GPUS)]
            for job in jobs:
                job.result()
        aggregate()
        status_path.write_text(json.dumps({"status": "COMPLETED", "stage": "aggregate"}, indent=2) + "\n")
    except BaseException as error:
        status_path.write_text(
            json.dumps({"status": "FAILED", "error": str(error)}, indent=2) + "\n"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(tuple(int(value) for value in args.worker.split(",") if value))
    elif args.aggregate:
        aggregate()
    elif args.pipeline:
        pipeline()
    else:
        parser.error("select --worker, --aggregate, or --pipeline")


if __name__ == "__main__":
    main()
