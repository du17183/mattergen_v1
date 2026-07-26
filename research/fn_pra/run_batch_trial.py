"""Run one resumable A0 multi-trajectory batch benchmark configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path
from statistics import median

import ase.io
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import open_dict
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.common.data.collate import collate
from mattergen.common.data.condition_factory import NumAtomsCrystalDataset
from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from mattergen.common.data.transform import SetProperty
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
from mattergen.common.utils.eval_utils import save_structures
from mattergen.generator import CrystalGenerator, structure_from_model_output
from research.fn_pra.phase1_common import atomic_json, now, sha256_file


CHECKPOINT_ROOT = Path(
    "/data/dxl/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
)
CHECKPOINT = CHECKPOINT_ROOT / "checkpoints/last.ckpt"
EXPECTED_CHECKPOINT_SHA = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
FIELDS = ("atomic_numbers", "cell", "pos")


def hash_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def hash_graph(item) -> dict[str, str]:
    fields = {field: hash_tensor(item[field]) for field in FIELDS}
    fields["combined"] = hashlib.sha256(
        json.dumps(fields, sort_keys=True).encode()
    ).hexdigest()
    return fields


def make_condition(seed: int):
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        dataset = NumAtomsCrystalDataset.from_num_atoms_distribution(
            num_atoms_distribution=NUM_ATOMS_DISTRIBUTIONS["ALEX_MP_20"],
            num_samples=1,
            transforms=[SetProperty("dft_mag_density", 0.10)],
        )
        return dataset[0]
    finally:
        np.random.set_state(numpy_state)


class Telemetry:
    def __init__(self, physical_gpu: int) -> None:
        self.physical_gpu = physical_gpu
        self.rows: list[dict] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for line in result.stdout.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    if int(values[0]) == self.physical_gpu:
                        self.rows.append(
                            {
                                "time": time.time(),
                                "memory_used_mib": int(values[1]),
                                "utilization_gpu_percent": int(values[2]),
                                "power_draw_w": float(values[3]),
                            }
                        )
                        break
            except Exception:
                pass
            self.stop_event.wait(0.5)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def summary(self) -> dict:
        if not self.rows:
            return {"samples": 0}
        return {
            "samples": len(self.rows),
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in self.rows),
            "mean_utilization_percent": float(
                np.mean([row["utilization_gpu_percent"] for row in self.rows])
            ),
            "max_utilization_percent": max(
                row["utilization_gpu_percent"] for row in self.rows
            ),
            "mean_power_w": float(np.mean([row["power_draw_w"] for row in self.rows])),
            "max_power_w": max(row["power_draw_w"] for row in self.rows),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--seed-start", type=int, default=15000)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--physical-gpu", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sampling_steps < 2:
        raise ValueError("sampling-steps must be at least 2")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    seeds = list(range(args.seed_start, args.seed_start + args.batch_size))
    config = {
        "created_at": now(),
        "method": "A0 frozen Adaptive CFG",
        "batch_size": args.batch_size,
        "seeds": seeds,
        "sampling_steps": args.sampling_steps,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "physical_gpu": args.physical_gpu,
        "target": {"dft_mag_density": 0.10},
        "base_guidance": 2.0,
        "adaptive_alpha": 0.50,
        "adaptive_ema": 0.95,
        "adaptive_epsilon": 1e-6,
        "guidance_min_scale": 0.0,
        "guidance_max_scale": 5.0,
        "trajectory_rng": "one independent torch.Generator stream per seed",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
    }
    atomic_json(output / "run_config.json", config)
    if config["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("official A0 checkpoint SHA256 mismatch")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    overrides = [
        "++lightning_module.diffusion_module.model.element_mask_func="
        "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
    ]
    if args.sampling_steps != 1000:
        overrides.append(
            "lightning_module.diffusion_module.corruption.discrete_corruptions."
            f"atomic_numbers.d3pm.schedule.num_steps={args.sampling_steps}"
        )
    checkpoint_info = MatterGenCheckpointInfo(
        model_path=CHECKPOINT_ROOT,
        load_epoch="last",
        config_overrides=overrides,
    )
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=args.batch_size,
        num_batches=1,
        num_atoms_distribution="ALEX_MP_20",
        diffusion_guidance_factor=2.0,
        properties_to_condition_on={"dft_mag_density": 0.10},
        deterministic=True,
        guidance_schedule="adaptive",
        guidance_min_scale=0.0,
        guidance_max_scale=5.0,
        guidance_adaptive_alpha=0.50,
        guidance_adaptive_ema=0.95,
        guidance_adaptive_eps=1e-6,
        sampling_config_overrides=[f"sampler_partial.N={args.sampling_steps}"],
        record_trajectories=False,
    )

    success = False
    error = None
    repeats: list[dict] = []
    telemetry = Telemetry(args.physical_gpu)
    telemetry.start()
    try:
        generator.prepare()
        conditions = collate([make_condition(seed) for seed in seeds])
        sampling_config = generator.load_sampling_config(
            batch_size=args.batch_size,
            num_batches=1,
        )
        with open_dict(sampling_config.sampler_partial):
            sampling_config.sampler_partial._target_ = (
                "research.fn_pra.independent_batch."
                "IndependentTrajectoryGuidedPredictorCorrector.from_pl_module"
            )
            sampling_config.sampler_partial.trajectory_seeds = seeds
        sampler = instantiate(sampling_config.sampler_partial)(pl_module=generator.model)

        initial_hashes: list[dict[str, str]] = []
        original_after_prior = sampler._on_after_sample_prior

        def capture_prior(batch) -> None:
            original_after_prior(batch)
            initial_hashes.clear()
            initial_hashes.extend(hash_graph(item) for item in batch.to_data_list())

        sampler._on_after_sample_prior = capture_prior
        last_mean = None
        total_iterations = args.warmups + args.repeats
        for iteration in range(total_iterations):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            cpu_start = time.process_time()
            started = time.monotonic()
            _, mean = sampler.sample(conditions, None)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            cpu_seconds = time.process_time() - cpu_start
            last_mean = mean
            row = {
                "iteration": iteration,
                "warmup": iteration < args.warmups,
                "elapsed_seconds": elapsed,
                "samples_per_hour": args.batch_size * 3600.0 / elapsed,
                "sample_latency_seconds": elapsed / args.batch_size,
                "cpu_seconds": cpu_seconds,
                "cpu_percent_one_core": 100.0 * cpu_seconds / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "physical_model_forward_count": sampler.physical_model_forward_count,
                "model_graphs_evaluated": sampler.model_graphs_evaluated,
                "initial_hashes": list(initial_hashes),
                "final_hashes": [hash_graph(item) for item in mean.to_data_list()],
            }
            if not row["warmup"]:
                repeats.append(row)

        assert last_mean is not None
        lengths, angles = lattice_matrix_to_params_torch(last_mean.cell)
        final_batch = last_mean.replace(lengths=lengths, angles=angles)
        structures = structure_from_model_output(
            final_batch["pos"].reshape(-1, 3),
            final_batch["atomic_numbers"].reshape(-1),
            final_batch["lengths"].reshape(-1, 3),
            final_batch["angles"].reshape(-1, 3),
            final_batch["num_atoms"].reshape(-1),
        )
        save_structures(output, structures)
        from mattergen.evaluation.metrics.structure import is_smact_valid, structure_validity

        validity = [
            {
                "structure_valid": bool(structure_validity(structure)),
                "composition_valid": bool(is_smact_valid(structure)),
            }
            for structure in structures
        ]
        deterministic_repeats = all(
            row["initial_hashes"] == repeats[0]["initial_hashes"]
            and row["final_hashes"] == repeats[0]["final_hashes"]
            for row in repeats
        )
        elapsed_values = [row["elapsed_seconds"] for row in repeats]
        summary = {
            "success": True,
            "batch_size": args.batch_size,
            "seeds": seeds,
            "structure_count": len(structures),
            "validity": validity,
            "generation_success_rate": 1.0,
            "deterministic_repeats_level1": deterministic_repeats,
            "median_batch_latency_seconds": median(elapsed_values),
            "median_sample_latency_seconds": median(elapsed_values) / args.batch_size,
            "median_samples_per_hour": args.batch_size * 3600.0 / median(elapsed_values),
            "physical_model_forward_count": repeats[0]["physical_model_forward_count"],
            "model_graphs_evaluated": repeats[0]["model_graphs_evaluated"],
            "repeats": repeats,
        }
        atomic_json(output / "summary.json", summary)
        success = True
    except BaseException:
        error = traceback.format_exc()
        (output / "error.txt").write_text(error, encoding="utf-8")
    finally:
        telemetry.stop()
        atomic_json(
            output / "telemetry.json",
            {"summary": telemetry.summary(), "samples": telemetry.rows},
        )
        atomic_json(
            output / "status.json",
            {"success": success, "error": error, "finished_at": now()},
        )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
