from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import torch

from research.rp_qtfg.common import atomic_json, atomic_text, now, sha256_file
from research.rp_qtfg.experiment_config import CONFIGS


CHECKPOINT = Path(
    "/data/dxl/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density"
)
CHECKPOINT_FILE = CHECKPOINT / "checkpoints/last.ckpt"
EXPECTED_CHECKPOINT_SHA = (
    "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
)
FIELDS = ("cell", "pos", "atomic_numbers")


def _hash_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _hash_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        "torch_cpu": _hash_tensor(torch.get_rng_state()),
        "torch_cuda": [
            _hash_tensor(item) for item in torch.cuda.get_rng_state_all()
        ],
    }
    state["combined"] = hashlib.sha256(
        json.dumps(state, sort_keys=True).encode()
    ).hexdigest()
    return state


class Telemetry:
    def __init__(self, physical_gpu: int):
        self.physical_gpu = physical_gpu
        self.rows: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                for line in result.stdout.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    if int(values[0]) == self.physical_gpu:
                        self.rows.append(
                            {
                                "time": time.time(),
                                "memory_used_mib": float(values[1]),
                                "utilization_gpu_percent": float(values[2]),
                                "power_draw_w": float(values[3]),
                            }
                        )
                        break
            except Exception:
                pass
            self.stop_event.wait(1.0)

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            return {"samples": 0}
        return {
            "samples": len(self.rows),
            "peak_memory_mib": max(row["memory_used_mib"] for row in self.rows),
            "mean_utilization_percent": float(
                np.mean([row["utilization_gpu_percent"] for row in self.rows])
            ),
            "max_utilization_percent": max(
                row["utilization_gpu_percent"] for row in self.rows
            ),
        }


def _validate_structure(path: Path) -> tuple[dict[str, Any], Any]:
    atoms = ase.io.read(path)
    arrays = (atoms.numbers, atoms.positions, atoms.cell.array)
    if len(atoms) == 0 or not all(np.isfinite(value).all() for value in arrays):
        raise RuntimeError("generated structure is empty or non-finite")
    if float(atoms.get_volume()) <= 0.1:
        raise RuntimeError("generated structure has invalid volume")
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.tobytes())
    return {
        "formula": atoms.get_chemical_formula(),
        "num_atoms": len(atoms),
        "atomic_numbers": atoms.numbers.tolist(),
        "final_structure_hash": digest.hexdigest(),
        "extxyz_sha256": sha256_file(path),
    }, atoms


def run(
    *,
    output: Path,
    seed: int,
    config_id: str,
    physical_gpu: int,
) -> int:
    if config_id not in CONFIGS:
        raise ValueError(config_id)
    config = CONFIGS[config_id]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    atomic_text(
        output / "command.txt",
        " ".join([os.path.realpath(os.sys.executable), *os.sys.argv]) + "\n",
    )
    actual_sha = sha256_file(CHECKPOINT_FILE)
    if actual_sha != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("A0 checkpoint SHA256 mismatch")
    atomic_json(
        output / "run_config.json",
        {
            "created_at": now(),
            "config": config.as_dict(),
            "seed": seed,
            "physical_gpu": physical_gpu,
            "target": {"dft_mag_density": 0.10},
            "base_guidance": 2.0,
            "guidance_schedule": "adaptive",
            "adaptive_alpha": 0.50,
            "adaptive_ema": 0.95,
            "adaptive_epsilon": 1e-6,
            "guidance_min_scale": 0.0,
            "guidance_max_scale": 5.0,
            "sampling_steps": 1000,
            "checkpoint_sha256": actual_sha,
        },
    )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)

    from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector

    initial: dict[str, str] = {}
    rng: dict[str, Any] = {}
    original_before = PredictorCorrector._on_before_sample_prior
    original_after = PredictorCorrector._on_after_sample_prior

    def before_prior(self, conditioning_data):
        original_before(self, conditioning_data)
        rng.update(_hash_rng_state())

    def after_prior(self, batch):
        original_after(self, batch)
        for field in FIELDS:
            initial[field] = _hash_tensor(batch[field])
        initial["combined"] = hashlib.sha256(
            json.dumps(initial, sort_keys=True).encode()
        ).hexdigest()

    PredictorCorrector._on_before_sample_prior = before_prior
    PredictorCorrector._on_after_sample_prior = after_prior
    telemetry = Telemetry(physical_gpu)
    telemetry.start()
    started = time.monotonic()
    cpu_started = time.process_time()
    error = None
    structure_summary: dict[str, Any] = {}
    validity = {
        "basic_structure_valid": False,
        "composition_valid": False,
    }
    success = False
    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        from mattergen.evaluation.metrics.structure import (
            is_smact_valid,
            structure_validity,
        )
        from mattergen.scripts.generate import main as generate

        trace_path = output / "rp_qtfg_summary.json"
        generate(
            output_path=str(output),
            model_path=str(CHECKPOINT),
            batch_size=1,
            num_batches=1,
            properties_to_condition_on={"dft_mag_density": 0.10},
            sampling_config_overrides=config.sampling_overrides(
                str(trace_path)
            ),
            record_trajectories=False,
            diffusion_guidance_factor=2.0,
            seed=seed,
            deterministic=True,
            guidance_schedule="adaptive",
            guidance_min_scale=0.0,
            guidance_max_scale=5.0,
            guidance_adaptive_alpha=0.50,
            guidance_adaptive_ema=0.95,
            guidance_adaptive_eps=1e-6,
        )
        if not rng or not initial:
            raise RuntimeError("initial-state audit hooks did not run")
        structure_summary, atoms = _validate_structure(
            output / "generated_crystals.extxyz"
        )
        structure = AseAtomsAdaptor.get_structure(atoms)
        validity = {
            "basic_structure_valid": bool(structure_validity(structure)),
            "composition_valid": bool(is_smact_valid(structure)),
        }
        atomic_json(
            output / "structure_hashes.json",
            {
                "rng_state_hash": rng["combined"],
                "initial_state_hash": initial["combined"],
                "initial_field_hashes": initial,
                **structure_summary,
            },
        )
        atomic_json(output / "rng_trace.json", rng)
        if config.enabled:
            physics = json.loads(trace_path.read_text())
            if physics["atomic_numbers_modified"]:
                raise RuntimeError("physics engine reported atomic modification")
            if physics["sampling_error"] is not None:
                raise RuntimeError("physics trace reported sampling error")
        success = True
    except BaseException:
        error = traceback.format_exc()
        atomic_text(output / "error.txt", error)
    finally:
        PredictorCorrector._on_before_sample_prior = original_before
        PredictorCorrector._on_after_sample_prior = original_after
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        telemetry.stop()
        elapsed = time.monotonic() - started
        atomic_json(
            output / "gpu_telemetry.json",
            {"samples": telemetry.rows, "summary": telemetry.summary()},
        )
        atomic_json(
            output / "run_summary.json",
            {
                "success": success,
                "return_code": 0 if success else 1,
                "config_id": config_id,
                "seed": seed,
                "physical_gpu": physical_gpu,
                "elapsed_seconds": elapsed,
                "process_cpu_seconds": time.process_time() - cpu_started,
                "peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else None
                ),
                "peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved()
                    if torch.cuda.is_available()
                    else None
                ),
                "telemetry": telemetry.summary(),
                **structure_summary,
                **validity,
                "error": error,
            },
        )
    return 0 if success else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config-id", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    return run(
        output=args.output_dir.resolve(),
        seed=args.seed,
        config_id=args.config_id,
        physical_gpu=args.physical_gpu,
    )


if __name__ == "__main__":
    raise SystemExit(main())
