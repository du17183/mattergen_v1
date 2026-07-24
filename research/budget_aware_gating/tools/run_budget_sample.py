#!/usr/bin/env python3
"""Run and validate one deterministic Corrector Gating sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import ase.io
import numpy as np
import torch


CHECKPOINT = Path(
    "/data/dxl/checkpoints/official/hf_mattergen/checkpoints/"
    "dft_mag_density"
)
FIELDS = ("cell", "pos", "atomic_numbers")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def hash_rng_state() -> dict:
    state = {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(
            repr(
                (
                    np.random.get_state()[0],
                    np.random.get_state()[1].tolist(),
                    np.random.get_state()[2:],
                )
            ).encode()
        ).hexdigest(),
        "torch_cpu": hash_tensor(torch.get_rng_state()),
        "torch_cuda": [
            hash_tensor(item) for item in torch.cuda.get_rng_state_all()
        ],
    }
    state["combined"] = hashlib.sha256(
        json.dumps(state, sort_keys=True).encode()
    ).hexdigest()
    return state


class Telemetry:
    def __init__(self, physical_gpu: int) -> None:
        self.physical_gpu = physical_gpu
        self.rows: list[dict] = []
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
                        "--query-gpu=index,memory.used,memory.free,"
                        "utilization.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                for line in result.stdout.splitlines():
                    fields = [item.strip() for item in line.split(",")]
                    if int(fields[0]) == self.physical_gpu:
                        self.rows.append(
                            {
                                "time": time.time(),
                                "memory_used_mib": int(fields[1]),
                                "memory_free_mib": int(fields[2]),
                                "utilization_gpu_percent": int(fields[3]),
                                "power_draw_w": float(fields[4]),
                            }
                        )
                        break
            except Exception:
                pass
            self.stop_event.wait(1.0)

    def summary(self) -> dict:
        if not self.rows:
            return {"samples": 0}
        utilization = [
            row["utilization_gpu_percent"] for row in self.rows
        ]
        memory = [row["memory_used_mib"] for row in self.rows]
        power = [row["power_draw_w"] for row in self.rows]
        return {
            "samples": len(self.rows),
            "memory_used_at_launch_mib": memory[0],
            "free_memory_at_launch_mib": self.rows[0][
                "memory_free_mib"
            ],
            "nvidia_smi_peak_used_mib": max(memory),
            "utilization_mean_percent": sum(utilization)
            / len(utilization),
            "utilization_max_percent": max(utilization),
            "power_draw_mean_w": sum(power) / len(power),
            "power_draw_max_w": max(power),
        }


def validate_structure(path: Path) -> tuple[dict, object]:
    frames = ase.io.read(path, ":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) != 1:
        raise ValueError(
            f"Expected one generated structure, got {len(frames)}"
        )
    atoms = frames[0]
    arrays = (
        np.asarray(atoms.get_atomic_numbers()),
        np.asarray(atoms.get_positions()),
        np.asarray(atoms.cell.array),
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Generated structure contains NaN or Inf")
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return (
        {
            "count": 1,
            "formula": atoms.get_chemical_formula(),
            "num_atoms": len(atoms),
            "final_structure_hash": digest.hexdigest(),
            "atomic_numbers": arrays[0].tolist(),
            "positions": arrays[1].tolist(),
            "cell": arrays[2].tolist(),
        },
        atoms,
    )


def validate_trace(path: Path, sampling_steps: int) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != sampling_steps:
        raise ValueError(
            f"Expected {sampling_steps} trace rows, got {len(rows)}"
        )
    numeric_checked = 0
    decisions: dict[str, int] = {}
    for row in rows:
        decisions[row["decision"]] = decisions.get(
            row["decision"], 0
        ) + 1
        for key, raw in row.items():
            if raw in ("", None) or key in {
                "decision",
                "fallback_reason",
                "rescue_reason",
            }:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            numeric_checked += 1
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite trace value in {key}: {raw}"
                )
    return {
        "rows": len(rows),
        "numeric_values_checked": numeric_checked,
        "decision_counts": decisions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--gating-enabled", action="store_true")
    parser.add_argument("--warmup", type=float, default=0.15)
    parser.add_argument("--min-progress", type=float, default=0.15)
    parser.add_argument("--max-progress", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--stable-steps", type=int, default=3)
    parser.add_argument("--calibration-interval", type=int, default=10)
    parser.add_argument("--max-skips", type=int, default=8)
    parser.add_argument("--fallback-threshold", type=float, default=0.20)
    parser.add_argument("--budget-aware-enabled", action="store_true")
    parser.add_argument("--max-skip-ratio", type=float, default=1.0)
    parser.add_argument(
        "--atomic-veto-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--atomic-threshold", type=float, default=0.05)
    parser.add_argument("--atomic-stable-steps", type=int, default=1)
    parser.add_argument(
        "--adaptive-calibration-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--calibration-min", type=int, default=4)
    parser.add_argument("--calibration-max", type=int, default=16)
    parser.add_argument(
        "--field-aggregation",
        choices=("all_fields", "weighted_max", "weighted_rms"),
        default="all_fields",
    )
    parser.add_argument(
        "--rescue-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trace", choices=("off", "disk"), default="off"
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=CHECKPOINT
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sampling_steps < 2:
        raise ValueError("sampling_steps must be >= 2")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_text(
        output / "command.txt",
        " ".join([sys.executable, *sys.argv]) + "\n",
    )
    checkpoint_file = args.checkpoint_root / "checkpoints" / "last.ckpt"
    config = {
        "route": "budget_aware_convergence_guided_corrector_scheduling",
        "config_id": args.config_id,
        "repeat_index": args.repeat_index,
        "seed": args.seed,
        "physical_gpu": args.physical_gpu,
        "checkpoint_root": str(args.checkpoint_root),
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "code_commit": subprocess.run(
            ["git", "-C", "/data/dxl/mattergen_v1", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "target": {"dft_mag_density": 0.10},
        "base_guidance": 2.0,
        "guidance_schedule": "adaptive",
        "adaptive_alpha": 0.50,
        "adaptive_ema": 0.95,
        "batch_size": 1,
        "sampling_steps": args.sampling_steps,
        "strict_deterministic": True,
        "corrector_gating": {
            "enabled": args.gating_enabled,
            "warmup_frac": args.warmup,
            "min_progress": args.min_progress,
            "max_progress": args.max_progress,
            "convergence_threshold": args.threshold,
            "consecutive_stable_steps": args.stable_steps,
            "calibration_interval": args.calibration_interval,
            "max_consecutive_skips": args.max_skips,
            "fallback_threshold": args.fallback_threshold,
            "rescue_enabled": args.rescue_enabled,
            "budget_aware_enabled": args.budget_aware_enabled,
            "max_skip_ratio": args.max_skip_ratio,
            "atomic_veto_enabled": args.atomic_veto_enabled,
            "atomic_stability_threshold": args.atomic_threshold,
            "atomic_min_stable_steps": args.atomic_stable_steps,
            "adaptive_calibration_enabled": (
                args.adaptive_calibration_enabled
            ),
            "calibration_interval_min": args.calibration_min,
            "calibration_interval_max": args.calibration_max,
            "field_aggregation": args.field_aggregation,
            "trace": args.trace,
        },
    }
    atomic_json(output / "run_config.json", config)

    trace_path = output / "corrector_trace.csv"
    corrector_summary_path = output / "corrector_summary.json"
    cfg_summary_path = output / "cfg_summary.json"
    rng_trace_path = output / "rng_trace.json"
    hashes_path = output / "structure_hashes.json"
    summary_path = output / "run_summary.json"
    telemetry_path = output / "gpu_telemetry.json"

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    intra_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    interop_threads = int(
        os.environ.get("MATTERGEN_INTEROP_THREADS", "4")
    )
    torch.set_num_threads(intra_threads)
    torch.set_num_interop_threads(interop_threads)

    initial: dict = {}
    rng: dict = {}
    from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector

    original_before = PredictorCorrector._on_before_sample_prior
    original_after = PredictorCorrector._on_after_sample_prior

    def before_prior(self, conditioning_data):
        original_before(self, conditioning_data)
        rng.update(hash_rng_state())

    def after_prior(self, batch):
        original_after(self, batch)
        for field in FIELDS:
            initial[field] = hash_tensor(batch[field])
        initial["combined"] = hashlib.sha256(
            json.dumps(initial, sort_keys=True).encode()
        ).hexdigest()

    PredictorCorrector._on_before_sample_prior = before_prior
    PredictorCorrector._on_after_sample_prior = after_prior

    telemetry = Telemetry(args.physical_gpu)
    telemetry.start()
    start = time.monotonic()
    cpu_start = time.process_time()
    success = False
    error_text = None
    structure: dict = {}
    corrector_summary: dict = {}
    cfg_summary: dict = {}
    trace_summary: dict = {}
    validity = {
        "basic_structure_valid": False,
        "composition_valid": False,
    }
    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        from mattergen.evaluation.metrics.structure import (
            is_smact_valid,
            structure_validity,
        )
        from mattergen.scripts.generate import main as generate

        model_overrides: list[str] = []
        if args.sampling_steps != 1000:
            model_overrides.append(
                "lightning_module.diffusion_module.corruption."
                "discrete_corruptions.atomic_numbers.d3pm.schedule."
                f"num_steps={args.sampling_steps}"
            )
        torch.cuda.reset_peak_memory_stats()
        generate(
            output_path=str(output),
            model_path=str(args.checkpoint_root),
            batch_size=1,
            num_batches=1,
            config_overrides=model_overrides,
            sampling_config_overrides=[
                f"sampler_partial.N={args.sampling_steps}"
            ],
            properties_to_condition_on={"dft_mag_density": 0.10},
            record_trajectories=False,
            diffusion_guidance_factor=2.0,
            seed=args.seed,
            deterministic=True,
            guidance_schedule="adaptive",
            guidance_warmup_frac=0.1,
            guidance_decay_frac=0.1,
            guidance_min_scale=0.0,
            guidance_max_scale=5.0,
            guidance_adaptive_alpha=0.50,
            guidance_adaptive_ema=0.95,
            guidance_adaptive_eps=1e-6,
            cfg_acceleration_enabled=False,
            cfg_trace_mode="off",
            cfg_summary_path=str(cfg_summary_path),
            corrector_gating_enabled=args.gating_enabled,
            corrector_warmup_frac=args.warmup,
            corrector_min_progress=args.min_progress,
            corrector_max_progress=args.max_progress,
            corrector_convergence_threshold=args.threshold,
            corrector_consecutive_stable_steps=args.stable_steps,
            corrector_calibration_interval=args.calibration_interval,
            corrector_max_consecutive_skips=args.max_skips,
            corrector_fallback_threshold=args.fallback_threshold,
            corrector_rescue_enabled=args.rescue_enabled,
            corrector_budget_aware_enabled=(
                args.budget_aware_enabled
            ),
            corrector_max_skip_ratio=args.max_skip_ratio,
            corrector_atomic_veto_enabled=args.atomic_veto_enabled,
            corrector_atomic_stability_threshold=(
                args.atomic_threshold
            ),
            corrector_atomic_min_stable_steps=(
                args.atomic_stable_steps
            ),
            corrector_adaptive_calibration_enabled=(
                args.adaptive_calibration_enabled
            ),
            corrector_calibration_interval_min=args.calibration_min,
            corrector_calibration_interval_max=args.calibration_max,
            corrector_field_aggregation=args.field_aggregation,
            corrector_trace_path=(
                str(trace_path) if args.trace == "disk" else None
            ),
            corrector_summary_path=str(corrector_summary_path),
        )
        if not rng or not initial:
            raise RuntimeError("RNG or initial-state hooks did not run")
        atomic_json(rng_trace_path, rng)
        structure, atoms = validate_structure(
            output / "generated_crystals.extxyz"
        )
        pymatgen_structure = AseAtomsAdaptor.get_structure(atoms)
        validity = {
            "basic_structure_valid": bool(
                structure_validity(pymatgen_structure)
            ),
            "composition_valid": bool(
                is_smact_valid(pymatgen_structure)
            ),
        }
        hashes = {
            "rng_state_hash": rng["combined"],
            "initial_atomic_numbers_hash": initial["atomic_numbers"],
            "initial_pos_hash": initial["pos"],
            "initial_cell_hash": initial["cell"],
            "initial_state_hash": initial["combined"],
            "final_structure_hash": structure[
                "final_structure_hash"
            ],
            "extxyz_sha256": sha256_file(
                output / "generated_crystals.extxyz"
            ),
            "atomic_numbers": structure["atomic_numbers"],
            "positions": structure["positions"],
            "cell": structure["cell"],
        }
        atomic_json(hashes_path, hashes)
        with corrector_summary_path.open(encoding="utf-8") as stream:
            corrector_summary = json.load(stream)
        with cfg_summary_path.open(encoding="utf-8") as stream:
            cfg_summary = json.load(stream)
        trace_summary = (
            validate_trace(trace_path, args.sampling_steps)
            if args.trace == "disk"
            else {"rows": 0, "numeric_values_checked": 0}
        )
        if corrector_summary["physical_model_forward_count"] != (
            corrector_summary["predictor_forward_count"]
            + corrector_summary["corrector_forward_count"]
        ):
            raise ValueError("Physical forward accounting mismatch")
        success = True
    except BaseException:
        error_text = traceback.format_exc()
        atomic_text(output / "error.txt", error_text)
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        process_cpu_seconds = time.process_time() - cpu_start
        telemetry.stop()
        telemetry_summary = telemetry.summary()
        atomic_json(
            telemetry_path,
            {"samples": telemetry.rows, "summary": telemetry_summary},
        )
        summary = {
            "success": success,
            "return_code": 0 if success else 1,
            "route": "budget_aware_convergence_guided_corrector_scheduling",
            "config_id": args.config_id,
            "repeat_index": args.repeat_index,
            "seed": args.seed,
            "physical_gpu": args.physical_gpu,
            "elapsed_seconds": elapsed,
            "process_cpu_seconds": process_cpu_seconds,
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
            "telemetry": telemetry_summary,
            "trace": trace_summary,
            "corrector": corrector_summary,
            "cfg": cfg_summary,
            "gating_enabled": args.gating_enabled,
            "budget_aware_enabled": args.budget_aware_enabled,
            "formula": structure.get("formula"),
            "num_atoms": structure.get("num_atoms"),
            **validity,
            "error": error_text,
        }
        atomic_json(summary_path, summary)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
