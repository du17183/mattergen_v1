#!/usr/bin/env python3
"""Run one deterministic A0/T1/T2 CG-TDR evaluation sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import ase.io
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def structure_record(path: Path) -> dict:
    atoms = ase.io.read(path)
    arrays = (
        np.asarray(atoms.numbers),
        np.asarray(atoms.positions),
        np.asarray(atoms.cell.array),
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Generated structure contains NaN or Inf")
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return {
        "formula": atoms.get_chemical_formula(),
        "num_atoms": len(atoms),
        "atomic_numbers": arrays[0].tolist(),
        "positions": arrays[1].tolist(),
        "cell": arrays[2].tolist(),
        "final_structure_hash": digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("A0", "T1", "T2"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cg-tdr-checkpoint")
    parser.add_argument(
        "--checkpoint-root",
        default="/data/dxl/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density",
    )
    args = parser.parse_args()
    if args.method != "A0" and not args.cg_tdr_checkpoint:
        raise ValueError("--cg-tdr-checkpoint is required for T1/T2")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "run_summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if old.get("success"):
            return 0
    metrics_path = output / "cg_tdr_metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(int(os.environ.get("MATTERGEN_INTEROP_THREADS", "2")))

    initial: dict[str, str] = {}
    from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector

    original_after = PredictorCorrector._on_after_sample_prior

    def after_prior(self, batch):
        original_after(self, batch)
        for field in ("atomic_numbers", "pos", "cell"):
            initial[field] = tensor_hash(batch[field])
        initial["combined"] = hashlib.sha256(
            json.dumps(initial, sort_keys=True).encode()
        ).hexdigest()

    PredictorCorrector._on_after_sample_prior = after_prior
    start = time.monotonic()
    try:
        from mattergen.scripts.generate import main as generate

        enabled = args.method != "A0"
        enable_cell = args.method == "T2"
        overrides = [
            f"sampler_partial.cg_tdr_enabled={str(enabled).lower()}",
            f"sampler_partial.cg_tdr_enable_cell={str(enable_cell).lower()}",
            f"sampler_partial.cg_tdr_metrics_path={metrics_path}",
        ]
        if enabled:
            overrides.append(
                f"sampler_partial.cg_tdr_checkpoint={Path(args.cg_tdr_checkpoint).resolve()}"
            )
        generate(
            output_path=str(output),
            model_path=args.checkpoint_root,
            batch_size=1,
            num_batches=1,
            properties_to_condition_on={"dft_mag_density": 0.10},
            sampling_config_name="cg_tdr",
            sampling_config_overrides=overrides,
            record_trajectories=False,
            diffusion_guidance_factor=2.0,
            seed=args.seed,
            deterministic=True,
            guidance_schedule="adaptive",
            guidance_warmup_frac=0.1,
            guidance_decay_frac=0.1,
            guidance_min_scale=0.0,
            guidance_max_scale=5.0,
            guidance_adaptive_alpha=0.5,
            guidance_adaptive_ema=0.95,
            guidance_adaptive_eps=1.0e-6,
        )
        structure = structure_record(output / "generated_crystals.extxyz")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        hashes = {
            "initial_atomic_numbers_hash": initial["atomic_numbers"],
            "initial_pos_hash": initial["pos"],
            "initial_cell_hash": initial["cell"],
            "initial_state_hash": initial["combined"],
            **structure,
        }
        atomic_json(output / "structure_hashes.json", hashes)
        atomic_json(
            summary_path,
            {
                "success": True,
                "method": args.method,
                "seed": args.seed,
                "physical_gpu": args.physical_gpu,
                "elapsed_seconds": time.monotonic() - start,
                "cg_tdr_enabled": enabled,
                "cg_tdr_enable_cell": enable_cell,
                "cg_tdr_checkpoint": args.cg_tdr_checkpoint,
                "formula": structure["formula"],
                "num_atoms": structure["num_atoms"],
                "metrics": metrics,
            },
        )
        return 0
    except BaseException:
        error = traceback.format_exc()
        (output / "error.txt").write_text(error, encoding="utf-8")
        atomic_json(
            summary_path,
            {
                "success": False,
                "method": args.method,
                "seed": args.seed,
                "physical_gpu": args.physical_gpu,
                "elapsed_seconds": time.monotonic() - start,
                "error": error,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
