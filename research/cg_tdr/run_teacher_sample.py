#!/usr/bin/env python3
"""Generate one deterministic A0 sample and its terminal CG-TDR feature record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import ase.io
import numpy as np
import torch


CHECKPOINT_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_outputs(output_dir: Path, teacher_record: Path, seed: int) -> dict:
    frames = ase.io.read(output_dir / "generated_crystals.extxyz", ":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) != 1:
        raise ValueError(f"Expected one structure, got {len(frames)}")
    atoms = frames[0]
    if not (
        np.isfinite(atoms.positions).all()
        and np.isfinite(atoms.cell.array).all()
        and np.isfinite(atoms.numbers).all()
    ):
        raise ValueError("Generated structure contains NaN or Inf")
    payload = torch.load(teacher_record, map_location="cpu", weights_only=False)
    required = {
        "pos",
        "cell",
        "atomic_numbers",
        "num_atoms",
        "batch_index",
        "node_features",
        "convergence",
        "conditional_score_pos",
        "conditional_score_cell",
        "adaptive_residual",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Teacher feature record is missing: {sorted(missing)}")
    if int(payload["seed"]) != seed:
        raise ValueError(f"Teacher record seed mismatch: {payload['seed']} != {seed}")
    if payload["node_features"].shape[0] != len(atoms):
        raise ValueError("Node feature count differs from generated atom count")
    if payload["convergence"].shape != (1, 8):
        raise ValueError(f"Unexpected convergence shape: {payload['convergence'].shape}")
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value.float()).all():
            raise ValueError(f"Non-finite tensor in teacher record: {key}")
    return {
        "formula": atoms.get_chemical_formula(),
        "num_atoms": len(atoms),
        "node_feature_shape": list(payload["node_features"].shape),
        "convergence": payload["convergence"].tolist(),
        "teacher_record_sha256": sha256_file(teacher_record),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument(
        "--checkpoint-root",
        default="/data/dxl/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    teacher_dir = Path(args.teacher_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    teacher_record = teacher_dir / f"seed_{args.seed}.pt"
    if summary_path.exists() and teacher_record.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("success"):
            return 0

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(int(os.environ.get("MATTERGEN_INTEROP_THREADS", "2")))

    start = time.monotonic()
    try:
        from mattergen.scripts.generate import main as generate

        structures = generate(
            output_path=str(output_dir),
            model_path=args.checkpoint_root,
            batch_size=1,
            num_batches=1,
            properties_to_condition_on={"dft_mag_density": 0.10},
            sampling_config_name="cg_tdr",
            sampling_config_overrides=[
                f"sampler_partial.cg_tdr_teacher_dump_dir={teacher_dir}",
                "sampler_partial.cg_tdr_enabled=false",
            ],
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
        if len(structures) != 1:
            raise RuntimeError(f"Expected one generated structure, got {len(structures)}")
        validation = validate_outputs(output_dir, teacher_record, args.seed)
        summary = {
            "success": True,
            "seed": args.seed,
            "physical_gpu": args.physical_gpu,
            "elapsed_seconds": time.monotonic() - start,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            **validation,
        }
        atomic_json(summary_path, summary)
        return 0
    except BaseException:
        error = traceback.format_exc()
        (output_dir / "error.txt").write_text(error, encoding="utf-8")
        atomic_json(
            summary_path,
            {
                "success": False,
                "seed": args.seed,
                "physical_gpu": args.physical_gpu,
                "elapsed_seconds": time.monotonic() - start,
                "error": error,
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
