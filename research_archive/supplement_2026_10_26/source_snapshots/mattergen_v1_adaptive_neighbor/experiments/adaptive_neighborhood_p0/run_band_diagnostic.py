#!/usr/bin/env python3
"""Generate fresh trajectories and measure block/band score sensitivity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
from types import MethodType
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/adaptive_neighborhood_p0"
MODEL_ROOT = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/"
    "hf_mattergen/checkpoints/dft_mag_density"
)
CHECKPOINT = MODEL_ROOT / "checkpoints/last.ckpt"
CHECKPOINT_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
TARGET_TIMESTEPS = (0.80, 0.50, 0.25, 0.10, 0.05, 0.02)
TARGET_INDICES = {200: 0.80, 500: 0.50, 750: 0.25, 900: 0.10, 950: 0.05, 980: 0.02}
BANDS = {
    "near": (0.0, 3.5),
    "mid": (3.5, 5.5),
    "far": (5.5, 7.0),
}
FIELDS = ("atomic_numbers", "pos", "cell")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def field_sensitivity(full: Any, masked: Any) -> dict[str, float]:
    result = {}
    for field in FIELDS:
        baseline = full[field].float()
        delta = baseline - masked[field].float()
        numerator = torch.sqrt(torch.mean(delta.square()))
        denominator = torch.sqrt(torch.mean(baseline.square())) + 1.0e-12
        result[field] = float((numerator / denominator).item())
    return result


class AtomAggregationBandMask:
    """Temporarily zero one distance band at one block's atom aggregation."""

    def __init__(self, gemnet: torch.nn.Module, block: int, band: str) -> None:
        self.gemnet = gemnet
        self.block = block
        self.band = band
        self.distance: torch.Tensor | None = None
        self.edge_count = 0
        self.masked_edge_count = 0
        self.handles: list[Any] = []

    def __enter__(self) -> "AtomAggregationBandMask":
        low, high = BANDS[self.band]

        def radial_pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            self.distance = inputs[0]

        def aggregation_pre_hook(
            _module: Any, inputs: tuple[Any, ...]
        ) -> tuple[Any, ...]:
            if self.distance is None:
                raise RuntimeError("radial distances were not captured")
            h, messages, rbf_h, idx_t = inputs
            distance = self.distance
            if len(distance) != len(messages):
                raise RuntimeError("distance/message edge count mismatch")
            if self.band == "near":
                selected = (distance >= low) & (distance < high)
            elif self.band == "mid":
                selected = (distance >= low) & (distance < high)
            else:
                selected = (distance >= low) & (distance <= high + 1.0e-6)
            self.edge_count = int(len(distance))
            self.masked_edge_count = int(selected.sum().item())
            keep = (~selected).to(messages.dtype).unsqueeze(-1)
            return h, messages * keep, rbf_h, idx_t

        self.handles.append(
            self.gemnet.radial_basis.register_forward_pre_hook(radial_pre_hook)
        )
        self.handles.append(
            self.gemnet.int_blocks[self.block].atom_update.register_forward_pre_hook(
                aggregation_pre_hook
            )
        )
        return self

    def __exit__(self, *_args: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def capture_sampler_states(sampler: Any) -> tuple[dict[float, Any], Any]:
    captured: dict[float, Any] = {}
    original = sampler._evaluate_exact_score

    def wrapped(self: Any, x: Any, t: torch.Tensor) -> Any:
        context = self.sampling_context
        index = int(context.get("sampling_step", -1))
        phase = str(context.get("phase", ""))
        if phase == "predictor" and index in TARGET_INDICES:
            target = TARGET_INDICES[index]
            if target not in captured:
                captured[target] = {
                    "state": x.clone().to("cpu"),
                    "actual_t": float(t.detach().reshape(-1)[0].cpu().item()),
                    "sampling_step": index,
                    "phase": phase,
                }
        return original(x, t)

    sampler._evaluate_exact_score = MethodType(wrapped, sampler)
    return captured, original


@torch.inference_mode()
def diagnose_state(
    sampler: Any,
    state: Any,
    actual_t: float,
    seed: int,
    requested_t: float,
) -> list[dict[str, Any]]:
    device = sampler._device
    state = state.to(device)
    t = torch.full((state.get_batch_size(),), actual_t, device=device)
    full = sampler._score_fn(state, t)
    gemnet = sampler.diffusion_module.model.gemnet
    rows: list[dict[str, Any]] = []
    for block in range(len(gemnet.int_blocks)):
        for band in BANDS:
            with AtomAggregationBandMask(gemnet, block, band) as mask:
                masked = sampler._score_fn(state, t)
            sensitivity = field_sensitivity(full, masked)
            rows.append(
                {
                    "seed": seed,
                    "requested_t_norm": requested_t,
                    "actual_t": actual_t,
                    "sampling_step": int(round((1.0 - actual_t) * 1000)),
                    "captured_phase": "predictor_after_corrector",
                    "block": block + 1,
                    "distance_band": band,
                    "band_lower_angstrom": BANDS[band][0],
                    "band_upper_angstrom": BANDS[band][1],
                    "edge_count_cfg_joint": mask.edge_count,
                    "masked_edge_count_cfg_joint": mask.masked_edge_count,
                    "atomic_sensitivity": sensitivity["atomic_numbers"],
                    "pos_sensitivity": sensitivity["pos"],
                    "cell_sensitivity": sensitivity["cell"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if not 94000 <= args.seed_start <= args.seed_end <= 94031:
        raise ValueError("diagnostic shard must be within frozen seeds 94000-94031")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real diagnostic")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("base checkpoint hash mismatch")

    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(MODEL_ROOT)),
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={"dft_mag_density": 0.2},
        diffusion_guidance_factor=2.0,
        guidance_schedule="constant",
        deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    if generator.model.diffusion_module.model.gemnet.cutoff != 7.0:
        raise RuntimeError("unexpected GemNet cutoff")
    if generator.model.diffusion_module.model.gemnet.max_neighbors != 50:
        raise RuntimeError("unexpected GemNet max_neighbors")

    shard_dir = OUT / "diagnostic_shards"
    state_root = OUT / "diagnostic_states"
    shard_dir.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_end + 1):
        seed_dir = OUT / "generation" / "B0" / str(seed)
        if seed_dir.exists():
            raise FileExistsError(f"refusing to overwrite {seed_dir}")
        temporary_seed_dir = seed_dir.parent / f".{seed}.{os.getpid()}.tmp"
        temporary_seed_dir.mkdir(parents=True, exist_ok=False)
        generator.seed = seed
        seed_everything(seed)
        generator._seed_sampling_rngs()
        sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
        if int(sampling_config.sampler_partial.N) != 1000:
            raise RuntimeError("diagnostic is not using 1000 diffusion steps")
        if str(sampling_config.sampler_partial.guidance_schedule) != "constant":
            raise RuntimeError("Adaptive CFG must be off")
        condition_loader = generator.get_condition_loader(sampling_config)
        sampler = instantiate(sampling_config.sampler_partial)(pl_module=generator.model)
        captured, _original = capture_sampler_states(sampler)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=condition_loader,
            properties_to_condition_on={"dft_mag_density": 0.2},
            output_path=temporary_seed_dir,
            cfg=generator.cfg,
            record_trajectories=False,
        )
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - started
        if sorted(captured) != sorted(TARGET_TIMESTEPS):
            raise RuntimeError(f"incomplete timestep capture for seed {seed}: {captured}")
        per_seed_rows: list[dict[str, Any]] = []
        saved_state_dir = state_root / str(seed)
        if saved_state_dir.exists():
            raise FileExistsError(f"refusing to overwrite {saved_state_dir}")
        temporary_state_dir = state_root / f".{seed}.{os.getpid()}.tmp"
        temporary_state_dir.mkdir(parents=True, exist_ok=False)
        diagnostic_started = time.perf_counter()
        for requested_t in TARGET_TIMESTEPS:
            item = captured[requested_t]
            torch.save(item, temporary_state_dir / f"t_{requested_t:.2f}.pt")
            per_seed_rows.extend(
                diagnose_state(
                    sampler,
                    item["state"],
                    item["actual_t"],
                    seed,
                    requested_t,
                )
            )
        torch.cuda.synchronize()
        diagnostic_seconds = time.perf_counter() - diagnostic_started
        all_rows.extend(per_seed_rows)
        summary = {
            "success": True,
            "seed": seed,
            "formula": structures[0].composition.reduced_formula,
            "num_atoms": len(structures[0]),
            "generation_seconds": generation_seconds,
            "diagnostic_seconds": diagnostic_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "sampling_score_calls": sampler.sampling_metrics["mattergen_score_calls"],
            "captured_timesteps": list(TARGET_TIMESTEPS),
            "sensitivity_rows": len(per_seed_rows),
            "adaptive_cfg_used": False,
            "dft_verified": False,
        }
        (temporary_seed_dir / "diagnostic_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_state_dir, saved_state_dir)
        os.replace(temporary_seed_dir, seed_dir)
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    csv_path = shard_dir / f"band_sensitivity_{args.seed_start}_{args.seed_end}.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = shard_dir / f"run_summaries_{args.seed_start}_{args.seed_end}.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
