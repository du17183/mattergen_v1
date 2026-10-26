"""Collect frozen GemNet-T block activations on new official sampling trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from hydra.utils import instantiate

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator, draw_samples_from_sampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/self_correcting_search_p0"
MODEL_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/checkpoints/dft_mag_density")
EXPECTED_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
ALL_SEEDS = tuple(range(98000, 98192))
REQUESTED_TIMESTEPS = (0.50, 0.25, 0.10, 0.05, 0.02)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for_seed(seed: int) -> str:
    if 98000 <= seed <= 98127:
        return "train"
    if 98128 <= seed <= 98159:
        return "validation"
    if 98160 <= seed <= 98191:
        return "test"
    raise ValueError(f"seed outside preregistered probe cohort: {seed}")


class HiddenRecorder:
    """Read-only forward hooks for the conditional half of a CFG=2 model call."""

    def __init__(self, sampler, gemnet: torch.nn.Module) -> None:
        self.sampler = sampler
        self.records: dict[tuple[int, int], np.ndarray] = {}
        self.actual_t: dict[int, float] = {}
        timesteps = torch.linspace(sampler._max_t, sampler._eps_t, sampler.N)
        self.sampler_timesteps = timesteps.cpu().numpy()
        self.step_to_requested = {
            int(torch.argmin(torch.abs(timesteps - requested)).item()): requested
            for requested in REQUESTED_TIMESTEPS
        }
        if len(self.step_to_requested) != len(REQUESTED_TIMESTEPS):
            raise RuntimeError("requested timesteps collapsed to duplicate sampler indices")
        if len(gemnet.int_blocks) != 4:
            raise RuntimeError(f"expected four GemNet-T blocks, found {len(gemnet.int_blocks)}")
        self.handles = [
            block.register_forward_hook(self._make_hook(block_index))
            for block_index, block in enumerate(gemnet.int_blocks)
        ]

    def _make_hook(self, block_index: int):
        def hook(_module, _inputs, output) -> None:
            context = self.sampler.sampling_context
            step = int(context.get("sampling_step", -1))
            if context.get("phase") != "corrector" or step not in self.step_to_requested:
                return
            key = (step, block_index)
            if key in self.records:
                raise RuntimeError(f"duplicate hidden activation for {key}")
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 2 or hidden.shape[1] != 512 or hidden.shape[0] % 2:
                raise RuntimeError(f"unexpected CFG hidden shape: {tuple(hidden.shape)}")
            # GuidedPredictorCorrector collates [unconditional, conditional]. Batch size is one.
            conditional = hidden[hidden.shape[0] // 2 :]
            pooled = conditional.float().mean(dim=0)
            if not bool(torch.isfinite(pooled).all().item()):
                raise RuntimeError("non-finite pooled hidden activation")
            self.records[key] = pooled.detach().cpu().numpy().astype(np.float32)
            self.actual_t[step] = float(self.sampler_timesteps[step])
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = []
        actual = []
        indices = []
        for requested in REQUESTED_TIMESTEPS:
            step = next(key for key, value in self.step_to_requested.items() if value == requested)
            block_rows = [self.records[(step, block)] for block in range(4)]
            rows.append(np.stack(block_rows))
            actual.append(self.actual_t[step])
            indices.append(step)
        hidden = np.stack(rows)  # [5 timesteps, 4 blocks, 512]
        return hidden, np.asarray(actual, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    seeds = tuple(args.seeds) if args.seeds else ALL_SEEDS[args.shard_index :: args.num_shards]
    if any(seed not in ALL_SEEDS for seed in seeds):
        raise ValueError("non-preregistered seed")
    if any(94000 <= seed <= 96031 for seed in seeds):
        raise RuntimeError("forbidden historical seed overlap")
    checkpoint_path = MODEL_ROOT / "checkpoints/last.ckpt"
    if sha256(checkpoint_path) != EXPECTED_SHA256:
        raise RuntimeError("official checkpoint hash mismatch")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo(model_path=str(MODEL_ROOT)),
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={"dft_mag_density": 0.2},
        diffusion_guidance_factor=2.0,
        guidance_schedule="constant",
        seed=seeds[0],
        deterministic=True,
        record_trajectories=False,
    )
    generator._configure_deterministic_mode()
    generator.prepare()
    generator.model.eval()
    for parameter in generator.model.parameters():
        parameter.requires_grad_(False)
    gemnet = generator.model.diffusion_module.model.gemnet

    for ordinal, seed in enumerate(seeds, start=1):
        run_dir = ROOT / "probe_dataset" / split_for_seed(seed) / str(seed)
        if (run_dir / "run_summary.json").is_file():
            print(json.dumps({"event": "skip", "seed": seed}), flush=True)
            continue
        partial = run_dir.with_name(run_dir.name + ".partial")
        if run_dir.exists() or partial.exists():
            raise RuntimeError(f"incomplete output retained: {run_dir} or {partial}")
        partial.mkdir(parents=True)

        generator.seed = seed
        generator._seed_sampling_rngs()
        sampling_config = generator.load_sampling_config(batch_size=1, num_batches=1)
        if int(sampling_config.sampler_partial.N) != 1000:
            raise RuntimeError("sampling steps changed")
        if float(sampling_config.sampler_partial.guidance_scale) != 2.0:
            raise RuntimeError("CFG changed")
        if str(sampling_config.sampler_partial.guidance_schedule) != "constant":
            raise RuntimeError("adaptive CFG is prohibited")
        condition_loader = generator.get_condition_loader(sampling_config)
        sampler = instantiate(sampling_config.sampler_partial)(pl_module=generator.model)
        recorder = HiddenRecorder(sampler, gemnet)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            structures = draw_samples_from_sampler(
                sampler=sampler,
                condition_loader=condition_loader,
                properties_to_condition_on={"dft_mag_density": 0.2},
                output_path=partial,
                cfg=generator.cfg,
                record_trajectories=False,
            )
            torch.cuda.synchronize()
            hidden, actual_t, step_indices = recorder.arrays()
        finally:
            recorder.close()
        elapsed = time.perf_counter() - started
        np.savez_compressed(
            partial / "hidden_states.npz",
            hidden=hidden,
            requested_t=np.asarray(REQUESTED_TIMESTEPS, dtype=np.float32),
            actual_t=actual_t,
            step_indices=step_indices,
        )
        summary = {
            "success": True,
            "seed": seed,
            "split": split_for_seed(seed),
            "target_dft_mag_density": 0.2,
            "guidance_scale": 2.0,
            "guidance_schedule": "constant",
            "adaptive_cfg_used": False,
            "sampling_steps": 1000,
            "corrector_steps_per_timestep": 1,
            "activation_phase": "pre_corrector_current_state",
            "activation_branch": "conditional_cfg_branch",
            "hidden_shape": list(hidden.shape),
            "requested_timesteps": list(REQUESTED_TIMESTEPS),
            "actual_timesteps": actual_t.tolist(),
            "sampling_step_indices": step_indices.tolist(),
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "mattergen_score_calls": int(sampler.sampling_metrics["mattergen_score_calls"]),
            "checkpoint_sha256": EXPECTED_SHA256,
            "formula": structures[0].composition.reduced_formula,
            "num_atoms": len(structures[0]),
            "surrogate_property_eval": True,
            "dft_verified": False,
        }
        if summary["mattergen_score_calls"] != 2000:
            raise RuntimeError("official score-call contract changed")
        (partial / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial, run_dir)
        print(json.dumps({"event": "complete", "seed": seed, "ordinal": ordinal,
                          "total": len(seeds), "seconds": elapsed}), flush=True)


if __name__ == "__main__":
    main()
