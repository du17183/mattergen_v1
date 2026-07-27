"""Shared deterministic C0/A0 native-batch generation for SPG Fast Gate."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Iterable, Sequence

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
from mattergen.generator import CrystalGenerator, structure_from_model_output
from research.spg_fastgate.common import (
    CHECKPOINT_ROOT,
    CHECKPOINT_SHA256,
    atomic_json,
    now,
    sha256_file,
)


FIELDS = ("atomic_numbers", "cell", "pos")
METHODS = ("C0", "A0")


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


def sha256_path(path: Path) -> str:
    return sha256_file(path)


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


def configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")


def build_generator(method: str, *, batch_size: int, sampling_steps: int = 1000):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    checkpoint = CHECKPOINT_ROOT / "checkpoints/last.ckpt"
    if sha256_path(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("official dft_mag_density checkpoint SHA256 mismatch")
    overrides = [
        "++lightning_module.diffusion_module.model.element_mask_func="
        "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
    ]
    if sampling_steps != 1000:
        overrides.append(
            "lightning_module.diffusion_module.corruption.discrete_corruptions."
            f"atomic_numbers.d3pm.schedule.num_steps={sampling_steps}"
        )
    checkpoint_info = MatterGenCheckpointInfo(
        model_path=CHECKPOINT_ROOT,
        load_epoch="last",
        config_overrides=overrides,
    )
    adaptive = method == "A0"
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=batch_size,
        num_batches=1,
        num_atoms_distribution="ALEX_MP_20",
        diffusion_guidance_factor=2.0,
        properties_to_condition_on={"dft_mag_density": 0.10},
        deterministic=True,
        guidance_schedule="adaptive" if adaptive else "constant",
        guidance_min_scale=0.0,
        guidance_max_scale=5.0,
        guidance_adaptive_alpha=0.50,
        guidance_adaptive_ema=0.95,
        guidance_adaptive_eps=1e-6,
        sampling_config_overrides=[f"sampler_partial.N={sampling_steps}"],
        record_trajectories=False,
    )
    generator.prepare()
    return generator


def build_sampler(generator: CrystalGenerator, method: str, seeds: Sequence[int]):
    sampling_config = generator.load_sampling_config(
        batch_size=len(seeds),
        num_batches=1,
    )
    with open_dict(sampling_config.sampler_partial):
        if len(seeds) == 1:
            target = (
                "research.spg_fastgate.native_single_batch."
                "NativeSingleTrajectoryGuidedPredictorCorrector.from_pl_module"
            )
        else:
            target = (
                "research.spg_fastgate.independent_batch."
                "IndependentTrajectoryGuidedPredictorCorrector.from_pl_module"
            )
        sampling_config.sampler_partial._target_ = target
        sampling_config.sampler_partial.trajectory_seeds = list(seeds)
        sampling_config.sampler_partial.guidance_schedule = (
            "adaptive" if method == "A0" else "constant"
        )
    return instantiate(sampling_config.sampler_partial)(pl_module=generator.model)


def model_output_to_structures(mean_batch):
    lengths, angles = lattice_matrix_to_params_torch(mean_batch.cell)
    final_batch = mean_batch.replace(lengths=lengths, angles=angles)
    return structure_from_model_output(
        final_batch["pos"].reshape(-1, 3),
        final_batch["atomic_numbers"].reshape(-1),
        final_batch["lengths"].reshape(-1, 3),
        final_batch["angles"].reshape(-1, 3),
        final_batch["num_atoms"].reshape(-1),
    )


def run_group(
    *,
    generator: CrystalGenerator,
    method: str,
    seeds: Sequence[int],
    output_root: Path | None = None,
    physical_gpu: int = 0,
    save_outputs: bool = True,
) -> dict:
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique sequence")
    sampler = build_sampler(generator, method, seeds)
    conditions = collate([make_condition(seed) for seed in seeds])
    initial_hashes: list[dict[str, str]] = []
    original_after_prior = sampler._on_after_sample_prior

    def capture_prior(batch) -> None:
        original_after_prior(batch)
        initial_hashes.clear()
        initial_hashes.extend(hash_graph(item) for item in batch.to_data_list())

    sampler._on_after_sample_prior = capture_prior
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    _, mean = sampler.sample(conditions, None)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    structures = model_output_to_structures(mean)
    mean_items = mean.to_data_list()
    if len(structures) != len(seeds) or len(initial_hashes) != len(seeds):
        raise RuntimeError("native batch did not preserve seed ordering")

    from mattergen.evaluation.metrics.structure import is_smact_valid, structure_validity

    rows = []
    for seed, structure, item, initial_hash in zip(
        seeds, structures, mean_items, initial_hashes, strict=True
    ):
        row = {
            "seed": seed,
            "method": method,
            "batch_size": len(seeds),
            "physical_gpu": physical_gpu,
            "success": True,
            "structure_valid": bool(structure_validity(structure)),
            "composition_valid": bool(is_smact_valid(structure)),
            "initial_hashes": initial_hash,
            "final_graph_hashes": hash_graph(item),
            "generation_seconds_batch": elapsed,
            "generation_seconds_per_sample": elapsed / len(seeds),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "physical_model_forward_count": sampler.physical_model_forward_count,
            "model_graphs_evaluated": sampler.model_graphs_evaluated,
            "created_at": now(),
        }
        if save_outputs:
            if output_root is None:
                raise ValueError("output_root is required when save_outputs=True")
            seed_dir = output_root / method / f"B{len(seeds)}" / f"seed_{seed}"
            if seed_dir.exists():
                existing = seed_dir / "status.json"
                if existing.is_file():
                    status = json.loads(existing.read_text(encoding="utf-8"))
                    if status.get("success") is True:
                        rows.append(status["summary"])
                        continue
                raise RuntimeError(f"refusing to overwrite incomplete task: {seed_dir}")
            seed_dir.mkdir(parents=True)
            structure_path = seed_dir / "generated_crystals.extxyz"
            ase.io.write(
                structure_path,
                AseAtomsAdaptor.get_atoms(structure),
                format="extxyz",
            )
            row["structure_path"] = str(structure_path)
            row["final_structure_hash"] = sha256_path(structure_path)
            atomic_json(seed_dir / "summary.json", row)
            atomic_json(
                seed_dir / "status.json",
                {"success": True, "finished_at": now(), "summary": row},
            )
        rows.append(row)
    return {
        "method": method,
        "batch_size": len(seeds),
        "seeds": list(seeds),
        "elapsed_seconds": elapsed,
        "samples_per_hour": len(seeds) * 3600.0 / elapsed,
        "rows": rows,
    }


def run_group_guarded(**kwargs) -> dict:
    output_root = kwargs.get("output_root")
    method = kwargs["method"]
    seeds = kwargs["seeds"]
    try:
        return run_group(**kwargs)
    except BaseException:
        if output_root is not None:
            for seed in seeds:
                seed_dir = Path(output_root) / method / f"B{len(seeds)}" / f"seed_{seed}"
                if not seed_dir.exists():
                    seed_dir.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    seed_dir / "status.json",
                    {
                        "success": False,
                        "finished_at": now(),
                        "error": traceback.format_exc(),
                    },
                )
        raise


def chunked(values: Sequence[int], size: int) -> Iterable[tuple[int, ...]]:
    for index in range(0, len(values), size):
        chunk = tuple(values[index : index + size])
        if len(chunk) == size:
            yield chunk
