"""Audit that batched trajectories consume independent seed-local RNG streams."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import open_dict

from mattergen.common.data.collate import collate
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator
from research.fn_pra.phase1_common import REPORTS, atomic_json, now
from research.fn_pra.run_batch_trial import CHECKPOINT_ROOT, hash_graph, make_condition


STEPS = 10


def state_hash(state: torch.Tensor) -> str:
    return hashlib.sha256(state.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_generator() -> CrystalGenerator:
    overrides = [
        "++lightning_module.diffusion_module.model.element_mask_func="
        "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}",
        "lightning_module.diffusion_module.corruption.discrete_corruptions."
        f"atomic_numbers.d3pm.schedule.num_steps={STEPS}",
    ]
    info = MatterGenCheckpointInfo(
        model_path=CHECKPOINT_ROOT,
        load_epoch="last",
        config_overrides=overrides,
    )
    generator = CrystalGenerator(
        checkpoint_info=info,
        diffusion_guidance_factor=2.0,
        properties_to_condition_on={"dft_mag_density": 0.10},
        deterministic=True,
        guidance_schedule="adaptive",
        guidance_min_scale=0.0,
        guidance_max_scale=5.0,
        guidance_adaptive_alpha=0.50,
        guidance_adaptive_ema=0.95,
        guidance_adaptive_eps=1e-6,
        sampling_config_overrides=[f"sampler_partial.N={STEPS}"],
        record_trajectories=False,
    )
    generator.prepare()
    return generator


def run(generator: CrystalGenerator, seeds: list[int]) -> dict:
    config = generator.load_sampling_config(len(seeds), 1)
    with open_dict(config.sampler_partial):
        config.sampler_partial._target_ = (
            "research.fn_pra.independent_batch."
            "IndependentTrajectoryGuidedPredictorCorrector.from_pl_module"
        )
        config.sampler_partial.trajectory_seeds = seeds
    sampler = instantiate(config.sampler_partial)(pl_module=generator.model)
    initial = []
    original_after = sampler._on_after_sample_prior

    def capture(batch):
        original_after(batch)
        initial.extend(hash_graph(item) for item in batch.to_data_list())

    sampler._on_after_sample_prior = capture
    _, mean = sampler.sample(collate([make_condition(seed) for seed in seeds]), None)
    return {
        "seeds": seeds,
        "initial": {str(seed): initial[index] for index, seed in enumerate(seeds)},
        "final": {
            str(seed): hash_graph(item)
            for seed, item in zip(seeds, mean.to_data_list(), strict=True)
        },
        "cpu_rng_end": {
            str(seed): state_hash(sampler._cpu_rng_states[index])
            for index, seed in enumerate(seeds)
        },
        "cuda_rng_end": {
            str(seed): state_hash(sampler._device_rng_states[index])
            for index, seed in enumerate(seeds)
        },
    }


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    generator = build_generator()
    singleton = run(generator, [15000])
    pair = run(generator, [15000, 15001])
    reversed_pair = run(generator, [15001, 15000])
    comparisons = {}
    for seed in (15000, 15001):
        sources = [pair, reversed_pair]
        if seed == 15000:
            sources.append(singleton)
        comparisons[str(seed)] = {
            "initial_identical_across_batch_context": len(
                {source["initial"][str(seed)]["combined"] for source in sources}
            )
            == 1,
            "cpu_rng_end_identical_across_batch_context": len(
                {source["cpu_rng_end"][str(seed)] for source in sources}
            )
            == 1,
            "cuda_rng_end_identical_across_batch_context": len(
                {source["cuda_rng_end"][str(seed)] for source in sources}
            )
            == 1,
            "final_level1_identical_across_batch_context": len(
                {source["final"][str(seed)]["combined"] for source in sources}
            )
            == 1,
        }
    report = {
        "created_at": now(),
        "sampling_steps": STEPS,
        "contexts": {
            "singleton": singleton,
            "pair": pair,
            "reversed_pair": reversed_pair,
        },
        "comparisons": comparisons,
        "rng_isolation_passed": all(
            values["initial_identical_across_batch_context"]
            and values["cpu_rng_end_identical_across_batch_context"]
            and values["cuda_rng_end_identical_across_batch_context"]
            for values in comparisons.values()
        ),
        "interpretation": (
            "Final hashes may differ because changing model batch shape/order changes floating-point "
            "score arithmetic; identical end RNG states prove this is not cross-seed RNG consumption."
        ),
    }
    atomic_json(REPORTS / "batch_rng_isolation_audit.json", report)


if __name__ == "__main__":
    main()
