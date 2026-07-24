# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
import os
import random
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import ase.io
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.condition_factory import ConditionLoader
from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
from mattergen.common.utils.eval_utils import (
    MatterGenCheckpointInfo,
    get_crystals_list,
    load_model_diffusion,
    make_structure,
    save_structures,
)
from mattergen.common.utils.globals import DEFAULT_SAMPLING_CONFIG_PATH, get_device
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.common.utils.data_classes import ProgressCallback


def draw_samples_from_sampler(
    sampler: PredictorCorrector,
    condition_loader: ConditionLoader,
    properties_to_condition_on: TargetProperty | None = None,
    output_path: Path | None = None,
    cfg: DictConfig | None = None,
    record_trajectories: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> list[Structure]:

    # Dict
    properties_to_condition_on = properties_to_condition_on or {}

    # we cannot conditional sample on something on which the model was not trained to condition on
    assert all([key in sampler.diffusion_module.model.cond_fields_model_was_trained_on for key in properties_to_condition_on.keys()])  # type: ignore

    all_samples_list = []
    all_trajs_list = []
    for batch_idx, (conditioning_data, mask) in enumerate(tqdm(condition_loader, desc="Generating samples")):
        if progress_callback is not None:
            progress_callback(progress=batch_idx / len(condition_loader))

        # generate samples
        if record_trajectories:
            sample, mean, intermediate_samples = sampler.sample_with_record(conditioning_data, mask)
            all_trajs_list.extend(list_of_time_steps_to_list_of_trajectories(intermediate_samples))
        else:
            sample, mean = sampler.sample(conditioning_data, mask)
        all_samples_list.extend(mean.to_data_list())

    if progress_callback is not None:
        # log 100% progress
        progress_callback(progress=1.0)

    all_samples = collate(all_samples_list)
    assert isinstance(all_samples, ChemGraph)
    lengths, angles = lattice_matrix_to_params_torch(all_samples.cell)
    all_samples = all_samples.replace(lengths=lengths, angles=angles)

    generated_strucs = structure_from_model_output(
        all_samples["pos"].reshape(-1, 3),
        all_samples["atomic_numbers"].reshape(-1),
        all_samples["lengths"].reshape(-1, 3),
        all_samples["angles"].reshape(-1, 3),
        all_samples["num_atoms"].reshape(-1),
    )

    if output_path is not None:
        assert cfg is not None
        # Save structures to disk in both a extxyz file and a compressed zip file.
        # do this before uploading to mongo in case there is an authentication error
        save_structures(output_path, generated_strucs)

        if record_trajectories:
            dump_trajectories(
                output_path=output_path,
                all_trajs_list=all_trajs_list,
            )

    return generated_strucs


def list_of_time_steps_to_list_of_trajectories(
    list_of_time_steps: list[ChemGraph],
) -> list[list[ChemGraph]]:
    # Rearrange the shapes of the recorded intermediate samples and predictions
    # We get a list of <num_timesteps> many ChemGraphBatches, each containing <batch_size>
    # many ChemGraphs. Instead, we group all the ChemGraphs of the same trajectory together,
    # i.e., we construct lists of <batch_size> many lists of
    # <num_timesteps * (1 + num_corrector_steps)> many ChemGraphs.

    # <num_timesteps * (1 + num_corrector_steps)> many lists of <batch_size> many ChemGraphs
    data_lists_per_timesteps = [x.to_data_list() for x in list_of_time_steps]

    # <batch_size> many lists of <num_timesteps * (1 + num_corrector_steps)> many ChemGraphs.
    data_lists_per_sample = [
        [data_lists_per_timesteps[ix_t][ix_traj] for ix_t in range(len(data_lists_per_timesteps))]
        for ix_traj in range(len(data_lists_per_timesteps[0]))
    ]
    return data_lists_per_sample


def dump_trajectories(
    output_path: Path,
    all_trajs_list: list[list[ChemGraph]],
) -> None:
    try:
        # We gather all trajectories in a single zip file as .extxyz files.
        # This way we can view them easily after downloading.
        with ZipFile(output_path / "generated_trajectories.zip", "w") as zip_obj:
            for ix, traj in enumerate(all_trajs_list):
                strucs = structures_from_trajectory(traj)
                ase_atoms = [AseAtomsAdaptor.get_atoms(crystal) for crystal in strucs]
                str_io = io.StringIO()
                ase.io.write(str_io, ase_atoms, format="extxyz")
                str_io.flush()
                zip_obj.writestr(f"gen_{ix}.extxyz", str_io.getvalue())
    except IOError as e:
        print(f"Got error {e} writing the trajectory to disk.")
    except ValueError as e:
        print(f"Got error ValueError '{e}' writing the trajectory to disk.")


def structure_from_model_output(
    frac_coords, atom_types, lengths, angles, num_atoms
) -> list[Structure]:
    structures = [
        make_structure(
            lengths=d["lengths"],
            angles=d["angles"],
            atom_types=d["atom_types"],
            frac_coords=d["frac_coords"],
        )
        for d in get_crystals_list(
            frac_coords.cpu(),
            atom_types.cpu(),
            lengths.cpu(),
            angles.cpu(),
            num_atoms.cpu(),
        )
    ]
    return structures


def structures_from_trajectory(traj: list[ChemGraph]) -> list[Structure]:
    all_strucs = []
    for batch in traj:
        cell = batch.cell
        lengths, angles = lattice_matrix_to_params_torch(cell)
        all_strucs.extend(
            structure_from_model_output(
                frac_coords=batch.pos,
                atom_types=batch.atomic_numbers,
                lengths=lengths,
                angles=angles,
                num_atoms=batch.num_atoms,
            )
        )

    return all_strucs


@dataclass
class CrystalGenerator:
    checkpoint_info: MatterGenCheckpointInfo

    # These may be set at runtime
    batch_size: int | None = None
    num_batches: int | None = None
    target_compositions_dict: list[dict[str, float]] | None = None
    num_atoms_distribution: str = "ALEX_MP_20"

    # Conditional generation
    diffusion_guidance_factor: float = 0.0
    properties_to_condition_on: TargetProperty | None = None

    # Reproducibility. A None seed retains the official unseeded behavior.
    seed: int | None = None
    deterministic: bool = False

    # Guidance scheduling and trace metadata.
    guidance_schedule: str = "constant"
    guidance_warmup_frac: float = 0.1
    guidance_decay_frac: float = 0.1
    guidance_min_scale: float = 0.0
    guidance_max_scale: float = 5.0
    guidance_adaptive_alpha: float = 0.5
    guidance_adaptive_ema: float = 0.95
    guidance_adaptive_eps: float = 1e-6
    guidance_trace_path: str | None = None
    guidance_run_id: str | None = None
    cfg_acceleration_enabled: bool = False
    cfg_warmup_frac: float = 0.15
    cfg_convergence_threshold: float = 0.05
    cfg_consecutive_stable_steps: int = 3
    cfg_calibration_interval: int = 10
    cfg_max_reuse_steps: int = 8
    cfg_extrapolation_enabled: bool = False
    cfg_extrapolation_order: int = 1
    cfg_fallback_threshold: float = 0.20
    cfg_min_progress: float = 0.0
    cfg_max_progress: float = 1.0
    cfg_trace_path: str | None = None
    cfg_trace_mode: str = "auto"
    cfg_summary_path: str | None = None
    corrector_gating_enabled: bool = False
    corrector_warmup_frac: float = 0.15
    corrector_min_progress: float = 0.15
    corrector_max_progress: float = 0.95
    corrector_convergence_threshold: float = 0.05
    corrector_consecutive_stable_steps: int = 3
    corrector_calibration_interval: int = 10
    corrector_max_consecutive_skips: int = 8
    corrector_fallback_threshold: float = 0.20
    corrector_rescue_enabled: bool = True
    corrector_budget_aware_enabled: bool = False
    corrector_max_skip_ratio: float = 1.0
    corrector_atomic_veto_enabled: bool = False
    corrector_atomic_stability_threshold: float = 0.05
    corrector_atomic_min_stable_steps: int = 1
    corrector_adaptive_calibration_enabled: bool = False
    corrector_calibration_interval_min: int = 4
    corrector_calibration_interval_max: int = 16
    corrector_field_aggregation: str = "all_fields"
    corrector_trace_path: str | None = None
    corrector_summary_path: str | None = None

    # Additional overrides, only has an effect when using a diffusion-codebase model
    sampling_config_overrides: list[str] | None = None

    # These only have an effect when using a legacy model
    num_samples_per_batch: int = 1
    niggli_reduction: bool = False

    # Config path, if None will default to DEFAULT_SAMPLING_CONFIG_PATH
    sampling_config_path: Path | None = None
    sampling_config_name: str = "default"

    record_trajectories: bool = True  # store all intermediate samples by default

    # These attributes are set when prepare() method is called.
    _model: DiffusionLightningModule | None = None
    _cfg: DictConfig | None = None

    # can be used to monitor progress of generation
    progress_callback: ProgressCallback | None = None

    def __post_init__(self) -> None:
        if self.seed is not None and not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be in the NumPy-compatible range [0, 2**32 - 1]")
        assert self.num_atoms_distribution in NUM_ATOMS_DISTRIBUTIONS, (
            f"num_atoms_distribution must be one of {list(NUM_ATOMS_DISTRIBUTIONS.keys())}, "
            f"but got {self.num_atoms_distribution}. To add your own distribution, "
            "please add it to mattergen.common.data.num_atoms_distribution.NUM_ATOMS_DISTRIBUTIONS."
        )
        if self.target_compositions_dict:
            assert self.cfg.lightning_module.diffusion_module.loss_fn.weights.get(
                "atomic_numbers", 0.0
            ) == 0.0 and "atomic_numbers" not in self.cfg.lightning_module.diffusion_module.corruption.get(
                "discrete_corruptions", {}
            ), "Input model appears to have been trained for crystal generation (i.e., with atom type denoising), not crystal structure prediction. Please use a model trained for crystal structure prediction instead."
            sampling_cfg = self._load_sampling_config(
                sampling_config_name=self.sampling_config_name,
                sampling_config_overrides=self.sampling_config_overrides,
                sampling_config_path=self.sampling_config_path,
            )
            if (
                "atomic_numbers" in sampling_cfg.sampler_partial.predictor_partials
                or "atomic_numbers" in sampling_cfg.sampler_partial.corrector_partials
            ):
                raise ValueError(
                    "Incompatible sampling config for crystal structure prediction: found atomic_numbers in predictor_partials or corrector_partials. Use the 'csp' sampling config instead, e.g., via --sampling-config-name=csp."
                )

    @property
    def model(self) -> DiffusionLightningModule:
        self.prepare()
        assert self._model is not None
        return self._model

    @property
    def cfg(self) -> DictConfig:
        self._cfg = self.checkpoint_info.config
        assert self._cfg is not None
        return self._cfg

    @property
    def num_structures_to_generate(self) -> int:
        """Returns the total number of structures to generate if `batch_size` and `num_batches` are specified at construction time;
        otherwise, raises an AssertionError.
        """
        assert self.batch_size is not None
        assert self.num_batches is not None
        return self.batch_size * self.num_batches

    @property
    def sampling_config(self) -> DictConfig:
        """Returns the sampling config if `batch_size` and `num_batches` are specified at construction time;
        otherwise, raises an AssertionError.
        """
        assert self.batch_size is not None
        assert self.num_batches is not None
        return self.load_sampling_config(
            batch_size=self.batch_size,
            num_batches=self.num_batches,
            target_compositions_dict=self.target_compositions_dict,
        )

    def get_condition_loader(
        self,
        sampling_config: DictConfig,
        target_compositions_dict: list[dict[str, float]] | None = None,
    ) -> ConditionLoader:
        condition_loader_partial = instantiate(sampling_config.condition_loader_partial)
        if not target_compositions_dict:
            return condition_loader_partial(properties=self.properties_to_condition_on)

        return condition_loader_partial(target_compositions_dict=target_compositions_dict)

    def load_sampling_config(
        self,
        batch_size: int,
        num_batches: int,
        target_compositions_dict: list[dict[str, float]] | None = None,
    ) -> DictConfig:
        """
        Create a sampling config from the given parameters.
        We specify certain sampling hyperparameters via the sampling config that is loaded via hydra.
        """
        if self.sampling_config_overrides is None:
            sampling_config_overrides = []
        else:
            # avoid modifying the original list
            sampling_config_overrides = self.sampling_config_overrides.copy()
        if not target_compositions_dict:
            # Default `condition_loader_partial` is
            # mattergen.common.data.condition_factory.get_number_of_atoms_condition_loader
            sampling_config_overrides += [
                f"+condition_loader_partial.num_atoms_distribution={self.num_atoms_distribution}",
                f"+condition_loader_partial.batch_size={batch_size}",
                f"+condition_loader_partial.num_samples={num_batches * batch_size}",
                f"sampler_partial.guidance_scale={self.diffusion_guidance_factor}",
                f"sampler_partial.guidance_schedule={self.guidance_schedule}",
                f"sampler_partial.guidance_warmup_frac={self.guidance_warmup_frac}",
                f"sampler_partial.guidance_decay_frac={self.guidance_decay_frac}",
                f"sampler_partial.guidance_min_scale={self.guidance_min_scale}",
                f"sampler_partial.guidance_max_scale={self.guidance_max_scale}",
                f"sampler_partial.guidance_adaptive_alpha={self.guidance_adaptive_alpha}",
                f"sampler_partial.guidance_adaptive_ema={self.guidance_adaptive_ema}",
                f"sampler_partial.guidance_adaptive_eps={self.guidance_adaptive_eps}",
                f"sampler_partial.cfg_acceleration_enabled={self.cfg_acceleration_enabled}",
                f"sampler_partial.cfg_warmup_frac={self.cfg_warmup_frac}",
                f"sampler_partial.cfg_convergence_threshold={self.cfg_convergence_threshold}",
                f"sampler_partial.cfg_consecutive_stable_steps={self.cfg_consecutive_stable_steps}",
                f"sampler_partial.cfg_calibration_interval={self.cfg_calibration_interval}",
                f"sampler_partial.cfg_max_reuse_steps={self.cfg_max_reuse_steps}",
                f"sampler_partial.cfg_extrapolation_enabled={self.cfg_extrapolation_enabled}",
                f"sampler_partial.cfg_extrapolation_order={self.cfg_extrapolation_order}",
                f"sampler_partial.cfg_fallback_threshold={self.cfg_fallback_threshold}",
                f"sampler_partial.cfg_min_progress={self.cfg_min_progress}",
                f"sampler_partial.cfg_max_progress={self.cfg_max_progress}",
                f"sampler_partial.cfg_trace_mode={self.cfg_trace_mode}",
                f"sampler_partial.corrector_gating_enabled={self.corrector_gating_enabled}",
                f"sampler_partial.corrector_warmup_frac={self.corrector_warmup_frac}",
                f"sampler_partial.corrector_min_progress={self.corrector_min_progress}",
                f"sampler_partial.corrector_max_progress={self.corrector_max_progress}",
                f"sampler_partial.corrector_convergence_threshold={self.corrector_convergence_threshold}",
                f"sampler_partial.corrector_consecutive_stable_steps={self.corrector_consecutive_stable_steps}",
                f"sampler_partial.corrector_calibration_interval={self.corrector_calibration_interval}",
                f"sampler_partial.corrector_max_consecutive_skips={self.corrector_max_consecutive_skips}",
                f"sampler_partial.corrector_fallback_threshold={self.corrector_fallback_threshold}",
                f"sampler_partial.corrector_rescue_enabled={self.corrector_rescue_enabled}",
                f"sampler_partial.corrector_budget_aware_enabled={self.corrector_budget_aware_enabled}",
                f"sampler_partial.corrector_max_skip_ratio={self.corrector_max_skip_ratio}",
                f"sampler_partial.corrector_atomic_veto_enabled={self.corrector_atomic_veto_enabled}",
                f"sampler_partial.corrector_atomic_stability_threshold={self.corrector_atomic_stability_threshold}",
                f"sampler_partial.corrector_atomic_min_stable_steps={self.corrector_atomic_min_stable_steps}",
                f"sampler_partial.corrector_adaptive_calibration_enabled={self.corrector_adaptive_calibration_enabled}",
                f"sampler_partial.corrector_calibration_interval_min={self.corrector_calibration_interval_min}",
                f"sampler_partial.corrector_calibration_interval_max={self.corrector_calibration_interval_max}",
                f"sampler_partial.corrector_field_aggregation={self.corrector_field_aggregation}",
            ]
            if self.seed is not None:
                sampling_config_overrides.append(f"sampler_partial.sample_seed={self.seed}")
            if self.guidance_trace_path is not None:
                sampling_config_overrides.append(
                    f"sampler_partial.guidance_trace_path={self.guidance_trace_path}"
                )
            if self.guidance_run_id is not None:
                sampling_config_overrides.append(f"sampler_partial.run_id={self.guidance_run_id}")
            if self.cfg_trace_path is not None:
                sampling_config_overrides.append(
                    f"sampler_partial.cfg_trace_path={self.cfg_trace_path}"
                )
            if self.cfg_summary_path is not None:
                sampling_config_overrides.append(
                    f"sampler_partial.cfg_summary_path={self.cfg_summary_path}"
                )
            if self.corrector_trace_path is not None:
                sampling_config_overrides.append(
                    f"sampler_partial.corrector_trace_path={self.corrector_trace_path}"
                )
            if self.corrector_summary_path is not None:
                sampling_config_overrides.append(
                    f"sampler_partial.corrector_summary_path={self.corrector_summary_path}"
                )
        else:
            # `condition_loader_partial` for fixed atom type (crystal structure prediction)
            num_structures_to_generate_per_composition = (
                num_batches * batch_size // len(target_compositions_dict)
            )
            sampling_config_overrides += [
                "condition_loader_partial._target_=mattergen.common.data.condition_factory.get_composition_data_loader",
                f"+condition_loader_partial.num_structures_to_generate_per_composition={num_structures_to_generate_per_composition}",
                f"+condition_loader_partial.batch_size={batch_size}",
            ]
        return self._load_sampling_config(
            sampling_config_overrides=sampling_config_overrides,
            sampling_config_path=self.sampling_config_path,
            sampling_config_name=self.sampling_config_name,
        )

    def _load_sampling_config(
        self,
        sampling_config_path: Path | None = None,
        sampling_config_name: str = "default",
        sampling_config_overrides: list[str] | None = None,
    ) -> DictConfig:
        if sampling_config_path is None:
            sampling_config_path = DEFAULT_SAMPLING_CONFIG_PATH

        if sampling_config_overrides is None:
            sampling_config_overrides = []

        with hydra.initialize_config_dir(os.path.abspath(str(sampling_config_path))):
            sampling_config = hydra.compose(
                config_name=sampling_config_name, overrides=sampling_config_overrides
            )
        return sampling_config

    def prepare(self) -> None:
        """Loads the model from checkpoint and prepares for generation."""
        if self._model is not None:
            return
        model = load_model_diffusion(self.checkpoint_info)
        model = model.to(get_device())
        self._model = model
        self._cfg = self.checkpoint_info.config

    def _configure_deterministic_mode(self) -> None:
        if not self.deterministic:
            return
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _seed_sampling_rngs(self) -> None:
        if self.seed is None:
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def generate(
        self,
        batch_size: int | None = None,
        num_batches: int | None = None,
        target_compositions_dict: list[dict[str, float]] | None = None,
        output_dir: str = "outputs",
    ) -> list[Structure]:
        # Prioritize the runtime provided batch_size, num_batches and target_compositions_dict
        batch_size = batch_size or self.batch_size
        num_batches = num_batches or self.num_batches
        target_compositions_dict = target_compositions_dict or self.target_compositions_dict
        assert batch_size is not None
        assert num_batches is not None

        # Deterministic kernels must be requested before the model performs CUDA
        # work. Load the model before setting the seed because module
        # construction may consume RNG. Set the seed before ConditionLoader
        # creation because it samples num_atoms with NumPy.
        self._configure_deterministic_mode()
        self.prepare()
        self._seed_sampling_rngs()

        # print config for debugging and reproducibility
        print("\nModel config:")
        print(OmegaConf.to_yaml(self.cfg, resolve=True))

        sampling_config = self.load_sampling_config(
            batch_size=batch_size,
            num_batches=num_batches,
            target_compositions_dict=target_compositions_dict,
        )

        print("\nSampling config:")
        print(OmegaConf.to_yaml(sampling_config, resolve=True))
        condition_loader = self.get_condition_loader(sampling_config, target_compositions_dict)

        sampler_partial = instantiate(sampling_config.sampler_partial)
        sampler = sampler_partial(pl_module=self.model)

        generated_structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=condition_loader,
            cfg=self.cfg,
            output_path=Path(output_dir),
            properties_to_condition_on=self.properties_to_condition_on,
            record_trajectories=self.record_trajectories,
            progress_callback=self.progress_callback,
        )

        return generated_structures
