from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.sde_lib import VPSDE
from mattergen.diffusion.data.batched_data import SimpleBatchedData
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor
from mattergen.diffusion.sampling.predictors_correctors import LangevinCorrector
from mattergen.diffusion.sampling.residual_adapter import (
    AdapterPrediction,
    FieldwiseResidualAdapter,
)
from mattergen.diffusion.sampling.residual_distillation import (
    ResidualDistillationController,
    TeacherShardWriter,
    build_residual_controller,
    iter_teacher_records,
    save_adapter_checkpoint,
)
from mattergen.diffusion.tests.test_reverse_sampling import get_diffusion_module


def make_adapter_batches():
    atom_index = torch.tensor([0, 0, 1, 1, 1])
    batch_index = {
        "pos": atom_index,
        "atomic_numbers": atom_index,
        "cell": None,
        "num_atoms": None,
    }
    pos_before = torch.tensor(
        [[0.98, 0.1, 0.2], [0.3, 0.4, 0.5], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    )
    pos_after = pos_before + 0.01
    pos_after[0, 0] = 0.01
    cell_before = torch.eye(3).repeat(2, 1, 1)
    cell_after = cell_before + 0.02
    atomic_numbers = torch.tensor([1, 8, 6, 14, 26])
    x_before = SimpleBatchedData(
        data={
            "pos": pos_before,
            "cell": cell_before,
            "atomic_numbers": atomic_numbers,
            "num_atoms": torch.tensor([2, 3]),
        },
        batch_idx=dict(batch_index),
    )
    x_after = SimpleBatchedData(
        data={
            "pos": pos_after,
            "cell": cell_after,
            "atomic_numbers": atomic_numbers.clone(),
            "num_atoms": torch.tensor([2, 3]),
        },
        batch_idx=dict(batch_index),
    )
    atomic_score = torch.randn(5, 101)
    atomic_score[:, -5:] = -1.0e10
    score_index = {"pos": atom_index, "atomic_numbers": atom_index, "cell": None}
    score_before = SimpleBatchedData(
        data={
            "pos": torch.randn(5, 3),
            "cell": torch.randn(2, 3, 3),
            "atomic_numbers": atomic_score,
        },
        batch_idx=dict(score_index),
    )
    score_after = score_before.replace(
        pos=score_before["pos"] + 0.1,
        cell=score_before["cell"] - 0.2,
        atomic_numbers=score_before["atomic_numbers"]
        + torch.where(atomic_score.abs() < 1.0e8, torch.full_like(atomic_score, 0.03), 0.0),
    )
    return x_before, x_after, score_before, score_after


def test_adapter_shapes_and_masked_logits_with_variable_atom_counts():
    x_before, x_after, score_before, _ = make_adapter_batches()
    model = FieldwiseResidualAdapter()
    prediction = model(
        x_before=x_before,
        x_after=x_after,
        score_before=score_before,
        t=torch.tensor([0.8, 0.8]),
        progress=0.2,
    )
    assert model.parameter_count < 10_000
    for field in ("pos", "cell", "atomic_numbers"):
        assert prediction.score[field].shape == score_before[field].shape
        assert torch.equal(prediction.score[field], score_before[field])
    assert prediction.context_features.shape == (2, 10)
    assert prediction.risk_features.shape == (2, 13)


def test_adapter_forward_is_reproducible_for_fixed_seed():
    x_before, x_after, score_before, _ = make_adapter_batches()

    def initialized_model():
        torch.manual_seed(20260904)
        model = FieldwiseResidualAdapter()
        for parameter in model.parameters():
            torch.nn.init.uniform_(parameter, -0.05, 0.05)
        return model

    first = initialized_model()(
        x_before=x_before,
        x_after=x_after,
        score_before=score_before,
        t=torch.tensor([0.4, 0.4]),
        progress=0.6,
    )
    second = initialized_model()(
        x_before=x_before,
        x_after=x_after,
        score_before=score_before,
        t=torch.tensor([0.4, 0.4]),
        progress=0.6,
    )
    for field in ("pos", "cell", "atomic_numbers"):
        assert torch.equal(first.score[field], second.score[field])
    assert torch.equal(first.risk_features, second.risk_features)


def test_adapter_off_is_numerically_identical_to_baseline():
    fields = ["pos", "cell", "atomic_numbers"]
    corruption = MultiCorruption(sdes={field: VPSDE() for field in fields})
    diffusion = get_diffusion_module(
        x0_mean=torch.tensor(0.0), x0_std=torch.tensor(1.0), multi_corruption=corruption
    )
    kwargs = dict(
        diffusion_module=diffusion,
        device=torch.device("cpu"),
        predictor_partials={field: AncestralSamplingPredictor for field in fields},
        corrector_partials={field: LangevinCorrector for field in fields},
        n_steps_corrector=1,
        N=4,
    )
    conditioning = SimpleBatchedData(
        data={field: torch.randn(3, 2) for field in fields},
        batch_idx={field: None for field in fields},
    )
    torch.manual_seed(1234)
    baseline = PredictorCorrector(**kwargs).sample(conditioning.clone())
    torch.manual_seed(1234)
    disabled = PredictorCorrector(
        **kwargs, corrector_residual_adapter={"enabled": False}
    ).sample(conditioning.clone())
    for baseline_batch, disabled_batch in zip(baseline, disabled):
        for field in fields:
            assert torch.equal(baseline_batch[field], disabled_batch[field])


def test_sampler_metrics_count_scores_saved_by_removing_corrector():
    fields = ["pos", "cell", "atomic_numbers"]
    corruption = MultiCorruption(sdes={field: VPSDE() for field in fields})
    diffusion = get_diffusion_module(
        x0_mean=torch.tensor(0.0), x0_std=torch.tensor(1.0), multi_corruption=corruption
    )
    sampler = PredictorCorrector(
        diffusion_module=diffusion,
        device=torch.device("cpu"),
        predictor_partials={field: AncestralSamplingPredictor for field in fields},
        corrector_partials={},
        n_steps_corrector=1,
        N=4,
    )
    conditioning = SimpleBatchedData(
        data={field: torch.randn(3, 2) for field in fields},
        batch_idx={field: None for field in fields},
    )
    sampler.sample(conditioning)
    assert sampler.sampling_metrics["mattergen_score_calls"] == 4
    assert sampler.sampling_metrics["theoretical_baseline_score_calls"] == 8
    assert sampler.sampling_metrics["saved_score_calls"] == 4
    assert sampler.sampling_metrics["forward_reduction"] == 0.5


def test_force_fallback_returns_exact_score(tmp_path: Path):
    x_before, x_after, score_before, score_after = make_adapter_batches()
    checkpoint = save_adapter_checkpoint(
        tmp_path / "adapter.pt", model=FieldwiseResidualAdapter()
    )
    controller = ResidualDistillationController(
        {
            "enabled": True,
            "mode": "adapter",
            "checkpoint_path": str(checkpoint),
            "coverage_target": 1.0,
            "force_fallback": True,
        },
        device=torch.device("cpu"),
    )
    actual = controller.score_after_corrector(
        exact_score_fn=lambda _x, _t: score_after,
        x_before=x_before,
        score_before=score_before,
        x_after=x_after,
        t=torch.tensor([0.5, 0.5]),
        sampling_step=1,
        progress=0.5,
    )
    for field in ("pos", "cell", "atomic_numbers"):
        assert torch.equal(actual[field], score_after[field])
    assert controller.metrics["fallback_calls"] == 1
    assert controller.metrics["saved_score_calls"] == 0


def test_nonfinite_adapter_output_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    x_before, x_after, score_before, score_after = make_adapter_batches()
    checkpoint = save_adapter_checkpoint(
        tmp_path / "adapter.pt", model=FieldwiseResidualAdapter()
    )
    controller = ResidualDistillationController(
        {
            "enabled": True,
            "mode": "adapter",
            "checkpoint_path": str(checkpoint),
            "coverage_target": 1.0,
        },
        device=torch.device("cpu"),
    )
    assert controller.adapter is not None
    original = controller.adapter.forward

    def nonfinite(**kwargs):
        prediction = original(**kwargs)
        residuals = dict(prediction.residuals)
        residuals["pos"] = torch.full_like(residuals["pos"], torch.nan)
        return AdapterPrediction(
            score=prediction.score,
            residuals=residuals,
            context_features=prediction.context_features,
            risk_features=prediction.risk_features,
        )

    monkeypatch.setattr(controller.adapter, "forward", nonfinite)
    actual = controller.score_after_corrector(
        exact_score_fn=lambda _x, _t: score_after,
        x_before=x_before,
        score_before=score_before,
        x_after=x_after,
        t=torch.tensor([0.5, 0.5]),
        sampling_step=1,
        progress=0.5,
    )
    assert torch.equal(actual["pos"], score_after["pos"])
    assert controller.metrics["fallback_calls"] == 1


def test_teacher_shards_round_trip(tmp_path: Path):
    x_before, x_after, score_before, score_after = make_adapter_batches()
    writer = TeacherShardWriter(
        output_dir=tmp_path / "teacher", seed=61000, split="train", shard_size=1
    )
    writer.add(
        sampling_step=7,
        t=torch.tensor([0.75, 0.75]),
        progress=0.25,
        x_before=x_before,
        score_before=score_before,
        x_after=x_after,
        score_after=score_after,
    )
    writer.close()
    records = list(iter_teacher_records([tmp_path / "teacher" / "manifest.json"]))
    assert len(records) == 1
    record = records[0]
    assert record["seed"] == 61000
    assert record["sampling_step"] == 7
    assert record["x_after"].get_batch_size() == 2
    assert torch.allclose(record["score_residual"]["pos"], torch.full((5, 3), 0.1), atol=1e-3)


def test_seed_plan_has_no_train_validation_test_or_historical_overlap():
    config_path = (
        Path(__file__).parents[3]
        / "research"
        / "corrector_distillation"
        / "experiment_config.json"
    )
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    split_sets = {
        name: set(range(bounds[0], bounds[1] + 1))
        for name, bounds in config["seed_splits"].items()
    }
    assert split_sets["stage_a_train"] <= split_sets["stage_b_train"]
    independent = ("stage_b_train", "validation", "stage_c_test")
    for index, left in enumerate(independent):
        for right in independent[index + 1 :]:
            assert not split_sets[left] & split_sets[right]
    historical = set()
    for start, end in config["excluded_historical_seed_ranges"]:
        historical |= set(range(start, end + 1))
    assert not set.union(*(split_sets[name] for name in independent)) & historical


def test_disabled_controller_is_not_constructed():
    assert build_residual_controller(None, device=torch.device("cpu")) is None
    assert (
        build_residual_controller({"enabled": False}, device=torch.device("cpu")) is None
    )
