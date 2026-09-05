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
    RISK_FEATURE_NAMES,
)
from mattergen.diffusion.sampling.residual_distillation import (
    ResidualDistillationController,
    TeacherShardWriter,
    build_residual_controller,
    iter_teacher_records,
    save_adapter_checkpoint,
)
from mattergen.diffusion.tests.test_reverse_sampling import get_diffusion_module
from research.corrector_distillation.train_adapter import timestep_weights


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


def test_forced_exact_full_sampler_is_identical_to_c0(tmp_path: Path):
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
    checkpoint = save_adapter_checkpoint(
        tmp_path / "forced_exact_adapter.pt", model=FieldwiseResidualAdapter()
    )
    torch.manual_seed(9123)
    baseline = PredictorCorrector(**kwargs).sample(conditioning.clone())
    torch.manual_seed(9123)
    forced = PredictorCorrector(
        **kwargs,
        corrector_residual_adapter={
            "enabled": True,
            "mode": "adapter",
            "checkpoint_path": str(checkpoint),
            "coverage_target": 1.0,
            "force_fallback": True,
        },
    ).sample(conditioning.clone())
    for baseline_batch, forced_batch in zip(baseline, forced):
        for field in fields:
            assert torch.equal(baseline_batch[field], forced_batch[field])


def test_dagger_target_is_exact_teacher_and_state_is_aligned(tmp_path: Path):
    x_before, x_after, score_before, score_after = make_adapter_batches()
    checkpoint = save_adapter_checkpoint(
        tmp_path / "dagger_adapter.pt", model=FieldwiseResidualAdapter()
    )
    dagger_dir = tmp_path / "dagger"
    controller = ResidualDistillationController(
        {
            "enabled": True,
            "mode": "adapter",
            "checkpoint_path": str(checkpoint),
            "coverage_target": 1.0,
            "sample_seed": 64000,
            "dagger_output_dir": str(dagger_dir),
            "dagger_split": "train",
        },
        device=torch.device("cpu"),
    )
    exact_calls = 0

    def exact(_x, _t):
        nonlocal exact_calls
        exact_calls += 1
        assert _x is x_after
        return score_after

    rollout_score = controller.score_after_corrector(
        exact_score_fn=exact,
        x_before=x_before,
        score_before=score_before,
        x_after=x_after,
        t=torch.tensor([0.5, 0.5]),
        sampling_step=17,
        progress=0.25,
    )
    controller.close()
    assert exact_calls == 1
    assert controller.metrics["dagger_teacher_calls"] == 1
    assert controller.metrics["fallback_calls"] == 0
    assert controller.metrics["adapter_accept_calls"] == 1
    assert torch.equal(rollout_score["pos"], score_before["pos"])
    record = next(iter_teacher_records([dagger_dir / "manifest.json"]))
    assert record["seed"] == 64000
    assert record["sampling_step"] == 17
    # DAgger shards intentionally use bfloat16 storage, so alignment is checked
    # to the storage precision rather than with bitwise float32 equality.
    assert torch.allclose(record["x_after"]["pos"], x_after["pos"], atol=2e-3)
    assert torch.allclose(record["score_after"]["pos"], score_after["pos"], atol=5e-3)
    assert torch.allclose(
        record["adapter_score_after"]["pos"], score_before["pos"], atol=5e-3
    )
    assert torch.allclose(
        record["prediction_error"]["pos"],
        score_before["pos"] - score_after["pos"],
        atol=5e-3,
    )
    assert bool(record["used_adapter"].all().item())


def test_late_exact_schedule_forces_full_teacher(tmp_path: Path):
    x_before, x_after, score_before, score_after = make_adapter_batches()
    checkpoint = save_adapter_checkpoint(
        tmp_path / "late_adapter.pt", model=FieldwiseResidualAdapter()
    )
    controller = ResidualDistillationController(
        {
            "enabled": True,
            "mode": "adapter",
            "checkpoint_path": str(checkpoint),
            "coverage_target": 1.0,
            "late_exact_start": 0.7,
        },
        device=torch.device("cpu"),
    )
    actual = controller.score_after_corrector(
        exact_score_fn=lambda _x, _t: score_after,
        x_before=x_before,
        score_before=score_before,
        x_after=x_after,
        t=torch.tensor([0.1, 0.1]),
        sampling_step=800,
        progress=0.8,
    )
    assert torch.equal(actual["pos"], score_after["pos"])
    assert controller.metrics["late_exact_calls"] == 1
    assert controller.metrics["fallback_calls"] == 1



def test_periodic_exact_anchor_every_k_adapter_steps_and_resets(tmp_path: Path):
    x_before, x_after, score_before, score_after = make_adapter_batches()
    checkpoint = save_adapter_checkpoint(
        tmp_path / "periodic_adapter.pt", model=FieldwiseResidualAdapter()
    )
    controller = ResidualDistillationController(
        {
            "enabled": True,
            "mode": "adapter",
            "checkpoint_path": str(checkpoint),
            "coverage_target": 1.0,
            "periodic_exact_anchor_k": 4,
        },
        device=torch.device("cpu"),
    )
    exact_calls = 0

    def exact(_x, _t):
        nonlocal exact_calls
        exact_calls += 1
        return score_after

    outputs = []
    for step in range(10):
        outputs.append(
            controller.score_after_corrector(
                exact_score_fn=exact,
                x_before=x_before,
                score_before=score_before,
                x_after=x_after,
                t=torch.tensor([0.5, 0.5]),
                sampling_step=step,
                progress=step / 20,
            )
        )
    assert exact_calls == 2
    assert torch.equal(outputs[4]["pos"], score_after["pos"])
    assert torch.equal(outputs[9]["pos"], score_after["pos"])
    assert controller.metrics["periodic_exact_calls"] == 2
    assert controller.metrics["saved_score_calls"] == 8
    assert controller.metrics["adapter_streak_max"] == 4
    assert controller.metrics["eligible_adapter_count_since_last_exact"] == 0


def test_late_exact_takes_priority_over_periodic_anchor(tmp_path: Path):
    x_before, x_after, score_before, score_after = make_adapter_batches()
    checkpoint = save_adapter_checkpoint(
        tmp_path / "late_periodic_adapter.pt", model=FieldwiseResidualAdapter()
    )
    controller = ResidualDistillationController(
        {
            "enabled": True,
            "mode": "adapter",
            "checkpoint_path": str(checkpoint),
            "coverage_target": 1.0,
            "late_exact_start": 0.7,
            "periodic_exact_anchor_k": 4,
        },
        device=torch.device("cpu"),
    )
    for step in range(4):
        controller.score_after_corrector(
            exact_score_fn=lambda _x, _t: score_after,
            x_before=x_before,
            score_before=score_before,
            x_after=x_after,
            t=torch.tensor([0.5, 0.5]),
            sampling_step=step,
            progress=0.1 * step,
        )
    actual = controller.score_after_corrector(
        exact_score_fn=lambda _x, _t: score_after,
        x_before=x_before,
        score_before=score_before,
        x_after=x_after,
        t=torch.tensor([0.1, 0.1]),
        sampling_step=800,
        progress=0.8,
    )
    assert torch.equal(actual["pos"], score_after["pos"])
    assert controller.metrics["late_exact_calls"] == 1
    assert controller.metrics["periodic_exact_calls"] == 0
    assert controller.metrics["eligible_adapter_count_since_last_exact"] == 0


def test_atomic_risk_takes_priority_over_periodic_anchor(tmp_path: Path):
    x_before, x_after, score_before, score_after = make_adapter_batches()
    checkpoint = save_adapter_checkpoint(
        tmp_path / "risk_periodic_adapter.pt",
        model=FieldwiseResidualAdapter(),
        calibration=field_calibration(),
    )
    controller = ResidualDistillationController(
        {
            "enabled": True,
            "mode": "adapter",
            "checkpoint_path": str(checkpoint),
            "coverage_target": 0.5,
            "risk_mode": "field",
            "risk_fields": ["atomic_numbers"],
            "periodic_exact_anchor_k": 4,
        },
        device=torch.device("cpu"),
    )
    for step in range(4):
        controller.score_after_corrector(
            exact_score_fn=lambda _x, _t: score_after,
            x_before=x_before,
            score_before=score_before,
            x_after=x_after,
            t=torch.tensor([0.5, 0.5]),
            sampling_step=step,
            progress=0.1 * step,
        )
    controller.calibration["field_coverage_thresholds"]["0.5"][
        "atomic_numbers"
    ] = -1.0
    actual = controller.score_after_corrector(
        exact_score_fn=lambda _x, _t: score_after,
        x_before=x_before,
        score_before=score_before,
        x_after=x_after,
        t=torch.tensor([0.5, 0.5]),
        sampling_step=4,
        progress=0.4,
    )
    assert torch.equal(actual["pos"], score_after["pos"])
    assert controller.metrics["field_risk_fallback_calls"] == 1
    assert controller.metrics["periodic_exact_calls"] == 0
    assert controller.metrics["eligible_adapter_count_since_last_exact"] == 0


def test_disabled_periodic_anchor_is_equivalent_to_v2(tmp_path: Path):
    x_before, x_after, score_before, score_after = make_adapter_batches()
    checkpoint = save_adapter_checkpoint(
        tmp_path / "disabled_periodic_adapter.pt", model=FieldwiseResidualAdapter()
    )
    base_config = {
        "enabled": True,
        "mode": "adapter",
        "checkpoint_path": str(checkpoint),
        "coverage_target": 1.0,
        "late_exact_start": 0.7,
    }
    v2 = ResidualDistillationController(base_config, device=torch.device("cpu"))
    disabled = ResidualDistillationController(
        {**base_config, "periodic_exact_anchor_k": None}, device=torch.device("cpu")
    )
    for step, progress in enumerate((0.1, 0.2, 0.6, 0.7, 0.9)):
        arguments = {
            "exact_score_fn": lambda _x, _t: score_after,
            "x_before": x_before,
            "score_before": score_before,
            "x_after": x_after,
            "t": torch.tensor([0.5, 0.5]),
            "sampling_step": step,
            "progress": progress,
        }
        expected = v2.score_after_corrector(**arguments)
        actual = disabled.score_after_corrector(**arguments)
        for field in ("pos", "cell", "atomic_numbers"):
            assert torch.equal(actual[field], expected[field])
    nondeterministic_timing_metrics = {
        "adapter_seconds",
        "exact_fallback_seconds",
        "dagger_teacher_seconds",
    }
    assert {
        key: value
        for key, value in disabled.metrics.items()
        if key not in nondeterministic_timing_metrics
    } == {
        key: value
        for key, value in v2.metrics.items()
        if key not in nondeterministic_timing_metrics
    }


def field_calibration(*, nan_weights: bool = False, include_thresholds: bool = True):
    dimension = len(RISK_FEATURE_NAMES)
    weights = [0.0] * dimension
    if nan_weights:
        weights[0] = float("nan")
    model = {
        "feature_mean": [0.0] * dimension,
        "feature_std": [1.0] * dimension,
        "linear_weights": weights,
        "linear_bias": 0.0,
    }
    calibration = {
        "field_risk_models": {
            field: dict(model) for field in ("pos", "cell", "atomic_numbers")
        }
    }
    if include_thresholds:
        calibration["field_coverage_thresholds"] = {
            "0.5": {
                field: 1.0 for field in ("pos", "cell", "atomic_numbers")
            }
        }
    return calibration


@pytest.mark.parametrize(
    "calibration",
    (
        field_calibration(nan_weights=True),
        field_calibration(include_thresholds=False),
    ),
    ids=("field-risk-nan", "field-threshold-missing"),
)
def test_invalid_field_risk_or_threshold_falls_back(
    tmp_path: Path, calibration: dict
):
    x_before, x_after, score_before, score_after = make_adapter_batches()
    checkpoint = save_adapter_checkpoint(
        tmp_path / f"field_adapter_{len(list(tmp_path.iterdir()))}.pt",
        model=FieldwiseResidualAdapter(),
        calibration=calibration,
    )
    controller = ResidualDistillationController(
        {
            "enabled": True,
            "mode": "adapter",
            "checkpoint_path": str(checkpoint),
            "coverage_target": 0.5,
            "risk_mode": "field",
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
    assert torch.equal(actual["pos"], score_after["pos"])
    assert controller.metrics["fallback_calls"] == 1


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


def test_v2_seed_plan_freezes_v1_and_holds_final_test_until_freeze():
    config_path = (
        Path(__file__).parents[3]
        / "experiments"
        / "corrector_residual_distillation_v2"
        / "configs"
        / "experiment.json"
    )
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    split_sets = {
        name: set(range(bounds[0], bounds[1] + 1))
        for name, bounds in config["seed_splits"].items()
    }
    names = list(split_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert not split_sets[left] & split_sets[right]
    v1_test = set(range(63000, 63032))
    assert not (split_sets["dagger_train"] | split_sets["validation"]) & v1_test
    assert config["final_test_used_before_freeze"] is False
    assert config["final_test_pre_freeze_audit"]["status"] == "not used"


def test_v2_timestep_weights_match_predeclared_schedule():
    progress = torch.tensor([0.0, 0.1999, 0.2, 0.6999, 0.7, 1.0])
    actual = timestep_weights(
        progress,
        early_weight=0.5,
        middle_weight=1.0,
        late_weight=2.0,
        early_end=0.2,
        late_start=0.7,
    )
    assert torch.equal(actual, torch.tensor([0.5, 0.5, 1.0, 1.0, 2.0, 2.0]))


def test_disabled_controller_is_not_constructed():
    assert build_residual_controller(None, device=torch.device("cpu")) is None
    assert (
        build_residual_controller({"enabled": False}, device=torch.device("cpu")) is None
    )
