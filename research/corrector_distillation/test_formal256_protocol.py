from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mattergen.diffusion.sampling.residual_adapter import FieldwiseResidualAdapter
from mattergen.diffusion.sampling.residual_distillation import (
    ResidualDistillationController,
    save_adapter_checkpoint,
)
from mattergen.diffusion.tests.test_residual_distillation import make_adapter_batches
from research.corrector_distillation.formal256_protocol import (
    ASSET_PATHS,
    EXPECTED_HASHES,
    FORMAL_SEEDS,
    HISTORICAL_SEED_RANGES,
    METHODS,
    benchmark_command,
    fixed_shards,
    load_frozen_config,
    seed_audit_payload,
    sha256,
    validate_frozen_protocol,
)


def test_frozen_config_and_asset_checksums() -> None:
    result = validate_frozen_protocol(verify_git=False)
    assert result["passed"], result["failures"]
    for name, path in ASSET_PATHS.items():
        assert sha256(path) == EXPECTED_HASHES[name]
    assert load_frozen_config()["adapter"]["risk_model_canonical_sha256"] == EXPECTED_HASHES[
        "risk_model_canonical"
    ]


def test_formal_seed_disjointness() -> None:
    formal = set(FORMAL_SEEDS)
    for ranges in HISTORICAL_SEED_RANGES.values():
        historical = {seed for start, end in ranges for seed in range(start, end + 1)}
        assert formal.isdisjoint(historical)
    assert seed_audit_payload()["passed"] is True


def test_formal_seed_interval_and_fixed_shards_are_immutable() -> None:
    assert FORMAL_SEEDS == tuple(range(67000, 67256))
    shards = fixed_shards()
    assert {gpu: (values[0], values[-1], len(values)) for gpu, values in shards.items()} == {
        0: (67000, 67031, 32),
        1: (67032, 67063, 32),
        2: (67064, 67095, 32),
        3: (67096, 67127, 32),
        4: (67128, 67159, 32),
        5: (67160, 67191, 32),
        6: (67192, 67223, 32),
        7: (67224, 67255, 32),
    }


def test_formal_c0_command_uses_untouched_default_sampler(tmp_path: Path) -> None:
    command = benchmark_command(
        checkpoint_root=tmp_path, output_root=tmp_path, label="C0", seed=67000
    )
    assert command[-4:] == ["--method", "C0", "--label", "C0"]
    prohibited = {
        "--adapter-checkpoint",
        "--coverage",
        "--risk-mode",
        "--risk-fields",
        "--late-exact-start",
        "--early-reuse-end",
    }
    assert prohibited.isdisjoint(command)
    assert METHODS == ("C0", "V2_Frozen_Atomic75_Late30")


def test_formal_v2_command_is_derived_from_exact_frozen_config(tmp_path: Path) -> None:
    config = load_frozen_config()
    command = benchmark_command(
        checkpoint_root=tmp_path,
        output_root=tmp_path,
        label="V2_Frozen_Atomic75_Late30",
        seed=67255,
    )
    joined = " ".join(command)
    assert str(config["adapter"]["checkpoint"]) in command
    assert "--coverage 0.75" in joined
    assert "--risk-mode field" in joined
    assert "--risk-fields atomic_numbers" in joined
    assert "--late-exact-start 0.7" in joined
    with pytest.raises(ValueError):
        benchmark_command(
            checkpoint_root=tmp_path, output_root=tmp_path, label="V1", seed=67000
        )
    with pytest.raises(ValueError):
        benchmark_command(
            checkpoint_root=tmp_path, output_root=tmp_path, label="C0", seed=67256
        )


def test_v2_forced_exact_returns_original_teacher_score(tmp_path: Path) -> None:
    x_before, x_after, score_before, score_after = make_adapter_batches()
    checkpoint = save_adapter_checkpoint(
        tmp_path / "adapter.pt", model=FieldwiseResidualAdapter()
    )
    controller = ResidualDistillationController(
        {
            "enabled": True,
            "mode": "adapter",
            "checkpoint_path": str(checkpoint),
            "coverage_target": 0.75,
            "risk_mode": "field",
            "risk_fields": ["atomic_numbers"],
            "late_exact_start": 0.7,
            "force_fallback": True,
        },
        device=torch.device("cpu"),
    )
    actual = controller.score_after_corrector(
        exact_score_fn=lambda _x, _t: score_after,
        x_before=x_before,
        score_before=score_before,
        x_after=x_after,
        t=torch.tensor([0.2, 0.2]),
        sampling_step=800,
        progress=0.8,
    )
    for field in ("pos", "cell", "atomic_numbers"):
        assert torch.equal(actual[field], score_after[field])
    assert controller.metrics["saved_score_calls"] == 0
    assert controller.metrics["fallback_calls"] == 1
