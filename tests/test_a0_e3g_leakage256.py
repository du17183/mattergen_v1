from __future__ import annotations

import numpy as np

from research import a0_e3g_leakage256 as leakage


def test_cohort_contract_is_disjoint_and_complete() -> None:
    assert len(leakage.TRAIN_SEEDS) == 64
    assert len(leakage.HELDOUT_SEEDS) == 192
    assert set(leakage.TRAIN_SEEDS).isdisjoint(leakage.HELDOUT_SEEDS)
    assert tuple(sorted(leakage.TRAIN_SEEDS + leakage.HELDOUT_SEEDS)) == (
        leakage.SEEDS
    )


def test_negative_bootstrap_gap_means_stronger_training_effect() -> None:
    train = np.full(64, -0.2)
    heldout = np.full(192, -0.05)
    low, high = leakage.bootstrap_gap(train, heldout)
    assert low <= high < 0


def test_pipeline_uses_importable_module_name() -> None:
    assert leakage.MODULE_NAME == "research.a0_e3g_leakage256"


def test_safety_leakage_overrides_unclear_mean_effect() -> None:
    state = leakage.classify_leakage(
        (-0.02, 0.08),
        -0.16,
        (-0.21, -0.10),
        1.0e-4,
    )
    assert state == "LEAKAGE_INFLATION_DETECTED"


def test_only_e3g_is_scheduled_for_new_relaxation() -> None:
    assert leakage.METHODS == ("A0_E3G",)
