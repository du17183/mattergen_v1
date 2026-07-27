import pytest

from research.cg_tdr.build_gate_v2_labels import percentile_scale, soft_target


def test_soft_target_uses_frozen_piecewise_quantiles():
    quantiles = (1.0, 2.0, 4.0)
    assert soft_target(0.5, quantiles) == 0.0
    assert soft_target(1.0, quantiles) == 0.0
    assert soft_target(1.5, quantiles) == pytest.approx(0.25)
    assert soft_target(2.0, quantiles) == pytest.approx(0.5)
    assert soft_target(3.0, quantiles) == pytest.approx(0.75)
    assert soft_target(4.0, quantiles) == pytest.approx(1.0)
    assert soft_target(5.0, quantiles) == 1.0


def test_soft_target_is_monotonic():
    quantiles = (0.2, 0.5, 0.9)
    targets = [soft_target(value / 100.0, quantiles) for value in range(121)]
    assert all(left <= right for left, right in zip(targets, targets[1:]))


def test_percentile_scale_ignores_negative_improvements():
    scale = percentile_scale([-10.0, -1.0, 0.0, 1.0, 2.0])
    assert scale > 0.0
    assert scale <= 2.0


def test_percentile_scale_is_strictly_positive_for_identity_data():
    assert percentile_scale([0.0, 0.0, -1.0]) == pytest.approx(1.0e-12)
