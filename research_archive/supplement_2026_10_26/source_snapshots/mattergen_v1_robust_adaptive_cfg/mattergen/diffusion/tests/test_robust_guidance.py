from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mattergen.diffusion.sampling.robust_guidance import RobustAdaptiveGuidanceController


def calibration(tmp_path: Path) -> tuple[Path, Path]:
    metadata = {
        "transform": {"time_bins": 2, "log_eps": 1e-12, "z_clip": 3.0},
        "online_statistics": {"ema_beta": 0.95},
        "confidence": {"signal_deadzone": 0.5, "signal_scale": 1.0, "agreement_k": 1.0},
    }
    json_path = tmp_path / "calibration.json"
    json_path.write_text(json.dumps(metadata))
    csv_path = tmp_path / "calibration.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["phase", "field", "time_bin", "mu_log_residual", "sigma_log_residual"],
        )
        writer.writeheader()
        for phase in ("corrector", "predictor"):
            for field in ("atomic", "position", "cell"):
                for time_bin in range(2):
                    writer.writerow(
                        {
                            "phase": phase,
                            "field": field,
                            "time_bin": time_bin,
                            "mu_log_residual": 0.0,
                            "sigma_log_residual": 1.0,
                        }
                    )
    return json_path, csv_path


def test_low_confidence_falls_back_to_two(tmp_path: Path) -> None:
    metadata, table = calibration(tmp_path)
    controller = RobustAdaptiveGuidanceController(
        calibration_json=metadata, calibration_csv=table, robust=True
    )
    decision = controller.evaluate(
        progress=0.2,
        phase="predictor",
        field_deltas={"atomic_numbers": 1.0, "pos": 1.0, "cell": 1.0},
    )
    assert decision.fallback_to_base
    assert decision.final_guidance == 2.0
    assert decision.confidence == 0.0


def test_median_rejects_single_field_outlier(tmp_path: Path) -> None:
    metadata, table = calibration(tmp_path)
    controller = RobustAdaptiveGuidanceController(
        calibration_json=metadata, calibration_csv=table, robust=False
    )
    decision = controller.evaluate(
        progress=0.2,
        phase="predictor",
        field_deltas={"atomic_numbers": 1.0, "pos": 1.0, "cell": 1e12},
    )
    assert decision.z_median == pytest.approx(0.0, abs=1e-10)
    assert decision.final_guidance == pytest.approx(2.0, abs=1e-10)
    assert decision.z_mean == pytest.approx(1.0)


def test_bounds_and_slew_limit(tmp_path: Path) -> None:
    metadata, table = calibration(tmp_path)
    controller = RobustAdaptiveGuidanceController(
        calibration_json=metadata, calibration_csv=table, robust=True
    )
    decision = controller.evaluate(
        progress=0.2,
        phase="predictor",
        field_deltas={"atomic_numbers": 100.0, "pos": 100.0, "cell": 100.0},
    )
    assert 1.5 <= decision.final_guidance <= 2.5
    assert decision.final_guidance == pytest.approx(2.05)
    assert decision.slew_limited


def test_phase_ema_is_separate_and_resettable(tmp_path: Path) -> None:
    metadata, table = calibration(tmp_path)
    controller = RobustAdaptiveGuidanceController(
        calibration_json=metadata, calibration_csv=table, robust=True
    )
    values = {"atomic_numbers": 10.0, "pos": 10.0, "cell": 10.0}
    controller.evaluate(progress=0.2, phase="predictor", field_deltas=values)
    assert controller.ema_by_phase["predictor"] is not None
    assert controller.ema_by_phase["corrector"] is None
    controller.reset()
    assert controller.ema_by_phase == {"corrector": None, "predictor": None}
