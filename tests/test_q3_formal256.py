from __future__ import annotations

from pathlib import Path

import numpy as np

from research import q3_formal256 as formal


def test_formal_seed_contract() -> None:
    assert formal.SEEDS == tuple(range(40000, 40256))
    assert len(formal.SEEDS) == 256
    assert formal.BASE_COMMIT == "87ec85c4ab353a362c8e2645cb0d14c3f6828672"


def test_holm_adjust_family_of_two() -> None:
    adjusted = formal.holm_adjust({"E3-A": 0.01, "E3-G": 0.04})
    assert adjusted == {"E3-A": 0.02, "E3-G": 0.04}


def test_holm_adjust_preserves_step_down_monotonicity() -> None:
    adjusted = formal.holm_adjust({"E3-A": 0.03, "E3-G": 0.04})
    assert adjusted == {"E3-A": 0.06, "E3-G": 0.06}


def test_bootstrap_ci_is_finite_and_directional() -> None:
    low, high = formal.bootstrap_ci(-np.ones(256))
    assert low == -1.0
    assert high == -1.0


def test_final_state_learned_gate() -> None:
    assert formal.select_final_state(True, True, True) == (
        "E3_G_FORMAL_CONFIRMED",
        "LEARNED_GATED_E3_PCR",
    )


def test_final_state_gate_unsupported() -> None:
    assert formal.select_final_state(True, True, False) == (
        "E3_REFINER_FORMAL_CONFIRMED_GATE_UNSUPPORTED",
        "SAFE_BOUNDED_E3_PCR",
    )


def test_final_state_always_only() -> None:
    assert formal.select_final_state(True, False, False) == (
        "E3_A_FORMAL_CONFIRMED",
        "ALWAYS_ON_SAFE_BOUNDED_E3_PCR",
    )


def test_final_state_no_go() -> None:
    assert formal.select_final_state(False, False, False) == (
        "E3_PCR_FORMAL_NO_GO",
        "NONE",
    )


def test_formal_subprocesses_use_module_entrypoint() -> None:
    source = Path("research/q3_formal256.py").read_text(encoding="utf-8")
    assert source.count("research.q3_formal256") >= 5
    assert "core.relax()" not in source


def test_formal_method_contract() -> None:
    assert formal.METHODS == ("C0", "ALWAYS_ON", "Q3_E3_PCR")
    assert formal.DISPLAY_NAMES["ALWAYS_ON"] == "E3-A"
    assert formal.DISPLAY_NAMES["Q3_E3_PCR"] == "E3-G"
