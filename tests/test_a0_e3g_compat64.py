from __future__ import annotations

import numpy as np
from ase import Atoms

from research import a0_e3g_compat64 as compat


def test_seed_contract() -> None:
    assert compat.SEEDS == tuple(range(41000, 41064))
    assert compat.METHODS == ("A0", "A0_E3G")


def test_frozen_a0_config_accepts_exact_contract() -> None:
    config = {
        "method": "adaptive",
        "target": {"dft_mag_density": 0.1},
        "base_guidance": 2.0,
        "batch_size": 1,
        "sampling_steps": 1000,
        "strict_deterministic": True,
        "guidance_parameters": {
            "min_scale": 0.0,
            "max_scale": 5.0,
            "adaptive_alpha": 0.5,
            "adaptive_ema": 0.95,
            "adaptive_eps": 1.0e-6,
        },
    }
    assert compat.validate_a0_config(config)
    config["guidance_parameters"]["adaptive_alpha"] = 0.5001
    assert not compat.validate_a0_config(config)


def test_e3g_disabled_is_exact_a0_copy() -> None:
    original = Atoms(
        numbers=[14, 14],
        positions=[[0.0, 0.0, 0.0], [1.2, 1.2, 1.2]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )
    candidate = original.copy()
    candidate.positions[1, 0] += 0.01
    output = compat.choose_gated_output(original, candidate, False)
    assert compat.core.structure_hash(output) == compat.core.structure_hash(original)
    assert output is not original


def test_e3g_enabled_selects_candidate() -> None:
    original = Atoms(
        numbers=[14],
        positions=[[0.0, 0.0, 0.0]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )
    candidate = original.copy()
    candidate.positions[0, 0] = 0.01
    assert compat.choose_gated_output(original, candidate, True) is candidate
