"""Deterministic geometry/safety tests for the frozen F0 correction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase import Atoms

from experiments.mattersim_late_force_guidance_p0.late_force_sampler import (
    bounded_cartesian_force_correction,
    cartesian_to_fractional_correction,
    position_correction_is_safe,
)


def main() -> None:
    cell = np.asarray([[4.0, 0.2, 0.1], [0.0, 5.0, 0.3], [0.0, 0.0, 6.0]])
    cart = np.asarray([[0.006, -0.002, 0.001], [-0.004, 0.003, -0.002]])
    fractional = cartesian_to_fractional_correction(cart, cell)
    assert np.allclose(fractional @ cell, cart, atol=1e-12)

    force = np.asarray([[100.0, 0.0,0.0], [-100.0, 5.0, 0.0], [-23.0, -5.0, 0.0]])
    bounded = bounded_cartesian_force_correction(
        force, force_reference_ev_a=0.07795149218357911,
        nominal_cart_step_a=0.005, hard_cart_cap_a=0.01,
    )
    assert np.linalg.norm(bounded, axis=1).max() <= 0.01 + 1e-12
    assert np.linalg.norm(bounded, axis=1).max() <= 0.02 + 1e-12
    assert np.allclose(bounded.mean(axis=0), 0.0, atol=1e-12)

    # The uncorrected candidate is safe (0.60 A); the proposed displacement
    # would reduce it to 0.40 A and therefore must be rejected/fall back.
    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]],
                  cell=10.0 * np.eye(3), pbc=True)
    unsafe_correction = np.asarray([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    safe, min_distance = position_correction_is_safe(
        atoms, unsafe_correction, min_distance_floor_a=0.5
    )
    assert not safe and min_distance < 0.5
    payload = {
        "cartesian_fractional_roundtrip_max_abs": float(np.abs(fractional @ cell - cart).max()),
        "max_per_atom_cartesian_correction_a": float(np.linalg.norm(bounded, axis=1).max()),
        "frozen_hard_per_atom_cap_a": 0.01,
        "absolute_protocol_cap_a": 0.02,
        "unsafe_candidate_original_min_distance_a": 0.6,
        "unsafe_candidate_min_distance_a": min_distance,
        "unsafe_candidate_rejected": not safe,
        "unsafe_candidate_action": "unmodified_official_score_fallback",
    }
    output = Path(__file__).resolve().parent / "guidance_geometry_test.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
