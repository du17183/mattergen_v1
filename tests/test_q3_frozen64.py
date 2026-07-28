from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import q3_frozen64 as frozen


def test_frozen_seed_contract() -> None:
    assert frozen.SEEDS == tuple(range(32000, 32064))
    assert set(frozen.SEEDS).isdisjoint(range(20000, 20064))
    assert set(frozen.SEEDS).isdisjoint(range(33000, 33128))


def test_frozen_config_matches_runtime_constants() -> None:
    config = json.loads(
        Path("configs/q3_e3_pcr_frozen64.json").read_text(encoding="utf-8")
    )
    assert config["gate"]["checkpoint_sha256"] == frozen.Q3_CHECKPOINT_SHA256
    assert config["gate"]["trainable_parameters"] == 129
    assert config["gate"]["threshold"] == 0.5
    assert config["refinement"]["steps"] == 5
    assert config["refinement"]["position_eta"] == 0.01
    assert config["refinement"]["per_step_radius_angstrom"] == 0.02
    assert config["evaluation"]["seed_start"] == frozen.SEEDS[0]
    assert config["evaluation"]["seed_end"] == frozen.SEEDS[-1]


def _force_frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"pre_relax_max_force_ev_ang": values})


def test_random_gate_ablation_is_deterministic_and_count_matched() -> None:
    baseline = _force_frame(np.linspace(0.1, 0.8, 64))
    always = _force_frame(np.linspace(0.08, 0.7, 64))
    first, first_summary = frozen.random_gate_ablation(baseline, always, 37)
    second, second_summary = frozen.random_gate_ablation(baseline, always, 37)
    pd.testing.assert_frame_equal(first, second)
    assert first_summary == second_summary
    assert (first["gate_on_count"] == 37).all()
    assert len(first) == 5


def test_bootstrap_ci_is_reproducible() -> None:
    values = np.linspace(-0.2, 0.1, 64)
    assert frozen.bootstrap_ci(values) == frozen.bootstrap_ci(values)


def test_structure_and_cell_hash_are_sensitive_and_stable() -> None:
    from ase import Atoms

    atoms = Atoms(
        numbers=[14, 14],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    assert frozen.structure_hash(atoms) == frozen.structure_hash(atoms.copy())
    assert frozen.cell_hash(atoms) == frozen.cell_hash(atoms.copy())
    changed = atoms.copy()
    changed.positions[1, 0] += 0.01
    assert frozen.structure_hash(changed) != frozen.structure_hash(atoms)
    assert frozen.cell_hash(changed) == frozen.cell_hash(atoms)
