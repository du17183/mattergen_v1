from __future__ import annotations

import numpy as np
import pandas as pd

from research.postgen_fastgate.oracle import one_trial, summarize_trials


def _frame() -> pd.DataFrame:
    rows = []
    for seed, ehull, rmsd, stable, nus in (
        (1, 0.20, 0.20, False, False),
        (2, 0.05, 0.15, True, True),
        (3, 0.15, 0.01, False, False),
        (4, 0.10, 0.10, True, False),
    ):
        rows.append(
            {
                "seed": seed,
                "energy_above_hull_per_atom": ehull,
                "rmsd_from_relaxation": rmsd,
                "stable": stable,
                "novel_unique_stable": nus,
                "comp_validity": True,
                "structure_validity": True,
                "novel": True,
                "novel_unique": True,
                "unique": True,
                "converged": True,
            }
        )
    return pd.DataFrame(rows)


def test_oracles_select_expected_extrema() -> None:
    result = one_trial(
        _frame(),
        pool_size=4,
        rng=np.random.default_rng(7),
    )
    assert result["oracle_min_ehull"]["energy_above_hull_per_atom"] == 0.05
    assert result["oracle_min_rmsd"]["rmsd_from_relaxation"] == 0.01
    assert result["oracle_stable_quality"]["stable"] == 1.0
    assert result["oracle_nus_quality"]["novel_unique_stable"] == 1.0


def test_summary_is_deterministic() -> None:
    left, left_rows = summarize_trials(
        _frame(), pool_size=4, trials=5, random_seed=11
    )
    right, right_rows = summarize_trials(
        _frame(), pool_size=4, trials=5, random_seed=11
    )
    assert left == right
    pd.testing.assert_frame_equal(left_rows, right_rows)


def test_oracle_gates_open_when_pool_contains_better_candidates() -> None:
    frame = pd.concat([_frame()] * 16, ignore_index=True)
    frame["seed"] = np.arange(len(frame))
    summary, _ = summarize_trials(
        frame, pool_size=4, trials=50, random_seed=23
    )
    assert summary["gates"]["Q1_UQ_PQR_ORACLE_GO"]
    assert summary["gates"]["Q2_RFR_ORACLE_GO"]
    assert summary["gates"]["Q6_NS_SETRANK_ORACLE_GO"]
