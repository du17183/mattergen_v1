from __future__ import annotations

import numpy as np
import pandas as pd

from research.postgen_fastgate.evaluate_quality import (
    q1_score,
    q2_score,
    q4_score,
    q6_score,
)


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pred_energy_above_hull_per_atom": [0.20, 0.05],
            "std_energy_above_hull_per_atom": [0.02, 0.01],
            "pred_rmsd_from_relaxation": [0.10, 0.02],
            "std_rmsd_from_relaxation": [0.01, 0.005],
            "prob_stable": [0.1, 0.9],
            "prob_novel_unique_stable": [0.1, 0.8],
            "prob_comp_validity": [0.2, 0.9],
            "prob_novel": [0.3, 0.9],
            "prob_unique": [0.5, 0.9],
            "std_prob_novel_unique_stable": [0.2, 0.05],
        }
    )


def test_all_frozen_scores_prefer_the_safe_quality_candidate() -> None:
    pool = _pool()
    for function in (q1_score, q2_score, q4_score, q6_score):
        score = function(pool)
        assert score.shape == (2,)
        assert np.isfinite(score).all()
        assert int(np.argmin(score)) == 1
