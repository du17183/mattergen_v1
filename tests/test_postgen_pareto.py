import numpy as np
import pandas as pd

from research.postgen_fastgate.pareto_eval import select_pools


def candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prob_novel": [0.8, 0.9, 0.1, 0.85],
            "std_prob_novel": [0.01, 0.01, 0.01, 0.01],
            "prob_unique": [0.8, 0.9, 0.9, 0.85],
            "std_prob_unique": [0.01, 0.01, 0.01, 0.01],
            "prob_comp_validity": [0.8, 0.9, 0.9, 0.85],
            "std_prob_comp_validity": [0.01, 0.01, 0.01, 0.01],
            "pred_energy_above_hull_per_atom": [0.1, 0.08, 0.02, 0.09],
            "std_energy_above_hull_per_atom": [0.01] * 4,
            "chgnet_mag_density": [0.2, 0.1, 0.1, 0.15],
            "prob_novel_unique_stable": [0.4, 0.8, 0.9, 0.5],
            "std_prob_novel_unique_stable": [0.01] * 4,
            "prob_stable": [0.4, 0.8, 0.9, 0.5],
            "std_prob_stable": [0.01] * 4,
        }
    )


def test_pareto_selector_excludes_novelty_conflict() -> None:
    selected, eligible = select_pools(candidate_frame(), np.array([[0, 1, 2, 3]]))
    assert not eligible[0, 2]
    assert selected.tolist() == [1]


def test_pareto_selector_keeps_baseline_eligible() -> None:
    _selected, eligible = select_pools(
        candidate_frame(), np.array([[0, 1, 2, 3]])
    )
    assert eligible[0, 0]
