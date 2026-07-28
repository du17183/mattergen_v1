from __future__ import annotations

import numpy as np
from ase import Atoms

from research.postgen_fastgate.features import split_for_seed, structure_features


def test_historical_split_boundaries() -> None:
    assert split_for_seed(20000) == "train"
    assert split_for_seed(20191) == "train"
    assert split_for_seed(20192) == "validation"
    assert split_for_seed(20223) == "validation"
    assert split_for_seed(20224) == "test"
    assert split_for_seed(20255) == "test"


def test_structure_features_are_finite_and_composition_normalized() -> None:
    atoms = Atoms(
        numbers=[8, 8, 14],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
        ],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    prediction = {
        "e": -5.0,
        "f": np.ones((3, 3)) * 0.1,
        "s": np.eye(3) * 0.2,
        "m": np.asarray([0.5, -0.5, 0.1]),
    }
    features = structure_features(atoms, prediction)
    assert all(np.isfinite(value) for value in features.values())
    fractions = [
        value
        for key, value in features.items()
        if key.startswith("element_fraction_")
    ]
    assert np.isclose(sum(fractions), 1.0)
    assert features["chgnet_mag_density"] > 0
