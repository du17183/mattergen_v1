from research.postgen_fastgate.new_eval import gate_decision, pool_for_seed


def test_frozen_pool_mapping() -> None:
    assert pool_for_seed(33000) == (0, 0)
    assert pool_for_seed(33003) == (0, 3)
    assert pool_for_seed(33124) == (31, 0)
    assert pool_for_seed(33127) == (31, 3)


def test_gate_requires_safety_and_positive_result() -> None:
    changes = {
        "structure_validity": 0.0,
        "composition_validity": 0.0,
        "stable": 0.03125,
        "nus": 0.0,
        "novel": 0.0,
        "unique": 0.0,
        "ehull": -0.005,
        "rmsd_relative": 0.0,
        "pre_relax_max_force_relative": 0.0,
    }
    assert gate_decision(changes, {"baseline": 0, "selected": 0}) == {
        "safety_gate": True,
        "positive_gate": True,
        "Q6_NS_SETRANK_FINAL_GO": True,
    }
    changes["novel"] = -0.03125
    assert not gate_decision(changes, {"baseline": 0, "selected": 0})[
        "Q6_NS_SETRANK_FINAL_GO"
    ]
