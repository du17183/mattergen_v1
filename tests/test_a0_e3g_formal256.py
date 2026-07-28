from research import a0_e3g_formal256 as formal


def complete_reasons(seed: int) -> list[str]:
    return formal.seed_disqualification_reasons(
        seed,
        structure_complete=True,
        generation_complete=True,
        relaxation_complete=True,
        official_metrics_complete=True,
        frozen_a0_config_match=True,
    )


def test_q3_training_overlap_is_rejected() -> None:
    assert complete_reasons(20000) == ["used_in_q3_gate_training"]
    assert complete_reasons(20063) == ["used_in_q3_gate_training"]


def test_remaining_registered_a0_seeds_are_seed_eligible() -> None:
    assert complete_reasons(20064) == []
    assert complete_reasons(20255) == []


def test_registered_batch_has_only_192_independent_seeds() -> None:
    eligible = sum(not complete_reasons(seed) for seed in formal.A0_SEEDS)
    assert eligible == 192


def test_partial_batch_has_source_data_incomplete_terminal_state() -> None:
    assert formal.determine_terminal_state(192) == "SOURCE_DATA_INCOMPLETE"


def test_disjoint_later_evaluation_ranges() -> None:
    registered = set(formal.A0_SEEDS)
    assert registered.isdisjoint(formal.Q3_FROZEN64_SEEDS)
    assert registered.isdisjoint(formal.Q3_FORMAL256_SEEDS)
    assert registered.isdisjoint(formal.COMPATIBILITY64_SEEDS)


def test_frozen_hashes_are_declared() -> None:
    assert len(formal.Q3_CHECKPOINT_SHA256) == 64
    assert len(formal.Q3_CONFIG_SHA256) == 64
    assert len(formal.MATTERGEN_SHA256) == 64
    assert len(formal.MATTERSIM_SHA256) == 64
