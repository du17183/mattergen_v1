from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from research.spg_static_mvp.equivalence_worker import compare_state
from research.spg_static_mvp.generation import (
    build_c0_generator,
    configure_determinism,
    find_gemnet,
)
from research.spg_static_mvp.reference_graph import build_reference_graph
from research.spg_static_mvp.static_builder import (
    StaticBucketConfig,
    StaticPeriodicGraphBuilder,
    install_static_builder,
)


ROOT = Path("/data/dxl/results/spg_static_mvp")
BUCKET = ROOT / "selected_bucket.json"
MANIFEST = ROOT / "equivalence/manifest.json"


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not BUCKET.is_file(),
    reason="SPG GPU integration artifacts are unavailable",
)


def _load_state(seed: int, state_index: int) -> dict:
    states = torch.load(
        ROOT / f"shape_states/seed_{seed}/states.pt",
        map_location="cpu",
        weights_only=False,
    )
    return states[state_index]


@pytest.fixture(scope="module")
def runtime():
    configure_determinism()
    generator = build_c0_generator(sampling_steps=2)
    gemnet = find_gemnet(generator.model)
    device = next(gemnet.parameters()).device
    builder = StaticPeriodicGraphBuilder(
        StaticBucketConfig.from_json(BUCKET), device
    )
    return gemnet, builder


@pytest.fixture(scope="module")
def manifest_states() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = manifest[:8]
    wanted.append({"seed": 24501, "state_index": 201})
    return [_load_state(item["seed"], item["state_index"]) for item in wanted]


def _assert_graph_exact(row: dict) -> None:
    assert row["used_static"]
    for key in (
        "raw_edge_order_match",
        "raw_offset_order_match",
        "edge_order_match",
        "edge_set_match",
        "offset_match",
        "neighbors_match",
        "id_swap_match",
        "triplet_set_match",
        "triplet_order_match",
        "triplet_ragged_match",
    ):
        assert row[key], (key, row)
    assert row["distance_max_error"] == 0.0
    assert row["vector_max_error"] == 0.0


def test_static_builder_disabled_restores_original_c0(runtime):
    gemnet, builder = runtime
    original = gemnet.generate_interaction_graph
    with install_static_builder(gemnet, builder):
        assert gemnet.generate_interaction_graph != original
    assert gemnet.generate_interaction_graph == original


def test_edge_offset_order_top50_and_triplets_exact(runtime, manifest_states):
    gemnet, builder = runtime
    for state in manifest_states:
        _assert_graph_exact(compare_state(gemnet, builder, state))


def test_boundary_tie_matches_original_cuda_sort(runtime):
    gemnet, builder = runtime
    row = compare_state(gemnet, builder, _load_state(24501, 201))
    _assert_graph_exact(row)


def test_fixed_masks_and_padding_do_not_enter_compact_output(runtime):
    gemnet, builder = runtime
    state = _load_state(24501, 5)
    device = next(gemnet.parameters()).device
    reference = build_reference_graph(
        gemnet,
        frac_positions=state["pos"].to(device),
        cell=state["cell"].to(device),
        num_atoms=state["num_atoms"].to(device),
    )
    result = builder.build(
        cart_positions=reference["cart_positions"],
        cell=state["cell"].to(device),
        num_atoms=state["num_atoms"].to(device),
        cutoff=float(gemnet.cutoff),
        max_neighbors=int(gemnet.max_neighbors),
        max_cell_images_per_dim=int(gemnet.max_cell_images_per_dim),
    )
    assert not builder.raw_edge_mask[result.raw_count :].any()
    assert not builder.triplet_mask[result.triplet_count :].any()
    compact_before = tuple(value.clone() for value in result.compact_gemnet_tuple())
    builder.id3_ba[result.triplet_count :].fill_(123)
    compact_after = result.compact_gemnet_tuple()
    assert all(torch.equal(a, b) for a, b in zip(compact_before, compact_after))


@pytest.mark.parametrize("atom_count", [8, 12])
def test_bucket_atom_boundaries(runtime, atom_count):
    stats = pd.concat(
        [pd.read_csv(path) for path in sorted((ROOT / "shape_states").glob("seed_*/shape_statistics.csv"))],
        ignore_index=True,
    )
    row = stats.loc[
        (stats["num_atoms"] == atom_count)
        & (stats[["rep_a1", "rep_a2", "rep_a3"]] <= 2).all(axis=1)
    ].iloc[0]
    state = _load_state(int(row.seed), int(row.state_index))
    _assert_graph_exact(compare_state(*runtime, state))


def test_high_periodic_images_low_volume_nonorthogonal_cells(runtime):
    stats = pd.concat(
        [pd.read_csv(path) for path in sorted((ROOT / "shape_states").glob("seed_*/shape_statistics.csv"))],
        ignore_index=True,
    )
    config = runtime[1].config
    eligible = stats.loc[
        stats["num_atoms"].between(config.num_atoms_min, config.num_atoms_max)
        & (stats[["rep_a1", "rep_a2", "rep_a3"]] <= 2).all(axis=1)
    ]
    picks = pd.concat(
        [
            eligible.sort_values("cell_volume").head(1),
            eligible.sort_values("cell_condition_number", ascending=False).head(1),
            eligible.loc[(eligible[["rep_a1", "rep_a2", "rep_a3"]] == 2).all(axis=1)].head(1),
        ]
    )
    for row in picks.itertuples(index=False):
        _assert_graph_exact(
            compare_state(*runtime, _load_state(int(row.seed), int(row.state_index)))
        )


def test_fallback_guards_cover_overflow_and_bad_cells(runtime):
    gemnet, builder = runtime
    state = _load_state(24501, 5)
    device = next(gemnet.parameters()).device
    reference = build_reference_graph(
        gemnet,
        frac_positions=state["pos"].to(device),
        cell=state["cell"].to(device),
        num_atoms=state["num_atoms"].to(device),
    )
    kwargs = dict(
        cart_positions=reference["cart_positions"],
        cell=state["cell"].to(device),
        num_atoms=state["num_atoms"].to(device),
        cutoff=float(gemnet.cutoff),
        max_neighbors=int(gemnet.max_neighbors),
        max_cell_images_per_dim=int(gemnet.max_cell_images_per_dim),
    )
    bad_atoms = dict(kwargs)
    bad_atoms["num_atoms"] = torch.tensor([7], device=device)
    assert builder.build(**bad_atoms).fallback_reason == "num_atoms_outside_bucket"
    bad_batch = dict(kwargs)
    bad_batch["num_atoms"] = torch.tensor([5, 5], device=device)
    assert builder.build(**bad_batch).fallback_reason == "batch_size_not_one"
    bad_cutoff = dict(kwargs)
    bad_cutoff["cutoff"] = 6.0
    assert builder.build(**bad_cutoff).fallback_reason == "graph_parameters_changed"
    singular = dict(kwargs)
    singular["cell"] = torch.zeros_like(kwargs["cell"])
    assert builder.build(**singular).fallback_reason == "singular_cell"
    nonfinite = dict(kwargs)
    nonfinite["cell"] = kwargs["cell"].clone()
    nonfinite["cell"][0, 0, 0] = torch.nan
    assert builder.build(**nonfinite).fallback_reason == "non_finite_cell"


def test_fallback_routes_to_original_builder(runtime):
    gemnet, builder = runtime
    state = _load_state(24500, 0)
    device = next(gemnet.parameters()).device
    reference = build_reference_graph(
        gemnet,
        frac_positions=state["pos"].to(device),
        cell=state["cell"].to(device),
        num_atoms=state["num_atoms"].to(device),
    )
    args = (
        reference["cart_positions"],
        state["cell"].to(device),
        state["num_atoms"].to(device),
        None,
        None,
        None,
    )
    expected = gemnet.generate_interaction_graph(*args)
    counters = {}
    with install_static_builder(gemnet, builder, counters):
        actual = gemnet.generate_interaction_graph(*args)
    assert counters["fallback"] == 1
    assert counters["fallback_reasons"] == {"num_atoms_outside_bucket": 1}
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def test_joint_cfg_duplicate_geometry_matches_original_batch(runtime):
    gemnet, builder = runtime
    state = _load_state(24501, 201)
    device = next(gemnet.parameters()).device
    positions = state["pos"].to(device)
    cell = state["cell"].to(device)
    num_atoms = state["num_atoms"].to(device)
    joint_positions = torch.cat([positions, positions], dim=0)
    joint_cell = torch.cat([cell, cell], dim=0)
    joint_num_atoms = torch.cat([num_atoms, num_atoms], dim=0)
    reference = build_reference_graph(
        gemnet,
        frac_positions=joint_positions,
        cell=joint_cell,
        num_atoms=joint_num_atoms,
    )
    result = builder.build(
        cart_positions=reference["cart_positions"],
        cell=joint_cell,
        num_atoms=joint_num_atoms,
        cutoff=float(gemnet.cutoff),
        max_neighbors=int(gemnet.max_neighbors),
        max_cell_images_per_dim=int(gemnet.max_cell_images_per_dim),
    )
    assert result.used_static
    assert result.batch_copies == 2
    expected = (
        reference["edge_index"],
        reference["neighbors"],
        reference["edge_distances"],
        reference["edge_vectors"],
        reference["id_swap"],
        reference["id3_ba"],
        reference["id3_ca"],
        reference["id3_ragged_idx"],
        reference["cell_offsets"],
    )
    actual = result.compact_gemnet_tuple()
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_trace_same_seed_and_random_tape_are_deterministic(runtime):
    gemnet, builder = runtime
    state = _load_state(24501, 5)
    torch.manual_seed(8128)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone()
    first = compare_state(gemnet, builder, state)
    cpu_after = torch.random.get_rng_state().clone()
    cuda_after = torch.cuda.get_rng_state().clone()
    second = compare_state(gemnet, builder, state)
    assert torch.equal(cpu_before, cpu_after)
    assert torch.equal(cuda_before, cuda_after)
    assert first == second
    torch.manual_seed(91)
    tape_a = torch.randn(16)
    compare_state(gemnet, builder, state)
    tape_b = torch.randn(16)
    torch.manual_seed(91)
    expected_a = torch.randn(16)
    expected_b = torch.randn(16)
    assert torch.equal(tape_a, expected_a)
    assert torch.equal(tape_b, expected_b)
