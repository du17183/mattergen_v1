from __future__ import annotations

import copy

import pytest

from research.mps_fastgate.benchmark import cross_equivalent, single_gpu_gate
from research.mps_fastgate.finalize import bitwise_audit, completion_skew_estimates
from research.mps_fastgate.runtime import worker_completion_skew_seconds


def _summary() -> dict:
    return {
        "result_index": [
            {
                "seed": 1,
                "round": 1,
                "worker_id": "w0",
                "elapsed_seconds": 2.0,
                "random_tape_hash": "r",
                "atomic_numbers_hash": "a",
                "final_structure_hash": "f",
                "positions_hash": "p",
                "cell_hash": "c",
            },
            {
                "seed": 2,
                "round": 1,
                "worker_id": "w1",
                "elapsed_seconds": 3.0,
                "random_tape_hash": "r2",
                "atomic_numbers_hash": "a2",
                "final_structure_hash": "f2",
                "positions_hash": "p2",
                "cell_hash": "c2",
            },
        ]
    }


def test_cross_equivalent_and_audit_detect_raw_tensor_change() -> None:
    reference = _summary()
    candidate = copy.deepcopy(reference)
    assert cross_equivalent(reference, candidate)
    assert bitwise_audit(reference, candidate)["bitwise_equivalent"]

    candidate["result_index"][1]["positions_hash"] = "changed"
    assert not cross_equivalent(reference, candidate)
    audit = bitwise_audit(reference, candidate)
    assert not audit["bitwise_equivalent"]
    assert audit["mismatch_count"] == 1


@pytest.mark.parametrize(
    ("bitwise", "success", "incremental", "expected"),
    [
        (False, True, 1.10, "MPS_NO_GO"),
        (True, False, 1.10, "MPS_NO_GO"),
        (True, True, 1.0299, "MPS_NO_GO"),
        (True, True, 1.03, "MPS_ENGINEERING_ONLY"),
        (True, True, 1.0499, "MPS_ENGINEERING_ONLY"),
        (True, True, 1.05, "RUN_EIGHT_GPU"),
    ],
)
def test_frozen_single_gpu_gate(bitwise: bool, success: bool, incremental: float, expected: str) -> None:
    assert single_gpu_gate(bitwise=bitwise, success=success, incremental=incremental) == expected


def test_worker_completion_skew_uses_each_workers_last_finish() -> None:
    rows = [
        {"worker_id": "w0", "finished_monotonic": 10.0},
        {"worker_id": "w1", "finished_monotonic": 11.0},
        {"worker_id": "w0", "finished_monotonic": 20.0},
        {"worker_id": "w1", "finished_monotonic": 22.5},
    ]
    assert worker_completion_skew_seconds(rows) == pytest.approx(2.5)


def test_reconstructed_completion_skew_groups_round_and_worker() -> None:
    summary = _summary()
    assert completion_skew_estimates(summary) == {1: pytest.approx(1.0)}
