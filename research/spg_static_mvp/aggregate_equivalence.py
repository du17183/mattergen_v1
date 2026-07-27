from __future__ import annotations

import json
from pathlib import Path

from research.spg_static_mvp.common import (
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    set_stage,
)


WORKERS = RESULTS / "equivalence/workers"
SUMMARY = RESULTS / "equivalence/equivalence_summary.json"
REPORT = REPORTS / "equivalence_report.md"
MATCH_KEYS = (
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
)


def row_passes(row: dict) -> bool:
    return (
        bool(row.get("used_static"))
        and all(bool(row.get(key)) for key in MATCH_KEYS)
        and float(row.get("distance_max_error", float("inf"))) == 0.0
        and float(row.get("vector_max_error", float("inf"))) == 0.0
    )


def aggregate(world_size: int = 8) -> dict:
    rows: list[dict] = []
    worker_states = []
    for rank in range(world_size):
        payload = json.loads(
            (WORKERS / f"rank_{rank}.json").read_text(encoding="utf-8")
        )
        if not payload.get("success"):
            raise RuntimeError(f"worker {rank} did not succeed")
        worker_states.append(int(payload["states"]))
        rows.extend(payload["rows"])
    denominator = max(len(rows), 1)
    first_failure = next((row for row in rows if not row_passes(row)), None)
    summary = {
        "passed": len(rows) == 10_000 and first_failure is None,
        "states": len(rows),
        "worker_states": worker_states,
        "static_states": sum(bool(row.get("used_static")) for row in rows),
        "fallback_states": sum(not bool(row.get("used_static")) for row in rows),
        "match_rates": {
            key: sum(bool(row.get(key)) for row in rows) / denominator
            for key in MATCH_KEYS
        },
        "distance_zero_rate": sum(
            float(row.get("distance_max_error", float("inf"))) == 0.0
            for row in rows
        )
        / denominator,
        "vector_zero_rate": sum(
            float(row.get("vector_max_error", float("inf"))) == 0.0
            for row in rows
        )
        / denominator,
        "max_distance_error": max(
            (float(row.get("distance_max_error", float("inf"))) for row in rows),
            default=float("inf"),
        ),
        "max_vector_error": max(
            (float(row.get("vector_max_error", float("inf"))) for row in rows),
            default=float("inf"),
        ),
        "first_failure": first_failure,
    }
    atomic_json(SUMMARY, summary)
    lines = [
        "# SPG static builder 10,000-state equivalence",
        "",
        f"- Passed: `{summary['passed']}`",
        f"- States: `{summary['states']}`",
        f"- Static states: `{summary['static_states']}`",
        f"- Fallback states: `{summary['fallback_states']}`",
        f"- First failure: `{summary['first_failure']}`",
        f"- Max distance error: `{summary['max_distance_error']}`",
        f"- Max vector error: `{summary['max_vector_error']}`",
        "",
        "| Field | Match rate |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {key} | {value:.6%} |"
        for key, value in summary["match_rates"].items()
    )
    lines.extend(
        [
            "",
            "The frozen manifest spans all eligible recorded seeds, atom counts",
            "8–12, sampling steps 0–999, and predictor/corrector states.",
            "",
        ]
    )
    atomic_text(REPORT, "\n".join(lines))
    set_stage(
        "state_equivalence_10000",
        "success" if summary["passed"] else "failed",
        "Completed strict real-state graph equivalence validation.",
        summary,
    )
    return summary


def main() -> int:
    summary = aggregate()
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
