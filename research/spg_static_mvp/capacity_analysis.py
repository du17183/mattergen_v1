from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.spg_static_mvp.common import (
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
)


PERCENTILES = (10, 25, 50, 75, 90, 95, 99, 100)


def summarize(values: pd.Series) -> dict[str, float]:
    return {
        f"p{percentile}": float(np.percentile(values, percentile))
        for percentile in PERCENTILES
    }


def main() -> int:
    bucket = json.loads(
        (RESULTS / "selected_bucket.json").read_text(encoding="utf-8")
    )
    frame = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(
                (RESULTS / "shape_states").glob("seed_*/shape_statistics.csv")
            )
        ],
        ignore_index=True,
    )
    mask = frame["num_atoms"].between(
        bucket["num_atoms_min"], bucket["num_atoms_max"]
    )
    mask &= (
        frame[["rep_a1", "rep_a2", "rep_a3"]]
        <= [bucket["max_rep_a1"], bucket["max_rep_a2"], bucket["max_rep_a3"]]
    ).all(axis=1)
    selected = frame.loc[mask].copy()
    candidate_capacity = float(bucket["max_candidate_pair_capacity"])
    edge_capacity = float(bucket["max_edge_capacity"])
    triplet_capacity = float(bucket["max_triplet_capacity"])
    selected["candidate_utilization"] = (
        selected["candidate_pair_images"] / candidate_capacity
    )
    selected["edge_utilization"] = selected["edge_count"] / edge_capacity
    selected["triplet_utilization"] = (
        selected["triplet_count"] / triplet_capacity
    )
    selected["candidate_padding_share"] = 1.0 - selected["candidate_utilization"]
    selected["edge_padding_share"] = 1.0 - selected["edge_utilization"]
    selected["triplet_padding_share"] = 1.0 - selected["triplet_utilization"]
    utilization = {
        "candidate": summarize(selected["candidate_utilization"]),
        "edge": summarize(selected["edge_utilization"]),
        "triplet": summarize(selected["triplet_utilization"]),
    }
    padding = {
        "candidate": summarize(selected["candidate_padding_share"]),
        "edge": summarize(selected["edge_padding_share"]),
        "triplet": summarize(selected["triplet_padding_share"]),
    }
    summary = {
        "completed_at": now(),
        "bucket_states": int(len(selected)),
        "capacities": {
            "candidate_pair_images": int(candidate_capacity),
            "edge": int(edge_capacity),
            "triplet": int(triplet_capacity),
        },
        "utilization": utilization,
        "padding": padding,
        "gemnet_input_semantics": "compacted_valid_edges_and_triplets_only",
        "INVALID_EDGE_COMPUTE_SHARE": 0.0,
        "INVALID_TRIPLET_COMPUTE_SHARE": 0.0,
        "PADDING_PERFORMANCE_RISK": (
            "builder_workspaces_process padded candidates/triplet masks; "
            "GemNet message passing receives only compact valid tensors"
        ),
        "EDGE_CAPACITY_UTILIZATION_P50": utilization["edge"]["p50"],
        "EDGE_CAPACITY_UTILIZATION_P95": utilization["edge"]["p95"],
        "TRIPLET_CAPACITY_UTILIZATION_P50": utilization["triplet"]["p50"],
        "TRIPLET_CAPACITY_UTILIZATION_P95": utilization["triplet"]["p95"],
    }
    atomic_json(RESULTS / "capacity/capacity_analysis.json", summary)
    columns = [
        "seed",
        "state_index",
        "num_atoms",
        "candidate_pair_images",
        "raw_edge_count",
        "edge_count",
        "triplet_count",
        "candidate_utilization",
        "edge_utilization",
        "triplet_utilization",
    ]
    output_csv = RESULTS / "capacity/capacity_states.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected[columns].to_csv(output_csv, index=False)
    lines = [
        "# SPG single-bucket capacity utilization",
        "",
        f"- States: `{len(selected)}`",
        f"- Candidate/edge/triplet capacities: `{int(candidate_capacity)}` / `{int(edge_capacity)}` / `{int(triplet_capacity)}`",
        "- GemNet receives compact valid edge/triplet tensors; fixed-capacity padding is not passed into message passing.",
        "- Invalid edge compute share in GemNet: `0%`",
        "- Invalid triplet compute share in GemNet: `0%`",
        "",
        "| Percentile | Candidate util. | Edge util. | Triplet util. |",
        "|---:|---:|---:|---:|",
    ]
    for percentile in PERCENTILES:
        key = f"p{percentile}"
        lines.append(
            f"| P{percentile} | {utilization['candidate'][key]:.3%} | "
            f"{utilization['edge'][key]:.3%} | {utilization['triplet'][key]:.3%} |"
        )
    lines.extend(
        [
            "",
            "The main risk is builder-side fixed candidate and triplet-mask work,",
            "not padded GemNet message passing. The microbenchmark must determine",
            "whether that workspace overhead erases the allocation/launch savings.",
            "",
        ]
    )
    atomic_text(REPORTS / "capacity_analysis.md", "\n".join(lines))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
