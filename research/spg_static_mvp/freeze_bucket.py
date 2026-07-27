from __future__ import annotations

import json

import pandas as pd

from research.spg_static_mvp.analyze_shapes import (
    REPORT,
    SHAPE_ROOT,
    SUMMARY_CSV,
    distribution_summary,
)
from research.spg_static_mvp.common import RESULTS, atomic_json, atomic_text, now


BUCKET_JSON = RESULTS / "selected_bucket.json"


def load_states() -> pd.DataFrame:
    paths = sorted(SHAPE_ROOT.glob("seed_*/shape_statistics.csv"))
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if len(frame) < 10_000:
        raise RuntimeError(f"shape audit has only {len(frame)} states")
    return frame


def freeze(frame: pd.DataFrame) -> dict:
    sample_atoms = frame.groupby("seed", as_index=False)["num_atoms"].first()
    candidates = []
    min_atoms = int(frame["num_atoms"].min())
    max_atoms = int(frame["num_atoms"].max())
    for low in range(min_atoms, max_atoms + 1):
        for high in range(low, min(max_atoms, low + 4) + 1):
            atom_mask = frame["num_atoms"].between(low, high)
            sample_mask = sample_atoms["num_atoms"].between(low, high)
            for max_rep in range(1, 6):
                mask = atom_mask & (
                    frame[["rep_a1", "rep_a2", "rep_a3"]] <= max_rep
                ).all(axis=1)
                if not mask.any():
                    continue
                capacity = int(high**2 * (2 * max_rep + 1) ** 3)
                utilization = float(
                    (frame.loc[mask, "candidate_pair_images"] / capacity).mean()
                )
                candidates.append(
                    {
                        "low": low,
                        "high": high,
                        "width": high - low + 1,
                        "max_rep": max_rep,
                        "state_coverage": float(mask.mean()),
                        "sample_coverage": float(sample_mask.mean()),
                        "validation_states": int(mask.sum()),
                        "candidate_capacity": capacity,
                        "candidate_padding_waste": 1.0 - utilization,
                        "atom_padding_waste": float(
                            ((high - frame.loc[mask, "num_atoms"]) / high).mean()
                        ),
                    }
                )
    eligible = [
        row
        for row in candidates
        if row["state_coverage"] >= 0.20
        and row["validation_states"] >= 10_000
        and row["candidate_padding_waste"] <= 0.40
    ]
    if not eligible:
        raise RuntimeError("no static bucket meets frozen coverage/padding constraints")
    choice = max(
        eligible,
        key=lambda row: (
            row["state_coverage"],
            -row["candidate_padding_waste"],
            -row["width"],
        ),
    )
    hit = frame["num_atoms"].between(choice["low"], choice["high"]) & (
        frame[["rep_a1", "rep_a2", "rep_a3"]] <= choice["max_rep"]
    ).all(axis=1)
    selected = frame.loc[hit]
    periodic_capacity = int((2 * choice["max_rep"] + 1) ** 3)
    return {
        "created_at": now(),
        "selected_bucket": (
            f"atoms_{choice['low']}_{choice['high']}_rep_le_{choice['max_rep']}"
        ),
        "num_atoms_min": int(choice["low"]),
        "num_atoms_max": int(choice["high"]),
        "max_rep_a1": int(choice["max_rep"]),
        "max_rep_a2": int(choice["max_rep"]),
        "max_rep_a3": int(choice["max_rep"]),
        "state_coverage": choice["state_coverage"],
        "sample_coverage": choice["sample_coverage"],
        "validation_states_available": choice["validation_states"],
        "total_states": int(len(frame)),
        "total_samples": int(frame["seed"].nunique()),
        "atom_padding_waste": choice["atom_padding_waste"],
        "padding_waste": choice["candidate_padding_waste"],
        "max_periodic_image_capacity": periodic_capacity,
        "max_candidate_pair_capacity": choice["candidate_capacity"],
        "max_raw_edge_capacity": int(selected["raw_edge_count"].max()),
        "max_edge_capacity": int(selected["edge_count"].max()),
        "max_triplet_capacity": int(selected["triplet_count"].max()),
        "selection_rule": (
            "maximum state coverage among contiguous <=5-atom and isotropic "
            "periodic-repetition buckets with candidate padding <=40%, "
            "coverage >=20%, and at least 10000 validation states"
        ),
    }


def main() -> dict:
    frame = load_states()
    bucket = freeze(frame)
    summary = distribution_summary(frame)
    summary.to_csv(SUMMARY_CSV, index=False)
    atomic_json(BUCKET_JSON, bucket)
    atom_counts = (
        frame.groupby("num_atoms")
        .agg(states=("state_index", "count"), samples=("seed", "nunique"))
        .reset_index()
    )
    atom_counts["state_coverage"] = atom_counts["states"] / len(frame)
    atom_counts["sample_coverage"] = atom_counts["samples"] / frame["seed"].nunique()
    lines = [
        "# SPG static MVP real trajectory shape distribution",
        "",
        f"- Real states: {len(frame)}",
        f"- Independent C0 samples: {frame['seed'].nunique()}",
        f"- Selected bucket: `{bucket['selected_bucket']}`",
        f"- State coverage: {bucket['state_coverage']:.4%}",
        f"- Sample coverage: {bucket['sample_coverage']:.4%}",
        f"- Candidate-capacity padding waste: {bucket['padding_waste']:.4%}",
        f"- Atom padding waste: {bucket['atom_padding_waste']:.4%}",
        f"- Candidate pair-image capacity: {bucket['max_candidate_pair_capacity']}",
        f"- Edge capacity: {bucket['max_edge_capacity']}",
        f"- Triplet capacity: {bucket['max_triplet_capacity']}",
        "",
        "## Quantiles",
        "",
        summary.to_markdown(index=False),
        "",
        "## Atom-count coverage",
        "",
        atom_counts.to_markdown(index=False),
        "",
        "The atom-count and periodic-repetition bucket is frozen before builder implementation.",
    ]
    atomic_text(REPORT, "\n".join(lines) + "\n")
    return bucket


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
