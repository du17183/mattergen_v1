from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.spg_static_mvp.common import REPORTS, RESULTS, atomic_json, atomic_text, now


SHAPE_ROOT = RESULTS / "shape_states"
SUMMARY_CSV = RESULTS / "shape_distribution.csv"
BUCKET_JSON = RESULTS / "selected_bucket.json"
REPORT = REPORTS / "shape_distribution.md"


def load_rows() -> pd.DataFrame:
    paths = sorted(SHAPE_ROOT.glob("seed_*/shape_statistics.csv"))
    if not paths:
        raise RuntimeError("no shape statistics found")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if len(frame) < 10_000:
        raise RuntimeError(f"shape audit has only {len(frame)} states")
    return frame


def choose_bucket(frame: pd.DataFrame) -> dict:
    total_states = len(frame)
    sample_atoms = frame.groupby("seed", as_index=False)["num_atoms"].first()
    total_samples = len(sample_atoms)
    candidates = []
    minimum = int(frame["num_atoms"].min())
    maximum = int(frame["num_atoms"].max())
    for low in range(minimum, maximum + 1):
        for high in range(low, min(maximum, low + 4) + 1):
            selected = frame["num_atoms"].between(low, high)
            selected_samples = sample_atoms["num_atoms"].between(low, high)
            if not selected.any():
                continue
            state_coverage = float(selected.mean())
            sample_coverage = float(selected_samples.mean())
            padding_waste = float(
                ((high - frame.loc[selected, "num_atoms"]) / high).mean()
            )
            candidates.append(
                {
                    "low": low,
                    "high": high,
                    "width": high - low + 1,
                    "state_coverage": state_coverage,
                    "sample_coverage": sample_coverage,
                    "padding_waste": padding_waste,
                    "validation_states": int(selected.sum()),
                }
            )
    eligible = [
        row
        for row in candidates
        if row["state_coverage"] >= 0.20
        and row["padding_waste"] <= 0.25
        and row["validation_states"] >= 10_000
    ]
    if not eligible:
        eligible = [
            row
            for row in candidates
            if row["state_coverage"] >= 0.20 and row["padding_waste"] <= 0.25
        ]
    if not eligible:
        eligible = candidates
    selected_bucket = max(
        eligible,
        key=lambda row: (
            row["state_coverage"],
            -row["padding_waste"],
            -row["width"],
        ),
    )
    selected_mask = frame["num_atoms"].between(
        selected_bucket["low"],
        selected_bucket["high"],
    )
    selected_frame = frame.loc[selected_mask]
    reps = {
        axis: int(selected_frame[axis].max())
        for axis in ("rep_a1", "rep_a2", "rep_a3")
    }
    max_periodic_images = int(
        (2 * reps["rep_a1"] + 1)
        * (2 * reps["rep_a2"] + 1)
        * (2 * reps["rep_a3"] + 1)
    )
    return {
        "created_at": now(),
        "selected_bucket": (
            str(selected_bucket["low"])
            if selected_bucket["low"] == selected_bucket["high"]
            else f"{selected_bucket['low']}-{selected_bucket['high']}"
        ),
        "num_atoms_min": int(selected_bucket["low"]),
        "num_atoms_max": int(selected_bucket["high"]),
        "state_coverage": selected_bucket["state_coverage"],
        "sample_coverage": selected_bucket["sample_coverage"],
        "padding_waste": selected_bucket["padding_waste"],
        "validation_states_available": selected_bucket["validation_states"],
        "total_states": total_states,
        "total_samples": total_samples,
        "max_rep_a1": reps["rep_a1"],
        "max_rep_a2": reps["rep_a2"],
        "max_rep_a3": reps["rep_a3"],
        "max_periodic_image_capacity": max_periodic_images,
        "max_candidate_pair_capacity": int(
            selected_bucket["high"] ** 2 * max_periodic_images
        ),
        "max_raw_edge_capacity": int(selected_frame["raw_edge_count"].max()),
        "max_edge_capacity": int(selected_frame["edge_count"].max()),
        "max_triplet_capacity": int(selected_frame["triplet_count"].max()),
        "selection_rule": (
            "maximum state coverage among contiguous <=5-atom buckets with "
            "padding <=25%, coverage >=20%, and preferably >=10000 validation states"
        ),
    }


def distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "num_atoms",
        "candidate_periodic_images",
        "candidate_pair_images",
        "raw_edge_count",
        "edge_count",
        "max_neighbors",
        "mean_neighbors",
        "triplet_count",
        "cell_volume",
        "cell_condition_number",
    )
    rows = []
    for metric in metrics:
        values = frame[metric]
        row = {"metric": metric, "count": int(values.count())}
        for quantile in (0.50, 0.75, 0.90, 0.95, 0.99):
            row[f"p{int(quantile * 100)}"] = float(values.quantile(quantile))
        row["max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> dict:
    frame = load_rows()
    bucket = choose_bucket(frame)
    summary = distribution_summary(frame)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
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
        f"- Selected bucket: `{bucket['selected_bucket']}` atoms",
        f"- State coverage: {bucket['state_coverage']:.4%}",
        f"- Sample coverage: {bucket['sample_coverage']:.4%}",
        f"- Padding waste: {bucket['padding_waste']:.4%}",
        f"- Edge capacity: {bucket['max_edge_capacity']}",
        f"- Triplet capacity: {bucket['max_triplet_capacity']}",
        f"- Candidate pair-image capacity: {bucket['max_candidate_pair_capacity']}",
        "",
        "## Quantiles",
        "",
        summary.to_markdown(index=False),
        "",
        "## Atom-count coverage",
        "",
        atom_counts.to_markdown(index=False),
        "",
        "The bucket is frozen before static-builder implementation.",
    ]
    atomic_text(REPORT, "\n".join(lines) + "\n")
    return bucket


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
