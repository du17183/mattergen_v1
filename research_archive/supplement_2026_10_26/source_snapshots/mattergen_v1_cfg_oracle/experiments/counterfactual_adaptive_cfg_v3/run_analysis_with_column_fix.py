"""Run the frozen Oracle analysis with a non-scientific pandas column fix."""

from __future__ import annotations

import pandas as pd

import analyze_oracle as frozen_analysis


def chosen_rows_without_duplicate_merge(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    label_column: str = "oracle_label",
) -> pd.DataFrame:
    """Select frozen labels after asserting both existing sources are identical."""
    keys = ["seed", "sampling_step"]
    if label_column not in frame.columns:
        return frozen_analysis.chosen_rows(frame, labels, label_column)
    observed = frame[keys + [label_column]].drop_duplicates()
    expected = labels[keys + [label_column]].drop_duplicates()
    check = observed.merge(
        expected,
        on=keys,
        how="outer",
        suffixes=("_candidate", "_state"),
        validate="one_to_one",
    )
    if len(check) != len(expected) or not (
        check[f"{label_column}_candidate"] == check[f"{label_column}_state"]
    ).all():
        raise RuntimeError("candidate/state Oracle labels disagree")
    return frame[frame["candidate_action"] == frame[label_column]].copy()


if __name__ == "__main__":
    frozen_analysis.chosen_rows = chosen_rows_without_duplicate_merge
    frozen_analysis.main()
