"""Build the requested P0-to-P1 direction replication table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_p1"
P0_ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_p0"
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260920


def direction(value: float, preferred: str, tolerance: float = 1e-12) -> str:
    if abs(value) <= tolerance:
        return "tie"
    favorable = value < 0 if preferred == "lower" else value > 0
    return "favorable" if favorable else "unfavorable"


def main() -> None:
    p0 = pd.read_csv(P0_ROOT / "paired_statistics.csv")
    p1 = pd.read_csv(ROOT / "paired_statistics.csv")
    p0 = p0[p0.comparison == "GBSA-TCL-TCL"].set_index("metric")
    p1 = p1[p1.comparison == "GBSA-TCL-TCL"].set_index("metric")
    metrics = (
        "e_hull", "stable", "nus", "rmsd", "atomic_force_mean",
        "structure_max_force", "relaxation_steps", "novel", "unique",
    )
    rows = []
    for metric in metrics:
        preferred = p1.loc[metric, "preferred_direction"]
        p0_delta = float(p0.loc[metric, "mean_delta_candidate_minus_baseline"])
        p1_delta = float(p1.loc[metric, "mean_delta_candidate_minus_baseline"])
        p0_direction = direction(p0_delta, preferred)
        p1_direction = direction(p1_delta, preferred)
        rows.append({
            "metric": metric, "preferred_direction": preferred,
            "p0_gbsa_minus_tcl_direction": p0_direction,
            "p1_gbsa_minus_tcl_direction": p1_direction,
            "p1_delta": p1_delta,
            "p1_ci95_low": float(p1.loc[metric, "bootstrap_ci95_low"]),
            "p1_ci95_high": float(p1.loc[metric, "bootstrap_ci95_high"]),
            "replicated": p1_direction == p0_direction,
        })

    p0_drift = pd.read_csv(P0_ROOT / "score_drift_per_sample.csv")
    p1_drift = pd.read_csv(ROOT / "score_drift_per_sample.csv")
    def paired_late(frame):
        late = frame[(frame.field == "pos") & (frame.stage == "late_70_100")]
        item = late.groupby(["comparison", "source_method", "seed"], as_index=False).relative_l2_diff.mean()
        pivot = item.pivot(index=["source_method", "seed"], columns="comparison", values="relative_l2_diff")
        return pivot["GBSA-TCL_vs_C0"] - pivot["TCL_vs_C0"]
    p0_values = paired_late(p0_drift).to_numpy()
    p1_values = paired_late(p1_drift).to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(p1_values), size=(N_BOOTSTRAP, len(p1_values)))
    boot = p1_values[indices].mean(axis=1)
    p0_delta = float(p0_values.mean())
    p1_delta = float(p1_values.mean())
    rows.append({
        "metric": "late_position_drift", "preferred_direction": "lower",
        "p0_gbsa_minus_tcl_direction": direction(p0_delta, "lower"),
        "p1_gbsa_minus_tcl_direction": direction(p1_delta, "lower"),
        "p1_delta": p1_delta,
        "p1_ci95_low": float(np.quantile(boot, .025)),
        "p1_ci95_high": float(np.quantile(boot, .975)),
        "replicated": direction(p1_delta, "lower") == direction(p0_delta, "lower"),
    })
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "replication_table.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
