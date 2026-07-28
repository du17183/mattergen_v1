#!/usr/bin/env python3
"""Compute deterministic K-candidate oracle ceilings from frozen C0 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "seed",
    "energy_above_hull_per_atom",
    "rmsd_from_relaxation",
    "novel_unique_stable",
    "stable",
    "comp_validity",
    "structure_validity",
    "novel",
    "novel_unique",
    "unique",
    "converged",
}

METRICS = (
    "energy_above_hull_per_atom",
    "rmsd_from_relaxation",
    "stable",
    "novel_unique_stable",
    "comp_validity",
    "structure_validity",
    "novel",
    "unique",
    "converged",
)


def _bool_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in (
        "novel_unique_stable",
        "stable",
        "comp_validity",
        "structure_validity",
        "novel",
        "novel_unique",
        "unique",
        "converged",
    ):
        if frame[column].dtype != bool:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False})
            )
        if frame[column].isna().any():
            raise ValueError(f"invalid boolean values in {column}")
    return frame


def load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = _bool_columns(frame)
    if frame["seed"].duplicated().any():
        raise ValueError("each seed must appear exactly once")
    if not np.isfinite(frame["energy_above_hull_per_atom"]).all():
        raise ValueError("non-finite E-hull")
    if not np.isfinite(frame["rmsd_from_relaxation"]).all():
        raise ValueError("non-finite RMSD")
    return frame.sort_values("seed").reset_index(drop=True)


def _first(pool: pd.DataFrame) -> pd.Series:
    return pool.iloc[0]


def _min_ehull(pool: pd.DataFrame) -> pd.Series:
    safe = pool[pool["structure_validity"] & pool["comp_validity"]]
    source = safe if len(safe) else pool
    return source.sort_values(
        ["energy_above_hull_per_atom", "rmsd_from_relaxation", "seed"],
        ascending=[True, True, True],
    ).iloc[0]


def _min_rmsd(pool: pd.DataFrame) -> pd.Series:
    safe = pool[pool["structure_validity"] & pool["comp_validity"]]
    source = safe if len(safe) else pool
    return source.sort_values(
        ["rmsd_from_relaxation", "energy_above_hull_per_atom", "seed"],
        ascending=[True, True, True],
    ).iloc[0]


def _stable_quality(pool: pd.DataFrame) -> pd.Series:
    return pool.sort_values(
        [
            "structure_validity",
            "comp_validity",
            "stable",
            "energy_above_hull_per_atom",
            "rmsd_from_relaxation",
            "seed",
        ],
        ascending=[False, False, False, True, True, True],
    ).iloc[0]


def _nus_quality(pool: pd.DataFrame) -> pd.Series:
    return pool.sort_values(
        [
            "structure_validity",
            "comp_validity",
            "novel_unique_stable",
            "stable",
            "novel_unique",
            "energy_above_hull_per_atom",
            "seed",
        ],
        ascending=[False, False, False, False, False, True, True],
    ).iloc[0]


SELECTORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "baseline_first": _first,
    "oracle_min_ehull": _min_ehull,
    "oracle_min_rmsd": _min_rmsd,
    "oracle_stable_quality": _stable_quality,
    "oracle_nus_quality": _nus_quality,
}


def _aggregate(rows: list[pd.Series]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    return {
        metric: float(frame[metric].astype(float).mean())
        for metric in METRICS
    }


def one_trial(
    frame: pd.DataFrame,
    *,
    pool_size: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    order = rng.permutation(len(frame))
    usable = len(order) - len(order) % pool_size
    order = order[:usable].reshape(-1, pool_size)
    selected: dict[str, list[pd.Series]] = {name: [] for name in SELECTORS}
    for indexes in order:
        pool = frame.iloc[indexes]
        for name, selector in SELECTORS.items():
            selected[name].append(selector(pool))
    return {name: _aggregate(rows) for name, rows in selected.items()}


def summarize_trials(
    frame: pd.DataFrame,
    *,
    pool_size: int,
    trials: int,
    random_seed: int,
) -> tuple[dict, pd.DataFrame]:
    if pool_size < 2:
        raise ValueError("pool_size must be at least two")
    if len(frame) < pool_size:
        raise ValueError("not enough rows for one pool")
    rng = np.random.default_rng(random_seed)
    trial_rows: list[dict] = []
    for trial in range(trials):
        result = one_trial(frame, pool_size=pool_size, rng=rng)
        baseline = result["baseline_first"]
        for selector, metrics in result.items():
            row = {"trial": trial, "selector": selector, **metrics}
            row["ehull_change_vs_first"] = (
                metrics["energy_above_hull_per_atom"]
                - baseline["energy_above_hull_per_atom"]
            )
            row["rmsd_relative_change_vs_first"] = (
                metrics["rmsd_from_relaxation"]
                / baseline["rmsd_from_relaxation"]
                - 1.0
            )
            row["stable_change_vs_first"] = metrics["stable"] - baseline["stable"]
            row["nus_change_vs_first"] = (
                metrics["novel_unique_stable"]
                - baseline["novel_unique_stable"]
            )
            trial_rows.append(row)
    details = pd.DataFrame(trial_rows)
    summary: dict[str, dict] = {}
    for selector, group in details.groupby("selector", sort=True):
        values: dict[str, dict[str, float]] = {}
        for column in (
            *METRICS,
            "ehull_change_vs_first",
            "rmsd_relative_change_vs_first",
            "stable_change_vs_first",
            "nus_change_vs_first",
        ):
            series = group[column].astype(float)
            values[column] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "q025": float(series.quantile(0.025)),
                "q975": float(series.quantile(0.975)),
            }
        summary[str(selector)] = values
    gates = {
        "Q1_UQ_PQR_ORACLE_GO": (
            summary["oracle_min_ehull"]["ehull_change_vs_first"]["mean"] <= -0.005
            or summary["oracle_stable_quality"]["stable_change_vs_first"]["mean"]
            >= 0.03125
            or summary["oracle_nus_quality"]["nus_change_vs_first"]["mean"]
            >= 0.03125
        ),
        "Q2_RFR_ORACLE_GO": (
            summary["oracle_min_rmsd"]["rmsd_relative_change_vs_first"]["mean"]
            <= -0.10
        ),
        "Q6_NS_SETRANK_ORACLE_GO": (
            summary["oracle_nus_quality"]["nus_change_vs_first"]["mean"]
            >= 0.03125
            and summary["oracle_nus_quality"]["stable_change_vs_first"]["mean"]
            >= -0.03125
        ),
    }
    payload = {
        "schema_version": 1,
        "rows": int(len(frame)),
        "pool_size": int(pool_size),
        "pools_per_trial": int(len(frame) // pool_size),
        "trials": int(trials),
        "random_seed": int(random_seed),
        "summary": summary,
        "gates": gates,
    }
    return payload, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path(
            "/data/dxl/reports/formal_256/C0/"
            "official_metrics_per_structure.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/dxl/results/postgen_fastgate/oracle"),
    )
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_metrics(args.metrics_csv)
    summary, details = summarize_trials(
        frame,
        pool_size=args.pool_size,
        trials=args.trials,
        random_seed=args.random_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    details.to_csv(args.output_dir / "oracle_trials.csv", index=False)
    print(json.dumps(summary["gates"], sort_keys=True))


if __name__ == "__main__":
    main()
