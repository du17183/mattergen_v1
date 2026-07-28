#!/usr/bin/env python3
"""Evaluate frozen learned selectors on held-out C0 candidate pools."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

OUTCOME_COLUMNS = (
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


def q1_score(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["pred_energy_above_hull_per_atom"].to_numpy(float)
        + 0.5 * frame["std_energy_above_hull_per_atom"].to_numpy(float)
        - 0.04 * frame["prob_stable"].to_numpy(float)
        - 0.06 * frame["prob_novel_unique_stable"].to_numpy(float)
        + 0.04 * (1.0 - frame["prob_comp_validity"].to_numpy(float))
        + 0.02 * (1.0 - frame["prob_novel"].to_numpy(float))
        + 0.02 * (1.0 - frame["prob_unique"].to_numpy(float))
    )


def q2_score(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["pred_rmsd_from_relaxation"].to_numpy(float)
        + frame["std_rmsd_from_relaxation"].to_numpy(float)
        + 0.05 * (1.0 - frame["prob_stable"].to_numpy(float))
        + 0.05 * (1.0 - frame["prob_comp_validity"].to_numpy(float))
        + 0.02 * (1.0 - frame["prob_novel"].to_numpy(float))
        + 0.02 * (1.0 - frame["prob_unique"].to_numpy(float))
    )


def q4_score(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["pred_energy_above_hull_per_atom"].to_numpy(float)
        + frame["std_energy_above_hull_per_atom"].to_numpy(float)
        + 0.05 * (1.0 - frame["prob_comp_validity"].to_numpy(float))
        + 0.02 * (1.0 - frame["prob_novel"].to_numpy(float))
        + 0.02 * (1.0 - frame["prob_unique"].to_numpy(float))
    )


def q6_score(frame: pd.DataFrame) -> np.ndarray:
    return -(
        frame["prob_novel_unique_stable"].to_numpy(float)
        + 0.25 * frame["prob_stable"].to_numpy(float)
        + 0.10 * frame["prob_novel"].to_numpy(float)
        + 0.10 * frame["prob_unique"].to_numpy(float)
        + 0.10 * frame["prob_comp_validity"].to_numpy(float)
        - 0.50 * frame["std_prob_novel_unique_stable"].to_numpy(float)
    )


SELECTORS: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "Q1_UQ_PQR": q1_score,
    "Q2_RFR": q2_score,
    "Q4_CPRC": q4_score,
    "Q6_NS_SETRANK_PROXY": q6_score,
}


def aggregate(rows: list[pd.Series]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    return {
        column: float(frame[column].astype(float).mean())
        for column in OUTCOME_COLUMNS
    }


def summarize(
    frame: pd.DataFrame,
    *,
    pool_size: int,
    trials: int,
    random_seed: int,
) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    rows: list[dict] = []
    for trial in range(trials):
        permutation = rng.permutation(len(frame))
        usable = len(permutation) - len(permutation) % pool_size
        pools = permutation[:usable].reshape(-1, pool_size)
        selections: dict[str, list[pd.Series]] = {
            "baseline_first": []
        }
        selections.update({name: [] for name in SELECTORS})
        for indexes in pools:
            pool = frame.iloc[indexes]
            selections["baseline_first"].append(pool.iloc[0])
            for name, score_function in SELECTORS.items():
                scores = score_function(pool)
                selections[name].append(pool.iloc[int(np.argmin(scores))])
        values = {
            name: aggregate(selected)
            for name, selected in selections.items()
        }
        baseline = values["baseline_first"]
        for name, metrics in values.items():
            rows.append(
                {
                    "trial": trial,
                    "selector": name,
                    **metrics,
                    "ehull_change": (
                        metrics["energy_above_hull_per_atom"]
                        - baseline["energy_above_hull_per_atom"]
                    ),
                    "rmsd_relative_change": (
                        metrics["rmsd_from_relaxation"]
                        / baseline["rmsd_from_relaxation"]
                        - 1.0
                    ),
                    "stable_change": metrics["stable"] - baseline["stable"],
                    "nus_change": (
                        metrics["novel_unique_stable"]
                        - baseline["novel_unique_stable"]
                    ),
                    "composition_change": (
                        metrics["comp_validity"]
                        - baseline["comp_validity"]
                    ),
                    "structure_change": (
                        metrics["structure_validity"]
                        - baseline["structure_validity"]
                    ),
                    "novel_change": metrics["novel"] - baseline["novel"],
                    "unique_change": metrics["unique"] - baseline["unique"],
                }
            )
    details = pd.DataFrame(rows)
    report: dict[str, dict] = {}
    for selector, group in details.groupby("selector", sort=True):
        report[selector] = {
            column: {
                "mean": float(group[column].mean()),
                "median": float(group[column].median()),
                "q025": float(group[column].quantile(0.025)),
                "q975": float(group[column].quantile(0.975)),
            }
            for column in (
                *OUTCOME_COLUMNS,
                "ehull_change",
                "rmsd_relative_change",
                "stable_change",
                "nus_change",
                "composition_change",
                "structure_change",
                "novel_change",
                "unique_change",
            )
        }
    gates: dict[str, bool] = {}
    for name in SELECTORS:
        value = report[name]
        safety = (
            value["structure_change"]["mean"] >= 0.0
            and value["composition_change"]["mean"] >= -0.03125
            and value["stable_change"]["mean"] >= -0.03125
            and value["nus_change"]["mean"] >= -0.03125
            and value["novel_change"]["mean"] >= -0.02
            and value["unique_change"]["mean"] >= -0.02
        )
        positive = (
            value["ehull_change"]["mean"] <= -0.005
            or value["stable_change"]["mean"] >= 0.03125
            or value["nus_change"]["mean"] >= 0.03125
            or value["rmsd_relative_change"]["mean"] <= -0.10
        )
        gates[f"{name}_LEARNED_OFFLINE_GO"] = bool(safety and positive)
    payload = {
        "schema_version": 1,
        "rows": len(frame),
        "pool_size": pool_size,
        "pools_per_trial": len(frame) // pool_size,
        "trials": trials,
        "random_seed": random_seed,
        "split": sorted(frame["split"].unique().tolist()),
        "method": sorted(frame["method"].unique().tolist()),
        "frozen_scores": {
            "Q1_UQ_PQR": "multi-task quality + ensemble uncertainty",
            "Q2_RFR": "RMSD risk + safety probabilities",
            "Q4_CPRC": "MatterSim E-hull prediction + uncertainty + safety",
            "Q6_NS_SETRANK_PROXY": "NUS/stable/novel/unique probabilities",
        },
        "summary": report,
        "gates": gates,
    }
    return payload, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "/data/dxl/results/postgen_fastgate/features/"
            "historical_features.csv"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "/data/dxl/results/postgen_fastgate/quality_model/"
            "predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/dxl/results/postgen_fastgate/learned_offline"
        ),
    )
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    predictions = pd.read_csv(args.predictions)
    frame = features.merge(
        predictions,
        on=["method", "seed", "split"],
        validate="one_to_one",
    )
    frame = frame[
        (frame["method"] == "C0") & (frame["split"] == "test")
    ].reset_index(drop=True)
    report, details = summarize(
        frame,
        pool_size=args.pool_size,
        trials=args.trials,
        random_seed=args.random_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / f".summary.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output_dir / "learned_offline_summary.json")
    details.to_csv(args.output_dir / "learned_offline_trials.csv", index=False)
    print(json.dumps(report["gates"], sort_keys=True))


if __name__ == "__main__":
    main()
