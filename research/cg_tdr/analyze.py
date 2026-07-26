#!/usr/bin/env python3
"""Official MatterGen metrics and paired CG-TDR Go/No-Go analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import binomtest, wilcoxon

from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.reference.correction_schemes import TRI110Compatibility2024
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.reference.reference_dataset_serializer import (
    LMDBBackedReferenceDatasetImpl,
)
from mattergen.evaluation.utils.lmdb_utils import lmdb_get
from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher


RESULTS = Path("/data/dxl/results/cg_tdr/phase0")
REPORTS = Path("/data/dxl/reports/cg_tdr/phase0")
GENERATION = RESULTS / "generation"
RELAX = RESULTS / "relax"
REFERENCE_LMDB = Path("/data/dxl/reference_assets/reference_TRI2024correction.lmdb")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def bootstrap_ci(values: np.ndarray, repeats: int = 20000) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [math.nan, math.nan]
    generator = np.random.default_rng(20260727)
    index = generator.integers(0, len(values), size=(repeats, len(values)))
    return [float(value) for value in np.percentile(values[index].mean(axis=1), [2.5, 97.5])]


def paired_stats(baseline: pd.DataFrame, candidate: pd.DataFrame, metric: str) -> dict[str, Any]:
    pair = baseline[["seed", metric]].merge(
        candidate[["seed", metric]],
        on="seed",
        suffixes=("_a0", "_candidate"),
        validate="one_to_one",
    )
    difference = (
        pair[f"{metric}_candidate"].to_numpy(float)
        - pair[f"{metric}_a0"].to_numpy(float)
    )
    try:
        test = wilcoxon(difference, zero_method="pratt") if np.any(difference) else None
    except ValueError:
        test = None
    leave_one_out = [
        {
            "removed_seed": int(pair.seed.iloc[index]),
            "mean_difference": float(np.delete(difference, index).mean()),
        }
        for index in range(len(difference))
    ]
    denominator = float(np.abs(difference).sum())
    return {
        "metric": metric,
        "paired_n": len(difference),
        "difference_definition": "candidate_minus_A0",
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "standard_deviation": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
        "bootstrap_95_ci": bootstrap_ci(difference),
        "wilcoxon_statistic": float(test.statistic) if test else 0.0,
        "wilcoxon_p_value": float(test.pvalue) if test else 1.0,
        "win_tie_loss": {
            "wins": int(np.sum(difference < -1.0e-12)),
            "ties": int(np.sum(np.abs(difference) <= 1.0e-12)),
            "losses": int(np.sum(difference > 1.0e-12)),
        },
        "leave_one_out": leave_one_out,
        "leave_one_out_range": [
            float(min(item["mean_difference"] for item in leave_one_out)),
            float(max(item["mean_difference"] for item in leave_one_out)),
        ],
        "max_sample_contribution": (
            float(np.abs(difference).max() / denominator) if denominator else 0.0
        ),
    }


def boolean_stats(baseline: pd.DataFrame, candidate: pd.DataFrame, metric: str) -> dict[str, Any]:
    pair = baseline[["seed", metric]].merge(
        candidate[["seed", metric]], on="seed", suffixes=("_a0", "_candidate")
    )
    left = pair[f"{metric}_a0"].astype(bool).to_numpy()
    right = pair[f"{metric}_candidate"].astype(bool).to_numpy()
    gains = int(np.sum(~left & right))
    losses = int(np.sum(left & ~right))
    discordant = gains + losses
    return {
        "metric": metric,
        "paired_n": len(pair),
        "gains": gains,
        "losses": losses,
        "ties": len(pair) - discordant,
        "exact_p_value": (
            float(binomtest(min(gains, losses), n=discordant, p=0.5).pvalue)
            if discordant
            else 1.0
        ),
    }


def reference_dataset() -> ReferenceDataset:
    def build_index(self: Any, _lmdb_path: Any) -> dict[str, Any]:
        result: dict[str, dict[str, Any]] = defaultdict(dict)
        with self.env.begin() as transaction:
            for system in lmdb_get(transaction, "chemical_systems"):
                formulas = lmdb_get(transaction, f"{system}.reduced_formulas")
                for formula in formulas:
                    result[system][formula] = lmdb_get(
                        transaction, f"{system}.{formula}.length"
                    )
        return dict(result)

    LMDBBackedReferenceDatasetImpl._build_num_entries_by_chemsys_reduced_formulas = build_index
    return ReferenceDataset(
        name="TRI2024correction",
        impl=LMDBBackedReferenceDatasetImpl(REFERENCE_LMDB, cleanup_dir=False),
    )


def methods_and_seeds(mode: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if mode == "eight":
        return ("A0", "T1", "T2"), tuple(range(23000, 23008))
    selected = str(
        json.loads((REPORTS / "eight_seed/selected_candidate.json").read_text())[
            "selected_config"
        ]
    )
    return ("A0", selected), tuple(range(23000, 23032))


def raw_frame(method: str, seeds: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        relax = json.loads((RELAX / method / str(seed) / "result.json").read_text())
        if relax.get("status") != "success":
            raise RuntimeError(f"Unsuccessful relaxation: {method}/{seed}")
        generation_dir = GENERATION / method / str(seed)
        generation = json.loads((generation_dir / "run_summary.json").read_text())
        hashes = json.loads((generation_dir / "structure_hashes.json").read_text())
        cg_metrics = json.loads((generation_dir / "cg_tdr_metrics.json").read_text())
        atoms = ase.io.read(relax["input_path"])
        if len(atoms) > 1:
            distances = atoms.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf)
            minimum_distance = float(distances.min())
        else:
            minimum_distance = math.inf
        rows.append(
            {
                **relax,
                "generation_success": bool(generation["success"]),
                "generation_elapsed_seconds": float(generation["elapsed_seconds"]),
                "initial_state_hash": hashes["initial_state_hash"],
                "atomic_sequence": tuple(hashes["atomic_numbers"]),
                "minimum_distance_angstrom": minimum_distance,
                "position_gate_mean": float(cg_metrics.get("position_gate_mean", 0.0)),
                "cell_gate_mean": float(cg_metrics.get("cell_gate_mean", 0.0)),
                "position_clipping_rate": float(
                    cg_metrics.get("position_clipping_rate", 0.0)
                ),
                "cell_fallback_rate": float(cg_metrics.get("cell_fallback_rate", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def official_frame(
    method: str,
    raw: pd.DataFrame,
    seeds: tuple[int, ...],
    reference: ReferenceDataset,
    out: Path,
) -> pd.DataFrame:
    relaxed = [ase.io.read(path) for path in raw.output_path]
    originals = [ase.io.read(path) for path in raw.input_path]
    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=[AseAtomsAdaptor.get_structure(atoms) for atoms in relaxed],
        energies=raw.relaxed_energy_ev.tolist(),
        reference=reference,
        original_structures=[AseAtomsAdaptor.get_structure(atoms) for atoms in originals],
        stability_threshold=0.1,
        structure_matcher=DefaultDisorderedStructureMatcher(),
        energy_correction_scheme=TRI110Compatibility2024(),
        n_failed_jobs=0,
    )
    method_out = out / method
    method_out.mkdir(parents=True, exist_ok=True)
    evaluator.compute_metrics("all", save_as=method_out / "official_metrics.json")
    official = evaluator.as_dataframe("all").drop(columns=["entry"], errors="ignore")
    official = official.reset_index(drop=True)
    official.insert(0, "seed", list(seeds))
    keep = [
        "seed",
        "energy_above_hull_per_atom",
        "rmsd_from_relaxation",
        "stable",
        "comp_validity",
        "structure_validity",
        "novel",
        "unique",
    ]
    return raw.merge(official[keep], on="seed", validate="one_to_one")


def summary(method: str, frame: pd.DataFrame) -> dict[str, Any]:
    nus = frame.stable.astype(bool) & frame.novel.astype(bool) & frame.unique.astype(bool)
    return {
        "method": method,
        "n": len(frame),
        "generation_success": float(frame.generation_success.mean()),
        "composition_validity": float(frame.comp_validity.mean()),
        "structure_validity": float(frame.structure_validity.mean()),
        "average_ehull": float(frame.energy_above_hull_per_atom.mean()),
        "stable_rate": float(frame.stable.mean()),
        "novel_rate": float(frame.novel.mean()),
        "unique_rate": float(frame.unique.mean()),
        "nus_rate": float(nus.mean()),
        "relaxation_rmsd_mean": float(frame.rmsd_from_relaxation.mean()),
        "initial_max_force_mean": float(frame.initial_max_force_ev_ang.mean()),
        "force_convergence_rate": float(frame.converged.mean()),
        "relaxation_failure_rate": 0.0,
        "generation_elapsed_median": float(frame.generation_elapsed_seconds.median()),
        "position_gate_mean": float(frame.position_gate_mean.mean()),
        "cell_gate_mean": float(frame.cell_gate_mean.mean()),
        "clipping_rate": float(frame.position_clipping_rate.mean()),
        "fallback_rate": float(frame.cell_fallback_rate.mean()),
        "severe_short_bonds": int((frame.minimum_distance_angstrom < 0.5).sum()),
    }


def comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    base_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    merged = baseline[
        ["seed", "initial_state_hash", "atomic_sequence", "rmsd_from_relaxation", "initial_max_force_ev_ang"]
    ].merge(
        candidate[
            ["seed", "initial_state_hash", "atomic_sequence", "rmsd_from_relaxation", "initial_max_force_ev_ang"]
        ],
        on="seed",
        suffixes=("_a0", "_candidate"),
    )
    initial_match = bool(
        (merged.initial_state_hash_a0 == merged.initial_state_hash_candidate).all()
    )
    atomic_match = bool(
        (merged.atomic_sequence_a0 == merged.atomic_sequence_candidate).all()
    )
    changes = {
        key: candidate_summary[key] - base_summary[key]
        for key in (
            "composition_validity",
            "structure_validity",
            "average_ehull",
            "stable_rate",
            "novel_rate",
            "unique_rate",
            "nus_rate",
            "relaxation_failure_rate",
        )
    }
    rmsd_relative = (
        candidate_summary["relaxation_rmsd_mean"]
        / base_summary["relaxation_rmsd_mean"]
        - 1.0
    )
    force_relative = (
        candidate_summary["initial_max_force_mean"]
        / base_summary["initial_max_force_mean"]
        - 1.0
    )
    latency = (
        candidate_summary["generation_elapsed_median"]
        / base_summary["generation_elapsed_median"]
        - 1.0
    )
    rmsd_wins = int(
        (
            merged.rmsd_from_relaxation_candidate
            < merged.rmsd_from_relaxation_a0 - 1.0e-12
        ).sum()
    )
    force_wins = int(
        (
            merged.initial_max_force_ev_ang_candidate
            < merged.initial_max_force_ev_ang_a0 - 1.0e-12
        ).sum()
    )
    common = (
        initial_match
        and atomic_match
        and candidate_summary["generation_success"] == 1.0
        and changes["structure_validity"] >= 0
        and changes["average_ehull"] <= 0.002
        and changes["relaxation_failure_rate"] <= 0
        and latency <= 0.10
        and candidate_summary["severe_short_bonds"] <= base_summary["severe_short_bonds"]
    )
    if mode == "eight":
        direction = (
            (rmsd_relative <= -0.03 and rmsd_wins >= 5)
            or (force_relative <= -0.03 and force_wins >= 5)
        )
        gate = (
            common
            and changes["composition_validity"] >= 0
            and changes["stable_rate"] >= 0
            and direction
        )
    else:
        safety = (
            common
            and changes["composition_validity"] >= -1.0 / 32.0
            and changes["stable_rate"] >= -1.0 / 32.0
            and changes["nus_rate"] >= -1.0 / 32.0
            and changes["novel_rate"] >= -0.02
            and changes["unique_rate"] >= -0.02
        )
        positive = (
            changes["average_ehull"] <= -0.005
            or changes["stable_rate"] >= 1.0 / 32.0
            or changes["nus_rate"] >= 1.0 / 32.0
            or rmsd_relative <= -0.10
            or force_relative <= -0.10
        )
        direction = positive
        gate = safety and positive
    return {
        "method": candidate_summary["method"],
        "initial_state_hashes_match": initial_match,
        "atomic_sequences_match": atomic_match,
        **{f"{key}_change": value for key, value in changes.items()},
        "rmsd_relative_change": rmsd_relative,
        "pre_relax_max_force_relative_change": force_relative,
        "rmsd_wins": rmsd_wins,
        "max_force_wins": force_wins,
        "latency_overhead": latency,
        "clear_improvement_direction": bool(direction),
        "gate_go": bool(gate),
    }


def analyze(mode: str) -> dict[str, Any]:
    methods, seeds = methods_and_seeds(mode)
    out = REPORTS / ("eight_seed" if mode == "eight" else "thirty_two_seed")
    out.mkdir(parents=True, exist_ok=True)
    reference = reference_dataset()
    frames: dict[str, pd.DataFrame] = {}
    summaries = []
    for method in methods:
        frame = official_frame(method, raw_frame(method, seeds), seeds, reference, out)
        frames[method] = frame
        frame.to_csv(out / f"{method}_per_structure.csv", index=False)
        summaries.append(summary(method, frame))
    base_summary = summaries[0]
    comparisons = [
        comparison(frames["A0"], frames[method], base_summary, summaries[index], mode)
        for index, method in enumerate(methods[1:], start=1)
    ]
    statistics = []
    for method in methods[1:]:
        for metric in (
            "energy_above_hull_per_atom",
            "rmsd_from_relaxation",
            "initial_max_force_ev_ang",
        ):
            row = paired_stats(frames["A0"], frames[method], metric)
            row["method"] = method
            statistics.append(row)
        for metric in ("stable", "comp_validity", "structure_validity", "novel", "unique"):
            row = boolean_stats(frames["A0"], frames[method], metric)
            row["method"] = method
            statistics.append(row)
    passing = [row for row in comparisons if row["gate_go"]]
    selected = None
    if mode == "eight" and passing:
        selected = min(
            passing,
            key=lambda row: (
                row["rmsd_relative_change"]
                + row["pre_relax_max_force_relative_change"]
                + 10.0 * row["average_ehull_change"],
                row["latency_overhead"],
            ),
        )["method"]
        atomic_json(out / "selected_candidate.json", {"selected_config": selected})
    elif mode == "thirtytwo":
        selected = methods[1]
    result = {
        "mode": mode,
        "seeds": list(seeds),
        "summaries": summaries,
        "comparisons": comparisons,
        "paired_statistics": statistics,
        "selected_config": selected,
        "CG_TDR_EIGHT_SEED_GO": bool(passing) if mode == "eight" else None,
        "CG_TDR_MVP_GO": bool(comparisons[0]["gate_go"]) if mode == "thirtytwo" else None,
        "CG_TDR_MVP_NO_GO": (
            not bool(comparisons[0]["gate_go"]) if mode == "thirtytwo" else None
        ),
        "SIXTY_FOUR_SEED_STARTED": False,
        "FORMAL_SEEDS_STARTED": False,
    }
    atomic_json(out / "analysis_report.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eight", "thirtytwo"), required=True)
    args = parser.parse_args()
    analyze(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
