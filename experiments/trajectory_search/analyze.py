"""Predeclared SAFE-A selection and paired 20k-bootstrap analysis."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from ase.io import read, write
import numpy as np
import pandas as pd

from experiments.trajectory_search.evaluate import GROUPS, RESULTS
from experiments.trajectory_search.run_generation import ORDER, registry


EPS = 1e-12
PROPERTY_TOL = 1e-6
BOOT_SEED = 2026092801
N_BOOT = 20_000
BUDGETS = {"C0": 1.0, "Independent_Best_of_2": 2.0, "Fixed_K2": 2.2, **{f"K{k}": 1.0 + 0.6 * k for k in range(5)}}


def methods(cohort: str) -> dict[str, tuple[str, ...]]:
    if cohort == "search_baseline":
        return {"C0": (), "Independent_Best_of_2": ("Independent_C0",), "Fixed_K2": ("GPulse", "PPulse")}
    return {f"K{k}": ORDER[:k] for k in range(5)}


def attach_branch_quality(cohort: str) -> pd.DataFrame:
    output = RESULTS / cohort
    values = pd.read_csv(output / "property_metrics_raw.csv")
    for column, default in (("e_hull", np.nan), ("stable", False), ("nus", False), ("novel", False), ("unique", False), ("quality_success", False), ("max_force", np.nan), ("rmsd", np.nan)):
        values[column] = default
    for group in GROUPS[cohort]:
        source = output / "evaluation_structures" / f"{group}.extxyz"
        if not source.exists():
            continue
        quality = output / "quality_branches" / group
        atoms = read(source, index=":")
        per = pd.read_csv(quality / "per_structure.csv")
        with gzip.open(quality / "official_detailed.json.gz", "rt") as stream:
            detailed = json.load(stream)
        if len(per) != len(detailed["energy_above_hull_per_atom"]):
            raise RuntimeError(f"quality length mismatch {group}")
        for output_index, row in per.reset_index(drop=True).iterrows():
            branch_id = str(atoms[int(row["index"])].info["branch_id"])
            match = values.index[values.branch_id == branch_id]
            if len(match) != 1:
                raise RuntimeError(f"missing branch {branch_id}")
            at = match[0]
            values.loc[at, "e_hull"] = float(detailed["energy_above_hull_per_atom"][output_index])
            for name, key in (("stable", "stable"), ("nus", "novel_unique_stable"), ("novel", "novel"), ("unique", "unique")):
                values.loc[at, name] = bool(detailed[key][output_index])
            values.loc[at, "rmsd"] = float(detailed["rmsd_from_relaxation"][output_index])
            values.loc[at, "max_force"] = float(row["pre_relaxation_max_force"])
            values.loc[at, "quality_success"] = True
    for column in ("structure_valid", "stable", "nus", "novel", "unique", "quality_success"):
        values[column] = values[column].fillna(False).astype(bool)
    values["evaluation_valid"] = values.structure_valid & values.quality_success
    values.to_csv(output / "evaluated_branches.csv", index=False)
    return values


def safe_select(c0: pd.Series, candidates: pd.DataFrame) -> tuple[pd.Series, str, int]:
    if candidates.empty:
        return c0, "C0", 0
    gain = float(c0.property_absolute_error) - candidates.property_absolute_error
    safe = (
        (gain > PROPERTY_TOL)
        & (candidates.evaluation_valid.astype(int) >= int(bool(c0.evaluation_valid)))
        & (candidates.stable.astype(int) >= int(bool(c0.stable)))
        & np.isfinite(candidates.e_hull)
        & np.isfinite(float(c0.e_hull))
        & (candidates.e_hull <= float(c0.e_hull) + 0.01 + EPS)
    )
    eligible = candidates[safe]
    if eligible.empty:
        return c0, "C0", 0
    winner = eligible.sort_values(["property_absolute_error", "policy_id"]).iloc[0]
    return winner, str(winner.policy_id), int(safe.sum())


def select(cohort: str) -> None:
    output = RESULTS / cohort
    if (output / "selected_outcomes.csv").exists():
        raise FileExistsError("selection already frozen")
    values = attach_branch_quality(cohort)
    seeds, _ = registry(cohort)
    choices = methods(cohort)
    selected = []
    for seed in seeds:
        frame = values[values.seed == seed]
        if len(frame) != len(GROUPS[cohort]):
            raise RuntimeError(f"candidate matrix mismatch seed {seed}")
        c0 = frame[frame.policy_id == "C0"].iloc[0]
        for method, policies in choices.items():
            candidate_frame = frame[frame.policy_id.isin(policies)]
            if len(candidate_frame) != len(policies):
                raise RuntimeError(f"candidate count mismatch {seed}/{method}")
            winner, terminal, eligible_count = safe_select(c0, candidate_frame)
            selected.append({**winner.to_dict(), "seed": seed, "result_method": method, "terminal_policy": terminal, "safe_candidate_count": eligible_count, "fallback_to_c0": terminal == "C0"})
    outcomes = pd.DataFrame(selected)
    outcomes.to_csv(output / "selected_outcomes.csv", index=False)
    structure_lookup = {str(atoms.info["branch_id"]): atoms for atoms in read(output / "candidates_all.extxyz", index=":")}
    mixed = output / "mixed_evaluation_structures"
    mixed.mkdir(exist_ok=False)
    for method in choices:
        atoms_list = []
        for _, row in outcomes[outcomes.result_method == method].sort_values("seed").iterrows():
            if not bool(row.structure_valid):
                continue
            atoms = structure_lookup[str(row.branch_id)].copy()
            atoms.info.update(sample_seed=int(row.seed), selected_method=method, terminal_policy=str(row.terminal_policy), branch_id=f"{method}:{int(row.seed)}:{row.terminal_policy}")
            atoms_list.append(atoms)
        if atoms_list:
            write(mixed / f"{method}.extxyz", atoms_list)
    summaries = [json.loads((output / "generation" / str(seed) / "run_summary.json").read_text()) for seed in seeds]
    accounting = {
        "cohort": cohort, "n": len(seeds), "total_study_acquisition_score_calls": sum(int(row["total_acquisition_score_calls"]) for row in summaries),
        "total_study_gpu_hours": sum(float(row["primary_elapsed_seconds"]) + float(row["independent_elapsed_seconds"]) for row in summaries) / 3600,
        "max_peak_allocated_bytes": max(int(row["peak_allocated_bytes"]) for row in summaries),
        "per_method_deployment_multiplier": {method: BUDGETS[method] for method in choices},
        "per_method_chgnet_and_mattersim_selector_calls": {method: 1 + len(policies) for method, policies in choices.items()},
    }
    (output / "compute_accounting.json").write_text(json.dumps(accounting, indent=2) + "\n")


def mixed_quality(cohort: str, method: str, n: int) -> dict[str, float]:
    output = RESULTS / cohort
    atoms = read(output / "mixed_evaluation_structures" / f"{method}.extxyz", index=":")
    quality = output / "quality_mixed" / method
    per = pd.read_csv(quality / "per_structure.csv")
    with gzip.open(quality / "official_detailed.json.gz", "rt") as stream:
        detailed = json.load(stream)
    if len(per) != len(atoms):
        raise RuntimeError(f"mixed quality length mismatch {method}")
    return {
        "e_hull": float(np.mean(detailed["energy_above_hull_per_atom"])),
        "stable": float(np.mean(detailed["stable"])),
        "novel": float(np.mean(detailed["novel"])),
        "unique": float(np.mean(detailed["unique"])),
        "nus": float(np.mean(detailed["novel_unique_stable"])),
        "validity": len(atoms) / n,
        "max_force": float(per.pre_relaxation_max_force.mean()),
        "rmsd": float(np.mean(detailed["rmsd_from_relaxation"])),
    }


def bootstrap(gain: np.ndarray, seed: int) -> dict:
    gain = np.asarray(gain, float)
    if not np.isfinite(gain).all():
        raise RuntimeError("nonfinite paired gain")
    rng = np.random.default_rng(seed)
    draws = gain[rng.integers(0, len(gain), size=(N_BOOT, len(gain)))].mean(axis=1)
    return {
        "n": len(gain), "mean": float(gain.mean()), "median": float(np.median(gain)),
        "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)),
        "wins": int((gain > EPS).sum()), "ties": int((np.abs(gain) <= EPS).sum()),
        "losses": int((gain < -EPS).sum()), "resamples": N_BOOT,
    }


def final(cohort: str) -> None:
    output = RESULTS / cohort
    if (output / "table_best_of_n.csv").exists() or (output / "table_budget_scaling.csv").exists():
        raise FileExistsError("final statistics already frozen")
    seeds, _ = registry(cohort)
    outcomes = pd.read_csv(output / "selected_outcomes.csv")
    choices = methods(cohort)
    reference = "C0" if cohort == "search_baseline" else "K0"
    pivot = outcomes.pivot(index="seed", columns="result_method", values="property_absolute_error").sort_index()
    if len(pivot) != len(seeds) or set(pivot.index) != set(seeds):
        raise RuntimeError("paired matrix incomplete")
    comparisons = {}
    table = []
    base_quality = mixed_quality(cohort, reference, len(seeds))
    for index, method in enumerate(choices):
        frame = outcomes[outcomes.result_method == method]
        quality = mixed_quality(cohort, method, len(seeds))
        gain = pivot[reference].to_numpy(float) - pivot[method].to_numpy(float)
        interval = bootstrap(gain, BOOT_SEED + index)
        comparisons[f"{reference}_minus_{method}"] = interval
        checks = {
            "e_hull": quality["e_hull"] - base_quality["e_hull"] <= .01 + EPS,
            "stable": base_quality["stable"] - quality["stable"] <= .05 + EPS,
            "nus": base_quality["nus"] - quality["nus"] <= .05 + EPS,
            "validity": base_quality["validity"] - quality["validity"] <= .05 + EPS,
        }
        table.append({
            "Method": method, "Budget": BUDGETS[method], "Property MAE": float(frame.property_absolute_error.mean()),
            "95% CI paired gain vs C0": f"[{interval['ci95_low']:.8f}, {interval['ci95_high']:.8f}]",
            "W": interval["wins"], "T": interval["ties"], "L": interval["losses"],
            "Stable": quality["stable"], "NUS": quality["nus"], "E-hull": quality["e_hull"],
            "Validity": quality["validity"], "Novel": quality["novel"], "Unique": quality["unique"],
            "quality_guardrails_pass": bool(all(checks.values())),
        })
    if cohort == "search_baseline":
        comparisons["Independent_Best_of_2_minus_Fixed_K2"] = bootstrap(pivot["Independent_Best_of_2"].to_numpy(float) - pivot["Fixed_K2"].to_numpy(float), BOOT_SEED + 100)
    pd.DataFrame(table).to_csv(output / ("table_best_of_n.csv" if cohort == "search_baseline" else "table_budget_scaling.csv"), index=False)
    pivot.reset_index().to_csv(output / "paired_results.csv", index=False)
    (output / "bootstrap_20k.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    (output / "final_summary.json").write_text(json.dumps({"n": len(seeds), "comparisons": comparisons, "surrogate_only": True, "dft_verified": False}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=tuple(GROUPS), required=True)
    parser.add_argument("--stage", choices=("select", "final"), required=True)
    args = parser.parse_args()
    if args.stage == "select":
        select(args.cohort)
    else:
        final(args.cohort)
