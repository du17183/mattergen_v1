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
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
)

from research.rp_qtfg.common import REPORTS, RESULTS, atomic_json, now, set_stage
from research.rp_qtfg.experiment_config import (
    CONFIGS,
    EIGHT_SEED_CONFIGS,
    EIGHT_SEEDS,
    THIRTY_TWO_SEEDS,
)


RELAX = RESULTS / "relax"
GENERATION = RESULTS / "generation"
OUT_ROOT = REPORTS
REFERENCE_LMDB = Path(
    "/data/dxl/reference_assets/reference_TRI2024correction.lmdb"
)


def _bootstrap_ci(
    values: np.ndarray,
    repeats: int = 20000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    random = np.random.default_rng(20260726)
    indices = random.integers(0, len(values), size=(repeats, len(values)))
    low, high = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def _paired_stats(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str,
) -> dict[str, Any]:
    pair = baseline[["seed", metric]].merge(
        candidate[["seed", metric]],
        on="seed",
        suffixes=("_a0", "_candidate"),
    )
    difference = pair[f"{metric}_candidate"].to_numpy(float) - pair[
        f"{metric}_a0"
    ].to_numpy(float)
    low, high = _bootstrap_ci(difference)
    try:
        test = (
            wilcoxon(difference, zero_method="pratt")
            if np.any(difference)
            else None
        )
    except ValueError:
        test = None
    return {
        "metric": metric,
        "paired_n": len(difference),
        "difference_definition": "candidate_minus_baseline",
        "mean_difference": float(np.mean(difference)),
        "median_difference": float(np.median(difference)),
        "bootstrap_95_ci": [low, high],
        "wilcoxon_statistic": float(test.statistic) if test else 0.0,
        "wilcoxon_p_value": float(test.pvalue) if test else 1.0,
        "candidate_wins": int(np.sum(difference < -1e-12)),
        "ties": int(np.sum(np.abs(difference) <= 1e-12)),
        "candidate_losses": int(np.sum(difference > 1e-12)),
    }


def _reference() -> ReferenceDataset:
    def build_index_single_env(
        self: Any,
        _lmdb_path: Any,
    ) -> dict[str, Any]:
        result: dict[str, dict[str, Any]] = defaultdict(dict)
        with self.env.begin() as transaction:
            for chemical_system in lmdb_get(
                transaction,
                "chemical_systems",
            ):
                formulas = lmdb_get(
                    transaction,
                    f"{chemical_system}.reduced_formulas",
                )
                for formula in formulas:
                    result[chemical_system][formula] = lmdb_get(
                        transaction,
                        f"{chemical_system}.{formula}.length",
                    )
        return dict(result)

    LMDBBackedReferenceDatasetImpl._build_num_entries_by_chemsys_reduced_formulas = (
        build_index_single_env
    )
    return ReferenceDataset(
        name="TRI2024correction",
        impl=LMDBBackedReferenceDatasetImpl(
            REFERENCE_LMDB,
            cleanup_dir=False,
        ),
    )


def _result_frame(method: str, seeds: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        result_path = RELAX / method / str(seed) / "result.json"
        result = json.loads(result_path.read_text())
        if result.get("status") != "success":
            raise RuntimeError(f"unsuccessful experiment result: {result_path}")
        generation_dir = GENERATION / method / str(seed)
        generation = json.loads((generation_dir / "run_summary.json").read_text())
        hashes = json.loads((generation_dir / "structure_hashes.json").read_text())
        trace_path = generation_dir / "rp_qtfg_summary.json"
        trace = json.loads(trace_path.read_text()) if trace_path.is_file() else {}
        atoms = ase.io.read(result["input_path"])
        if len(atoms) > 1:
            distances = atoms.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf)
            minimum_distance = float(np.min(distances))
        else:
            minimum_distance = math.inf
        result.update({
            "generation_success": bool(generation["success"]),
            "generation_elapsed_seconds": float(generation["elapsed_seconds"]),
            "generation_composition_valid": bool(generation["composition_valid"]),
            "generation_structure_valid": bool(generation["basic_structure_valid"]),
            "initial_state_hash": hashes["initial_state_hash"],
            "minimum_distance_angstrom": minimum_distance,
            "chgnet_forward_count": int(trace.get("chgnet_forward_count", 0)),
            "chgnet_backward_count": int(trace.get("chgnet_backward_count", 0)),
            "backtracking_count": int(trace.get("backtracking_count", 0)),
            "fallback_count": int(trace.get("fallback_count", 0)),
            "conflict_count": int(trace.get("conflict_count", 0)),
            "clipping_count": int(trace.get("clipping_count", 0)),
            "eligible_calls": int(trace.get("eligible_calls", 0)),
            "guided_calls": int(trace.get("guided_calls", 0)),
            "atomic_numbers_modified": bool(trace.get("atomic_numbers_modified", False)),
        })
        rows.append(result)
    return pd.DataFrame(rows)


def _configs_and_seeds(mode: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if mode == "eight":
        return EIGHT_SEED_CONFIGS, EIGHT_SEEDS
    if mode == "thirtytwo":
        selected_path = REPORTS / "eight_seed/selected_candidate.json"
        selected = str(json.loads(selected_path.read_text())["selected_config"])
        return ("A0", selected), THIRTY_TWO_SEEDS
    raise ValueError(mode)


def _official_frame(
    method: str,
    raw: pd.DataFrame,
    reference: ReferenceDataset,
    seeds: tuple[int, ...],
    out: Path,
) -> pd.DataFrame:
    cached = out / f"{method}_per_structure.csv"
    if cached.is_file():
        frame = pd.read_csv(cached)
        if tuple(frame.seed.astype(int)) == seeds:
            return frame
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
    keep = ["seed", "energy_above_hull_per_atom", "rmsd_from_relaxation", "stable", "comp_validity", "structure_validity", "novel", "unique"]
    return raw.merge(official[keep], on="seed", validate="one_to_one")


def _summary(method: str, frame: pd.DataFrame) -> dict[str, Any]:
    eligible = int(frame.eligible_calls.sum())
    field_opportunities = eligible * (2 if method != "A0" and CONFIGS[method].guidance_fields == "position_cell" else 1)
    nus = frame.stable.astype(bool) & frame.novel.astype(bool) & frame.unique.astype(bool)
    return {
        "method": method, "n": len(frame),
        "generation_success": float(frame.generation_success.mean()),
        "generation_elapsed_median": float(frame.generation_elapsed_seconds.median()),
        "initial_energy_per_atom_mean": float(frame.initial_energy_per_atom_ev.mean()),
        "initial_max_force_mean": float(frame.initial_max_force_ev_ang.mean()),
        "relaxation_rmsd_mean": float(frame.rmsd_from_relaxation.mean()),
        "average_ehull": float(frame.energy_above_hull_per_atom.mean()),
        "stable_rate": float(frame.stable.mean()),
        "composition_validity": float(frame.comp_validity.mean()),
        "structure_validity": float(frame.structure_validity.mean()),
        "novel_rate": float(frame.novel.mean()), "unique_rate": float(frame.unique.mean()),
        "nus_rate": float(nus.mean()), "force_convergence_rate": float(frame.converged.mean()),
        "relaxation_failure_rate": 0.0,
        "severe_short_bond_count": int((frame.minimum_distance_angstrom < 0.5).sum()),
        "atomic_numbers_modified": bool(frame.atomic_numbers_modified.any()),
        "chgnet_forward_count": int(frame.chgnet_forward_count.sum()),
        "chgnet_backward_count": int(frame.chgnet_backward_count.sum()),
        "backtracking_mean": float(frame.backtracking_count.mean()),
        "fallback_rate": float(frame.fallback_count.sum() / max(eligible, 1)),
        "conflict_rate": float(frame.conflict_count.sum() / max(field_opportunities, 1)),
        "clipping_rate": float(frame.clipping_count.sum() / max(field_opportunities, 1)),
    }


def _boolean_paired(baseline: pd.DataFrame, candidate: pd.DataFrame, metric: str) -> dict[str, Any]:
    pair = baseline[["seed", metric]].merge(candidate[["seed", metric]], on="seed", suffixes=("_a0", "_candidate"))
    left = pair[f"{metric}_a0"].astype(bool).to_numpy()
    right = pair[f"{metric}_candidate"].astype(bool).to_numpy()
    gains = int(np.sum(np.logical_not(left) & right))
    losses = int(np.sum(left & np.logical_not(right)))
    discordant = gains + losses
    p_value = float(binomtest(min(gains, losses), n=discordant, p=0.5).pvalue) if discordant else 1.0
    return {"metric": metric, "paired_n": len(pair), "gains": gains, "losses": losses, "ties": len(pair) - discordant, "exact_p_value": p_value}


def _comparison(baseline: pd.DataFrame, candidate: pd.DataFrame, baseline_summary: dict[str, Any], candidate_summary: dict[str, Any], mode: str) -> dict[str, Any]:
    pair = baseline[["seed", "initial_energy_per_atom_ev", "initial_max_force_ev_ang", "rmsd_from_relaxation", "energy_above_hull_per_atom"]].merge(
        candidate[["seed", "initial_energy_per_atom_ev", "initial_max_force_ev_ang", "rmsd_from_relaxation", "energy_above_hull_per_atom"]],
        on="seed", suffixes=("_a0", "_candidate"), validate="one_to_one",
    )
    energy_better = pair.initial_energy_per_atom_ev_candidate < pair.initial_energy_per_atom_ev_a0 - 1e-8
    force_better = pair.initial_max_force_ev_ang_candidate < pair.initial_max_force_ev_ang_a0 - 1e-8
    rmsd_better = pair.rmsd_from_relaxation_candidate < pair.rmsd_from_relaxation_a0 - 1e-8
    any_better = energy_better | force_better | rmsd_better
    changes = {key: candidate_summary[key] - baseline_summary[key] for key in ("average_ehull", "stable_rate", "composition_validity", "structure_validity", "novel_rate", "unique_rate", "nus_rate", "relaxation_failure_rate")}
    rmsd_relative = candidate_summary["relaxation_rmsd_mean"] / baseline_summary["relaxation_rmsd_mean"] - 1.0
    force_relative = candidate_summary["initial_max_force_mean"] / baseline_summary["initial_max_force_mean"] - 1.0
    latency_relative = candidate_summary["generation_elapsed_median"] / baseline_summary["generation_elapsed_median"] - 1.0
    common_safe = (
        candidate_summary["generation_success"] == 1.0
        and candidate_summary["structure_validity"] >= baseline_summary["structure_validity"]
        and candidate_summary["severe_short_bond_count"] <= baseline_summary["severe_short_bond_count"]
        and not candidate_summary["atomic_numbers_modified"]
        and changes["average_ehull"] <= 0.002
    )
    clear_direction = bool((rmsd_relative <= -0.03 and float(rmsd_better.mean()) >= 0.625) or (force_relative <= -0.03 and float(force_better.mean()) >= 0.625))
    if mode == "eight":
        gate_go = bool(common_safe and candidate_summary["structure_validity"] == 1.0 and changes["composition_validity"] >= 0.0 and changes["stable_rate"] >= 0.0 and clear_direction)
    else:
        safety = bool(common_safe and changes["composition_validity"] >= -1.0 / 32.0 and changes["stable_rate"] >= -1.0 / 32.0 and changes["nus_rate"] >= -1.0 / 32.0 and changes["novel_rate"] >= -0.02 and changes["unique_rate"] >= -0.02 and changes["relaxation_failure_rate"] <= 0.0)
        positive = bool(changes["average_ehull"] <= -0.005 or changes["stable_rate"] >= 1.0 / 32.0 or changes["nus_rate"] >= 1.0 / 32.0 or rmsd_relative <= -0.10 or force_relative <= -0.10)
        gate_go = safety and positive
    return {
        "method": candidate_summary["method"],
        "matterSim_energy_improvement_rate": float(energy_better.mean()),
        "matterSim_force_improvement_rate": float(force_better.mean()),
        "relaxation_rmsd_improvement_rate": float(rmsd_better.mean()),
        "any_primary_improvement_rate": float(any_better.mean()),
        "ehull_change_candidate_minus_a0": float(changes["average_ehull"]),
        "stable_change": float(changes["stable_rate"]), "composition_change": float(changes["composition_validity"]),
        "structure_change": float(changes["structure_validity"]), "novel_change": float(changes["novel_rate"]),
        "unique_change": float(changes["unique_rate"]), "nus_change": float(changes["nus_rate"]),
        "rmsd_relative_change": float(rmsd_relative),
        "pre_relax_max_force_relative_change": float(force_relative),
        "latency_overhead": float(latency_relative),
        "latency_risk_over_30_percent": bool(latency_relative > 0.30),
        "clear_improvement_direction": clear_direction, "clear_improvement_rule": "mean RMSD or pre-relax max force improves at least 3 percent and at least 5 of 8 pairs improve", "gate_go": bool(gate_go),
    }


def run(mode: str) -> dict[str, Any]:
    atomic_json(
        REPORTS / "launcher.json",
        {
            "pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgid(0),
            "sid": os.getsid(0), "user": os.environ.get("USER", ""),
            "cwd": os.getcwd(), "exe": os.readlink(f"/proc/{os.getpid()}/exe"),
            "command": " ".join(os.sys.argv), "mode": mode, "started_at": now(),
        },
    )
    configs, seeds = _configs_and_seeds(mode)
    out = OUT_ROOT / ("eight_seed" if mode == "eight" else "thirty_two")
    out.mkdir(parents=True, exist_ok=True)
    reference = _reference()
    frames: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    for method in configs:
        raw = _result_frame(method, seeds)
        frame = _official_frame(method, raw, reference, seeds, out)
        frame.to_csv(out / f"{method}_per_structure.csv", index=False)
        frames[method] = frame
        summaries.append(_summary(method, frame))
    try:
        reference.impl.cleanup(cleanup_dir=False)
    except Exception:
        pass
    baseline = frames["A0"]
    baseline_summary = next(row for row in summaries if row["method"] == "A0")
    initial_hash_mismatches = []
    for seed in seeds:
        hashes = {method: str(frames[method].loc[frames[method].seed == seed, "initial_state_hash"].iloc[0]) for method in configs}
        if len(set(hashes.values())) != 1:
            initial_hash_mismatches.append({"seed": seed, "hashes": hashes})
    comparisons: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for method in configs[1:]:
        candidate = frames[method]
        candidate_summary = next(row for row in summaries if row["method"] == method)
        comparison = _comparison(baseline, candidate, baseline_summary, candidate_summary, mode)
        if initial_hash_mismatches:
            comparison["gate_go"] = False
        comparisons.append(comparison)
        for metric in ("initial_energy_per_atom_ev", "initial_max_force_ev_ang", "rmsd_from_relaxation", "energy_above_hull_per_atom"):
            stats = _paired_stats(baseline, candidate, metric)
            stats["method"] = method
            paired.append(stats)
        for metric in ("stable", "comp_validity", "structure_validity", "novel", "unique"):
            stats = _boolean_paired(baseline, candidate, metric)
            stats["method"] = method
            paired.append(stats)
    passing = [row for row in comparisons if row["gate_go"]]
    selected = None
    if mode == "eight" and passing:
        selected = min(passing, key=lambda row: (row["rmsd_relative_change"] + row["pre_relax_max_force_relative_change"] + 10.0 * row["ehull_change_candidate_minus_a0"], row["latency_overhead"]))["method"]
    elif mode == "thirtytwo":
        selected = configs[1]
    gate_go = bool(passing) if mode == "eight" else bool(comparisons[0]["gate_go"])
    result = {
        "created_at": now(), "mode": mode, "seeds": list(seeds),
        "initial_state_hashes_match": not initial_hash_mismatches,
        "initial_state_hash_mismatches": initial_hash_mismatches,
        "selected_config": selected,
        "RP_QTFG_EIGHT_SEED_GO": gate_go if mode == "eight" else True,
        "RP_QTFG_MVP_GO": gate_go if mode == "thirtytwo" else None,
        "RP_QTFG_MVP_NO_GO": (not gate_go) if mode == "thirtytwo" else None,
        "summaries": summaries, "comparisons": comparisons, "paired_statistics": paired,
        "interpretation": {"stability_source": "MatterSim-5M surrogate", "dft_verified": False, "magnetic_property_verified": False, "latency_target": "<=30 percent overhead is advisory and reported as an efficiency risk"},
    }
    atomic_json(out / "analysis_report.json", result)
    atomic_json(out / "paired_statistics.json", paired)
    pd.DataFrame(summaries).to_csv(out / "method_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "comparisons.csv", index=False)
    if mode == "eight":
        atomic_json(out / "selected_candidate.json", {"created_at": now(), "selected_config": selected, "config": CONFIGS[selected].as_dict() if selected else None, "RP_QTFG_EIGHT_SEED_GO": gate_go})
    report = [
        f"# RP-QTFG {mode} paired evaluation", "",
        f"- Seeds: {seeds[0]}–{seeds[-1]} ({len(seeds)} per method).",
        f"- Initial-state pairing passed: `{not initial_hash_mismatches}`.",
        f"- Selected config: `{selected}`.", f"- Gate decision: `{gate_go}`.",
        "- Evaluator: independent MatterSim-5M with TRI2024 compatibility; DFT verified: false.",
        "", "## Method summary", "", pd.DataFrame(summaries).to_markdown(index=False),
        "", "## Comparisons", "", pd.DataFrame(comparisons).to_markdown(index=False),
    ]
    report_name = "eight_seed_report.md" if mode == "eight" else "mvp_report.md"
    (out / report_name).write_text("\n".join(report) + "\n", encoding="utf-8")
    if mode == "eight":
        set_stage("eight_seed_review", "success" if gate_go else "stop_for_review", f"8-seed review selected {selected}." if gate_go else "8-seed No-Go: no candidate met every frozen smoke gate.", {"RP_QTFG_EIGHT_SEED_GO": gate_go, "selected": selected})
    else:
        set_stage("metrics", "success", "32-seed paired metrics and statistics completed.", {"methods": list(configs), "seeds": len(seeds)})
        set_stage("mvp_go_no_go", "stop_for_review", f"RP-QTFG MVP decision: {gate_go}.", {"RP_QTFG_MVP_GO": gate_go, "RP_QTFG_MVP_NO_GO": not gate_go})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eight", "thirtytwo"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.mode), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
