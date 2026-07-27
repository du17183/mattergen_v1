"""Official metrics, paired statistics, and Fast Gate quality decisions."""

from __future__ import annotations

import json
import math
import subprocess
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
from research.spg_fastgate.common import (
    REPORTS,
    RESULTS,
    PROJECT,
    atomic_json,
    atomic_text,
    base_environment,
    now,
    read_json,
    set_stage,
)


REFERENCE_LMDB = Path("/data/dxl/reference_assets/reference_TRI2024correction.lmdb")
QUALITY_CONFIGS = ("C0_B1", "C0_B4", "A0_B1", "A0_B4")
QUALITY_SEEDS = tuple(range(24064, 24128))
BF16_CONFIGS = (
    "C0_FP32",
    "C0_FIELD_SAFE_BF16",
    "A0_FP32",
    "A0_FIELD_SAFE_BF16",
)
BF16_SEEDS = tuple(range(24128, 24136))
CONTINUOUS_METRICS = (
    "energy_above_hull_per_atom",
    "rmsd_from_relaxation",
    "initial_energy_per_atom_ev",
    "initial_max_force_ev_ang",
    "target_error",
)
BOOLEAN_METRICS = (
    "stable",
    "metastable_le_0_2",
    "comp_validity",
    "structure_validity",
    "novel",
    "unique",
    "nus",
    "msun_le_0_2",
    "converged",
    "hit_0.02",
)


def bootstrap_ci(values: np.ndarray, repeats: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    random = np.random.default_rng(20260727)
    indices = random.integers(0, len(values), size=(repeats, len(values)))
    low, high = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def reference_dataset() -> ReferenceDataset:
    def build_index_single_env(self: Any, _lmdb_path: Any) -> dict[str, Any]:
        result: dict[str, dict[str, Any]] = defaultdict(dict)
        with self.env.begin() as transaction:
            for chemical_system in lmdb_get(transaction, "chemical_systems"):
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
        impl=LMDBBackedReferenceDatasetImpl(REFERENCE_LMDB, cleanup_dir=False),
    )


def config_metadata(config: str) -> tuple[str, int, str, str]:
    if "_B" in config:
        method, batch = config.split("_B", maxsplit=1)
        return "quality", int(batch), method, "FP32"
    method, precision = config.split("_", maxsplit=1)
    return "bf16", 1, method, precision


def raw_frame(config: str, seeds: tuple[int, ...]) -> pd.DataFrame:
    family, batch_size, method, precision = config_metadata(config)
    rows = []
    for seed in seeds:
        result_path = RESULTS / "relaxed" / family / config / f"seed_{seed}/result.json"
        result = read_json(result_path)
        if result.get("status") != "success":
            raise RuntimeError(f"invalid relaxation result: {result_path}")
        if family == "quality":
            generation_dir = (
                RESULTS
                / "quality_generation"
                / method
                / f"B{batch_size}"
                / f"seed_{seed}"
            )
        else:
            generation_dir = (
                RESULTS
                / "bf16_generation"
                / precision
                / method
                / "B1"
                / f"seed_{seed}"
            )
        generation = read_json(generation_dir / "summary.json")
        atoms = ase.io.read(result["input_path"])
        if len(atoms) > 1:
            distances = atoms.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf)
            minimum_distance = float(np.min(distances))
        else:
            minimum_distance = math.inf
        rows.append(
            {
                **result,
                "config": config,
                "method": method,
                "batch_size": batch_size,
                "precision": precision,
                "generation_success": bool(generation["success"]),
                "generation_elapsed_seconds": float(
                    generation["generation_seconds_per_sample"]
                ),
                "generation_composition_valid": bool(
                    generation["composition_valid"]
                ),
                "generation_structure_valid": bool(
                    generation["structure_valid"]
                ),
                "initial_state_hash": generation["initial_hashes"]["combined"],
                "minimum_distance_angstrom": minimum_distance,
            }
        )
    return pd.DataFrame(rows)


def official_frame(
    config: str,
    raw: pd.DataFrame,
    reference: ReferenceDataset,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    output = REPORTS / "quality"
    cached = output / f"{config}_official_base.csv"
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
        original_structures=[
            AseAtomsAdaptor.get_structure(atoms) for atoms in originals
        ],
        stability_threshold=0.1,
        structure_matcher=DefaultDisorderedStructureMatcher(),
        energy_correction_scheme=TRI110Compatibility2024(),
        n_failed_jobs=0,
    )
    method_output = output / config
    method_output.mkdir(parents=True, exist_ok=True)
    evaluator.compute_metrics("all", save_as=method_output / "official_metrics.json")
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
    merged = raw.merge(official[keep], on="seed", validate="one_to_one")
    merged.to_csv(cached, index=False)
    return merged


def magnetic_metrics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tasks = []
    for config, frame in frames.items():
        for row in frame.itertuples():
            tasks.append(
                {
                    "config": config,
                    "seed": int(row.seed),
                    "structure_path": str(row.input_path),
                }
            )
    output_path = REPORTS / "quality/magnetic_metrics.csv"
    expected = {(row["config"], row["seed"]) for row in tasks}
    if output_path.is_file():
        cached = pd.read_csv(output_path)
        observed = set(zip(cached.config, cached.seed.astype(int), strict=True))
        if observed == expected:
            return cached
    manifest = REPORTS / "quality/chgnet_magnetic_manifest.json"
    atomic_json(manifest, {"tasks": tasks, "target_density": 0.10})
    teacher_python = Path("/data/dxl/envs/fn_pra_teacher/bin/python")
    if not teacher_python.is_file():
        raise RuntimeError(f"CHGNet teacher Python not found: {teacher_python}")
    subprocess.run(
        [
            str(teacher_python),
            "-m",
            "research.spg_fastgate.chgnet_magnetic_worker",
            "--manifest",
            str(manifest),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT,
        env=base_environment(0),
        check=True,
    )
    output = pd.read_csv(output_path)
    observed = set(zip(output.config, output.seed.astype(int), strict=True))
    if observed != expected:
        raise RuntimeError("CHGNet magnetic worker output does not match manifest")
    return output


def summary(config: str, frame: pd.DataFrame) -> dict[str, Any]:
    nus = frame.stable.astype(bool) & frame.novel.astype(bool) & frame.unique.astype(bool)
    metastable = frame.energy_above_hull_per_atom <= 0.2
    msun = metastable & frame.novel.astype(bool) & frame.unique.astype(bool)
    return {
        "config": config,
        "n": len(frame),
        "generation_success": float(frame.generation_success.mean()),
        "generation_elapsed_median": float(frame.generation_elapsed_seconds.median()),
        "generation_composition_validity": float(
            frame.generation_composition_valid.mean()
        ),
        "generation_structure_validity": float(
            frame.generation_structure_valid.mean()
        ),
        "average_ehull": float(frame.energy_above_hull_per_atom.mean()),
        "median_ehull": float(frame.energy_above_hull_per_atom.median()),
        "stable_rate": float(frame.stable.mean()),
        "metastable_le_0_2_rate": float(metastable.mean()),
        "nus_rate": float(nus.mean()),
        "msun_le_0_2_rate": float(msun.mean()),
        "novel_rate": float(frame.novel.mean()),
        "unique_rate": float(frame.unique.mean()),
        "rmsd_mean": float(frame.rmsd_from_relaxation.mean()),
        "rmsd_median": float(frame.rmsd_from_relaxation.median()),
        "pre_relax_max_force_mean": float(frame.initial_max_force_ev_ang.mean()),
        "pre_relax_max_force_median": float(frame.initial_max_force_ev_ang.median()),
        "force_convergence_rate": float(frame.converged.mean()),
        "relaxation_failure_rate": 0.0,
        "hit_0_01": float(frame["hit_0.01"].mean()),
        "hit_0_02": float(frame["hit_0.02"].mean()),
        "hit_0_05": float(frame["hit_0.05"].mean()),
        "target_error_mean": float(frame.target_error.mean()),
        "severe_short_bond_count": int(
            (frame.minimum_distance_angstrom < 0.5).sum()
        ),
    }


def paired_stats(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str,
    comparison: str,
) -> dict[str, Any]:
    pair = baseline[["seed", metric]].merge(
        candidate[["seed", metric]],
        on="seed",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    difference = (
        pair[f"{metric}_candidate"].to_numpy(float)
        - pair[f"{metric}_baseline"].to_numpy(float)
    )
    low, high = bootstrap_ci(difference)
    try:
        test = wilcoxon(difference, zero_method="pratt") if np.any(difference) else None
    except ValueError:
        test = None
    leave_one_out = [
        float(np.delete(difference, index).mean())
        for index in range(len(difference))
    ]
    denominator = float(np.abs(difference).sum())
    return {
        "comparison": comparison,
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
        "maximum_single_sample_contribution": (
            float(np.abs(difference).max() / denominator)
            if denominator > 0
            else 0.0
        ),
        "leave_one_out_mean_min": min(leave_one_out) if leave_one_out else math.nan,
        "leave_one_out_mean_max": max(leave_one_out) if leave_one_out else math.nan,
    }


def boolean_stats(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str,
    comparison: str,
) -> dict[str, Any]:
    left = baseline.sort_values("seed")[metric].astype(bool).to_numpy()
    right = candidate.sort_values("seed")[metric].astype(bool).to_numpy()
    gains = int(np.sum(~left & right))
    losses = int(np.sum(left & ~right))
    discordant = gains + losses
    return {
        "comparison": comparison,
        "metric": metric,
        "paired_n": len(left),
        "gains": gains,
        "losses": losses,
        "ties": len(left) - discordant,
        "exact_p_value": (
            float(binomtest(min(gains, losses), n=discordant, p=0.5).pvalue)
            if discordant
            else 1.0
        ),
    }


def enrich_flags(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["metastable_le_0_2"] = frame.energy_above_hull_per_atom <= 0.2
    frame["nus"] = (
        frame.stable.astype(bool)
        & frame.novel.astype(bool)
        & frame.unique.astype(bool)
    )
    frame["msun_le_0_2"] = (
        frame.metastable_le_0_2
        & frame.novel.astype(bool)
        & frame.unique.astype(bool)
    )
    return frame


def quality_decision(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_summary: dict,
    candidate_summary: dict,
    throughput_speedup: float,
) -> dict[str, Any]:
    ehull_change = candidate_summary["average_ehull"] - baseline_summary["average_ehull"]
    stable_change = candidate_summary["stable_rate"] - baseline_summary["stable_rate"]
    nus_change = candidate_summary["nus_rate"] - baseline_summary["nus_rate"]
    composition_change = (
        candidate_summary["generation_composition_validity"]
        - baseline_summary["generation_composition_validity"]
    )
    structure_change = (
        candidate_summary["generation_structure_validity"]
        - baseline_summary["generation_structure_validity"]
    )
    rmsd_pair = baseline[["seed", "rmsd_from_relaxation"]].merge(
        candidate[["seed", "rmsd_from_relaxation"]],
        on="seed",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    rmsd_difference = (
        rmsd_pair.rmsd_from_relaxation_candidate.to_numpy()
        - rmsd_pair.rmsd_from_relaxation_baseline.to_numpy()
    )
    rmsd_ci = bootstrap_ci(rmsd_difference)
    rmsd_no_systematic_worsening = bool(
        float(rmsd_difference.mean()) <= 0.01
        and not (rmsd_ci[0] > 0.0 and float(rmsd_difference.mean()) > 0.0)
    )
    hit_change = candidate_summary["hit_0_02"] - baseline_summary["hit_0_02"]
    initial_hash_match = bool(
        np.array_equal(
            baseline.sort_values("seed").initial_state_hash.to_numpy(),
            candidate.sort_values("seed").initial_state_hash.to_numpy(),
        )
    )
    gates = {
        "initial_random_tape_match": initial_hash_match,
        "generation_success_not_lower": candidate_summary["generation_success"]
        >= baseline_summary["generation_success"],
        "structure_validity_not_lower": structure_change >= 0.0,
        "composition_decline_le_1_of_64": composition_change >= -1.0 / 64.0,
        "ehull_degradation_le_0_002": ehull_change <= 0.002,
        "stable_decline_le_1_of_64": stable_change >= -1.0 / 64.0,
        "nus_decline_le_1_of_64": nus_change >= -1.0 / 64.0,
        "rmsd_no_systematic_worsening": rmsd_no_systematic_worsening,
        "relaxation_failure_not_increased": True,
        "hit_0_02_decline_le_1_of_64": hit_change >= -1.0 / 64.0,
        "throughput_ge_2x": throughput_speedup >= 2.0,
    }
    return {
        "baseline": baseline_summary["config"],
        "candidate": candidate_summary["config"],
        "throughput_speedup": throughput_speedup,
        "ehull_change": ehull_change,
        "stable_change": stable_change,
        "nus_change": nus_change,
        "composition_change": composition_change,
        "structure_change": structure_change,
        "rmsd_mean_change_angstrom": float(rmsd_difference.mean()),
        "rmsd_bootstrap_95_ci": list(rmsd_ci),
        "hit_0_02_change": hit_change,
        "gates": gates,
        "quality_equivalent": all(gates.values()),
    }


def bf16_decision(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_summary: dict,
    candidate_summary: dict,
) -> dict[str, Any]:
    speedup = (
        baseline_summary["generation_elapsed_median"]
        / candidate_summary["generation_elapsed_median"]
    )
    gates = {
        "speedup_ge_1_05": speedup >= 1.05,
        "generation_success_not_lower": candidate_summary["generation_success"]
        >= baseline_summary["generation_success"],
        "structure_not_lower": candidate_summary["generation_structure_validity"]
        >= baseline_summary["generation_structure_validity"],
        "composition_not_lower": candidate_summary["generation_composition_validity"]
        >= baseline_summary["generation_composition_validity"],
        "ehull_degradation_le_0_002": candidate_summary["average_ehull"]
        - baseline_summary["average_ehull"]
        <= 0.002,
        "stable_not_lower": candidate_summary["stable_rate"]
        >= baseline_summary["stable_rate"],
        "nus_not_lower": candidate_summary["nus_rate"] >= baseline_summary["nus_rate"],
        "rmsd_not_obviously_worse": candidate_summary["rmsd_mean"]
        - baseline_summary["rmsd_mean"]
        <= 0.01,
    }
    return {
        "baseline": baseline_summary["config"],
        "candidate": candidate_summary["config"],
        "endpoint_speedup": speedup,
        "gates": gates,
        "go": all(gates.values()),
    }


def main() -> int:
    set_stage(
        "b4_quality_metrics",
        "running",
        "Computing official metrics, CHGNet magnetic proxy, and paired equivalence statistics.",
    )
    reference = reference_dataset()
    frames = {}
    for config in QUALITY_CONFIGS:
        frames[config] = official_frame(
            config,
            raw_frame(config, QUALITY_SEEDS),
            reference,
            QUALITY_SEEDS,
        )
    state_probe = read_json(RESULTS / "bf16_state_probe.json", {})
    if state_probe.get("FIELD_SAFE_BF16_STATE_GO"):
        for config in BF16_CONFIGS:
            frames[config] = official_frame(
                config,
                raw_frame(config, BF16_SEEDS),
                reference,
                BF16_SEEDS,
            )
    try:
        reference.impl.cleanup(cleanup_dir=False)
    except Exception:
        pass
    magnetic = magnetic_metrics(frames)
    for config, frame in list(frames.items()):
        frame = frame.merge(
            magnetic[magnetic.config == config].drop(columns=["config"]),
            on="seed",
            validate="one_to_one",
        )
        frame = enrich_flags(frame)
        frame.to_csv(REPORTS / "quality" / f"{config}_per_structure.csv", index=False)
        frames[config] = frame
    summaries = {config: summary(config, frame) for config, frame in frames.items()}
    performance = read_json(RESULTS / "performance_baseline.json")
    performance_index = {
        (row["method"], int(row["batch_size"])): row
        for row in performance["rows"]
    }
    decisions = {}
    paired = []
    for method in ("C0", "A0"):
        baseline_name = f"{method}_B1"
        candidate_name = f"{method}_B4"
        comparison = f"{candidate_name}_vs_{baseline_name}"
        speedup = (
            performance_index[(method, 4)]["fixed8_samples_per_hour"]
            / performance_index[(method, 1)]["fixed8_samples_per_hour"]
        )
        decisions[method] = quality_decision(
            frames[baseline_name],
            frames[candidate_name],
            summaries[baseline_name],
            summaries[candidate_name],
            speedup,
        )
        for metric in CONTINUOUS_METRICS:
            paired.append(
                paired_stats(
                    frames[baseline_name],
                    frames[candidate_name],
                    metric,
                    comparison,
                )
            )
        for metric in BOOLEAN_METRICS:
            paired.append(
                boolean_stats(
                    frames[baseline_name],
                    frames[candidate_name],
                    metric,
                    comparison,
                )
            )
    native_go = bool(
        decisions["C0"]["quality_equivalent"]
        and decisions["A0"]["quality_equivalent"]
    )
    bf16_decisions = {}
    if state_probe.get("FIELD_SAFE_BF16_STATE_GO"):
        for method in ("C0", "A0"):
            baseline_name = f"{method}_FP32"
            candidate_name = f"{method}_FIELD_SAFE_BF16"
            bf16_decisions[method] = bf16_decision(
                frames[baseline_name],
                frames[candidate_name],
                summaries[baseline_name],
                summaries[candidate_name],
            )
            comparison = f"{candidate_name}_vs_{baseline_name}"
            for metric in CONTINUOUS_METRICS:
                paired.append(
                    paired_stats(
                        frames[baseline_name],
                        frames[candidate_name],
                        metric,
                        comparison,
                    )
                )
            for metric in BOOLEAN_METRICS:
                paired.append(
                    boolean_stats(
                        frames[baseline_name],
                        frames[candidate_name],
                        metric,
                        comparison,
                    )
                )
    bf16_go = bool(
        bf16_decisions
        and all(row["go"] for row in bf16_decisions.values())
    )
    result = {
        "created_at": now(),
        "quality_seeds": list(QUALITY_SEEDS),
        "bf16_seeds": list(BF16_SEEDS),
        "summaries": summaries,
        "quality_decisions": decisions,
        "bf16_decisions": bf16_decisions,
        "paired_statistics": paired,
        "C0_B4_QUALITY_EQUIVALENT": decisions["C0"]["quality_equivalent"],
        "A0_B4_QUALITY_EQUIVALENT": decisions["A0"]["quality_equivalent"],
        "B4_THROUGHPUT_GO": all(
            row["throughput_speedup"] >= 2.0 for row in decisions.values()
        ),
        "NATIVE_BATCHING_GO": native_go,
        "FIELD_SAFE_BF16_GO": bf16_go,
        "STABILITY_SOURCE": "MatterSim-5M surrogate",
        "DFT_VERIFIED": False,
        "MAGNETIC_PROPERTY_DFT_VERIFIED": False,
    }
    atomic_json(REPORTS / "b4_quality_report.json", result)
    atomic_json(REPORTS / "paired_statistics.json", paired)
    pd.DataFrame(summaries.values()).to_csv(
        RESULTS / "quality_metrics.csv", index=False
    )
    report_lines = [
        "# SPG Fast Gate B4 quality equivalence",
        "",
        f"- C0 B4 equivalent: `{result['C0_B4_QUALITY_EQUIVALENT']}`",
        f"- A0 B4 equivalent: `{result['A0_B4_QUALITY_EQUIVALENT']}`",
        f"- Native batching GO: `{native_go}`",
        f"- Field-safe BF16 GO: `{bf16_go}`",
        "",
        "## Method summary",
        "",
        pd.DataFrame(summaries.values()).to_markdown(index=False),
        "",
        "## B4 decisions",
        "",
        "```json",
        json.dumps(decisions, indent=2, ensure_ascii=False),
        "```",
        "",
        "MatterSim-5M and CHGNet magnetic density are surrogate evaluations; no DFT verification was performed.",
    ]
    atomic_text(REPORTS / "b4_quality_report.md", "\n".join(report_lines) + "\n")
    atomic_text(
        REPORTS / "bf16_report.md",
        "# Field-safe BF16 endpoint gate\n\n"
        f"- FIELD_SAFE_BF16_STATE_GO: `{state_probe.get('FIELD_SAFE_BF16_STATE_GO', False)}`\n"
        f"- FIELD_SAFE_BF16_GO: `{bf16_go}`\n\n"
        f"```json\n{json.dumps(bf16_decisions, indent=2, ensure_ascii=False)}\n```\n",
    )
    set_stage(
        "b4_quality_metrics",
        "success",
        f"B4 quality decisions completed; native batching GO={native_go}.",
        result,
    )
    set_stage(
        "bf16_eight_seed",
        "success",
        f"BF16 endpoint decision completed; FIELD_SAFE_BF16_GO={bf16_go}.",
        {"FIELD_SAFE_BF16_GO": bf16_go, "decisions": bf16_decisions},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
