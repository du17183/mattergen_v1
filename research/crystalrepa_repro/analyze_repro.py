from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from statistics import median

import ase.io
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from research.crystalrepa_repro.common import REPORTS, RESULTS, atomic_json, atomic_text, now, set_stage

TOOLS = Path("/data/dxl/tools/innovation2_next")
sys.path.insert(0, str(TOOLS))
import analyze_corrector_32 as legacy  # noqa: E402

METHODS = ("U0", "R1")
SEEDS = tuple(range(17000, 17064))
RELAX_JSON = RESULTS / "progress/mattersim_relax_progress.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_rows() -> tuple[dict[str, pd.DataFrame], dict[str, list[dict]]]:
    state = read_json(RELAX_JSON)
    if len(state["tasks"]) != 128 or any(task["status"] != "success" for task in state["tasks"]):
        raise RuntimeError("Analysis requires 128 successful relaxation tasks")
    frames, generation = {}, {}
    for method in METHODS:
        rows, gen_rows = [], []
        for seed in SEEDS:
            task = next(item for item in state["tasks"] if item["config"] == method and int(item["seed"]) == seed)
            relax = read_json(Path(task["output_path"]).parent / "relax_summary.json")
            generated = RESULTS / "generation" / method / f"seed_{seed}"
            gen = read_json(generated / "summary.json")
            relaxed_atoms = ase.io.read(task["output_path"])
            original_atoms = ase.io.read(task["input_path"])
            structure = AseAtomsAdaptor.get_structure(relaxed_atoms)
            rows.append({
                "config": method, "seed": seed, "task_id": task["task_id"],
                "energy_ev": float(relax["energy_ev"]), "energy_per_atom_ev": float(relax["energy_per_atom_ev"]),
                "maximum_force_ev_ang": float(relax["maximum_force_ev_ang"]), "converged": bool(relax["converged"]),
                "relax_elapsed_seconds": float(relax["elapsed_seconds"]), "steps": int(relax["steps"]),
                "formula": structure.composition.reduced_formula, "chemical_system": structure.composition.chemical_system,
                "_relaxed_atoms": relaxed_atoms, "_original_atoms": original_atoms,
            })
            gen_rows.append({
                "seed": seed, "generation_seconds": float(gen["generation_seconds"]),
                "composition_valid": bool(gen["composition_valid"]), "structure_valid": bool(gen["structure_valid"]),
                "teacher_used_at_inference": bool(gen["teacher_used_at_inference"]),
                "projection_loaded_at_inference": bool(gen["projection_loaded_at_inference"]),
            })
        frames[method] = pd.DataFrame(rows)
        generation[method] = gen_rows
    return frames, generation


def rename_stats(row: dict) -> dict:
    row = dict(row)
    row["comparison"] = "R1-U0"
    if "G3_wins" in row:
        row["R1_wins"] = row.pop("G3_wins")
        row["R1_losses"] = row.pop("G3_losses")
    return row


def install_safe_smact_policy() -> None:
    import mattergen.evaluation.metrics.structure as structure_metrics

    original = structure_metrics.is_smact_valid
    if getattr(original, "_crystalrepa_safe", False):
        return

    def safe_is_smact_valid(structure):
        try:
            return original(structure)
        except TypeError as error:
            if "SMACT validity checker failed" not in str(error):
                raise
            return False

    safe_is_smact_valid._crystalrepa_safe = True
    structure_metrics.is_smact_valid = safe_is_smact_valid
    atomic_json(
        REPORTS / "smact_metric_policy.json",
        {
            "created_at": now(),
            "policy": "Missing SMACT oxidation-state data is composition invalid (False), not an inference or relaxation failure.",
            "exception_scope": "TypeError containing 'SMACT validity checker failed' only",
        },
    )


def official_metrics_resume(frames: dict[str, pd.DataFrame]) -> tuple[dict, list]:
    metrics, errors, remaining = {}, [], {}
    for method in METHODS:
        detail = REPORTS / method / "official_metrics_per_structure.csv"
        aggregate = REPORTS / method / "official_metrics.json"
        if detail.is_file() and aggregate.is_file():
            frames[method] = pd.read_csv(detail)
            metrics[method] = legacy.metric_values(read_json(aggregate))
        else:
            error = REPORTS / method / "official_metrics_error.json"
            if error.is_file():
                frozen = REPORTS / "frozen" / f"initial_{method}_official_metrics_error.json"
                frozen.parent.mkdir(parents=True, exist_ok=True)
                if not frozen.exists():
                    error.replace(frozen)
                else:
                    index = 2
                    while True:
                        alternate = frozen.with_name(
                            f"{frozen.stem}_{index}{frozen.suffix}"
                        )
                        if not alternate.exists():
                            error.replace(alternate)
                            break
                        index += 1
            remaining[method] = frames[method]
    if remaining:
        original_configs = legacy.CONFIGS
        legacy.CONFIGS = tuple(remaining)
        try:
            computed, new_errors = legacy.official_metrics(remaining)
        finally:
            legacy.CONFIGS = original_configs
        frames.update(remaining)
        metrics.update(computed)
        errors.extend(new_errors)
    return metrics, errors


def reference_terminal_elements() -> set[str]:
    def build_index_single_environment(self, _lmdb_path):
        result = {}
        with self.env.begin() as transaction:
            systems = legacy.lmdb_get(transaction, "chemical_systems")
            for system in systems:
                result[system] = {}
                formulas = legacy.lmdb_get(
                    transaction, f"{system}.reduced_formulas"
                )
                for formula in formulas:
                    result[system][formula] = legacy.lmdb_get(
                        transaction, f"{system}.{formula}.length"
                    )
        return result

    legacy.LMDBBackedReferenceDatasetImpl._build_num_entries_by_chemsys_reduced_formulas = (
        build_index_single_environment
    )
    implementation = legacy.LMDBBackedReferenceDatasetImpl(
        legacy.REFERENCE_LMDB, cleanup_dir=False
    )
    try:
        with implementation.env.begin() as transaction:
            systems = legacy.lmdb_get(transaction, "chemical_systems")
        return {str(system) for system in systems if "-" not in str(system)}
    finally:
        implementation.cleanup(cleanup_dir=False)


def repair_reference_gap_metrics(frames: dict[str, pd.DataFrame]) -> None:
    """Recover energy metrics on the maximal reference-supported subset.

    The official evaluator disables its entire energy capability when any
    generated chemical system lacks a terminal reference. We retain full-set
    structure/composition metrics, calculate E-hull only where all terminals
    exist, and calculate relaxation RMSD for all rows with MatterGen's official
    RMSD function (which does not require a thermodynamic reference).
    """
    required = {"energy_above_hull_per_atom", "rmsd_from_relaxation"}
    for method in METHODS:
        detail_path = REPORTS / method / "official_metrics_per_structure.csv"
        aggregate_path = REPORTS / method / "official_metrics.json"
        if not detail_path.is_file() or not aggregate_path.is_file():
            continue
        full_detail = pd.read_csv(detail_path)
        if required.issubset(full_detail.columns):
            continue
        terminals = reference_terminal_elements()
        original = frames[method]
        supported = original["chemical_system"].map(
            lambda value: set(str(value).split("-")).issubset(terminals)
        )
        unsupported_rows = original.loc[
            ~supported, ["seed", "formula", "chemical_system"]
        ].to_dict("records")
        frozen_dir = REPORTS / "frozen"
        frozen_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            detail_path,
            frozen_dir / f"{method}_full_non_energy_metrics_per_structure.csv",
        )
        shutil.copy2(
            aggregate_path,
            frozen_dir / f"{method}_full_non_energy_metrics.json",
        )
        label = f"{method}_reference_supported"
        supported_detail = REPORTS / label / "official_metrics_per_structure.csv"
        if supported_detail.is_file() and required.issubset(
            pd.read_csv(supported_detail, nrows=1).columns
        ):
            subset_frame = pd.read_csv(supported_detail)
        else:
            subset_frames = {label: original.loc[supported].copy()}
            original_configs = legacy.CONFIGS
            legacy.CONFIGS = (label,)
            try:
                legacy.official_metrics(subset_frames)
            finally:
                legacy.CONFIGS = original_configs
            subset_frame = subset_frames[label]
        extra_columns = [
            column
            for column in subset_frame.columns
            if column not in full_detail.columns and not column.startswith("_")
        ]
        merged = full_detail.merge(
            subset_frame[["seed", *extra_columns]], on="seed", how="left"
        )
        from mattergen.evaluation.utils.utils import (
            compute_rmsd_angstrom,
            preprocess_structure,
        )

        direct_rmsd = {}
        for _, row in original.iterrows():
            relaxed = AseAtomsAdaptor.get_structure(row["_relaxed_atoms"])
            initial = AseAtomsAdaptor.get_structure(row["_original_atoms"])
            direct_rmsd[int(row["seed"])] = compute_rmsd_angstrom(
                relaxed, preprocess_structure(initial)
            )
        evaluator_rmsd = merged.get("rmsd_from_relaxation", pd.Series(dtype=float))
        evaluator_by_seed = dict(zip(merged["seed"], evaluator_rmsd))
        comparisons = [
            abs(float(evaluator_by_seed[seed]) - value)
            for seed, value in direct_rmsd.items()
            if seed in evaluator_by_seed and np.isfinite(evaluator_by_seed[seed])
        ]
        merged["rmsd_from_relaxation"] = merged["seed"].map(direct_rmsd)
        temporary = detail_path.with_name(f".{detail_path.name}.{os.getpid()}.tmp")
        merged.to_csv(temporary, index=False)
        os.replace(temporary, detail_path)
        atomic_json(
            REPORTS / f"{method}_reference_coverage.json",
            {
                "created_at": now(),
                "reference_terminal_count": len(terminals),
                "total_rows": len(original),
                "ehull_supported_rows": int(supported.sum()),
                "ehull_missing_rows": int((~supported).sum()),
                "rmsd_supported_rows": len(direct_rmsd),
                "unsupported_rows": unsupported_rows,
                "missing_terminal_elements": sorted(
                    {
                        element
                        for system in original.loc[~supported, "chemical_system"]
                        for element in str(system).split("-")
                        if element not in terminals
                    }
                ),
                "ehull_policy": "Mean over reference-supported rows; missing rows remain NaN. Metastable/stable tests count missing E-hull as False.",
                "rmsd_policy": "Official MatterGen compute_rmsd_angstrom on all 64 initial/relaxed pairs.",
                "rmsd_official_subset_max_abs_difference": max(comparisons)
                if comparisons
                else None,
            },
        )
        frames[method] = merged


def main() -> None:
    set_stage("metrics", "running", "Computing TRI2024-corrected metrics at metastable and stable thresholds.")
    frames, generation = load_rows()
    legacy.REPORT = REPORTS
    legacy.CONFIGS = METHODS
    legacy.SEEDS = list(SEEDS)
    install_safe_smact_policy()
    repair_reference_gap_metrics(frames)
    raw_metrics, metric_errors = official_metrics_resume(frames)
    quality = {}
    for method in METHODS:
        frame = frames[method]
        gen = generation[method]
        ehull = frame["energy_above_hull_per_atom"].astype(float)
        metastable = ehull <= 0.1
        stable = ehull <= 0.0
        novel = frame["novel"].astype(bool)
        unique = frame["unique"].astype(bool)
        frame["metastable_paper"] = metastable
        frame["stable_paper"] = stable
        frame["msun_paper"] = metastable & novel & unique
        frame["sun_paper"] = stable & novel & unique
        quality[method] = {
            "method": method, "generation_success": 1.0,
            "composition_validity": float(np.mean([item["composition_valid"] for item in gen])),
            "structure_validity": float(np.mean([item["structure_valid"] for item in gen])),
            "relax_success": 1.0, "force_convergence": float(frame["converged"].mean()),
            "average_ehull_ev_atom": float(ehull.mean()),
            "ehull_finite_count": int(np.isfinite(ehull).sum()),
            "ehull_missing_count": int((~np.isfinite(ehull)).sum()),
            "metastable_rate": float(metastable.mean()), "stable_rate": float(stable.mean()),
            "novel_rate": float(novel.mean()), "unique_rate": float(unique.mean()),
            "msun_rate": float(frame["msun_paper"].mean()), "sun_rate": float(frame["sun_paper"].mean()),
            "relaxation_rmsd": float(frame["rmsd_from_relaxation"].mean()),
            "median_generation_seconds": median(item["generation_seconds"] for item in gen),
        }
    quality_rows = [quality[method] for method in METHODS]
    pd.DataFrame(quality_rows).to_csv(REPORTS / "quality_comparison.csv", index=False)
    atomic_json(REPORTS / "quality_comparison.json", {"created_at": now(), "methods": quality_rows, "official_metric_errors": metric_errors, "metastable_threshold": 0.1, "stable_threshold": 0.0})
    set_stage("metrics", "success", "Unified generation, relaxation, E-hull, MSUN/SUN metrics complete.", quality)

    set_stage("paired_statistics", "running", "Computing paired bootstrap, Wilcoxon, and exact binary tests.")
    u0, r1 = frames["U0"].sort_values("seed"), frames["R1"].sort_values("seed")
    paired = []
    for column in ("energy_above_hull_per_atom", "rmsd_from_relaxation", "maximum_force_ev_ang"):
        paired.append(rename_stats(legacy.continuous_stats(u0[column].to_numpy(), r1[column].to_numpy(), column, True)))
    paired.append(rename_stats(legacy.continuous_stats(
        np.asarray([item["generation_seconds"] for item in generation["U0"]]),
        np.asarray([item["generation_seconds"] for item in generation["R1"]]), "generation_seconds", True,
    )))
    for column in ("metastable_paper", "stable_paper", "msun_paper", "sun_paper", "comp_validity", "structure_validity", "converged"):
        paired.append(rename_stats(legacy.binary_stats(u0[column].to_numpy(), r1[column].to_numpy(), column)))
    pd.DataFrame(paired).to_csv(REPORTS / "paired_statistics.csv", index=False)
    atomic_json(REPORTS / "paired_statistics.json", {"created_at": now(), "rows": paired})
    set_stage("paired_statistics", "success", "Paired statistics completed for 64 matched seeds.", {"rows": len(paired)})

    a, b = quality["U0"], quality["R1"]
    overhead = b["median_generation_seconds"] / a["median_generation_seconds"] - 1
    engineering = {
        "generation_success_100pct": b["generation_success"] == 1.0,
        "structure_validity_not_lower": b["structure_validity"] >= a["structure_validity"],
        "composition_drop_le_2pp": b["composition_validity"] >= a["composition_validity"] - 0.02,
        "relax_failure_increment_le_2pp": b["relax_success"] >= a["relax_success"] - 0.02,
        "inference_overhead_le_5pct": overhead <= 0.05,
        "teacher_absent_inference": not any(item["teacher_used_at_inference"] for item in generation["R1"]),
        "projection_absent_inference": not any(item["projection_loaded_at_inference"] for item in generation["R1"]),
    }
    positive = {
        "ehull_drop_ge_0_005": b["average_ehull_ev_atom"] <= a["average_ehull_ev_atom"] - 0.005,
        "stable_gain_ge_1_5pp": b["stable_rate"] >= a["stable_rate"] + 0.015,
        "metastable_gain_ge_2pp": b["metastable_rate"] >= a["metastable_rate"] + 0.02,
        "msun_or_sun_gain_ge_1_5pp": max(b["msun_rate"] - a["msun_rate"], b["sun_rate"] - a["sun_rate"]) >= 0.015,
        "rmsd_drop_ge_10pct": b["relaxation_rmsd"] <= 0.9 * a["relaxation_rmsd"],
    }
    no_systematic_harm = {
        "ehull_not_worse_0_015": b["average_ehull_ev_atom"] <= a["average_ehull_ev_atom"] + 0.015,
        "metastable_not_lower_3pp": b["metastable_rate"] >= a["metastable_rate"] - 0.03,
        "stable_not_lower_3pp": b["stable_rate"] >= a["stable_rate"] - 0.03,
        "msun_not_lower_3pp": b["msun_rate"] >= a["msun_rate"] - 0.03,
        "sun_not_lower_3pp": b["sun_rate"] >= a["sun_rate"] - 0.03,
        "rmsd_not_worse_10pct": b["relaxation_rmsd"] <= 1.1 * a["relaxation_rmsd"],
    }
    engineering_go = all(engineering.values())
    scientific_go = engineering_go and any(positive.values()) and all(no_systematic_harm.values())
    decision = {
        "created_at": now(), "REPA_REPRO_ENGINEERING_GO": engineering_go,
        "REPA_REPRO_SCIENTIFIC_GO": scientific_go, "REPA_BASE_REPRODUCED": scientific_go,
        "REPA_REPRO_NO_GO": not scientific_go, "engineering_checks": engineering,
        "scientific_positive_signals": positive, "no_systematic_harm": no_systematic_harm,
        "inference_overhead_fraction": overhead, "U0": a, "R1": b,
        "controlled_deviation": "CHGNet 0.3.0 is not one of the paper's ten teachers",
        "teacher_evaluator_independent": True, "stability_source": "MatterSim-5M surrogate", "DFT_VERIFIED": False,
    }
    atomic_json(REPORTS / "repro_decision.json", decision)
    atomic_text(REPORTS / "crystalrepa_repro_final_report.md", "# CrystalREPA unconditional MP-20 reproduction\n\n" + f"- `REPA_REPRO_ENGINEERING_GO={engineering_go}`\n- `REPA_REPRO_SCIENTIFIC_GO={scientific_go}`\n- `REPA_BASE_REPRODUCED={scientific_go}`\n- `REPA_REPRO_NO_GO={not scientific_go}`\n\n" + pd.DataFrame(quality_rows).to_markdown(index=False) + "\n\nControlled limitation: CHGNet 0.3.0 is an independent frozen teacher but is not one of the ten teachers reported in CrystalREPA. Results are MatterSim-5M surrogate estimates, not DFT.\n")
    set_stage("repro_go_no_go", "success", f"Reproduction decision complete: base_reproduced={scientific_go}.", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
