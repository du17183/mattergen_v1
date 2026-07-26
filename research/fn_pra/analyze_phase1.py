"""Official TRI2024-corrected FN-PRA Phase-1 quality and paired analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median

import ase.io
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from research.fn_pra.phase1_common import (
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
    set_stage,
)


TOOLS = Path("/data/dxl/tools/innovation2_next")
sys.path.insert(0, str(TOOLS))
import analyze_corrector_32 as legacy  # noqa: E402


METHODS = ("A0", "P1")
SEEDS = tuple(range(15000, 15032))
EIGHT_ROOT = RESULTS / "generation/eight_seed"
THIRTY_TWO_ROOT = RESULTS / "generation/thirty_two_seed"
RELAX_JSON = RESULTS / "progress/mattersim_relax_progress.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generation_dir(method: str, seed: int) -> Path:
    root = EIGHT_ROOT if seed < 15008 else THIRTY_TWO_ROOT
    return root / method / f"seed_{seed}"


def metric_values(metrics: dict) -> dict:
    return {
        key: value["value"] if isinstance(value, dict) and "value" in value else value
        for key, value in metrics.items()
    }


def load_rows() -> tuple[dict[str, pd.DataFrame], dict[str, list[dict]], list[dict]]:
    relax_state = read_json(RELAX_JSON)
    if len(relax_state["tasks"]) != 64 or any(
        task["status"] != "success" for task in relax_state["tasks"]
    ):
        raise RuntimeError("analysis requires 64 successful relaxation tasks")
    per_method = {}
    generation = {}
    nonconverged = []
    for method in METHODS:
        relax_rows = []
        generation_rows = []
        for seed in SEEDS:
            task = next(
                item
                for item in relax_state["tasks"]
                if item["config"] == method and int(item["seed"]) == seed
            )
            relax_summary = read_json(Path(task["output_path"]).parent / "relax_summary.json")
            generated = generation_dir(method, seed)
            generation_summary = read_json(generated / "summary.json")
            relaxed_atoms = ase.io.read(task["output_path"])
            original_atoms = ase.io.read(task["input_path"])
            relaxed_structure = AseAtomsAdaptor.get_structure(relaxed_atoms)
            relax_rows.append(
                {
                    "config": method,
                    "seed": seed,
                    "task_id": task["task_id"],
                    "energy_ev": float(relax_summary["energy_ev"]),
                    "energy_per_atom_ev": float(relax_summary["energy_per_atom_ev"]),
                    "maximum_force_ev_ang": float(
                        relax_summary["maximum_force_ev_ang"]
                    ),
                    "converged": bool(relax_summary["converged"]),
                    "relax_elapsed_seconds": float(relax_summary["elapsed_seconds"]),
                    "steps": int(relax_summary["steps"]),
                    "formula": relaxed_structure.composition.reduced_formula,
                    "chemical_system": relaxed_structure.composition.chemical_system,
                    "input_hash": relax_summary["input_hash"],
                    "output_hash": relax_summary["output_hash"],
                    "_relaxed_atoms": relaxed_atoms,
                    "_original_atoms": original_atoms,
                }
            )
            generation_rows.append(
                {
                    "seed": seed,
                    "generation_seconds": float(generation_summary["generation_seconds"]),
                    "composition_valid": bool(generation_summary["composition_valid"]),
                    "structure_valid": bool(generation_summary["structure_valid"]),
                    "initial_state_hash": generation_summary["initial_hashes"]["combined"],
                    "teacher_used_at_inference": bool(
                        generation_summary["teacher_used_at_inference"]
                    ),
                    "projection_heads_loaded_at_inference": bool(
                        generation_summary["projection_heads_loaded_at_inference"]
                    ),
                }
            )
            if not relax_summary["converged"]:
                nonconverged.append(
                    {
                        "method": method,
                        "seed": seed,
                        "status": "relax_valid_but_not_converged",
                        "maximum_force_ev_ang": relax_summary["maximum_force_ev_ang"],
                        "steps": relax_summary["steps"],
                    }
                )
        per_method[method] = pd.DataFrame(relax_rows)
        generation[method] = generation_rows
    return per_method, generation, nonconverged


def paired_row(row: dict) -> dict:
    updated = dict(row)
    updated["comparison"] = "P1-A0"
    for old, new in (
        ("G3_wins", "P1_wins"),
        ("G3_losses", "P1_losses"),
    ):
        if old in updated:
            updated[new] = updated.pop(old)
    return updated


def main() -> int:
    set_stage(
        "phase1_analysis",
        "running",
        "Computing official TRI2024-corrected metrics and paired Phase-1 decisions.",
    )
    per_method, generation, nonconverged = load_rows()
    legacy.REPORT = REPORTS
    legacy.CONFIGS = METHODS
    legacy.SEEDS = list(SEEDS)
    official, errors = legacy.official_metrics(per_method)
    official = {method: metric_values(values) for method, values in official.items()}

    quality_rows = []
    for method in METHODS:
        frame = per_method[method]
        gen = generation[method]
        row = {
            "method": method,
            "generation_success_rate": 1.0,
            "generation_composition_validity": float(
                np.mean([item["composition_valid"] for item in gen])
            ),
            "generation_structure_validity": float(
                np.mean([item["structure_valid"] for item in gen])
            ),
            "relax_success_rate": 1.0,
            "force_convergence_rate": float(frame["converged"].astype(bool).mean()),
            "formula_diversity": float(frame["formula"].nunique() / len(frame)),
            "chemical_system_diversity": float(
                frame["chemical_system"].nunique() / len(frame)
            ),
            "median_generation_seconds": median(
                item["generation_seconds"] for item in gen
            ),
            "mean_relax_seconds": float(frame["relax_elapsed_seconds"].mean()),
            **official[method],
        }
        quality_rows.append(row)
    quality = {row["method"]: row for row in quality_rows}
    pd.DataFrame(quality_rows).to_csv(REPORTS / "quality_comparison.csv", index=False)
    atomic_json(
        REPORTS / "quality_comparison.json",
        {
            "created_at": now(),
            "STABILITY_SOURCE": "MatterSim-5M surrogate",
            "DFT_VERIFIED": False,
            "PROPERTY_TARGET_VERIFIED": False,
            "stability_threshold_ev_atom": 0.1,
            "methods": quality_rows,
            "official_metric_errors": errors,
        },
    )

    paired = []
    continuous = (
        ("energy_above_hull_per_atom", True),
        ("rmsd_from_relaxation", True),
        ("relax_elapsed_seconds", True),
        ("maximum_force_ev_ang", True),
    )
    for column, lower_is_better in continuous:
        paired.append(
            paired_row(
                legacy.continuous_stats(
                    per_method["A0"][column].to_numpy(),
                    per_method["P1"][column].to_numpy(),
                    column,
                    lower_is_better,
                )
            )
        )
    paired.append(
        paired_row(
            legacy.continuous_stats(
                np.asarray(
                    [item["generation_seconds"] for item in generation["A0"]],
                    dtype=float,
                ),
                np.asarray(
                    [item["generation_seconds"] for item in generation["P1"]],
                    dtype=float,
                ),
                "generation_seconds",
                True,
            )
        )
    )
    for column in (
        "stable",
        "novel_unique_stable",
        "comp_validity",
        "structure_validity",
        "converged",
    ):
        paired.append(
            paired_row(
                legacy.binary_stats(
                    per_method["A0"][column].to_numpy(),
                    per_method["P1"][column].to_numpy(),
                    column,
                )
            )
        )
    pd.DataFrame(paired).to_csv(REPORTS / "paired_statistics.csv", index=False)
    atomic_json(REPORTS / "paired_statistics.json", {"rows": paired})
    pd.DataFrame(nonconverged).to_csv(
        REPORTS / "relax_valid_but_not_converged.csv", index=False
    )

    a0 = quality["A0"]
    p1 = quality["P1"]
    eight = read_json(REPORTS / "eight_seed_generation.json")
    generation_overhead = (
        p1["median_generation_seconds"] / a0["median_generation_seconds"] - 1.0
    )
    engineering_checks = {
        "generation_success_100_percent": p1["generation_success_rate"] == 1.0,
        "structure_validity_not_lower": p1["generation_structure_validity"]
        >= a0["generation_structure_validity"],
        "composition_validity_drop_le_2pp": p1["generation_composition_validity"]
        >= a0["generation_composition_validity"] - 0.02,
        "relax_failure_increment_le_2pp": p1["relax_success_rate"]
        >= a0["relax_success_rate"] - 0.02,
        "median_generation_overhead_le_5pct": generation_overhead <= 0.05,
        "teacher_absent_inference": not any(
            item["teacher_used_at_inference"] for item in generation["P1"]
        ),
        "projection_heads_absent_inference": not any(
            item["projection_heads_loaded_at_inference"] for item in generation["P1"]
        ),
        "level1_deterministic": all(
            all(value.values()) for value in eight["determinism_repeat"].values()
        ),
    }
    positive_signals = {
        "mean_ehull_drop_ge_0_005": p1["avg_energy_above_hull_per_atom"]
        <= a0["avg_energy_above_hull_per_atom"] - 0.005,
        "stable_gain_ge_1_5pp": p1["frac_stable_structures"]
        >= a0["frac_stable_structures"] + 0.015,
        "nus_gain_ge_1_5pp": p1["frac_novel_unique_stable_structures"]
        >= a0["frac_novel_unique_stable_structures"] + 0.015,
        "relax_rmsd_drop_ge_5pct": p1["avg_rmsd_from_relaxation"]
        <= 0.95 * a0["avg_rmsd_from_relaxation"],
    }
    systematic_quality_checks = {
        "ehull_not_worse_over_0_015": p1["avg_energy_above_hull_per_atom"]
        <= a0["avg_energy_above_hull_per_atom"] + 0.015,
        "stable_not_lower_over_3pp": p1["frac_stable_structures"]
        >= a0["frac_stable_structures"] - 0.03,
        "nus_not_lower_over_3pp": p1["frac_novel_unique_stable_structures"]
        >= a0["frac_novel_unique_stable_structures"] - 0.03,
        "rmsd_not_worse_over_10pct": p1["avg_rmsd_from_relaxation"]
        <= 1.10 * a0["avg_rmsd_from_relaxation"],
    }
    engineering_go = all(engineering_checks.values())
    scientific_go = bool(
        engineering_go
        and any(positive_signals.values())
        and all(systematic_quality_checks.values())
    )
    decision = {
        "created_at": now(),
        "PHASE1_ENGINEERING_GO": engineering_go,
        "PHASE1_SCIENTIFIC_GO": scientific_go,
        "FN_PRA_PHASE1_NO_GO": not scientific_go,
        "engineering_checks": engineering_checks,
        "positive_signals": positive_signals,
        "systematic_quality_checks": systematic_quality_checks,
        "generation_overhead_fraction": generation_overhead,
        "A0": a0,
        "P1": p1,
        "STABILITY_SOURCE": "MatterSim-5M surrogate",
        "DFT_VERIFIED": False,
        "PROPERTY_TARGET_VERIFIED": False,
        "teacher": "CHGNet 0.3.0",
        "evaluator": "MatterSim-5M",
        "teacher_evaluator_circularity": False,
    }
    atomic_json(REPORTS / "phase1_decision.json", decision)
    atomic_text(
        REPORTS / "phase1_final_report.md",
        f"""# FN-PRA Phase-1 final report

## Decision

- `PHASE1_ENGINEERING_GO={engineering_go}`
- `PHASE1_SCIENTIFIC_GO={scientific_go}`
- `FN_PRA_PHASE1_NO_GO={not scientific_go}`

## A0 versus P1

| Metric | A0 | P1 | Difference |
|---|---:|---:|---:|
| Median generation time (s) | {a0["median_generation_seconds"]:.4f} | {p1["median_generation_seconds"]:.4f} | {100*generation_overhead:+.3f}% |
| Composition validity | {a0["generation_composition_validity"]:.4f} | {p1["generation_composition_validity"]:.4f} | {100*(p1["generation_composition_validity"]-a0["generation_composition_validity"]):+.3f} pp |
| Structure validity | {a0["generation_structure_validity"]:.4f} | {p1["generation_structure_validity"]:.4f} | {100*(p1["generation_structure_validity"]-a0["generation_structure_validity"]):+.3f} pp |
| Mean E-hull (eV/atom) | {a0["avg_energy_above_hull_per_atom"]:.6f} | {p1["avg_energy_above_hull_per_atom"]:.6f} | {p1["avg_energy_above_hull_per_atom"]-a0["avg_energy_above_hull_per_atom"]:+.6f} |
| Stable fraction | {a0["frac_stable_structures"]:.4f} | {p1["frac_stable_structures"]:.4f} | {100*(p1["frac_stable_structures"]-a0["frac_stable_structures"]):+.3f} pp |
| NUS fraction | {a0["frac_novel_unique_stable_structures"]:.4f} | {p1["frac_novel_unique_stable_structures"]:.4f} | {100*(p1["frac_novel_unique_stable_structures"]-a0["frac_novel_unique_stable_structures"]):+.3f} pp |
| Mean relaxation RMSD | {a0["avg_rmsd_from_relaxation"]:.6f} | {p1["avg_rmsd_from_relaxation"]:.6f} | {p1["avg_rmsd_from_relaxation"]-a0["avg_rmsd_from_relaxation"]:+.6f} |

All stability quantities are MatterSim-5M surrogate results with TRI2024
correction. No DFT verification or direct magnetic-property verification was
performed in Phase-1.
""",
    )
    set_stage(
        "phase1_analysis",
        "success",
        f"Phase-1 analysis complete; engineering_go={engineering_go}, scientific_go={scientific_go}.",
        decision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
