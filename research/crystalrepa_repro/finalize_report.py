"""Build the exact final handoff summary from frozen reproduction outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.crystalrepa_repro.common import REPORTS, atomic_json, atomic_text, now
from research.crystalrepa_repro.configuration import CHECKPOINT, CHECKPOINT_SHA256


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pp(value: float) -> float:
    return 100.0 * value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-commit", default="PENDING")
    parser.add_argument("--draft-pr", default="PENDING")
    args = parser.parse_args()
    paper = read(REPORTS / "frozen/paper_config.json")
    training = read(REPORTS / "training_summary_10000.json")
    inference = read(REPORTS / "inference_checkpoint_manifest.json")
    generation = read(REPORTS / "sixty_four_generation.json")
    relaxation = read(REPORTS / "mattersim_relaxation_summary.json")
    quality_rows = read(REPORTS / "quality_comparison.json")["methods"]
    quality = {row["method"]: row for row in quality_rows}
    paired = read(REPORTS / "paired_statistics.json")
    decision = read(REPORTS / "repro_decision.json")
    determinism = read(REPORTS / "eight_seed_generation.json")
    reference_coverage = read(REPORTS / "R1_reference_coverage.json")
    u0, r1 = quality["U0"], quality["R1"]
    paired_ehull = next(
        row
        for row in paired["rows"]
        if row["metric"] == "energy_above_hull_per_atom"
    )
    summary = {
        "created_at": now(),
        "CRYSTALREPA_REPRO_COMPLETED": True,
        "PAPER_CONFIG_SOURCE": paper["source"],
        "BASE_MODEL": "Official unconditional MP-20 MatterGen",
        "BASE_CHECKPOINT": str(CHECKPOINT),
        "BASE_CHECKPOINT_SHA256": CHECKPOINT_SHA256,
        "ALIGNMENT_BLOCK": int(paper["mattergen_mp20"]["alignment_block_1_indexed"]),
        "TEACHER": "CHGNet 0.3.0 (controlled local deviation)",
        "TEACHER_LAYER": "atom_fea before final CHGNet convolution",
        "EA_NCE_TEMPERATURE": float(paper["mattergen_mp20"]["temperature"]),
        "ALIGNMENT_WEIGHT": float(paper["mattergen_mp20"]["alignment_weight"]),
        "TRAINING_STEPS": int(training["max_steps"]),
        "BEST_CHECKPOINT": inference["best_training_checkpoint"],
        "BEST_VALIDATION_LOSS": inference["best_validation_loss"],
        "TRAINABLE_PARAMETERS": int(training["trainable_parameters"]),
        "TESTS_PASSED": "4/4 targeted + integration + 8-rank DDP; 5 unrelated baseline failures documented",
        "DETERMINISM_LEVEL": {
            "paired_initial_state": bool(determinism["pairing"]["passed"]),
            "exact_repeat": determinism["determinism"],
        },
        "U0_GENERATION": f"{generation['success']['U0']}/64",
        "R1_GENERATION": f"{generation['success']['R1']}/64",
        "U0_RELAX": "64/64",
        "R1_RELAX": "64/64",
        "RELAX_CONVERGED_TOTAL": int(relaxation["converged"]),
        "RELAX_VALID_AT_CAP_TOTAL": int(relaxation["valid_but_not_converged"]),
        "U0_COMPOSITION_VALIDITY": u0["composition_validity"],
        "R1_COMPOSITION_VALIDITY": r1["composition_validity"],
        "COMPOSITION_CHANGE_PP": pp(r1["composition_validity"] - u0["composition_validity"]),
        "U0_STRUCTURE_VALIDITY": u0["structure_validity"],
        "R1_STRUCTURE_VALIDITY": r1["structure_validity"],
        "STRUCTURE_CHANGE_PP": pp(r1["structure_validity"] - u0["structure_validity"]),
        "U0_AVG_EHULL": u0["average_ehull_ev_atom"],
        "R1_AVG_EHULL": r1["average_ehull_ev_atom"],
        "EHULL_CHANGE": r1["average_ehull_ev_atom"] - u0["average_ehull_ev_atom"],
        "EHULL_REFERENCE_COVERAGE": {
            "U0": f"{u0['ehull_finite_count']}/64",
            "R1": f"{r1['ehull_finite_count']}/64",
            "R1_missing_terminal_elements": reference_coverage[
                "missing_terminal_elements"
            ],
            "policy": reference_coverage["ehull_policy"],
        },
        "PAIRED_EHULL": {
            "sample_count": paired_ehull["paired_sample_count"],
            "mean_difference": paired_ehull["paired_mean_difference"],
            "bootstrap_95_ci": [
                paired_ehull["bootstrap_95_ci_low"],
                paired_ehull["bootstrap_95_ci_high"],
            ],
            "wilcoxon_p_value": paired_ehull["wilcoxon_p_value"],
        },
        "U0_METASTABLE": u0["metastable_rate"],
        "R1_METASTABLE": r1["metastable_rate"],
        "METASTABLE_CHANGE_PP": pp(r1["metastable_rate"] - u0["metastable_rate"]),
        "U0_STABLE": u0["stable_rate"],
        "R1_STABLE": r1["stable_rate"],
        "STABLE_CHANGE_PP": pp(r1["stable_rate"] - u0["stable_rate"]),
        "U0_NUS_OR_SUN": {"MSUN": u0["msun_rate"], "SUN": u0["sun_rate"]},
        "R1_NUS_OR_SUN": {"MSUN": r1["msun_rate"], "SUN": r1["sun_rate"]},
        "NUS_OR_SUN_CHANGE_PP": {
            "MSUN": pp(r1["msun_rate"] - u0["msun_rate"]),
            "SUN": pp(r1["sun_rate"] - u0["sun_rate"]),
        },
        "U0_RELAX_RMSD": u0["relaxation_rmsd"],
        "R1_RELAX_RMSD": r1["relaxation_rmsd"],
        "RMSD_CHANGE": r1["relaxation_rmsd"] - u0["relaxation_rmsd"],
        "INFERENCE_OVERHEAD": decision["inference_overhead_fraction"],
        "PAIRED_STATISTICS": {
            "rows": len(paired["rows"]),
            "report": str(REPORTS / "paired_statistics.csv"),
        },
        "REPA_REPRO_ENGINEERING_GO": decision["REPA_REPRO_ENGINEERING_GO"],
        "REPA_REPRO_SCIENTIFIC_GO": decision["REPA_REPRO_SCIENTIFIC_GO"],
        "REPA_BASE_REPRODUCED": decision["REPA_BASE_REPRODUCED"],
        "REPA_REPRO_NO_GO": decision["REPA_REPRO_NO_GO"],
        "FINAL_REPORT": str(REPORTS / "crystalrepa_repro_final_report.md"),
        "GITHUB_BRANCH": "feature/crystalrepa-repro",
        "GITHUB_COMMIT": args.github_commit,
        "DRAFT_PR": args.draft_pr,
        "CONDITIONAL_FN_PRA_STARTED": False,
        "FORMAL_SEEDS_STARTED": False,
        "GPU_WORKERS": 0,
        "OTHER_PROCESSES_TERMINATED": False,
        "SIGKILL_USED": False,
        "LIMITATIONS": [
            "CHGNet 0.3.0 is not one of the ten CrystalREPA paper Teachers.",
            "The 10,000-step cap is much shorter than the paper's 1,900 epochs.",
            "Stability is a MatterSim-5M surrogate estimate, not DFT.",
            "The paired development set contains 64 seeds, not the paper's 1,024 samples over five runs.",
            "TRI2024 has no terminal references for Pm, Pu, Tc, or U; R1 E-hull uses 59/64 reference-supported rows and paired E-hull statistics use the same 59 seeds.",
        ],
        "NEXT_ACTION": (
            "After human review, isolate conditional FN-PRA V2 interactions."
            if decision["REPA_BASE_REPRODUCED"]
            else "Do not stack more conditional REPA modules; switch to field-confidence three-field Self-Conditioning."
        ),
    }
    atomic_json(REPORTS / "final_summary.json", summary)
    table = (
        "| Metric | U0 | R1 | Change |\n|---|---:|---:|---:|\n"
        f"| Composition validity | {u0['composition_validity']:.4f} | {r1['composition_validity']:.4f} | {summary['COMPOSITION_CHANGE_PP']:+.3f} pp |\n"
        f"| Structure validity | {u0['structure_validity']:.4f} | {r1['structure_validity']:.4f} | {summary['STRUCTURE_CHANGE_PP']:+.3f} pp |\n"
        f"| Mean E-hull (eV/atom) | {u0['average_ehull_ev_atom']:.6f} | {r1['average_ehull_ev_atom']:.6f} | {summary['EHULL_CHANGE']:+.6f} |\n"
        f"| Metastable | {u0['metastable_rate']:.4f} | {r1['metastable_rate']:.4f} | {summary['METASTABLE_CHANGE_PP']:+.3f} pp |\n"
        f"| Stable | {u0['stable_rate']:.4f} | {r1['stable_rate']:.4f} | {summary['STABLE_CHANGE_PP']:+.3f} pp |\n"
        f"| Relaxation RMSD | {u0['relaxation_rmsd']:.6f} | {r1['relaxation_rmsd']:.6f} | {summary['RMSD_CHANGE']:+.6f} |\n"
    )
    text = (
        "# CrystalREPA unconditional MP-20 reproduction — final handoff\n\n"
        f"- `REPA_REPRO_ENGINEERING_GO={decision['REPA_REPRO_ENGINEERING_GO']}`\n"
        f"- `REPA_REPRO_SCIENTIFIC_GO={decision['REPA_REPRO_SCIENTIFIC_GO']}`\n"
        f"- `REPA_BASE_REPRODUCED={decision['REPA_BASE_REPRODUCED']}`\n"
        f"- `REPA_REPRO_NO_GO={decision['REPA_REPRO_NO_GO']}`\n\n"
        + table
        + (
            "\nE-hull coverage is U0 "
            f"{u0['ehull_finite_count']}/64 and R1 {r1['ehull_finite_count']}/64. "
            "TRI2024 lacks terminal references for Pm, Pu, Tc, and U; the paired "
            f"{paired_ehull['paired_sample_count']}-seed E-hull mean difference is "
            f"{paired_ehull['paired_mean_difference']:+.6f} eV/atom "
            f"(95% bootstrap CI [{paired_ehull['bootstrap_95_ci_low']:+.6f}, "
            f"{paired_ehull['bootstrap_95_ci_high']:+.6f}], "
            f"Wilcoxon p={paired_ehull['wilcoxon_p_value']:.6f}).\n\n"
        )
        + "The frozen Teacher is CHGNet 0.3.0, a controlled deviation from the paper. "
        "MatterSim-5M is used only as the independent evaluation surrogate.\n"
    )
    atomic_text(REPORTS / "crystalrepa_repro_final_report.md", text)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
