#!/usr/bin/env python3
"""Freeze the completed CG-TDR Phase-0 evaluation and No-Go decision."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULTS = Path("/data/dxl/results/cg_tdr/phase0")
REPORTS = Path("/data/dxl/reports/cg_tdr/phase0")
FINAL_JSON = REPORTS / "cg_tdr_eval_final.json"
FINAL_REPORT = REPORTS / "cg_tdr_eval_final.md"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def relative(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1.0


def main() -> int:
    tests = json.loads(
        (RESULTS / "test_baseline_comparison.json").read_text()
    )
    residual = json.loads(
        (RESULTS / "residual_learning_summary.json").read_text()
    )
    labels = json.loads((RESULTS / "gate_v2_labels.json").read_text())
    training = json.loads(
        (RESULTS / "training_v2/training_summary.json").read_text()
    )
    v1 = json.loads(
        (REPORTS / "v1_eight_seed/v1_decision.json").read_text()
    )
    v2 = json.loads(
        (REPORTS / "v2_eight_seed/v2_decision.json").read_text()
    )
    identity = residual["identity_baseline"]
    a0 = v2["summaries"]["A0"]
    v2p = v2["summaries"]["V2P"]
    v2c = v2["summaries"]["V2C"]
    payload = {
        "CG_TDR_EVAL_COMPLETED": True,
        "MAIN_TEST_PASSED": tests["main"]["passed"],
        "MAIN_TEST_FAILED": tests["main"]["failed"],
        "FEATURE_TEST_PASSED": tests["feature"]["passed"],
        "FEATURE_TEST_FAILED": tests["feature"]["failed"],
        "TEST_FAILURE_SET_IDENTICAL": tests["TEST_FAILURE_SET_IDENTICAL"],
        "PRE_EXISTING_TEST_FAILURES": tests["PRE_EXISTING_TEST_FAILURES"],
        "CG_TDR_INTRODUCED_TEST_FAILURES": tests[
            "CG_TDR_INTRODUCED_TEST_FAILURES"
        ],
        "FULL_TEST_BLOCKER_CLEARED": tests["FULL_TEST_BLOCKER_CLEARED"],
        "ZERO_POSITION_LOSS": identity["zero_position_loss"],
        "MODEL_POSITION_LOSS": identity["model_position_loss"],
        "POSITION_LOSS_IMPROVEMENT": identity["position_loss_improvement"],
        "ZERO_CELL_LOSS": identity["zero_cell_loss"],
        "MODEL_CELL_LOSS": identity["model_cell_loss"],
        "CELL_LOSS_IMPROVEMENT": identity["cell_loss_improvement"],
        "POSITION_COSINE_MEAN": residual["position"]["cosine_mean"],
        "POSITION_COSINE_MEDIAN": residual["position"]["cosine_median"],
        "POSITION_POSITIVE_COSINE_RATE": residual["position"][
            "positive_cosine_rate"
        ],
        "POSITION_MAGNITUDE_RATIO": residual["position"][
            "magnitude_ratio_median"
        ],
        "CELL_COSINE_MEAN": residual["cell"]["cosine_mean"],
        "CELL_COSINE_MEDIAN": residual["cell"]["cosine_median"],
        "CELL_POSITIVE_COSINE_RATE": residual["cell"]["positive_cosine_rate"],
        "CELL_MAGNITUDE_RATIO": residual["cell"]["magnitude_ratio_median"],
        "V1_POSITION_GATE_MEAN": residual["position_gate"]["mean"],
        "V1_POSITION_GATE_STD": residual["position_gate"]["std"],
        "V1_CELL_GATE_MEAN": residual["cell_gate"]["mean"],
        "V1_CELL_GATE_STD": residual["cell_gate"]["std"],
        "GATE_SELECTIVITY_VALID": residual["GATE_SELECTIVITY_VALID"],
        "V1_A0_GENERATION": 8,
        "V1_T1_GENERATION": 8,
        "V1_T2_GENERATION": 8,
        "V1_A0_RELAX": 8,
        "V1_T1_RELAX": 8,
        "V1_T2_RELAX": 8,
        "V1_SUMMARIES": v1["summaries"],
        "CG_TDR_V1_EIGHT_SEED_SAFE": v1["CG_TDR_V1_EIGHT_SEED_SAFE"],
        "CG_TDR_V1_EIGHT_SEED_POSITIVE": v1[
            "CG_TDR_V1_EIGHT_SEED_POSITIVE"
        ],
        "CG_TDR_V1_DIRECT_GO": v1["CG_TDR_V1_DIRECT_GO"],
        "CG_TDR_GATE_V2_REQUIRED": v1["CG_TDR_GATE_V2_REQUIRED"],
        "CG_TDR_V1_FATAL_NO_GO": v1["CG_TDR_V1_FATAL_NO_GO"],
        "GATE_V2_IMPLEMENTED": True,
        "GATE_V2_TRAINING_SEED": training["training_seed"],
        "GATE_V2_STEPS": training["steps_completed"],
        "GATE_V2_BEST_STEP": training["best_step"],
        "GATE_V2_HIGH_CONFIDENCE_RATE": labels["splits"]["train"][
            "position_high_confidence_rate_ge_0_5"
        ],
        "GATE_V2_ZERO_LOW_CONFIDENCE_RATE": labels["splits"]["train"][
            "position_zero_low_rate_le_0_1"
        ],
        "GATE_V2_UTILITY_CORRELATION": training["test_metrics"][
            "position_gate_utility_spearman"
        ],
        "GATE_V2_CELL_UTILITY_CORRELATION": training["test_metrics"][
            "cell_gate_utility_spearman"
        ],
        "CG_TDR_GATE_V2_VALID": v2["CG_TDR_GATE_V2_VALID"],
        "CG_TDR_V2_EIGHT_SEED_GO": v2["CG_TDR_V2_EIGHT_SEED_GO"],
        "V2_SUMMARIES": v2["summaries"],
        "V2P_CHANGES": {
            "ehull_ev_atom": v2p["average_ehull"] - a0["average_ehull"],
            "stable_rate": v2p["stable_rate"] - a0["stable_rate"],
            "nus_rate": v2p["nus_rate"] - a0["nus_rate"],
            "rmsd_mean_relative": relative(
                v2p["relaxation_rmsd_mean"], a0["relaxation_rmsd_mean"]
            ),
            "rmsd_median_relative": relative(
                v2p["relaxation_rmsd_median"],
                a0["relaxation_rmsd_median"],
            ),
            "max_force_mean_relative": relative(
                v2p["initial_max_force_mean"], a0["initial_max_force_mean"]
            ),
            "max_force_median_relative": relative(
                v2p["initial_max_force_median"],
                a0["initial_max_force_median"],
            ),
            "latency_relative": relative(
                v2p["generation_elapsed_median"],
                a0["generation_elapsed_median"],
            ),
        },
        "V2C_CHANGES": {
            "ehull_ev_atom": v2c["average_ehull"] - a0["average_ehull"],
            "stable_rate": v2c["stable_rate"] - a0["stable_rate"],
            "nus_rate": v2c["nus_rate"] - a0["nus_rate"],
            "rmsd_mean_relative": relative(
                v2c["relaxation_rmsd_mean"], a0["relaxation_rmsd_mean"]
            ),
            "rmsd_median_relative": relative(
                v2c["relaxation_rmsd_median"],
                a0["relaxation_rmsd_median"],
            ),
            "max_force_mean_relative": relative(
                v2c["initial_max_force_mean"], a0["initial_max_force_mean"]
            ),
            "max_force_median_relative": relative(
                v2c["initial_max_force_median"],
                a0["initial_max_force_median"],
            ),
            "latency_relative": relative(
                v2c["generation_elapsed_median"],
                a0["generation_elapsed_median"],
            ),
        },
        "THIRTY_TWO_SEED_STARTED": False,
        "CG_TDR_MVP_GO": False,
        "CG_TDR_MVP_NO_GO": True,
        "CG_TDR_ROUTE_STOPPED": True,
        "SIXTY_FOUR_SEED_STARTED": False,
        "FORMAL_SEEDS_STARTED": False,
        "TARGETED_TESTS_PASSED": 24,
        "FINAL_REPORT": str(FINAL_REPORT),
        "TEST_REPORT": str(REPORTS / "test_baseline_comparison.md"),
        "IDENTITY_REPORT": str(REPORTS / "identity_baseline.md"),
        "RESIDUAL_DIAGNOSTIC_REPORT": str(
            REPORTS / "residual_learning_diagnostics.md"
        ),
        "V1_REPORT": str(REPORTS / "v1_eight_seed/v1_report.md"),
        "V2_REPORT": str(REPORTS / "v2_eight_seed/v2_report.md"),
        "GPU_WORKERS": 0,
        "OTHER_PROCESSES_TERMINATED": False,
        "SIGKILL_USED": False,
        "LIMITATIONS": [
            "Eight-seed development screen; statistical power is limited.",
            "MatterSim-5M is an independent surrogate evaluator, not DFT verification.",
            "Per-process peak VRAM telemetry was not present in the frozen V1 runner.",
            "The V2 gate became selective, but its learned position direction remained weak.",
        ],
        "NEXT_ACTION": (
            "Stop CG-TDR after the single allowed Gate V2 repair. Do not run "
            "32/64/256 seeds; select a different second-innovation route."
        ),
    }
    atomic_json(FINAL_JSON, payload)
    v1_a0 = v1["summaries"]["A0"]
    v1_t1 = v1["summaries"]["T1"]
    v1_t2 = v1["summaries"]["T2"]
    lines = [
        "# CG-TDR Phase 0 final evaluation",
        "",
        "## Final decision",
        "",
        "```text",
        "CG_TDR_GATE_V2_VALID=True",
        "CG_TDR_V2_EIGHT_SEED_GO=False",
        "CG_TDR_MVP_GO=False",
        "CG_TDR_MVP_NO_GO=True",
        "CG_TDR_ROUTE_STOPPED=True",
        "THIRTY_TWO_SEED_STARTED=False",
        "SIXTY_FOUR_SEED_STARTED=False",
        "FORMAL_SEEDS_STARTED=False",
        "```",
        "",
        "The utility-calibrated Gate V2 fixed the engineering defect (near-always-on gates), but no candidate reached a frozen positive quality threshold. The only permitted V2 repair is exhausted, so the CG-TDR route stops at eight seeds.",
        "",
        "## Test attribution",
        "",
        f"- main: {tests['main']['passed']} passed, {tests['main']['failed']} failed",
        f"- feature/cg-tdr: {tests['feature']['passed']} passed, {tests['feature']['failed']} failed",
        "- Failure node IDs and exception types are identical.",
        "- `CG_TDR_INTRODUCED_TEST_FAILURES=0`",
        "- CG-TDR/V2 targeted tests: 24/24 passed.",
        "",
        "## V1 residual diagnosis",
        "",
        f"- Position loss: zero {identity['zero_position_loss']:.8g}, V1 {identity['model_position_loss']:.8g} ({identity['position_loss_improvement']:+.2%} improvement).",
        f"- Cell loss: zero {identity['zero_cell_loss']:.8g}, V1 {identity['model_cell_loss']:.8g} ({identity['cell_loss_improvement']:+.2%} improvement).",
        f"- Position cosine mean/median: {residual['position']['cosine_mean']:.4f}/{residual['position']['cosine_median']:.4f}.",
        f"- Position/cell gate >0.9: {residual['position_gate']['fraction_gt_0_9']:.1%}/{residual['cell_gate']['fraction_gt_0_9']:.1%}.",
        "",
        "## V1 eight-seed result",
        "",
        "| Method | E-hull | Stable | NUS | RMSD mean | RMSD median | Max force mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| A0 | {v1_a0['average_ehull']:.6f} | {v1_a0['stable_rate']:.1%} | {v1_a0['nus_rate']:.1%} | {v1_a0['relaxation_rmsd_mean']:.6f} | {v1_a0['relaxation_rmsd_median']:.6f} | {v1_a0['initial_max_force_mean']:.6f} |",
        f"| T1 | {v1_t1['average_ehull']:.6f} | {v1_t1['stable_rate']:.1%} | {v1_t1['nus_rate']:.1%} | {v1_t1['relaxation_rmsd_mean']:.6f} | {v1_t1['relaxation_rmsd_median']:.6f} | {v1_t1['initial_max_force_mean']:.6f} |",
        f"| T2 | {v1_t2['average_ehull']:.6f} | {v1_t2['stable_rate']:.1%} | {v1_t2['nus_rate']:.1%} | {v1_t2['relaxation_rmsd_mean']:.6f} | {v1_t2['relaxation_rmsd_median']:.6f} | {v1_t2['initial_max_force_mean']:.6f} |",
        "",
        "V1 was neither safe nor positive: T1 worsened median RMSD by 18.30%; T2 worsened mean RMSD by 10.84% and mean maximum force by 20.02%.",
        "",
        "## Gate V2 repair",
        "",
        f"- Train high-confidence target rate: {labels['splits']['train']['position_high_confidence_rate_ge_0_5']:.2%}",
        f"- Train zero/low target rate: {labels['splits']['train']['position_zero_low_rate_le_0_1']:.2%}",
        f"- Best step: {training['best_step']} of {training['steps_completed']}; seed {training['training_seed']}",
        f"- Test gate--utility Spearman: position {training['test_metrics']['position_gate_utility_spearman']:.3f}, cell {training['test_metrics']['cell_gate_utility_spearman']:.3f}",
        f"- Inference gate std: position {v2['inference_gate_statistics']['V2P']['position_gate_mean_std']:.3f}, cell {v2['inference_gate_statistics']['V2C']['cell_gate_mean_std']:.3f}",
        "",
        "## V2 eight-seed result",
        "",
        "| Method | E-hull | Stable | NUS | RMSD mean | RMSD median | Max force mean | Max force median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| A0 | {a0['average_ehull']:.6f} | {a0['stable_rate']:.1%} | {a0['nus_rate']:.1%} | {a0['relaxation_rmsd_mean']:.6f} | {a0['relaxation_rmsd_median']:.6f} | {a0['initial_max_force_mean']:.6f} | {a0['initial_max_force_median']:.6f} |",
        f"| V2P | {v2p['average_ehull']:.6f} | {v2p['stable_rate']:.1%} | {v2p['nus_rate']:.1%} | {v2p['relaxation_rmsd_mean']:.6f} | {v2p['relaxation_rmsd_median']:.6f} | {v2p['initial_max_force_mean']:.6f} | {v2p['initial_max_force_median']:.6f} |",
        f"| V2C | {v2c['average_ehull']:.6f} | {v2c['stable_rate']:.1%} | {v2c['nus_rate']:.1%} | {v2c['relaxation_rmsd_mean']:.6f} | {v2c['relaxation_rmsd_median']:.6f} | {v2c['initial_max_force_mean']:.6f} | {v2c['initial_max_force_median']:.6f} |",
        "",
        "V2P is quality-safe but effectively flat and misses every positive threshold. V2C remains unsafe due to +18.29% median RMSD. Stable, NUS, composition validity, and structure validity are unchanged for both.",
        "",
        "## Limitations and next action",
        "",
        "- Eight-seed development screen only.",
        "- MatterSim-5M results are surrogate evaluation, not DFT proof.",
        "- The one allowed Gate V2 repair has been used; no V3 is permitted.",
        "- Stop CG-TDR and move to a different second-innovation candidate.",
        "",
    ]
    FINAL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    progress = {
        "stage": "stop_for_review",
        "overall_status": "completed",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "test_baseline_comparison_completed": True,
        "identity_baseline_completed": True,
        "residual_diagnostics_completed": True,
        "v1_generation_success": 24,
        "v1_relax_success": 24,
        "gate_v2_training_completed": True,
        "v2_generation_success": 16,
        "v2_relax_success": 16,
        "CG_TDR_GATE_V2_VALID": True,
        "CG_TDR_V2_EIGHT_SEED_GO": False,
        "CG_TDR_MVP_GO": False,
        "CG_TDR_MVP_NO_GO": True,
        "CG_TDR_ROUTE_STOPPED": True,
        "eight_seed_started": True,
        "thirty_two_seed_started": False,
        "sixty_four_seed_started": False,
        "formal_seeds_started": False,
        "active_workers": {},
        "gpu_count": 8,
    }
    atomic_json(RESULTS / "progress/master_progress.json", progress)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
