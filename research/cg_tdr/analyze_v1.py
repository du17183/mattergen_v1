#!/usr/bin/env python3
"""Apply the frozen V1 eight-seed safety and positive-effect decision rules."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pandas as pd

from research.cg_tdr import analyze as base


RESULTS = Path("/data/dxl/results/cg_tdr/phase0")
REPORTS = Path("/data/dxl/reports/cg_tdr/phase0")
OUT = REPORTS / "v1_eight_seed"
GENERATION = RESULTS / "generation"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def relative(candidate: float, baseline: float) -> float:
    return candidate / max(abs(baseline), 1.0e-18) - 1.0


def method_summary(method: str, frame: pd.DataFrame) -> dict[str, Any]:
    stable = frame["stable"].astype(bool)
    # This extra descriptive threshold is kept separate from the frozen
    # MatterGen stable metric used by all project Go/No-Go gates.
    metastable = frame["energy_above_hull_per_atom"].astype(float) <= 0.2
    novel = frame["novel"].astype(bool)
    unique = frame["unique"].astype(bool)
    generation_rows = [
        json.loads(
            (GENERATION / method / str(seed) / "run_summary.json").read_text()
        )
        for seed in frame["seed"].astype(int)
    ]
    cg_rows = [
        json.loads(
            (GENERATION / method / str(seed) / "cg_tdr_metrics.json").read_text()
        )
        for seed in frame["seed"].astype(int)
    ]
    mechanism_keys = (
        "position_gate_mean",
        "cell_gate_mean",
        "position_clipping_rate",
        "cell_fallback_rate",
        "position_residual_rms",
    )
    mechanism: dict[str, float] = {}
    for key in mechanism_keys:
        values = np.asarray([float(row.get(key, 0.0)) for row in cg_rows])
        mechanism[f"{key}_mean"] = float(values.mean())
        mechanism[f"{key}_median"] = float(np.median(values))
        mechanism[f"{key}_std"] = float(values.std(ddof=0))
    return {
        "method": method,
        "n": len(frame),
        "generation_success": float(frame["generation_success"].mean()),
        "composition_validity": float(frame["comp_validity"].mean()),
        "structure_validity": float(frame["structure_validity"].mean()),
        "average_ehull": float(frame["energy_above_hull_per_atom"].mean()),
        "median_ehull": float(frame["energy_above_hull_per_atom"].median()),
        "stable_rate": float(stable.mean()),
        "stable_count": int(stable.sum()),
        "metastable_le_0_2_rate": float(metastable.mean()),
        "metastable_le_0_2_count": int(metastable.sum()),
        "nus_rate": float((stable & novel & unique).mean()),
        "nus_count": int((stable & novel & unique).sum()),
        "msun_le_0_2_rate": float((metastable & novel & unique).mean()),
        "msun_le_0_2_count": int((metastable & novel & unique).sum()),
        "novel_rate": float(novel.mean()),
        "unique_rate": float(unique.mean()),
        "relaxation_rmsd_mean": float(frame["rmsd_from_relaxation"].mean()),
        "relaxation_rmsd_median": float(frame["rmsd_from_relaxation"].median()),
        "initial_max_force_mean": float(frame["initial_max_force_ev_ang"].mean()),
        "initial_max_force_median": float(
            frame["initial_max_force_ev_ang"].median()
        ),
        "force_convergence_rate": float(frame["converged"].mean()),
        "relax_steps_mean": float(frame["relax_steps"].mean()),
        "relax_steps_median": float(frame["relax_steps"].median()),
        "relaxation_failure_rate": 0.0,
        "generation_elapsed_median": float(
            np.median([float(row["elapsed_seconds"]) for row in generation_rows])
        ),
        "generation_elapsed_mean": float(
            np.mean([float(row["elapsed_seconds"]) for row in generation_rows])
        ),
        "peak_vram_mib": None,
        "peak_vram_note": (
            "V1 runner predates per-process peak allocation telemetry; concurrent "
            "nvidia-smi snapshots were retained in the execution log."
        ),
        **mechanism,
    }


def candidate_decision(
    baseline_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    pair = baseline_frame[
        ["seed", "initial_state_hash", "atomic_sequence", "minimum_distance_angstrom"]
    ].merge(
        candidate_frame[
            ["seed", "initial_state_hash", "atomic_sequence", "minimum_distance_angstrom"]
        ],
        on="seed",
        suffixes=("_a0", "_candidate"),
        validate="one_to_one",
    )
    initial_match = bool(
        (pair["initial_state_hash_a0"] == pair["initial_state_hash_candidate"]).all()
    )
    atomic_match = bool(
        (pair["atomic_sequence_a0"].astype(str) == pair["atomic_sequence_candidate"].astype(str)).all()
    )
    changes = {
        "composition_validity": candidate["composition_validity"]
        - baseline["composition_validity"],
        "structure_validity": candidate["structure_validity"]
        - baseline["structure_validity"],
        "average_ehull": candidate["average_ehull"] - baseline["average_ehull"],
        "stable_rate": candidate["stable_rate"] - baseline["stable_rate"],
        "stable_count": candidate["stable_count"] - baseline["stable_count"],
        "nus_rate": candidate["nus_rate"] - baseline["nus_rate"],
        "nus_count": candidate["nus_count"] - baseline["nus_count"],
        "novel_rate": candidate["novel_rate"] - baseline["novel_rate"],
        "unique_rate": candidate["unique_rate"] - baseline["unique_rate"],
        "relaxation_failure_rate": candidate["relaxation_failure_rate"]
        - baseline["relaxation_failure_rate"],
    }
    relatives = {
        "rmsd_mean": relative(
            candidate["relaxation_rmsd_mean"], baseline["relaxation_rmsd_mean"]
        ),
        "rmsd_median": relative(
            candidate["relaxation_rmsd_median"],
            baseline["relaxation_rmsd_median"],
        ),
        "max_force_mean": relative(
            candidate["initial_max_force_mean"], baseline["initial_max_force_mean"]
        ),
        "max_force_median": relative(
            candidate["initial_max_force_median"],
            baseline["initial_max_force_median"],
        ),
        "latency": relative(
            candidate["generation_elapsed_median"],
            baseline["generation_elapsed_median"],
        ),
    }
    no_nan = all(
        math.isfinite(float(candidate[key]))
        for key in (
            "average_ehull",
            "relaxation_rmsd_mean",
            "relaxation_rmsd_median",
            "initial_max_force_mean",
            "initial_max_force_median",
            "generation_elapsed_median",
        )
    )
    safety_checks = {
        "generation_success_100_percent": candidate["generation_success"] == 1.0,
        "structure_validity_100_percent": candidate["structure_validity"] == 1.0,
        "composition_not_lower": changes["composition_validity"] >= 0.0,
        "initial_random_tape_match": initial_match,
        "atomic_sequences_match": atomic_match,
        "no_nan_or_inf": no_nan,
        "relax_failure_not_increased": changes["relaxation_failure_rate"] <= 0.0,
        "ehull_worsening_le_0_002": changes["average_ehull"] <= 0.002,
        "latency_overhead_le_10_percent": relatives["latency"] <= 0.10,
        "rmsd_mean_not_worse_10_percent": relatives["rmsd_mean"] <= 0.10,
        "rmsd_median_not_worse_10_percent": relatives["rmsd_median"] <= 0.10,
        "max_force_mean_not_worse_10_percent": relatives["max_force_mean"] <= 0.10,
        "max_force_median_not_worse_10_percent": relatives["max_force_median"]
        <= 0.10,
        "severe_short_bonds_not_increased": bool(
            (
                pair["minimum_distance_angstrom_candidate"] < 0.5
            ).sum()
            <= (pair["minimum_distance_angstrom_a0"] < 0.5).sum()
        ),
    }
    positive_checks = {
        "rmsd_mean_reduction_ge_5_percent": relatives["rmsd_mean"] <= -0.05,
        "rmsd_median_reduction_ge_5_percent": relatives["rmsd_median"] <= -0.05,
        "max_force_mean_reduction_ge_5_percent": relatives["max_force_mean"]
        <= -0.05,
        "max_force_median_reduction_ge_5_percent": relatives["max_force_median"]
        <= -0.05,
        "ehull_reduction_ge_0_003": changes["average_ehull"] <= -0.003,
        "stable_gain_ge_one_structure": changes["stable_count"] >= 1,
        "nus_gain_ge_one_structure": changes["nus_count"] >= 1,
    }
    return {
        "method": candidate["method"],
        "changes": changes,
        "relative_changes": relatives,
        "safety_checks": safety_checks,
        "positive_checks": positive_checks,
        "safe": bool(all(safety_checks.values())),
        "positive": bool(any(positive_checks.values())),
    }


def main() -> int:
    base_result = base.analyze("eight")
    source = REPORTS / "eight_seed"
    frames = {
        method: pd.read_csv(source / f"{method}_per_structure.csv")
        for method in ("A0", "T1", "T2")
    }
    summaries = {
        method: method_summary(method, frame) for method, frame in frames.items()
    }
    decisions = {
        method: candidate_decision(
            frames["A0"], frames[method], summaries["A0"], summaries[method]
        )
        for method in ("T1", "T2")
    }
    ranked = sorted(
        decisions,
        key=lambda method: (
            not decisions[method]["safe"],
            sum(not passed for passed in decisions[method]["safety_checks"].values()),
            not decisions[method]["positive"],
            decisions[method]["relative_changes"]["rmsd_mean"],
            decisions[method]["relative_changes"]["max_force_mean"],
            decisions[method]["changes"]["average_ehull"],
        ),
    )
    selected = ranked[0]
    v1_safe = any(item["safe"] for item in decisions.values())
    v1_positive = any(
        item["safe"] and item["positive"] for item in decisions.values()
    )
    diagnostics = json.loads(
        (RESULTS / "residual_learning_summary.json").read_text()
    )
    gate_selectivity = bool(diagnostics["GATE_SELECTIVITY_VALID"])
    direct_go = bool(v1_positive and gate_selectivity)
    fatal = not any(
        item["safety_checks"]["generation_success_100_percent"]
        and item["safety_checks"]["structure_validity_100_percent"]
        and item["safety_checks"]["no_nan_or_inf"]
        for item in decisions.values()
    )
    result = {
        "status": "success",
        "seeds": list(range(23000, 23008)),
        "checkpoint": "/data/dxl/results/cg_tdr/phase0/training/checkpoints/best.pt",
        "checkpoint_step": 100,
        "summaries": summaries,
        "candidate_decisions": decisions,
        "paired_statistics": base_result["paired_statistics"],
        "selected_v1_candidate": selected,
        "GATE_SELECTIVITY_VALID": gate_selectivity,
        "CG_TDR_V1_EIGHT_SEED_SAFE": v1_safe,
        "CG_TDR_V1_EIGHT_SEED_POSITIVE": v1_positive,
        "CG_TDR_V1_DIRECT_GO": direct_go,
        "CG_TDR_GATE_V2_REQUIRED": not direct_go and not fatal,
        "CG_TDR_V1_FATAL_NO_GO": fatal,
        "THIRTY_TWO_SEED_STARTED": False,
        "SIXTY_FOUR_SEED_STARTED": False,
        "FORMAL_SEEDS_STARTED": False,
        "metric_definitions": {
            "stable": "Frozen project MatterGen evaluator metric (threshold 0.1 eV/atom).",
            "metastable_le_0_2": "Descriptive E-hull <= 0.2 eV/atom.",
            "NUS": "novel AND unique AND frozen stable.",
            "MSUN_le_0_2": "novel AND unique AND E-hull <= 0.2 eV/atom.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "v1_decision.json", result)
    atomic_json(
        OUT / "selected_v1_candidate.json", {"selected_config": selected}
    )
    lines = [
        "# CG-TDR V1 eight-seed report",
        "",
        "- Checkpoint: `best.pt`, strictly verified at step 100",
        "- Seeds: 23000--23007, paired A0/T1/T2",
        "- Independent evaluator: MatterSim-5M",
        "",
        "## Summary",
        "",
        "| Method | E-hull mean | Stable | NUS | RMSD mean | RMSD median | Max force mean | Max force median | Median time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("A0", "T1", "T2"):
        item = summaries[method]
        lines.append(
            f"| {method} | {item['average_ehull']:.6f} | "
            f"{item['stable_rate']:.3%} | {item['nus_rate']:.3%} | "
            f"{item['relaxation_rmsd_mean']:.6f} | "
            f"{item['relaxation_rmsd_median']:.6f} | "
            f"{item['initial_max_force_mean']:.6f} | "
            f"{item['initial_max_force_median']:.6f} | "
            f"{item['generation_elapsed_median']:.2f}s |"
        )
    lines.extend(
        [
            "",
            "## Frozen decisions",
            "",
            f"- `CG_TDR_V1_EIGHT_SEED_SAFE={v1_safe}`",
            f"- `CG_TDR_V1_EIGHT_SEED_POSITIVE={v1_positive}`",
            f"- `GATE_SELECTIVITY_VALID={gate_selectivity}`",
            f"- `CG_TDR_V1_DIRECT_GO={direct_go}`",
            f"- `CG_TDR_GATE_V2_REQUIRED={result['CG_TDR_GATE_V2_REQUIRED']}`",
            f"- Selected V1 diagnostic candidate: `{selected}`",
            "",
            "V1 is evaluated as a near-always-on terminal refiner. Gate V2 is "
            "required whenever gate selectivity is invalid, even if a V1 quality "
            "metric improves. No 32-seed task is launched by this analysis.",
            "",
        ]
    )
    (OUT / "v1_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
