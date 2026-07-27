#!/usr/bin/env python3
"""Evaluate Gate V2 on the same frozen eight development seeds."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.cg_tdr import analyze as base
from research.cg_tdr.analyze_v1 import candidate_decision, method_summary


RESULTS = Path("/data/dxl/results/cg_tdr/phase0")
REPORTS = Path("/data/dxl/reports/cg_tdr/phase0")
V1_SOURCE = REPORTS / "eight_seed"
OUT = REPORTS / "v2_eight_seed"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def inference_gate(method: str) -> dict[str, float]:
    rows = [
        json.loads(
            (
                RESULTS
                / f"generation/{method}/{seed}/cg_tdr_metrics.json"
            ).read_text()
        )
        for seed in range(23000, 23008)
    ]
    result = {}
    for key in ("position_gate_mean", "cell_gate_mean"):
        values = np.asarray([float(row.get(key, 0.0)) for row in rows])
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_std"] = float(values.std(ddof=0))
        result[f"{key}_gt_0_9_rate"] = float(np.mean(values > 0.9))
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = base.reference_dataset()
    frames = {
        method: pd.read_csv(V1_SOURCE / f"{method}_per_structure.csv")
        for method in ("A0", "T1")
    }
    for method in ("V2P", "V2C"):
        frame = base.official_frame(
            method,
            base.raw_frame(method, tuple(range(23000, 23008))),
            tuple(range(23000, 23008)),
            reference,
            OUT,
        )
        frame.to_csv(OUT / f"{method}_per_structure.csv", index=False)
        frames[method] = frame
    summaries = {
        method: method_summary(method, frame) for method, frame in frames.items()
    }
    decisions = {
        method: candidate_decision(
            frames["A0"], frames[method], summaries["A0"], summaries[method]
        )
        for method in ("V2P", "V2C")
    }
    training = json.loads(
        (RESULTS / "training_v2/training_summary.json").read_text()
    )
    training_metrics = training["test_metrics"]
    gates = {method: inference_gate(method) for method in ("V2P", "V2C")}
    v1_identity_rate = 0.0
    for method in ("V2P", "V2C"):
        gate = gates[method]
        position_selective = bool(
            gate["position_gate_mean_std"] > 0.008384357197599718
            and gate["position_gate_mean_gt_0_9_rate"] < 1.0
        )
        cell_selective = bool(
            method == "V2P"
            or (
                gate["cell_gate_mean_std"] > 0.01319300510654945
                and gate["cell_gate_mean_gt_0_9_rate"] < 1.0
            )
        )
        identity_improved = bool(
            training_metrics["position_identity_output_rate"] > v1_identity_rate
            and (
                method == "V2P"
                or training_metrics["cell_identity_output_rate"] > v1_identity_rate
            )
        )
        decisions[method]["gate_checks"] = {
            "training_gate_v2_valid": bool(training["CG_TDR_GATE_V2_VALID"]),
            "position_gate_std_greater_than_v1": position_selective,
            "cell_gate_std_greater_than_v1_if_enabled": cell_selective,
            "gate_gt_0_9_rate_lower_than_v1": bool(
                gate["position_gate_mean_gt_0_9_rate"] < 1.0
                and (
                    method == "V2P"
                    or gate["cell_gate_mean_gt_0_9_rate"] < 1.0
                )
            ),
            "identity_output_rate_improved": identity_improved,
            "latency_overhead_le_10_percent": decisions[method][
                "safety_checks"
            ]["latency_overhead_le_10_percent"],
        }
        decisions[method]["v2_go"] = bool(
            decisions[method]["safe"]
            and decisions[method]["positive"]
            and all(decisions[method]["gate_checks"].values())
        )
    passing = [method for method in ("V2P", "V2C") if decisions[method]["v2_go"]]
    selected = (
        min(
            passing,
            key=lambda method: (
                decisions[method]["changes"]["average_ehull"],
                decisions[method]["relative_changes"]["rmsd_mean"],
                decisions[method]["relative_changes"]["max_force_mean"],
            ),
        )
        if passing
        else None
    )
    paired_statistics = []
    for method in ("V2P", "V2C"):
        for metric in (
            "energy_above_hull_per_atom",
            "rmsd_from_relaxation",
            "initial_max_force_ev_ang",
        ):
            row = base.paired_stats(frames["A0"], frames[method], metric)
            row["method"] = method
            paired_statistics.append(row)
        for metric in (
            "stable",
            "comp_validity",
            "structure_validity",
            "novel",
            "unique",
        ):
            row = base.boolean_stats(frames["A0"], frames[method], metric)
            row["method"] = method
            paired_statistics.append(row)
    gate_valid = bool(
        training["CG_TDR_GATE_V2_VALID"]
        and any(
            all(decisions[method]["gate_checks"].values())
            for method in ("V2P", "V2C")
        )
    )
    go = bool(selected)
    result = {
        "status": "success",
        "seeds": list(range(23000, 23008)),
        "checkpoint": training["best_checkpoint"],
        "checkpoint_step": training["best_step"],
        "training_seed": training["training_seed"],
        "summaries": summaries,
        "inference_gate_statistics": gates,
        "candidate_decisions": decisions,
        "paired_statistics": paired_statistics,
        "selected_v2_candidate": selected,
        "CG_TDR_GATE_V2_VALID": gate_valid,
        "CG_TDR_V2_EIGHT_SEED_GO": go,
        "CG_TDR_ROUTE_STOPPED": not go,
        "CG_TDR_MVP_GO": False,
        "CG_TDR_MVP_NO_GO": not go,
        "THIRTY_TWO_SEED_STARTED": False,
        "SIXTY_FOUR_SEED_STARTED": False,
        "FORMAL_SEEDS_STARTED": False,
    }
    atomic_json(OUT / "v2_decision.json", result)
    if selected:
        atomic_json(
            OUT / "selected_v2_candidate.json",
            {"selected_config": selected},
        )
    lines = [
        "# CG-TDR Gate V2 eight-seed report",
        "",
        "- Same paired seeds: 23000--23007",
        "- A0 and V1/T1 results were reused; successful tasks were not rerun.",
        "- Independent evaluator: MatterSim-5M",
        "",
        "| Method | E-hull mean | Stable | NUS | RMSD mean | RMSD median | Max force mean | Max force median | Median time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("A0", "T1", "V2P", "V2C"):
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
            "## Decision",
            "",
            f"- `CG_TDR_GATE_V2_VALID={gate_valid}`",
            f"- `CG_TDR_V2_EIGHT_SEED_GO={go}`",
            f"- `CG_TDR_ROUTE_STOPPED={not go}`",
            f"- Selected V2 candidate: `{selected}`",
            "",
            (
                "The eight-seed Gate V2 passed; only the frozen 32-seed extension "
                "is now permitted."
                if go
                else "No Gate V2 candidate passed every frozen safety, positive, "
                "and selectivity gate. The CG-TDR route is stopped and no 32-seed "
                "task is permitted."
            ),
            "",
        ]
    )
    (OUT / "v2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
