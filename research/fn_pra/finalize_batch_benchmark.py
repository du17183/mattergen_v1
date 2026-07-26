"""Merge single-GPU and fixed-eight-GPU batch benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd

from research.fn_pra.phase1_common import (
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
    set_stage,
)


ROOT = RESULTS / "batch_benchmark"
BATCHES = (1, 2, 4, 8)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_equivalence(reference, candidate) -> dict:
    same_count = len(reference) == len(candidate)
    same_numbers = same_count and np.array_equal(reference.numbers, candidate.numbers)
    if not same_count:
        return {
            "same_atom_count": False,
            "same_atomic_numbers": False,
            "position_max_abs_ang": None,
            "cell_max_abs_ang": None,
            "level2_numeric_equivalence": False,
        }
    position_max = float(np.max(np.abs(reference.positions - candidate.positions)))
    cell_max = float(np.max(np.abs(reference.cell.array - candidate.cell.array)))
    level2 = bool(
        same_numbers
        and np.allclose(reference.positions, candidate.positions, rtol=1e-4, atol=1e-4)
        and np.allclose(reference.cell.array, candidate.cell.array, rtol=1e-4, atol=1e-4)
    )
    return {
        "same_atom_count": True,
        "same_atomic_numbers": bool(same_numbers),
        "position_max_abs_ang": position_max,
        "cell_max_abs_ang": cell_max,
        "level2_numeric_equivalence": level2,
    }


def main() -> int:
    summaries = {
        batch: read_json(ROOT / f"formal_b{batch}/summary.json") for batch in BATCHES
    }
    telemetry = {
        batch: read_json(ROOT / f"formal_b{batch}/telemetry.json")["summary"]
        for batch in BATCHES
    }
    fixed = read_json(REPORTS / "fixed8_batch_benchmark.json")
    fixed_by_batch = {
        int(row["batch_size_per_gpu"]): row for row in fixed["rows"]
    }
    rng_audit = read_json(REPORTS / "batch_rng_isolation_audit.json")
    reference_frames = ase.io.read(ROOT / "formal_b1/generated_crystals.extxyz", ":")
    if not isinstance(reference_frames, list):
        reference_frames = [reference_frames]
    reference = reference_frames[0]
    baseline_throughput = summaries[1]["median_samples_per_hour"]
    baseline_fixed_throughput = fixed_by_batch[1]["fixed8_samples_per_hour"]
    baseline_validity = np.mean(
        [row["structure_valid"] for row in summaries[1]["validity"]]
    )
    rows = []
    for batch in BATCHES:
        frames = ase.io.read(ROOT / f"formal_b{batch}/generated_crystals.extxyz", ":")
        if not isinstance(frames, list):
            frames = [frames]
        equivalence = final_equivalence(reference, frames[0])
        summary = summaries[batch]
        first_repeat = summary["repeats"][0]
        initial_exact = (
            first_repeat["initial_hashes"][0]
            == summaries[1]["repeats"][0]["initial_hashes"][0]
        )
        final_level1 = (
            first_repeat["final_hashes"][0]
            == summaries[1]["repeats"][0]["final_hashes"][0]
        )
        structure_validity = float(
            np.mean([row["structure_valid"] for row in summary["validity"]])
        )
        composition_validity = float(
            np.mean([row["composition_valid"] for row in summary["validity"]])
        )
        throughput_gain = summary["median_samples_per_hour"] / baseline_throughput - 1.0
        fixed_gain = (
            fixed_by_batch[batch]["fixed8_samples_per_hour"]
            / baseline_fixed_throughput
            - 1.0
        )
        hard_gate = bool(
            batch == 1
            or (
                throughput_gain >= 0.15
                and fixed_gain >= 0.15
                and summary["generation_success_rate"] == 1.0
                and structure_validity >= baseline_validity
                and initial_exact
                and equivalence["level2_numeric_equivalence"]
                and rng_audit["rng_isolation_passed"]
            )
        )
        rows.append(
            {
                "batch_size": batch,
                "single_gpu_median_batch_latency_seconds": summary[
                    "median_batch_latency_seconds"
                ],
                "single_gpu_median_sample_latency_seconds": summary[
                    "median_sample_latency_seconds"
                ],
                "single_gpu_samples_per_hour": summary["median_samples_per_hour"],
                "single_gpu_throughput_gain_vs_b1": throughput_gain,
                "fixed8_samples_per_hour": fixed_by_batch[batch][
                    "fixed8_samples_per_hour"
                ],
                "fixed8_throughput_gain_vs_b1": fixed_gain,
                "gpu_utilization_mean_percent": telemetry[batch].get(
                    "mean_utilization_percent"
                ),
                "gpu_power_mean_w": telemetry[batch].get("mean_power_w"),
                "peak_vram_allocated_bytes": max(
                    repeat["peak_allocated_bytes"] for repeat in summary["repeats"]
                ),
                "cpu_percent_one_core_median": float(
                    np.median(
                        [repeat["cpu_percent_one_core"] for repeat in summary["repeats"]]
                    )
                ),
                "physical_model_forward_count": summary[
                    "physical_model_forward_count"
                ],
                "generation_success_rate": summary["generation_success_rate"],
                "structure_validity": structure_validity,
                "composition_validity": composition_validity,
                "deterministic_repeats_level1": summary[
                    "deterministic_repeats_level1"
                ],
                "same_seed_initial_level1": initial_exact,
                "same_seed_final_level1": final_level1,
                **equivalence,
                "rng_isolation_passed": rng_audit["rng_isolation_passed"],
                "hard_gate_passed": hard_gate,
            }
        )
    passing = [row for row in rows if row["batch_size"] > 1 and row["hard_gate_passed"]]
    selected = max(passing, key=lambda row: row["fixed8_samples_per_hour"]) if passing else rows[0]
    recommendation = {
        "created_at": now(),
        "BATCH_ENGINEERING_GO": bool(passing),
        "recommended_batch_size": selected["batch_size"],
        "reason": (
            "Selected the highest-throughput batch satisfying speed, validity, RNG isolation, "
            "and final Level-2 equivalence."
            if passing
            else "No batch_size>1 satisfied the required final same-seed Level-2 equivalence; retain native batch_size=1."
        ),
        "rng_isolation_passed": rng_audit["rng_isolation_passed"],
        "rows": rows,
    }
    pd.DataFrame(rows).to_csv(REPORTS / "batch_benchmark.csv", index=False)
    atomic_json(REPORTS / "batch_recommendation.json", recommendation)
    table = pd.DataFrame(rows)[
        [
            "batch_size",
            "single_gpu_samples_per_hour",
            "single_gpu_throughput_gain_vs_b1",
            "fixed8_samples_per_hour",
            "fixed8_throughput_gain_vs_b1",
            "same_seed_final_level1",
            "level2_numeric_equivalence",
            "hard_gate_passed",
        ]
    ].to_markdown(index=False)
    atomic_text(
        REPORTS / "batch_benchmark.md",
        f"""# A0 lossless multi-trajectory batch benchmark

{table}

- Each configuration used three warmups and at least five timed repeats.
- Single-GPU measured sample count was fixed at 40 per configuration.
- The fixed-eight-GPU benchmark used the same per-GPU protocol.
- Per-seed initial states and final RNG states are independent of batch membership/order:
  `{rng_audit["rng_isolation_passed"]}`.
- Recommended batch size: `{recommendation["recommended_batch_size"]}`.
- `BATCH_ENGINEERING_GO={recommendation["BATCH_ENGINEERING_GO"]}`.

Final Level-2 equivalence requires identical atom types and Cartesian
positions/cell numerically close at `rtol=1e-4, atol=1e-4`. This is deliberately
stricter than merely preserving validity or distributional quality.
""",
    )
    set_stage(
        "batch_benchmark",
        "success",
        f"Single-GPU and fixed8 batch benchmark complete; BATCH_ENGINEERING_GO={recommendation['BATCH_ENGINEERING_GO']}.",
        recommendation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
