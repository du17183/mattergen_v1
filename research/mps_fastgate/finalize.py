from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from research.mps_fastgate.common import REPORTS, RESULTS, atomic_json, atomic_text, set_stage


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FIELDS = (
    "random_tape_hash",
    "atomic_numbers_hash",
    "final_structure_hash",
    "positions_hash",
    "cell_hash",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bitwise_audit(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref = {(row["seed"], row["round"]): row for row in reference["result_index"]}
    cand = {(row["seed"], row["round"]): row for row in candidate["result_index"]}
    keys = sorted(ref.keys() | cand.keys())
    mismatches: list[dict[str, Any]] = []
    matches = {field: 0 for field in FIELDS}
    for key in keys:
        if key not in ref or key not in cand:
            mismatches.append({"seed": key[0], "round": key[1], "missing_pair": True})
            continue
        bad_fields = []
        for field in FIELDS:
            if ref[key][field] == cand[key][field]:
                matches[field] += 1
            else:
                bad_fields.append(field)
        if bad_fields:
            mismatches.append({"seed": key[0], "round": key[1], "fields": bad_fields})
    return {
        "pair_count": len(keys),
        "field_match_count": matches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "bitwise_equivalent": not mismatches and ref.keys() == cand.keys(),
    }


def completion_skew_estimates(config: dict[str, Any]) -> dict[int, float]:
    by_round: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in config["result_index"]:
        by_round[int(row["round"])][row["worker_id"]] += float(row["elapsed_seconds"])
    return {
        round_index: max(worker_totals.values()) - min(worker_totals.values())
        if len(worker_totals) > 1 else 0.0
        for round_index, worker_totals in sorted(by_round.items())
    }


def csv_rows(config: dict[str, Any], skews: dict[int, float]) -> list[dict[str, Any]]:
    rows = []
    for round_row in config["rounds"]:
        rows.append(
            {
                "config_id": config["config_id"],
                "mps_enabled": config["mps_enabled"],
                "workers_per_gpu": config["workers_per_gpu"],
                "active_thread_percentage": config["active_thread_percentage"],
                "round": round_row["round"],
                "wall_seconds": round_row["wall_seconds"],
                "samples_per_hour": round_row["samples_per_hour"],
                "success_count": round_row["success_count"],
                "failure_count": round_row["failure_count"],
                "p50_latency_seconds": round_row["p50_latency_seconds"],
                "p95_latency_seconds": round_row["p95_latency_seconds"],
                "worker_completion_skew_estimate_seconds": skews[round_row["round"]],
                "cpu_utilization_equivalent_percent": round_row["cpu_utilization_equivalent_percent"],
                "throughput_median_samples_per_hour": config["throughput_median_samples_per_hour"],
                "within_config_bitwise": config["within_config_bitwise"],
                "gpu_utilization_mean_percent": config["telemetry"]["gpu_utilization_mean_percent"],
                "peak_memory_mib": config["telemetry"]["peak_memory_mib"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def round_table(config: dict[str, Any], skews: dict[int, float]) -> list[str]:
    lines = [
        "| Round | Throughput (samples/h) | Wall (s) | P50 (s) | P95 (s) | Worker skew est. (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in config["rounds"]:
        lines.append(
            f"| {row['round']} | {row['samples_per_hour']:.4f} | {row['wall_seconds']:.2f} "
            f"| {row['p50_latency_seconds']:.2f} | {row['p95_latency_seconds']:.2f} "
            f"| {skews[row['round']]:.3f} |"
        )
    return lines


def main() -> int:
    availability = load_json(RESULTS / "mps_availability.json")
    final = load_json(RESULTS / "final_summary.json")
    s0 = load_json(RESULTS / "runs/S0_off_w2/summary.json")
    s1 = load_json(RESULTS / "runs/S1_mps_w2_p50/summary.json")
    audit = bitwise_audit(s0, s1)
    s0_skews = completion_skew_estimates(s0)
    s1_skews = completion_skew_estimates(s1)
    incremental_percent = (final["S1_INCREMENTAL_SPEEDUP"] - 1.0) * 100.0
    historical_two = 526.8200043796986 / 8.0
    historical_drift = (final["S0_THROUGHPUT"] / historical_two - 1.0) * 100.0

    if final["FINAL_STATE"] != "MPS_NO_GO":
        raise RuntimeError(f"unexpected frozen state: {final['FINAL_STATE']}")
    if not audit["bitwise_equivalent"]:
        raise RuntimeError("S0/S1 bitwise audit failed")
    if s0["failure_count"] or s1["failure_count"]:
        raise RuntimeError("generation failures are incompatible with frozen report")

    derived = {
        "S0_WORKER_COMPLETION_SKEW_ESTIMATE_BY_ROUND": s0_skews,
        "S1_WORKER_COMPLETION_SKEW_ESTIMATE_BY_ROUND": s1_skews,
        "S0_WORKER_COMPLETION_SKEW_ESTIMATE_MEDIAN": statistics.median(s0_skews.values()),
        "S1_WORKER_COMPLETION_SKEW_ESTIMATE_MEDIAN": statistics.median(s1_skews.values()),
        "S0_HISTORICAL_TWO_WORKER_DRIFT_PERCENT": historical_drift,
        "S1_INCREMENTAL_PERCENT": incremental_percent,
        "NOTE": "Skew is reconstructed from per-worker summed task latency because the original raw summary field was mislabeled; throughput and latency are unaffected.",
    }
    atomic_json(RESULTS / "bitwise_audit.json", audit)
    atomic_json(RESULTS / "derived_metrics.json", derived)

    final.update(
        {
            "S0_P50_LATENCY": s0["p50_latency_median_seconds"],
            "S0_P95_LATENCY": s0["p95_latency_median_seconds"],
            "S0_SUCCESS_COUNT": s0["success_count"],
            "S1_SUCCESS_COUNT": s1["success_count"],
            "S0_FAILURE_COUNT": s0["failure_count"],
            "S1_FAILURE_COUNT": s1["failure_count"],
            "S1_INCREMENTAL_PERCENT": incremental_percent,
            "BITWISE_PAIR_COUNT": audit["pair_count"],
            "BITWISE_MISMATCH_COUNT": audit["mismatch_count"],
            "S0_GPU_UTILIZATION_MEAN_PERCENT": s0["telemetry"]["gpu_utilization_mean_percent"],
            "S1_GPU_UTILIZATION_MEAN_PERCENT": s1["telemetry"]["gpu_utilization_mean_percent"],
            "S0_PEAK_MEMORY_MIB": s0["telemetry"]["peak_memory_mib"],
            "S1_PEAK_MEMORY_MIB": s1["telemetry"]["peak_memory_mib"],
        }
    )
    atomic_json(RESULTS / "final_summary.json", final)

    rows = csv_rows(s0, s0_skews) + csv_rows(s1, s1_skews)
    write_csv(REPORTS / "single_gpu_results.csv", rows)

    correctness_lines = [
        "| Check | Matching pairs | Total pairs |",
        "|---|---:|---:|",
        *[f"| {field} | {audit['field_match_count'][field]} | {audit['pair_count']} |" for field in FIELDS],
    ]
    report_lines = [
        "# NVIDIA MPS MatterGen fast-gate final report",
        "",
        "## Decision",
        "",
        "`FINAL_STATE=MPS_NO_GO`",
        "",
        f"MPS preserved bitwise outputs but changed median throughput from {final['S0_THROUGHPUT']:.4f} "
        f"to {final['S1_THROUGHPUT']:.4f} samples/hour ({incremental_percent:+.3f}%). "
        "This is below the frozen 3% engineering gate and the 5% paper gate.",
        "",
        "## Frozen protocol",
        "",
        "- GPU: NVIDIA RTX PRO 5000 72GB Blackwell, GPU 0",
        f"- Driver: {availability['driver']}; driver-reported CUDA: {availability['driver_reported_cuda']}; PyTorch CUDA: {availability['project_torch_cuda']}",
        "- Model: C0 original MatterGen, batch size 1 per process, FP32, full Predictor/Corrector",
        "- S0: MPS OFF, 2 persistent workers",
        "- S1: MPS ON, 2 persistent workers, 50% active threads per client",
        "- Seeds: 27000-27015; three timed repeats; 48 trajectories/configuration",
        "- Model load and one real forward warm-up excluded from timed windows",
        "",
        "## Aggregate result",
        "",
        "| Metric | S0 MPS OFF | S1 MPS ON |",
        "|---|---:|---:|",
        f"| Median throughput (samples/h) | {final['S0_THROUGHPUT']:.4f} | {final['S1_THROUGHPUT']:.4f} |",
        f"| Median P50 latency (s) | {s0['p50_latency_median_seconds']:.3f} | {s1['p50_latency_median_seconds']:.3f} |",
        f"| Median P95 latency (s) | {s0['p95_latency_median_seconds']:.3f} | {s1['p95_latency_median_seconds']:.3f} |",
        f"| Mean GPU utilization (%) | {s0['telemetry']['gpu_utilization_mean_percent']:.3f} | {s1['telemetry']['gpu_utilization_mean_percent']:.3f} |",
        f"| Peak GPU memory (MiB) | {s0['telemetry']['peak_memory_mib']:.0f} | {s1['telemetry']['peak_memory_mib']:.0f} |",
        f"| Successful trajectories | {s0['success_count']}/48 | {s1['success_count']}/48 |",
        f"| Failures | {s0['failure_count']} | {s1['failure_count']} |",
        "",
        f"S1/S0 incremental speedup: **{final['S1_INCREMENTAL_SPEEDUP']:.6f}x ({incremental_percent:+.3f}%)**.  "
        f"S1 versus the historical one-worker reference: **{final['S1_TOTAL_SPEEDUP_VS_ONE_WORKER']:.6f}x**.",
        "",
        "## S0 repeats",
        "",
        *round_table(s0, s0_skews),
        "",
        "## S1 repeats",
        "",
        *round_table(s1, s1_skews),
        "",
        "## Correctness",
        "",
        *correctness_lines,
        "",
        "Within each configuration, all three repeats were also bitwise identical. Scientific outputs were recorded once per configuration; no MatterSim evaluation was needed because raw outputs matched exactly.",
        "",
        "## Gate and cleanup",
        "",
        "- S1 incremental throughput was below 3%, so S2 and 8-GPU confirmation were not started.",
        "- MPS control/server were stopped cooperatively; project MPS pipe/log directories were cleaned.",
        "- GPU workers after exit: 0; other processes terminated: false; SIGKILL used: false.",
        "",
        "## Limitations",
        "",
        f"- The historical one-worker reference was reused because S0 drift was {historical_drift:+.3f}%, below the frozen 10% retest threshold.",
        "- The original worker finish-spread field was mislabeled. The tables use a reconstructed per-worker summed-latency skew estimate; primary wall-clock throughput and latency are unaffected.",
        "- This result applies to the tested C0 batch-1 persistent-worker workload and this driver/PyTorch stack.",
        "",
    ]
    report = "\n".join(report_lines)
    atomic_text(REPORTS / "final_report.md", report)
    atomic_text(REPORTS / "single_gpu_report.md", report)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name, payload in {
        "mps_availability.json": availability,
        "final_summary.json": final,
        "bitwise_audit.json": audit,
        "derived_metrics.json": derived,
    }.items():
        atomic_json(ARTIFACTS / name, payload)
    atomic_text(ARTIFACTS / "final_report.md", report)
    write_csv(ARTIFACTS / "single_gpu_results.csv", rows)
    set_stage("stop_for_review", "stop_for_review", "Fast-gate complete; waiting for human review.", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
