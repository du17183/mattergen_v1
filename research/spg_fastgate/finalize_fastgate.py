"""Amdahl analysis and final SPG-MatterGen Fast Gate decision."""

from __future__ import annotations

import json
import csv
from io import StringIO
from pathlib import Path

import pandas as pd

from research.spg_fastgate.common import (
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
    read_json,
    set_stage,
)


def performance_index(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {
        (str(row["method"]), int(row["batch_size"])): row
        for row in rows
    }


def nsys_intervals(path: str, *, api: bool = False) -> list[tuple[float, float]]:
    """Read timestamp intervals from an Nsight CSV with a textual preamble."""

    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Start" in line and ("Duration" in line or "Dur" in line) and "," in line
        ),
        None,
    )
    if header_index is None:
        return []
    reader = csv.DictReader(StringIO("\n".join(lines[header_index:])))
    fieldnames = reader.fieldnames or []
    if api:
        start_name = next(
            (name for name in fieldnames if name.lower().startswith("api start")),
            None,
        )
        duration_name = next(
            (
                name
                for name in fieldnames
                if name.lower().startswith(("api dur", "api duration"))
            ),
            None,
        )
    else:
        start_name = next(
            (name for name in fieldnames if name.lower().startswith("start")),
            None,
        )
        duration_name = next(
            (
                name
                for name in fieldnames
                if name.lower().startswith(("duration", "dur"))
            ),
            None,
        )
    if start_name is None or duration_name is None:
        return []
    intervals = []
    for row in reader:
        try:
            start = float(str(row[start_name]).replace(",", ""))
            duration = float(str(row[duration_name]).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if duration >= 0:
            intervals.append((start, start + duration))
    return intervals


def interval_coverage(intervals: list[tuple[float, float]]) -> tuple[float | None, float | None]:
    """Return covered and gap shares over the first-to-last event span."""

    if not intervals:
        return None, None
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    span = merged[-1][1] - merged[0][0]
    if span <= 0:
        return None, None
    covered = sum(end - start for start, end in merged)
    covered_share = min(1.0, max(0.0, covered / span))
    return covered_share, 1.0 - covered_share


def main() -> int:
    set_stage(
        "amdahl_analysis",
        "running",
        "Combining performance, profiler, BF16, compile, and quality evidence.",
    )
    performance = read_json(RESULTS / "performance_baseline.json")
    profiler = read_json(RESULTS / "profiler_breakdown.json")
    compile_report = read_json(RESULTS / "compile_audit.json")
    bf16_state = read_json(RESULTS / "bf16_state_probe.json")
    quality = read_json(REPORTS / "b4_quality_report.json")
    rows = performance["rows"]
    indexed = performance_index(rows)
    c0_b4_speedup = (
        indexed[("C0", 4)]["fixed8_samples_per_hour"]
        / indexed[("C0", 1)]["fixed8_samples_per_hour"]
    )
    a0_b4_speedup = (
        indexed[("A0", 4)]["fixed8_samples_per_hour"]
        / indexed[("A0", 1)]["fixed8_samples_per_hour"]
    )
    pytorch_profiles = profiler["pytorch"]
    if "scope_totals" in pytorch_profiles:
        pytorch_profiles = {"C0": pytorch_profiles, "A0": pytorch_profiles}

    def method_shares(method: str) -> dict[str, float]:
        scopes = pytorch_profiles[method]["scope_totals"]
        total_cuda = scopes.get("spg::sampling_total", {}).get(
            "cuda_time_total_us", 0.0
        )

        def share(scope: str) -> float:
            value = scopes.get(scope, {}).get("cuda_time_total_us", 0.0)
            return float(value / total_cuda) if total_cuda else 0.0

        return {
            "periodic_graph": share("spg::periodic_neighbor_construction"),
            "triplet": share("spg::triplet_construction"),
            "interaction_graph_total": share("spg::interaction_graph_total"),
            "scatter": share("spg::scatter_segment"),
            "gemnet": share("spg::gemnet_forward"),
            "gemnet_blocks": share("spg::gemnet_interaction_block"),
            "cfg": share("spg::cfg_score_total"),
            "predictor": share("spg::predictor_update"),
            "corrector": share("spg::corrector_update"),
        }

    shares_by_method = {
        method: method_shares(method) for method in ("C0", "A0")
    }
    c0_shares = shares_by_method["C0"]
    periodic_share = c0_shares["periodic_graph"]
    triplet_share = c0_shares["triplet"]
    graph_total_share = c0_shares["interaction_graph_total"]
    scatter_share = c0_shares["scatter"]
    gemnet_share = c0_shares["gemnet"]
    block_share = c0_shares["gemnet_blocks"]
    cfg_share = c0_shares["cfg"]
    predictor_share = c0_shares["predictor"]
    corrector_share = c0_shares["corrector"]
    graph_triplet_share = max(
        min(
            1.0,
            max(
                row["interaction_graph_total"],
                row["periodic_graph"] + row["triplet"],
            ),
        )
        for row in shares_by_method.values()
    )
    c0_b1_utilization = float(
        indexed[("C0", 1)]["mean_gpu_utilization_percent"]
    )
    trace_tables = profiler["nsight"].get("trace_tables", {})
    gpu_intervals = nsys_intervals(
        trace_tables.get("cuda_gpu_trace", {}).get("path", "")
    ) if trace_tables.get("cuda_gpu_trace") else []
    api_intervals = nsys_intervals(
        trace_tables.get("cuda_kern_exec_trace", {}).get("path", ""),
        api=True,
    ) if trace_tables.get("cuda_kern_exec_trace") else []
    nsys_process_gpu_active_share, nsys_process_gpu_idle_share = interval_coverage(
        gpu_intervals
    )
    api_busy_share, cpu_launch_gap_share = interval_coverage(api_intervals)
    gpu_active_share_proxy = c0_b1_utilization / 100.0
    if cpu_launch_gap_share is None:
        cpu_launch_gap_share = max(0.0, 1.0 - gpu_active_share_proxy)
    kernel_count_estimate = sum(
        int(row.get("count", 0))
        for row in pytorch_profiles["C0"]["top_cuda_operators"]
        if row.get("self_cuda_time_total_us", 0.0) > 0
    )
    # Conservative static graph model: graph/triplet path becomes 3x faster.
    estimated_static_graph_speedup = (
        1.0
        / (1.0 - graph_triplet_share + graph_triplet_share / 3.0)
        if graph_triplet_share < 1.0
        else 3.0
    )
    evidence_conditions = {
        "periodic_plus_triplet_ge_15_percent": graph_triplet_share >= 0.15,
        "dynamic_compile_blocker": bool(
            compile_report.get("STATIC_GRAPH_REQUIRED")
        ),
        "kernel_fragmentation": kernel_count_estimate >= 1000,
        "single_worker_gpu_utilization_low": c0_b1_utilization < 60.0,
    }
    static_worth = bool(
        estimated_static_graph_speedup >= 1.08
        and any(evidence_conditions.values())
    )
    native_go = bool(quality["NATIVE_BATCHING_GO"])
    bf16_go = bool(quality["FIELD_SAFE_BF16_GO"])
    partial_compile = bool(compile_report["PARTIAL_COMPILE_WORKS"])
    endpoint_speedups = {
        method: row["endpoint_speedup"]
        for method, row in quality.get("bf16_decisions", {}).items()
    }
    fastgate_go = bool(
        native_go
        or static_worth
        or (partial_compile and c0_b1_utilization < 60.0)
    )
    result = {
        "created_at": now(),
        "C0_B1_THROUGHPUT": indexed[("C0", 1)]["fixed8_samples_per_hour"],
        "C0_B4_THROUGHPUT": indexed[("C0", 4)]["fixed8_samples_per_hour"],
        "C0_B8_THROUGHPUT": indexed[("C0", 8)]["fixed8_samples_per_hour"],
        "A0_B1_THROUGHPUT": indexed[("A0", 1)]["fixed8_samples_per_hour"],
        "A0_B4_THROUGHPUT": indexed[("A0", 4)]["fixed8_samples_per_hour"],
        "A0_B8_THROUGHPUT": indexed[("A0", 8)]["fixed8_samples_per_hour"],
        "C0_B4_SPEEDUP": c0_b4_speedup,
        "A0_B4_SPEEDUP": a0_b4_speedup,
        "C0_B4_QUALITY_EQUIVALENT": quality[
            "C0_B4_QUALITY_EQUIVALENT"
        ],
        "A0_B4_QUALITY_EQUIVALENT": quality[
            "A0_B4_QUALITY_EQUIVALENT"
        ],
        "NATIVE_BATCHING_GO": native_go,
        "BF16_ATOMIC_COSINE": bf16_state.get(
            "field_safe_bf16_comparison", {}
        )
        .get("fields", {})
        .get("atomic_numbers", {})
        .get("cosine"),
        "BF16_POSITION_COSINE": bf16_state.get(
            "field_safe_bf16_comparison", {}
        )
        .get("fields", {})
        .get("pos", {})
        .get("cosine"),
        "BF16_CELL_COSINE": bf16_state.get(
            "field_safe_bf16_comparison", {}
        )
        .get("fields", {})
        .get("cell", {})
        .get("cosine"),
        "BF16_FORWARD_SPEEDUP": bf16_state.get(
            "field_safe_bf16_comparison", {}
        ).get("speedup"),
        "BF16_ENDPOINT_SPEEDUP": min(endpoint_speedups.values())
        if endpoint_speedups
        else None,
        "BF16_ENDPOINT_SPEEDUPS": endpoint_speedups,
        "FIELD_SAFE_BF16_GO": bf16_go,
        "COMPILE_GRAPH_BREAKS": compile_report.get(
            "total_graph_breaks_observed", 0
        ),
        "COMPILE_RECOMPILES": compile_report.get(
            "max_recompiles_observed", 0
        ),
        "PARTIAL_COMPILE_SPEEDUP": max(
            compile_report.get("best_block_speedup", 0.0),
            compile_report.get("best_other_speedup", 0.0),
        ),
        "PARTIAL_COMPILE_WORKS": partial_compile,
        "STATIC_GRAPH_REQUIRED": compile_report["STATIC_GRAPH_REQUIRED"],
        "PERIODIC_GRAPH_TIME_SHARE": periodic_share,
        "TRIPLET_TIME_SHARE": triplet_share,
        "INTERACTION_GRAPH_TOTAL_TIME_SHARE": graph_total_share,
        "SCATTER_TIME_SHARE": scatter_share,
        "GEMNET_TIME_SHARE": gemnet_share,
        "GEMNET_BLOCK_TIME_SHARE": block_share,
        "CFG_TIME_SHARE": cfg_share,
        "PREDICTOR_TIME_SHARE": predictor_share,
        "CORRECTOR_TIME_SHARE": corrector_share,
        "TIME_SHARES_BY_METHOD": shares_by_method,
        "CPU_LAUNCH_GAP_SHARE": cpu_launch_gap_share,
        "GPU_ACTIVE_SHARE": gpu_active_share_proxy,
        "GPU_ACTIVE_SHARE_SOURCE": "C0_B1_NVIDIA_SMI_MEAN_UTILIZATION_PROXY",
        "GPU_IDLE_SHARE": 1.0 - gpu_active_share_proxy,
        "NSYS_PROCESS_GPU_ACTIVE_SHARE": nsys_process_gpu_active_share,
        "NSYS_PROCESS_GPU_IDLE_SHARE": nsys_process_gpu_idle_share,
        "NSYS_API_BUSY_SHARE": api_busy_share,
        "SM_ACTIVE_SHARE": None,
        "MEMORY_BANDWIDTH_UTILIZATION": None,
        "GPU_HARDWARE_METRICS_CAPTURED": False,
        "CUDA_OPERATOR_CALL_COUNT_ESTIMATE": kernel_count_estimate,
        "STATIC_PERIODIC_GRAPH_WORTH_BUILDING": static_worth,
        "ESTIMATED_STATIC_GRAPH_SPEEDUP": estimated_static_graph_speedup,
        "static_graph_evidence": evidence_conditions,
        "SPG_MATTERGEN_FASTGATE_GO": fastgate_go,
        "SPG_MATTERGEN_FASTGATE_NO_GO": not fastgate_go,
        "STATIC_GRAPH_STARTED": False,
        "FORMAL_256_STARTED": False,
        "limitations": [
            "GPU hardware counters could not be captured because NVGPUCTRPERM denied access.",
            "GPU_ACTIVE_SHARE is the frozen C0-B1 nvidia-smi mean-utilization telemetry proxy, not an SM-active hardware counter.",
            "NSYS_PROCESS_GPU_ACTIVE_SHARE covers the full profiled process, including startup, and is reported separately from steady-state telemetry.",
            "CPU launch-gap share is derived from a representative 10-step Nsight API trace; raw traces are retained outside Git.",
            "Static graph speedup is an Amdahl estimate, not an implemented result.",
            "B4 and BF16 quality are MatterSim/CHGNet surrogate evaluations, not DFT.",
            "No 256-seed formal validation was started.",
        ],
    }
    atomic_json(REPORTS / "fastgate_final.json", result)
    profiler_lines = [
        "# SPG Fast Gate profiler breakdown",
        "",
        "| Method | Periodic | Triplet | Graph total | Scatter | GemNet | Blocks | CFG | Predictor | Corrector |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            "| {method} | {periodic_graph:.4%} | {triplet:.4%} | "
            "{interaction_graph_total:.4%} | {scatter:.4%} | {gemnet:.4%} | "
            "{gemnet_blocks:.4%} | {cfg:.4%} | {predictor:.4%} | "
            "{corrector:.4%} |".format(method=method, **row)
            for method, row in shares_by_method.items()
        ],
        "",
        f"- C0-B1 nvidia-smi mean-utilization proxy: {gpu_active_share_proxy:.4%}",
        (
            "- Nsight full-process GPU activity coverage: "
            f"{nsys_process_gpu_active_share:.4%}"
            if nsys_process_gpu_active_share is not None
            else "- Nsight full-process GPU activity coverage: unavailable"
        ),
        f"- Nsight CPU launch gap share: {cpu_launch_gap_share:.4%}",
        "- SM active / memory bandwidth hardware counters: unavailable (NVGPUCTRPERM)",
        f"- CUDA operator call count estimate in profiled run: {kernel_count_estimate}",
    ]
    atomic_text(
        REPORTS / "profiler_breakdown.md",
        "\n".join(profiler_lines) + "\n",
    )
    compile_lines = [
        "# SPG Fast Gate partial compile audit",
        "",
        f"- PARTIAL_COMPILE_WORKS: `{partial_compile}`",
        f"- Best interaction-block speedup: {compile_report.get('best_block_speedup', 0.0):.4f}×",
        f"- Best other hotspot speedup: {compile_report.get('best_other_speedup', 0.0):.4f}×",
        f"- Observed graph breaks: {compile_report.get('total_graph_breaks_observed', 0)}",
        f"- Maximum recompiles observed: {compile_report.get('max_recompiles_observed', 0)}",
        f"- Full score model fullgraph success: `{compile_report['full_score_model_fullgraph']['fullgraph_success']}`",
        f"- STATIC_GRAPH_REQUIRED: `{compile_report['STATIC_GRAPH_REQUIRED']}`",
    ]
    atomic_text(REPORTS / "compile_report.md", "\n".join(compile_lines) + "\n")
    summary_table = pd.DataFrame(rows)[
        [
            "method",
            "batch_size",
            "fixed8_samples_per_hour",
            "speedup_vs_b1",
            "median_sample_latency_seconds",
            "mean_gpu_utilization_percent",
        ]
    ]
    final_lines = [
        "# SPG-MatterGen Fast Gate final report",
        "",
        f"- SPG_MATTERGEN_FASTGATE_GO: `{fastgate_go}`",
        f"- NATIVE_BATCHING_GO: `{native_go}`",
        f"- FIELD_SAFE_BF16_GO: `{bf16_go}`",
        f"- PARTIAL_COMPILE_WORKS: `{partial_compile}`",
        f"- STATIC_GRAPH_REQUIRED: `{compile_report['STATIC_GRAPH_REQUIRED']}`",
        f"- STATIC_PERIODIC_GRAPH_WORTH_BUILDING: `{static_worth}`",
        f"- Estimated single-bucket static-graph speedup: {estimated_static_graph_speedup:.4f}×",
        "",
        "## Performance",
        "",
        summary_table.to_markdown(index=False),
        "",
        "## Quality decisions",
        "",
        "```json",
        json.dumps(quality["quality_decisions"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Recommended next action",
        "",
        (
            "Implement only the highest-coverage static periodic-graph bucket MVP, then run an 8-seed exactness and endpoint timing gate."
            if static_worth
            else "Do not implement the full static graph engine; retain only components that passed this Fast Gate."
        ),
        "",
        "The Fast Gate did not start a static graph implementation or 256-seed formal validation.",
    ]
    atomic_text(REPORTS / "final_report.md", "\n".join(final_lines) + "\n")
    set_stage(
        "amdahl_analysis",
        "success",
        f"Static periodic graph worth building={static_worth}; estimated speedup={estimated_static_graph_speedup:.3f}x.",
        result,
    )
    set_stage(
        "final_go_no_go",
        "stop_for_review",
        f"SPG-MatterGen Fast Gate decision: GO={fastgate_go}.",
        result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
