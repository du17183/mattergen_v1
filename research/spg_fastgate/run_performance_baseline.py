"""Run and aggregate the six fixed-eight-GPU native batch baselines."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from statistics import mean, median

from research.spg_fastgate.common import (
    LOGS,
    PROJECT,
    PYTHON,
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    base_environment,
    now,
    set_stage,
)


CONFIGS = (("C0", 1), ("C0", 4), ("C0", 8), ("A0", 1), ("A0", 4), ("A0", 8))
RAW = RESULTS / "performance/raw"
CSV_PATH = RESULTS / "performance_baseline.csv"
REPORT_PATH = REPORTS / "performance_baseline.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_config(method: str, batch_size: int) -> None:
    processes: list[tuple[subprocess.Popen, object]] = []
    for gpu in range(8):
        output = RAW / f"{method}_B{batch_size}" / f"gpu{gpu}"
        status = output / "status.json"
        if status.is_file() and read_json(status).get("success") is True:
            continue
        log = LOGS / "performance" / f"{method.lower()}_b{batch_size}_gpu{gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [
                str(PYTHON),
                "-m",
                "research.spg_fastgate.benchmark_worker",
                "--method",
                method,
                "--batch-size",
                str(batch_size),
                "--physical-gpu",
                str(gpu),
            ],
            cwd=PROJECT,
            env=base_environment(gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((process, stream))
    return_codes = []
    for process, stream in processes:
        return_codes.append(process.wait())
        stream.close()
    if any(return_code != 0 for return_code in return_codes):
        raise RuntimeError(f"{method}-B{batch_size} benchmark failed: {return_codes}")


def aggregate() -> list[dict]:
    rows = []
    for method, batch_size in CONFIGS:
        summaries = [
            read_json(RAW / f"{method}_B{batch_size}" / f"gpu{gpu}/summary.json")
            for gpu in range(8)
        ]
        formal_by_gpu = [
            [row for row in summary["rows"] if not row["warmup"]]
            for summary in summaries
        ]
        measured_samples = 8 * 3 * batch_size
        steady_state_wall = max(
            sum(row["wall_seconds"] for row in formal) for formal in formal_by_gpu
        )
        all_formal = [row for formal in formal_by_gpu for row in formal]
        rows.append(
            {
                "method": method,
                "batch_size": batch_size,
                "gpu_count": 8,
                "measured_samples": measured_samples,
                "fixed8_samples_per_hour": measured_samples * 3600.0 / steady_state_wall,
                "mean_sample_latency_seconds": mean(
                    row["sample_latency_seconds"] for row in all_formal
                ),
                "median_sample_latency_seconds": median(
                    row["sample_latency_seconds"] for row in all_formal
                ),
                "first_sample_latency_seconds": median(
                    summary["rows"][0]["sample_latency_seconds"]
                    for summary in summaries
                ),
                "steady_state_sample_latency_seconds": median(
                    row["sample_latency_seconds"] for row in all_formal
                ),
                "median_batch_wall_seconds": median(
                    row["wall_seconds"] for row in all_formal
                ),
                "median_batch_cuda_seconds": median(
                    row["cuda_event_seconds"] for row in all_formal
                ),
                "mean_gpu_utilization_percent": mean(
                    summary["telemetry"].get("mean_utilization_percent", 0.0)
                    for summary in summaries
                ),
                "mean_gpu_power_w": mean(
                    summary["telemetry"].get("mean_power_w", 0.0)
                    for summary in summaries
                ),
                "median_cpu_process_seconds": median(
                    row["cpu_seconds"] for row in all_formal
                ),
                "max_peak_allocated_bytes": max(
                    row["peak_allocated_bytes"] for row in all_formal
                ),
                "max_peak_reserved_bytes": max(
                    row["peak_reserved_bytes"] for row in all_formal
                ),
            }
        )
    for method in ("C0", "A0"):
        baseline = next(
            row["fixed8_samples_per_hour"]
            for row in rows
            if row["method"] == method and row["batch_size"] == 1
        )
        for row in rows:
            if row["method"] == method:
                row["speedup_vs_b1"] = row["fixed8_samples_per_hour"] / baseline
    return rows


def write_reports(rows: list[dict]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# SPG-MatterGen Fast Gate performance baseline",
        "",
        "| Method | Batch | Fixed-8 throughput (samples/h) | Speedup vs B1 | Median sample latency (s) | GPU util (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['batch_size']} | "
            f"{row['fixed8_samples_per_hour']:.3f} | {row['speedup_vs_b1']:.3f}× | "
            f"{row['median_sample_latency_seconds']:.3f} | "
            f"{row['mean_gpu_utilization_percent']:.2f} |"
        )
    atomic_text(REPORT_PATH, "\n".join(lines) + "\n")
    atomic_json(
        RESULTS / "performance_baseline.json",
        {"created_at": now(), "rows": rows},
    )


def main() -> int:
    set_stage(
        "performance_baseline",
        "running",
        "Running C0/A0 B1/B4/B8 fixed-eight-GPU baselines.",
        {"performance_seeds": list(range(24000, 24016)), "configs": CONFIGS},
    )
    try:
        for method, batch_size in CONFIGS:
            run_config(method, batch_size)
        rows = aggregate()
        write_reports(rows)
    except BaseException as error:
        set_stage(
            "performance_baseline",
            "failed",
            f"Performance baseline failed: {error}",
        )
        raise
    set_stage(
        "performance_baseline",
        "success",
        "C0/A0 B1/B4/B8 performance baselines completed.",
        {"rows": rows},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
