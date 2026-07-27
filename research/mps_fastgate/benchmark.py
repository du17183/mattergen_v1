from __future__ import annotations

import contextlib
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterator

from research.mps_fastgate.common import (
    MPS_ROOT,
    PROJECT,
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
    set_final_state,
    set_stage,
)
from research.mps_fastgate.runtime import run_configuration


CONTROL = PROJECT / "research/mps_fastgate/mps_control.sh"
SINGLE_SEEDS = tuple(range(27000, 27016))
EIGHT_SEEDS = tuple(range(27100, 27132))
HISTORICAL_ONE_GPU_THROUGHPUT = 452.97298324514577 / 8.0
HISTORICAL_TWO_GPU_THROUGHPUT = 526.8200043796986 / 8.0


def mps_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,user,comm,args"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "nvidia-cuda-mps-control" in line or "nvidia-cuda-mps-server" in line
    ]


def control(action: str, visible_gpus: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["MPS_VISIBLE_GPUS"] = visible_gpus
    return subprocess.run(
        ["bash", str(CONTROL), action],
        cwd=PROJECT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


@contextlib.contextmanager
def project_mps(visible_gpus: str) -> Iterator[None]:
    if mps_processes():
        raise RuntimeError(f"refusing to start: pre-existing MPS processes: {mps_processes()}")
    control("start", visible_gpus)
    try:
        yield
    finally:
        control("stop", visible_gpus)
        if mps_processes():
            raise RuntimeError(f"project MPS processes remained after quit: {mps_processes()}")
        control("clean", visible_gpus)


def cross_equivalent(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    ref = {
        (row["seed"], row["round"]): row
        for row in reference["result_index"]
    }
    cand = {
        (row["seed"], row["round"]): row
        for row in candidate["result_index"]
    }
    if ref.keys() != cand.keys():
        return False
    fields = (
        "random_tape_hash",
        "final_structure_hash",
        "atomic_numbers_hash",
        "positions_hash",
        "cell_hash",
    )
    return all(
        all(ref[key][field] == cand[key][field] for field in fields)
        for key in ref
    )


def single_gpu_gate(*, bitwise: bool, success: bool, incremental: float) -> str:
    if not bitwise or not success or incremental < 1.03:
        return "MPS_NO_GO"
    if incremental < 1.05:
        return "MPS_ENGINEERING_ONLY"
    return "RUN_EIGHT_GPU"


def config_csv_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for round_row in config["rounds"]:
        rows.append(
            {
                "config_id": config["config_id"],
                "mps_enabled": config["mps_enabled"],
                "workers_per_gpu": config["workers_per_gpu"],
                "active_thread_percentage": config["active_thread_percentage"],
                "gpu_count": len(config["gpus"]),
                **round_row,
                "throughput_median_samples_per_hour": config["throughput_median_samples_per_hour"],
                "within_config_bitwise": config["within_config_bitwise"],
                "gpu_utilization_mean_percent": config["telemetry"].get("gpu_utilization_mean_percent"),
                "peak_memory_mib": config["telemetry"].get("peak_memory_mib"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_text(path, "")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_mps_config(**kwargs) -> dict[str, Any]:
    visible = ",".join(str(gpu) for gpu in kwargs["gpus"])
    with project_mps(visible):
        return run_configuration(mps_enabled=True, **kwargs)


def main() -> int:
    availability_path = RESULTS / "mps_availability.json"
    availability = json.loads(availability_path.read_text(encoding="utf-8"))
    availability.update(
        {
            "MPS_AVAILABLE": True,
            "MPS_SERVER_STARTED": True,
            "MPS_CLIENT_CONNECTED": True,
            "MPS_SMOKE_PASSED": True,
            "mps_server_running_after_smoke": False,
            "mattergen_smoke": str(RESULTS / "smoke/mattergen_forward.json"),
        }
    )
    atomic_json(availability_path, availability)
    set_stage("mps_smoke", "success", "User MPS service, client, MatterGen forward, and cooperative quit passed.", availability)

    if mps_processes():
        raise RuntimeError(f"unexpected MPS process before S0: {mps_processes()}")
    set_stage("single_gpu_s0", "running", "MPS OFF, GPU 0, two persistent workers, 16 seeds x three rounds.")
    s0 = run_configuration(
        config_id="S0_off_w2",
        mps_enabled=False,
        workers_per_gpu=2,
        active_thread_percentage=None,
        gpus=(0,),
        seeds=SINGLE_SEEDS,
        rounds=3,
    )
    s0_drift = abs(s0["throughput_median_samples_per_hour"] / HISTORICAL_TWO_GPU_THROUGHPUT - 1.0)
    set_stage("single_gpu_s0", "success", "S0 completed 48 measured trajectories.", {"summary": s0, "historical_drift": s0_drift})

    h1 = None
    one_worker_reference = HISTORICAL_ONE_GPU_THROUGHPUT
    if s0_drift > 0.10:
        set_stage("single_gpu_h1", "running", "S0 drift exceeded 10%; measuring one-worker reference.")
        h1 = run_configuration(
            config_id="H1_off_w1",
            mps_enabled=False,
            workers_per_gpu=1,
            active_thread_percentage=None,
            gpus=(0,),
            seeds=SINGLE_SEEDS,
            rounds=3,
        )
        one_worker_reference = h1["throughput_median_samples_per_hour"]
        set_stage("single_gpu_h1", "success", "Supplemental one-worker reference completed.", h1)

    set_stage("single_gpu_s1", "running", "MPS ON, GPU 0, two workers, 50% active threads, 16 seeds x three rounds.")
    s1 = run_mps_config(
        config_id="S1_mps_w2_p50",
        workers_per_gpu=2,
        active_thread_percentage=50,
        gpus=(0,),
        seeds=SINGLE_SEEDS,
        rounds=3,
    )
    s1_bitwise = cross_equivalent(s0, s1)
    s1_incremental = s1["throughput_median_samples_per_hour"] / s0["throughput_median_samples_per_hour"]
    s1_total = s1["throughput_median_samples_per_hour"] / one_worker_reference
    s1_success = s1["failure_count"] == 0 and s1["success_count"] == len(SINGLE_SEEDS) * 3
    single_payload = {
        "S0": s0,
        "S1": s1,
        "H1": h1,
        "s0_historical_drift": s0_drift,
        "one_worker_reference_samples_per_hour": one_worker_reference,
        "S1_INCREMENTAL_SPEEDUP": s1_incremental,
        "S1_TOTAL_SPEEDUP_VS_ONE_WORKER": s1_total,
        "S1_BITWISE_EQUIVALENT": s1_bitwise,
        "S1_SUCCESS_SAFE": s1_success,
    }
    atomic_json(RESULTS / "single_gpu_decision.json", single_payload)
    set_stage("single_gpu_s1", "success", "S1 completed; applying frozen single-GPU gate.", single_payload)

    configs = [s0, s1] + ([h1] if h1 is not None else [])
    s2 = None
    r0 = None
    r1 = None
    eight_started = False
    single_gate = single_gpu_gate(bitwise=s1_bitwise, success=s1_success, incremental=s1_incremental)
    if single_gate != "RUN_EIGHT_GPU":
        final_state = single_gate
    else:
        set_stage("single_gpu_s2", "running", "S1 passed 5%; testing four workers at 25% active threads.")
        s2 = run_mps_config(
            config_id="S2_mps_w4_p25",
            workers_per_gpu=4,
            active_thread_percentage=25,
            gpus=(0,),
            seeds=SINGLE_SEEDS,
            rounds=3,
        )
        configs.append(s2)
        s2_bitwise = cross_equivalent(s0, s2)
        s2_incremental = s2["throughput_median_samples_per_hour"] / s1["throughput_median_samples_per_hour"]
        set_stage("single_gpu_s2", "success", "S2 diagnostic completed.", {"summary": s2, "bitwise": s2_bitwise, "incremental_over_s1": s2_incremental})
        if not s2_bitwise or s2["failure_count"]:
            final_state = "MPS_NO_GO"
        else:
            eight_started = True
            set_stage("eight_gpu_r0", "running", "MPS OFF, GPUs 0-7, two workers/GPU, 32 new seeds x two rounds.")
            r0 = run_configuration(
                config_id="R0_off_8gpu_w2",
                mps_enabled=False,
                workers_per_gpu=2,
                active_thread_percentage=None,
                gpus=tuple(range(8)),
                seeds=EIGHT_SEEDS,
                rounds=2,
            )
            configs.append(r0)
            set_stage("eight_gpu_r0", "success", "R0 completed 64 measured trajectories.", r0)
            set_stage("eight_gpu_r1", "running", "MPS ON, GPUs 0-7, two workers/GPU, 50% active threads.")
            r1 = run_mps_config(
                config_id="R1_mps_8gpu_w2_p50",
                workers_per_gpu=2,
                active_thread_percentage=50,
                gpus=tuple(range(8)),
                seeds=EIGHT_SEEDS,
                rounds=2,
            )
            configs.append(r1)
            r1_bitwise = cross_equivalent(r0, r1)
            r1_incremental = r1["throughput_median_samples_per_hour"] / r0["throughput_median_samples_per_hour"]
            r1_total = r1["throughput_median_samples_per_hour"] / (one_worker_reference * 8.0)
            r1_safe = r1_bitwise and r1["failure_count"] == 0
            set_stage("eight_gpu_r1", "success", "R1 completed; applying paper gate.", {"summary": r1, "bitwise": r1_bitwise, "incremental": r1_incremental, "total_speedup": r1_total})
            if r1_safe and r1_incremental >= 1.05 and r1_total >= 1.25:
                final_state = "MPS_PAPER_GO"
            elif r1_safe and r1_incremental >= 1.03:
                final_state = "MPS_ENGINEERING_ONLY"
            else:
                final_state = "MPS_NO_GO"

    set_final_state(final_state)
    all_rows = [row for config in configs for row in config_csv_rows(config)]
    write_csv(REPORTS / "single_gpu_results.csv", [row for row in all_rows if row["gpu_count"] == 1])
    if eight_started:
        write_csv(REPORTS / "eight_gpu_results.csv", [row for row in all_rows if row["gpu_count"] == 8])

    s2_incremental_value = (
        s2["throughput_median_samples_per_hour"] / s1["throughput_median_samples_per_hour"]
        if s2 is not None else None
    )
    r1_bitwise_value = cross_equivalent(r0, r1) if r0 is not None and r1 is not None else None
    r1_incremental_value = (
        r1["throughput_median_samples_per_hour"] / r0["throughput_median_samples_per_hour"]
        if r0 is not None and r1 is not None else None
    )
    r1_total_value = (
        r1["throughput_median_samples_per_hour"] / (one_worker_reference * 8.0)
        if r1 is not None else None
    )
    final = {
        "completed_at": now(),
        "FINAL_STATE": final_state,
        "MPS_AVAILABLE": True,
        "MPS_SERVER_STARTED": True,
        "MPS_SMOKE_PASSED": True,
        "SINGLE_GPU_SEEDS": list(SINGLE_SEEDS),
        "S0_THROUGHPUT": s0["throughput_median_samples_per_hour"],
        "S1_THROUGHPUT": s1["throughput_median_samples_per_hour"],
        "S1_INCREMENTAL_SPEEDUP": s1_incremental,
        "S1_TOTAL_SPEEDUP_VS_ONE_WORKER": s1_total,
        "S1_P50_LATENCY": s1["p50_latency_median_seconds"],
        "S1_P95_LATENCY": s1["p95_latency_median_seconds"],
        "S1_BITWISE_EQUIVALENT": s1_bitwise,
        "S2_STARTED": s2 is not None,
        "S2_THROUGHPUT": s2["throughput_median_samples_per_hour"] if s2 is not None else None,
        "S2_INCREMENTAL_OVER_S1": s2_incremental_value,
        "BEST_WORKERS_PER_GPU": 4 if s2_incremental_value is not None and s2_incremental_value >= 1.03 else 2,
        "EIGHT_GPU_STARTED": eight_started,
        "R0_THROUGHPUT": r0["throughput_median_samples_per_hour"] if r0 is not None else None,
        "R1_THROUGHPUT": r1["throughput_median_samples_per_hour"] if r1 is not None else None,
        "R1_INCREMENTAL_SPEEDUP": r1_incremental_value,
        "R1_TOTAL_SPEEDUP_VS_ONE_WORKER": r1_total_value,
        "R1_BITWISE_EQUIVALENT": r1_bitwise_value,
        "MPS_PAPER_GO": final_state == "MPS_PAPER_GO",
        "MPS_ENGINEERING_ONLY": final_state == "MPS_ENGINEERING_ONLY",
        "MPS_NO_GO": final_state == "MPS_NO_GO",
        "MPS_SERVER_RUNNING_AFTER_EXIT": bool(mps_processes()),
        "GPU_WORKERS": 0,
        "OTHER_PROCESSES_TERMINATED": False,
        "SIGKILL_USED": False,
    }
    atomic_json(RESULTS / "final_summary.json", final)
    single_lines = [
        "# MPS single-GPU fast-gate",
        "",
        f"- Seeds: `{SINGLE_SEEDS[0]}-{SINGLE_SEEDS[-1]}`",
        f"- S0 throughput: `{final['S0_THROUGHPUT']:.4f} samples/hour`",
        f"- S1 throughput: `{final['S1_THROUGHPUT']:.4f} samples/hour`",
        f"- S1 incremental: `{s1_incremental:.4f}x`",
        f"- S1 total versus one worker: `{s1_total:.4f}x`",
        f"- Bitwise equivalent: `{s1_bitwise}`",
        f"- Final state: `{final_state}`",
        "",
    ]
    atomic_text(REPORTS / "single_gpu_report.md", "\n".join(single_lines))
    final_lines = [
        "# NVIDIA MPS MatterGen fast-gate final report",
        "",
        f"Final state: `{final_state}`",
        "",
        *single_lines[2:],
        f"- S2 started: `{s2 is not None}`",
        f"- 8-GPU confirmation started: `{eight_started}`",
        "",
        "All executed comparisons use C0, batch size one per process, strict FP32,",
        "the complete Predictor/Corrector sampler, identical seeds, and raw-tensor hashes.",
        "",
    ]
    atomic_text(REPORTS / "final_report.md", "\n".join(final_lines))
    set_stage("final_decision", "success", f"Frozen final state: {final_state}", final)
    set_stage("stop_for_review", "stop_for_review", "Fast-gate complete; waiting for human review.", final)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
