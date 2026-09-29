"""Resume-safe stage runner; launch in tmux with an external master log."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


PROJECT = Path(__file__).resolve().parents[2]
FIELD = Path("/mnt/datasets-livsyn/dxl/mattergen_v1_field_cfg")
MAIN = Path("/mnt/datasets-livsyn/dxl/mattergen_v1")
ROOT = PROJECT / "experiments/trajectory_search"
GPUS = (2, 3, 4, 5, 6, 7)
COHORTS = ("search_baseline", "budget_scaling")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_snapshot() -> dict[int, dict[str, object]]:
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True)
    apps_output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True
    )
    active: dict[str, list[int]] = {}
    for row in apps_output.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            active.setdefault(parts[1], []).append(int(parts[0]))
    snapshot: dict[int, dict[str, object]] = {}
    for row in output.splitlines():
        index, uuid, used, utilization = [part.strip() for part in row.split(",")]
        snapshot[int(index)] = {
            "uuid": uuid,
            "memory_used_mib": int(used),
            "utilization_percent": int(utilization),
            "active_compute_pids": active.get(uuid, []),
        }
    return snapshot


def assert_idle(stage: str) -> None:
    snapshot = gpu_snapshot()
    problems = []
    for gpu in GPUS:
        item = snapshot.get(gpu)
        if item is None or int(item["memory_used_mib"]) > 1000 or item["active_compute_pids"]:
            problems.append(f"GPU {gpu}: {item}")
    audit = ROOT / "protocol" / f"gpu_audit_{stage}.json"
    audit.write_text(json.dumps({
        "utc": now(), "requested_gpus": GPUS, "snapshot": snapshot, "problems": problems,
        "check_rule": "memory <= 1000 MiB and no active compute PID; utilization alone is not occupancy",
    }, indent=2) + "\n")
    if problems:
        raise RuntimeError("refusing occupied GPU(s): " + ", ".join(problems))


def environment(gpu: int | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(PYTHONPATH=f"{FIELD}:{PROJECT}:{MAIN}", XDG_CACHE_HOME=str(MAIN / ".cache"), HF_HOME=str(MAIN / ".cache/huggingface"), TORCH_HOME=str(MAIN / ".cache/torch"), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    if gpu is not None:
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), SEARCH_PHYSICAL_GPU=str(gpu))
    return env


def run_single(cohort: str, label: str, module: str, stage: str, gpu: int) -> None:
    output = PROJECT / "results" / cohort
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", module, "--cohort", cohort, "--stage", stage]
    if stage.startswith("quality"):
        cmd += ["--gpus", ",".join(map(str, GPUS))]
    log = logs / f"{label}.log"
    with log.open("a") as stream:
        stream.write(f"\n[{now()}] START {' '.join(cmd)}\n")
        stream.flush()
        result = subprocess.run(cmd, cwd=FIELD, env=environment(gpu), stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"[{now()}] EXIT {result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"stage {cohort}/{label} failed; see {log}")


def generate(cohort: str) -> None:
    seeds = range(2100000000, 2100000128) if cohort == "search_baseline" else range(2100010000, 2100010128)
    output = PROJECT / "results" / cohort
    if all((output / "generation" / str(seed) / "run_summary.json").exists() for seed in seeds):
        return
    assert_idle(f"{cohort}_generation")
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    try:
        for index, gpu in enumerate(GPUS):
            cmd = [sys.executable, "-m", "experiments.trajectory_search.run_generation", "--cohort", cohort, "--shard-index", str(index), "--shard-count", str(len(GPUS))]
            stream = (logs / f"generation_gpu{gpu}.log").open("a")
            streams.append(stream)
            stream.write(f"\n[{now()}] START {' '.join(cmd)}\n")
            stream.flush()
            processes.append((gpu, subprocess.Popen(cmd, cwd=FIELD, env=environment(gpu), stdout=stream, stderr=subprocess.STDOUT)))
        errors = []
        for gpu, process in processes:
            code = process.wait()
            if code:
                errors.append(f"GPU {gpu} worker exit {code}")
        if errors:
            raise RuntimeError("; ".join(errors))
    finally:
        for stream in streams:
            stream.close()


def one_cohort(cohort: str) -> None:
    output = PROJECT / "results" / cohort
    generate(cohort)
    if not (output / "property_metrics_raw.csv").exists():
        assert_idle(f"{cohort}_property")
        run_single(cohort, "prepare", "experiments.trajectory_search.evaluate", "prepare", GPUS[0])
    if not all((output / "quality_branches" / group / "official_detailed.json.gz").exists() for group in (("C0", "GPulse", "PPulse", "Independent_C0") if cohort == "search_baseline" else ("C0", "GPulse", "PPulse", "APulse", "CPulse"))):
        assert_idle(f"{cohort}_branch_quality")
        run_single(cohort, "quality_branches", "experiments.trajectory_search.evaluate", "quality-branches", GPUS[0])
    if not (output / "selected_outcomes.csv").exists():
        run_single(cohort, "selection", "experiments.trajectory_search.analyze", "select", GPUS[0])
    expected = ("C0", "Independent_Best_of_2", "Fixed_K2") if cohort == "search_baseline" else tuple(f"K{k}" for k in range(5))
    if not all((output / "quality_mixed" / method / "official_detailed.json.gz").exists() for method in expected):
        assert_idle(f"{cohort}_mixed_quality")
        run_single(cohort, "quality_mixed", "experiments.trajectory_search.evaluate", "quality-mixed", GPUS[0])
    table = output / ("table_best_of_n.csv" if cohort == "search_baseline" else "table_budget_scaling.csv")
    if not table.exists():
        run_single(cohort, "final_statistics", "experiments.trajectory_search.analyze", "final", GPUS[0])
    figure = output / ("fig_best_of_n.pdf" if cohort == "search_baseline" else "fig_budget_scaling.pdf")
    if not figure.exists():
        cmd = [sys.executable, "-m", "experiments.trajectory_search.plot_report", "--cohort", cohort]
        with (output / "logs" / "plot.log").open("a") as stream:
            result = subprocess.run(cmd, cwd=FIELD, env=environment(), stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"plot failed: {cohort}")


def main() -> None:
    for cohort in COHORTS:
        print(f"[{now()}] cohort START {cohort}", flush=True)
        one_cohort(cohort)
        print(f"[{now()}] cohort COMPLETE {cohort}", flush=True)
    report = PROJECT / "docs/trajectory_search_report.md"
    if not report.exists():
        subprocess.run([sys.executable, "-m", "experiments.trajectory_search.plot_report", "--report"], cwd=FIELD, env=environment(), check=True)
    print(f"[{now()}] SEARCH_STUDY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
