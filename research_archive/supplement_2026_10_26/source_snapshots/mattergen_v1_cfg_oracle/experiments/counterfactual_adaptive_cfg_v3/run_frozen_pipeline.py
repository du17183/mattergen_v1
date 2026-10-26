"""Execute the preregistered Counterfactual Adaptive-CFG gate chain."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/counterfactual_adaptive_cfg_v3"
ORACLE = ROOT / "oracle"
LOGS = ROOT / "logs"
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"
GPU_POOL = (0, 1, 3, 4, 5, 6, 7)


def status(stage: str, state: str = "RUNNING", **extra) -> None:
    payload = {"status": state, "stage": stage, **extra}
    (ROOT / "pipeline_status.json").write_text(json.dumps(payload, indent=2) + "\n")


def run(command: list[str], *, environment: dict[str, str] | None = None, log_name: str | None = None) -> None:
    env = dict(os.environ)
    if environment:
        env.update(environment)
    if log_name is None:
        result = subprocess.run(command, cwd=PROJECT, env=env)
    else:
        with (LOGS / log_name).open("a", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}")


def assert_gpu_pool_idle() -> None:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    observed = {}
    for line in output.splitlines():
        index, used = (int(value.strip()) for value in line.split(","))
        observed[index] = used
    busy = {index: observed[index] for index in GPU_POOL if observed.get(index, 10**9) >= 5000}
    if busy:
        raise RuntimeError(f"authorized Oracle GPU pool is no longer idle: {busy}")
    audit = {"allowed_gpu_pool": list(GPU_POOL), "excluded_gpu": 2, "memory_used_mib": observed}
    (ROOT / "resource_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def generation() -> None:
    seeds = tuple(json.loads((ORACLE / "oracle_seeds.json").read_text())["seeds"])
    environment = {
        "PYTHONPATH": str(PROJECT),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    }

    def worker(gpu: int, shard: tuple[int, ...]) -> None:
        run(
            [PYTHON, str(ROOT / "run_oracle_generation.py"), "--seeds", ",".join(map(str, shard))],
            environment=environment | {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "CFG_ORACLE_PHYSICAL_GPU": str(gpu),
            },
            log_name=f"oracle_generation_gpu_{gpu}.log",
        )

    with ThreadPoolExecutor(max_workers=len(GPU_POOL)) as pool:
        futures = [
            pool.submit(worker, gpu, tuple(seeds[index::len(GPU_POOL)]))
            for index, gpu in enumerate(GPU_POOL)
        ]
        for future in futures:
            future.result()


def write_final_report(headroom: dict, predictability: dict | None) -> None:
    if headroom["ORACLE_HEADROOM"] == "FAIL":
        states = {
            "ORACLE_HEADROOM": "FAIL",
            "ORACLE_PREDICTABILITY": "NOT_RUN",
            "V3_P0": "NOT_RUN",
            "V3_FORMAL256": "NOT_RUN",
            "INNOVATION1_FINAL_STATUS": "MIXED",
        }
    elif predictability and predictability["ORACLE_PREDICTABILITY"] == "FAIL":
        states = {
            "ORACLE_HEADROOM": "GO",
            "ORACLE_PREDICTABILITY": "FAIL",
            "V3_P0": "NOT_RUN",
            "V3_FORMAL256": "NOT_RUN",
            "INNOVATION1_FINAL_STATUS": "MIXED",
        }
    else:
        return
    report = "# Counterfactual Adaptive CFG V3 final report\n\n"
    report += "The experiment followed the frozen Oracle → Headroom → Predictability gate chain.\n\n"
    report += "```json\n" + json.dumps(states, indent=2) + "\n```\n\n"
    report += "All Oracle outputs were retained. No V1/V2 rescue tuning or post-outcome gate change was performed. "
    report += "All reported property and energy/stability values are surrogate evaluations; `DFT_VERIFIED=False`.\n"
    (ROOT / "final_report.md").write_text(report)


def main() -> None:
    try:
        LOGS.mkdir(exist_ok=True)
        status("preflight")
        assert_gpu_pool_idle()
        run([PYTHON, "-m", "py_compile",
             "mattergen/diffusion/sampling/counterfactual_oracle.py",
             str(ROOT / "run_oracle_generation.py"),
             str(ROOT / "prepare_oracle_outputs.py"),
             str(ROOT / "run_oracle_quality.py"),
             str(ROOT / "analyze_oracle.py"),
             str(ROOT / "run_predictability.py")])
        status("oracle_generation")
        generation()
        status("oracle_property_evaluation")
        run(
            [PYTHON, str(ROOT / "prepare_oracle_outputs.py")],
            environment={
                "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(PROJECT),
                "MPLCONFIGDIR": str(ROOT / ".matplotlib"), "PYTHONUNBUFFERED": "1",
            },
            log_name="oracle_property_evaluation.log",
        )
        status("oracle_quality_evaluation")
        run([PYTHON, str(ROOT / "run_oracle_quality.py")], log_name="oracle_quality_driver.log")
        status("oracle_headroom_analysis")
        run(
            [PYTHON, str(ROOT / "analyze_oracle.py")],
            environment={"MPLCONFIGDIR": str(ROOT / ".matplotlib"), "PYTHONUNBUFFERED": "1"},
            log_name="oracle_analysis.log",
        )
        headroom = json.loads((ORACLE / "oracle_summary.json").read_text())
        if headroom["ORACLE_HEADROOM"] != "GO":
            write_final_report(headroom, None)
            status("stopped_after_oracle_headroom", "COMPLETED", decision=headroom)
            return
        status("oracle_predictability")
        run(
            [PYTHON, str(ROOT / "run_predictability.py")],
            environment={"MPLCONFIGDIR": str(ROOT / ".matplotlib"), "PYTHONUNBUFFERED": "1"},
            log_name="predictability.log",
        )
        predictability = json.loads((ROOT / "controller/decision_summary.json").read_text())
        if predictability["ORACLE_PREDICTABILITY"] != "GO":
            write_final_report(headroom, predictability)
            status("stopped_after_predictability", "COMPLETED", decision=predictability)
            return
        status(
            "v3_implementation_required",
            "GATE_GO",
            decision=predictability,
            note="Both scientific gates passed; implement frozen sparse V3 before fresh P0.",
        )
    except BaseException as error:
        status("technical_failure", "FAILED", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
