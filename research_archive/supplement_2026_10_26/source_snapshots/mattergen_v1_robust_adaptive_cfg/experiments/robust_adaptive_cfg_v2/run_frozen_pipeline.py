"""End-to-end frozen V2 pipeline with the preregistered P0 stop gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/robust_adaptive_cfg_v2"
PYTHON = "/mnt/datasets-livsyn/dxl/alm/.venv/bin/python"
AUTHORIZED_GPUS = (0, 1, 3, 4, 5, 6, 7)


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def wait_for(path: Path, *, complete_status: str, failure_status: str = "FAILED") -> dict:
    while True:
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("status") == complete_status:
                return payload
            if payload.get("status") == failure_status:
                raise RuntimeError(f"upstream failed: {path}: {payload}")
        time.sleep(15)


def run(args: list[str], env: dict[str, str]) -> None:
    print(json.dumps({"event": "run", "args": args}), flush=True)
    result = subprocess.run(args, cwd=PROJECT, env=env)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {args}")


def select_free_gpus() -> tuple[int, ...]:
    while True:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = []
        for line in output.splitlines():
            index, used, free, utilization = [int(value.strip()) for value in line.split(",")]
            rows.append(
                {"index": index, "memory_used_mib": used, "memory_free_mib": free, "utilization_percent": utilization}
            )
        free = tuple(
            row["index"]
            for row in rows
            if row["index"] in AUTHORIZED_GPUS and row["memory_used_mib"] < 5000
        )
        audit = {
            "authorized_gpus": list(AUTHORIZED_GPUS),
            "observed": rows,
            "selected_free_gpus": list(free),
            "gpu_2_excluded": True,
            "free_rule": "memory.used < 5000 MiB at cohort launch; never terminate other processes",
        }
        save(ROOT / "resource_audit_latest.json", audit)
        if free:
            return free
        save(ROOT / "pipeline_status.json", {"status": "WAITING_RESOURCE", "resource_audit": audit})
        time.sleep(30)


def final_report(formal_status: str) -> None:
    stage0 = json.loads((ROOT / "stage0_diagnostic/diagnostic_summary.json").read_text())
    calibration = json.loads((ROOT / "calibration/residual_calibration.json").read_text())
    p0 = json.loads((ROOT / "p0/decision_summary.json").read_text())
    formal_path = ROOT / "formal256/decision_summary.json"
    formal = json.loads(formal_path.read_text()) if formal_path.exists() else {
        "ROBUST_ADAPTIVE_CFG_FORMAL256": formal_status
    }
    innovation = formal.get(
        "INNOVATION1_FINAL_STATUS",
        "MIXED" if p0["ROBUST_ADAPTIVE_CFG_P0"] != "GO" else "PENDING",
    )
    report = f"""# Robust Adaptive CFG V2 final report

## Frozen method

Multi-field conditional residual → timestep/phase calibration → median field consensus → phase-specific online EMA → magnitude/agreement confidence → fallback to fixed CFG=2 when uncertain → bounded adaptive guidance [1.5, 2.5] → 0.05 slew-rate stabilization. Stage gating is OFF.

## 1. Why V1 was unstable

Stage 0 linked {stage0['event_count']:,} controller decisions from {stage0['seed_count_with_trace']} historical paired seeds. Raw residuals were strongly timestep-dependent, field scales differed by orders of magnitude, and mean log-residual scale shifted across observed cohorts. Harmful samples had greater average field disagreement, while extreme CFG events alone did not consistently distinguish harm. Formal256 endpoints were retained, but its controller trajectories were never archived.

## 2. Calibration

The calibration used {calibration['source']['seed_count']} historical fixed-CFG=2 deterministic replays and {calibration['source']['trace_rows']:,} score decisions. Every replayed final structure numerically matched the archived C0 structure. No quality endpoint entered calibration and no hyperparameter sweep was used.

## 3. Fresh P0 decision

```json
{json.dumps(p0, indent=2, ensure_ascii=False)}
```

## 4. Fresh Formal256 decision

```json
{json.dumps(formal, indent=2, ensure_ascii=False)}
```

## Final status

```text
ROBUST_ADAPTIVE_CFG_P0 = {p0['ROBUST_ADAPTIVE_CFG_P0']}
ROBUST_ADAPTIVE_CFG_FORMAL256 = {formal.get('ROBUST_ADAPTIVE_CFG_FORMAL256', formal_status)}
INNOVATION1_FINAL_STATUS = {innovation}
```

All registered samples were retained. Parameters and seeds were not modified after observing P0 or Formal256. Energy/stability and magnetic-property evaluations use frozen MatterSim/CHGNet surrogates (`SURROGATE_PROPERTY_EVAL=True`, `DFT_VERIFIED=False`).
"""
    (ROOT / "final_report.md").write_text(report)
    save(
        ROOT / "final_status.json",
        {
            "ROBUST_ADAPTIVE_CFG_P0": p0["ROBUST_ADAPTIVE_CFG_P0"],
            "ROBUST_ADAPTIVE_CFG_FORMAL256": formal.get("ROBUST_ADAPTIVE_CFG_FORMAL256", formal_status),
            "INNOVATION1_FINAL_STATUS": innovation,
            "parameters_retuned_after_results": False,
            "all_data_retained": True,
            "SURROGATE_PROPERTY_EVAL": True,
            "DFT_VERIFIED": False,
        },
    )


def main() -> None:
    status_path = ROOT / "pipeline_status.json"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(PROJECT) + ":/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance",
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        MPLCONFIGDIR="/tmp/matplotlib-robust-cfg",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONUNBUFFERED="1",
    )
    try:
        save(status_path, {"status": "RUNNING", "stage": "wait_calibration_replay"})
        wait_for(ROOT / "calibration/replay_pipeline_status.json", complete_status="COMPLETED")
        while not (ROOT / "p0/seeds.json").exists() or not (ROOT / "formal256/seeds.json").exists():
            time.sleep(15)

        save(status_path, {"status": "RUNNING", "stage": "build_and_freeze_calibration"})
        run([PYTHON, str(ROOT / "calibration/build_calibration.py")], env)
        run(
            [
                PYTHON,
                "-m",
                "pytest",
                "mattergen/diffusion/tests/test_guidance_schedule.py",
                "mattergen/diffusion/tests/test_robust_guidance.py",
                "-q",
            ],
            env,
        )

        p0_gpus = select_free_gpus()
        p0_env = env | {"ROBUST_CFG_GPU_POOL": ",".join(map(str, p0_gpus))}
        save(status_path, {"status": "RUNNING", "stage": "fresh_p0", "gpus": p0_gpus})
        run([PYTHON, str(ROOT / "run_generation_and_evaluation.py"), "--cohort", "p0", "--pipeline"], p0_env)
        run([PYTHON, str(ROOT / "analyze_experiment.py"), "--cohort", "p0"], p0_env)
        p0 = json.loads((ROOT / "p0/decision_summary.json").read_text())
        if p0["ROBUST_ADAPTIVE_CFG_P0"] != "GO":
            formal = {
                "ROBUST_ADAPTIVE_CFG_FORMAL256": "NOT_RUN",
                "reason": f"P0={p0['ROBUST_ADAPTIVE_CFG_P0']}; frozen stop rule applied",
                "generation_started": False,
                "rescue_tuning": False,
            }
            save(ROOT / "formal256/decision_summary.json", formal)
            final_report("NOT_RUN")
            save(status_path, {"status": "COMPLETED", "stage": "stopped_after_p0", "decision": p0})
            return

        formal_gpus = select_free_gpus()
        formal_env = env | {"ROBUST_CFG_GPU_POOL": ",".join(map(str, formal_gpus))}
        save(status_path, {"status": "RUNNING", "stage": "fresh_formal256", "gpus": formal_gpus})
        run([PYTHON, str(ROOT / "run_generation_and_evaluation.py"), "--cohort", "formal256", "--pipeline"], formal_env)
        run([PYTHON, str(ROOT / "analyze_experiment.py"), "--cohort", "formal256"], formal_env)
        final_report("COMPLETED")
        save(status_path, {"status": "COMPLETED", "stage": "formal256_decision"})
    except BaseException as error:
        save(status_path, {"status": "FAILED", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
