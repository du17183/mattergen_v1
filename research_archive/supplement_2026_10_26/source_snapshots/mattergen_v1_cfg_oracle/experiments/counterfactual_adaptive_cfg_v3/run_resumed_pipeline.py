"""Resume the frozen pipeline after a resource-only GPU reassignment."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from run_frozen_pipeline import (
    LOGS,
    ORACLE,
    PROJECT,
    PYTHON,
    ROOT,
    run,
    status,
    write_final_report,
)


PROPERTY_GPU = 1
QUALITY_GPUS = (1, 3, 4, 5, 6, 7)


def assert_generation_complete() -> None:
    seeds = list(map(int, json.loads((ORACLE / "oracle_seeds.json").read_text())["seeds"]))
    if len(seeds) != 32 or len(set(seeds)) != 32:
        raise RuntimeError("registered Oracle seed list changed")
    for seed in seeds:
        source = ORACLE / "generation" / str(seed)
        summary = json.loads((source / "run_summary.json").read_text())
        if not summary.get("success") or summary.get("seed") != seed:
            raise RuntimeError(f"incomplete Oracle generation: {seed}")
        if (
            summary.get("short_branches") != 24
            or summary.get("full_branches") != 6
            or summary.get("decision_states") != 8
            or summary.get("full_rollout_states") != 2
        ):
            raise RuntimeError(f"Oracle count mismatch: {seed}")
        for check in summary.get("pairing_checks", []):
            if (
                not check.get("atomic_numbers_equal")
                or check.get("max_fractional_position_abs_error") != 0.0
                or check.get("max_cell_abs_error") != 0.0
            ):
                raise RuntimeError(f"Oracle CFG2 pairing mismatch: {seed}")
    partial = list((ORACLE / "generation").glob("*.partial"))
    if partial:
        raise RuntimeError(f"retained generation partials: {partial}")


def assert_gpus_idle(gpus: tuple[int, ...]) -> None:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    observed: dict[int, dict[str, int]] = {}
    for line in output.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        observed[index] = {"memory_used_mib": memory, "utilization_percent": utilization}
    busy = {
        gpu: observed.get(gpu)
        for gpu in gpus
        if gpu not in observed
        or observed[gpu]["memory_used_mib"] >= 5000
        or observed[gpu]["utilization_percent"] >= 50
    }
    if busy:
        raise RuntimeError(f"reassigned GPU pool is no longer idle: {busy}")
    (ROOT / "resource_reassignment_audit.json").write_text(
        json.dumps(
            {
                "reason": "GPU 0 became occupied by unrelated PID 3105853 after generation",
                "property_gpu": PROPERTY_GPU,
                "quality_gpus": list(QUALITY_GPUS),
                "excluded_gpus": [0, 2],
                "observed_before_resume": observed,
                "scientific_protocol_changed": False,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    try:
        LOGS.mkdir(exist_ok=True)
        assert_generation_complete()
        assert_gpus_idle(QUALITY_GPUS)
        status("oracle_property_evaluation_resumed_gpu1")
        run(
            [PYTHON, str(ROOT / "run_property_from_aggregates.py")],
            environment={
                "CUDA_VISIBLE_DEVICES": str(PROPERTY_GPU),
                "PYTHONPATH": str(PROJECT),
                "MPLCONFIGDIR": str(ROOT / ".matplotlib"),
                "PYTHONUNBUFFERED": "1",
            },
            log_name="oracle_property_evaluation.log",
        )
        assert_gpus_idle(QUALITY_GPUS)
        status("oracle_quality_evaluation_reassigned")
        run(
            [PYTHON, str(ROOT / "run_oracle_quality_reassigned.py")],
            log_name="oracle_quality_driver_reassigned.log",
        )
        status("oracle_headroom_analysis")
        run(
            [PYTHON, str(ROOT / "analyze_oracle.py")],
            environment={
                "MPLCONFIGDIR": str(ROOT / ".matplotlib"),
                "PYTHONUNBUFFERED": "1",
            },
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
            environment={
                "MPLCONFIGDIR": str(ROOT / ".matplotlib"),
                "PYTHONUNBUFFERED": "1",
            },
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
            note="Both frozen scientific gates passed; implement sparse V3 before fresh P0.",
        )
    except BaseException as error:
        status("technical_failure", "FAILED", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
