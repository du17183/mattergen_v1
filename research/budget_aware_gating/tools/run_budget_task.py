#!/usr/bin/env python3
"""Standardized one-structure runner for the Budget-Aware Corrector Gating study."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import yaml

TOOLS = Path("/data/dxl/tools/budget_aware_gating")
sys.path.insert(0, str(TOOLS))
import run_budget_sample as sample  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def arg_value(name: str, default: str | None = None) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except ValueError:
        return default


def main() -> int:
    output_raw = arg_value("--output-dir")
    if output_raw is None:
        raise ValueError("--output-dir is required")
    output = Path(output_raw).resolve()
    gpu_slot = int(arg_value("--gpu-slot", "0"))
    workers_per_gpu = int(arg_value("--workers-per-gpu", "1"))
    filtered: list[str] = [sys.argv[0]]
    skip_next = False
    for index, value in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if value in {"--gpu-slot", "--workers-per-gpu"}:
            skip_next = True
            continue
        filtered.append(value)
    sys.argv = filtered

    # CrystalGenerator.__init__ contains checkpoint/model construction; generate()
    # is the sampling interval. This measures the boundary without changing either.
    from mattergen.generator import CrystalGenerator

    timing = {"model_load_seconds": 0.0}
    original_init = CrystalGenerator.__init__

    def timed_init(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return original_init(self, *args, **kwargs)
        finally:
            timing["model_load_seconds"] += time.monotonic() - started

    CrystalGenerator.__init__ = timed_init
    wall_start = time.time()
    monotonic_start = time.monotonic()
    try:
        code = sample.main()
    finally:
        CrystalGenerator.__init__ = original_init
    wall_finish = time.time()
    total = time.monotonic() - monotonic_start

    if not output.exists():
        return code
    summary_path = output / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "start_time_unix": wall_start,
                "finish_time_unix": wall_finish,
                "model_load_seconds": timing["model_load_seconds"],
                "sampling_seconds": max(0.0, total - timing["model_load_seconds"]),
                "gpu_slot": gpu_slot,
                "workers_per_gpu": workers_per_gpu,
                "worker_pid": os.getpid(),
                "worker_pgid": os.getpgid(0),
                "cpu_utilization_percent_of_one_core": (
                    100.0
                    * float(summary.get("process_cpu_seconds", 0.0))
                    / max(float(summary.get("elapsed_seconds", total)), 1e-9)
                ),
            }
        )
        atomic_json(summary_path, summary)

    config_path = output / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["gpu_slot"] = gpu_slot
        config["workers_per_gpu"] = workers_per_gpu
        config["adaptive_epsilon"] = 1e-6
        config["guidance_min_scale"] = 0.0
        config["guidance_max_scale"] = 5.0
        atomic_json(config_path, config)
        atomic_text(
            output / "resolved_config.yaml",
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        )

    cfg_path = output / "cfg_summary.json"
    corrector_path = output / "corrector_summary.json"
    if cfg_path.exists() and corrector_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        corrector = json.loads(corrector_path.read_text(encoding="utf-8"))
        atomic_json(
            output / "nfe_summary.json",
            {"cfg": cfg, "corrector_gating": corrector},
        )
        atomic_json(output / "corrector_gating_summary.json", corrector)

    telemetry_path = output / "gpu_telemetry.json"
    if telemetry_path.exists():
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        rows = telemetry.get("samples", [])
        csv_path = output / "telemetry.csv"
        tmp = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
        fields = (
            "time",
            "memory_used_mib",
            "memory_free_mib",
            "utilization_gpu_percent",
            "power_draw_w",
        )
        with tmp.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in fields} for row in rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, csv_path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
