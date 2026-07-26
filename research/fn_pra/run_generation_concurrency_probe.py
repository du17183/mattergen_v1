"""Verify equal safe A0/P1 generation concurrency before the 32-seed stage."""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from research.fn_pra.phase1_common import (
    LOGS,
    PROJECT,
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
)


PYTHON = "/data/dxl/envs/mattergen_py310/bin/python"
ROOT = RESULTS / "generation/concurrency_probe"
METHODS = ("A0", "P1")
WORKERS = (1, 2, 4)
SEED = 15000
GPU = 4


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_slot(method: str, workers: int, slot: int) -> dict:
    output = ROOT / method / f"workers_{workers}" / f"slot_{slot}"
    status = output / "status.json"
    if status.is_file() and read_json(status).get("success") is True:
        return read_json(output / "summary.json")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite incomplete concurrency probe: {output}")
    log = LOGS / f"concurrency_probe_{method.lower()}_w{workers}_s{slot}.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(GPU),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    command = [
        PYTHON,
        "-m",
        "research.fn_pra.run_paired_sample",
        "--output-dir",
        str(output),
        "--method",
        method,
        "--seed",
        str(SEED),
        "--physical-gpu",
        str(GPU),
        "--sampling-steps",
        "1000",
        "--repeat-index",
        str(100 + workers * 10 + slot),
    ]
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0 or not status.is_file() or not read_json(status).get("success"):
        raise RuntimeError(f"concurrency probe failed: {method} workers={workers} slot={slot}")
    return read_json(output / "summary.json")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for workers in WORKERS:
        for method in METHODS:
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(run_slot, method, workers, slot) for slot in range(workers)
                ]
                summaries = [future.result() for future in as_completed(futures)]
            wall = time.monotonic() - started
            rows.append(
                {
                    "method": method,
                    "workers_per_gpu": workers,
                    "physical_gpu": GPU,
                    "success": len(summaries),
                    "wall_seconds": wall,
                    "structures_per_hour": workers * 3600.0 / wall,
                    "max_peak_allocated_bytes_per_process": max(
                        row["peak_allocated_bytes"] for row in summaries
                    ),
                    "max_peak_reserved_bytes_per_process": max(
                        row["peak_reserved_bytes"] for row in summaries
                    ),
                    "all_structure_valid": all(row["structure_valid"] for row in summaries),
                    "all_initial_hashes_identical": len(
                        {row["initial_hashes"]["combined"] for row in summaries}
                    )
                    == 1,
                }
            )
            atomic_json(
                REPORTS / "generation_concurrency_probe.partial.json",
                {"updated_at": now(), "rows": rows},
            )
    safe = all(
        row["success"] == row["workers_per_gpu"] and row["all_structure_valid"] for row in rows
    )
    report = {
        "created_at": now(),
        "gpu": GPU,
        "workers_tested": list(WORKERS),
        "methods": list(METHODS),
        "rows": rows,
        "selected_workers_per_gpu": 4 if safe else 2,
        "selected_total_concurrency": 32 if safe else 16,
        "passed": safe,
    }
    atomic_json(REPORTS / "generation_concurrency_probe.json", report)
    atomic_text(
        REPORTS / "generation_concurrency_probe.md",
        "# A0/P1 generation concurrency probe\n\n"
        f"- Tested workers/GPU: {list(WORKERS)} on the same physical GPU.\n"
        f"- Both methods passed all requested slots: {safe}.\n"
        f"- Selected maximum workers/GPU: {report['selected_workers_per_gpu']}.\n"
        f"- Selected maximum total concurrency: {report['selected_total_concurrency']}.\n",
    )
    return 0 if safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
