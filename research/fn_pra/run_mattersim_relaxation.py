"""Run 64 FN-PRA Phase-1 relaxations with the verified persistent MatterSim worker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np

from research.fn_pra.phase1_common import (
    LOGS,
    PROJECT,
    REPORTS,
    RESULTS,
    atomic_json,
    atomic_text,
    now,
    set_stage,
    sha256_file,
)


TOOLS = Path("/data/dxl/tools/innovation2_next")
sys.path.insert(0, str(TOOLS))
import run_corrector_32 as core  # noqa: E402


PYTHON = Path("/data/dxl/envs/mattergen_py310/bin/python")
EIGHT_ROOT = RESULTS / "generation/eight_seed"
THIRTY_TWO_ROOT = RESULTS / "generation/thirty_two_seed"
RELAXED = RESULTS / "relaxed"
PROGRESS = RESULTS / "progress"
RELAX_JSON = PROGRESS / "mattersim_relax_progress.json"
RELAX_CSV = PROGRESS / "mattersim_relax_progress.csv"
STOP_MARKER = PROGRESS / "STOP_REQUESTED"
MATTERSIM = Path("/data/dxl/mattersim_weights/mattersim-v1.0.0-5M.pth")
MATTERSIM_SHA = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
SOURCE_CONFIG = Path("/data/dxl/reports/guidance_stage7_eval/relax_config_frozen.json")
METHODS = ("A0", "P1")
SEEDS = tuple(range(15000, 15032))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generation_dir(method: str, seed: int) -> Path:
    root = EIGHT_ROOT if seed < 15008 else THIRTY_TWO_ROOT
    return root / method / f"seed_{seed}"


def configure_core() -> None:
    core.ROOT = Path("/data/dxl")
    core.PROJECT = PROJECT
    core.RESULT = RESULTS
    core.REPORT = REPORTS
    core.LOG = LOGS
    core.PROGRESS = PROGRESS
    core.RELAXED = RELAXED
    core.PYTHON = PYTHON
    core.RELAX_JSON = RELAX_JSON
    core.RELAX_CSV = RELAX_CSV
    core.STOP_MARKER = STOP_MARKER
    core.SEEDS = list(SEEDS)
    core.CONFIGS = METHODS
    core.relax_initial = relax_initial


def freeze_config() -> dict:
    if sha256_file(MATTERSIM) != MATTERSIM_SHA:
        raise RuntimeError("MatterSim-5M checkpoint SHA mismatch")
    source = read_json(SOURCE_CONFIG)
    source.update(
        {
            "source_config": str(SOURCE_CONFIG),
            "source_config_sha256": sha256_file(SOURCE_CONFIG),
            "checkpoint": str(MATTERSIM),
            "checkpoint_sha256": MATTERSIM_SHA,
            "optimizer": "FIRE",
            "cell_filter": "EXPCELLFILTER",
            "fmax_ev_ang": 0.05,
            "max_steps": 500,
            "stability_threshold_ev_atom": 0.1,
            "TRI2024_reference": "/data/dxl/reference_assets/reference_TRI2024correction.gz",
            "TRI2024_reference_sha256": "3631b54625f2a5410fb83aab16fda78073a2a713e8457e3beec523d0682315f5",
            "structure_matcher": "DefaultDisorderedStructureMatcher",
            "energy_correction": "TRI110Compatibility2024",
            "workers_per_gpu": 2,
            "total_concurrency": 16,
            "freeze_time": now(),
            "STABILITY_SOURCE": "MatterSim-5M surrogate",
            "DFT_VERIFIED": False,
            "PROPERTY_TARGET_VERIFIED": False,
        }
    )
    atomic_json(REPORTS / "mattersim_config.json", source)
    digest = sha256_file(REPORTS / "mattersim_config.json")
    atomic_text(REPORTS / "mattersim_config.sha256", f"{digest}  mattersim_config.json\n")
    return source


def relax_initial() -> dict:
    tasks = []
    config_hash = sha256_file(REPORTS / "mattersim_config.json")
    for method in METHODS:
        for seed in SEEDS:
            generated = generation_dir(method, seed)
            status = read_json(generated / "status.json")
            summary = read_json(generated / "summary.json")
            if status.get("success") is not True or summary.get("success") is not True:
                raise RuntimeError(f"generation prerequisite failed: {method} seed {seed}")
            input_path = generated / "generated_crystals.extxyz"
            atoms = ase.io.read(input_path)
            if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
                raise RuntimeError(f"non-finite generation input: {method} seed {seed}")
            tasks.append(
                {
                    "task_id": f"{method}_seed_{seed}",
                    "config": method,
                    "seed": seed,
                    "status": "pending",
                    "attempt": 0,
                    "real_failure_count": 0,
                    "input_path": str(input_path),
                    "input_hash": summary["final_structure_hash"],
                    "output_path": str(
                        RELAXED / method / f"seed_{seed}" / "relaxed_structure.extxyz"
                    ),
                    "output_hash": None,
                    "physical_gpu": None,
                    "gpu_slot": None,
                    "worker_pid": None,
                    "worker_pgid": None,
                    "checkpoint": str(MATTERSIM),
                    "checkpoint_sha256": MATTERSIM_SHA,
                    "relax_config_hash": config_hash,
                    "start_time": None,
                    "finish_time": None,
                    "elapsed": None,
                    "steps": None,
                    "energy": None,
                    "energy_per_atom": None,
                    "maximum_force": None,
                    "converged": None,
                    "return_code": None,
                    "failure_reason": "",
                }
            )
    return {
        "schema_version": 1,
        "created_at": now(),
        "updated_at": now(),
        "full_status": "pending",
        "selected_workers_per_gpu": 2,
        "selected_total_concurrency": 16,
        "tasks": tasks,
    }


def worker(gpu: int, slot: int) -> int:
    configure_core()
    return core.relax_worker(gpu, slot)


def launch() -> int:
    configure_core()
    freeze_config()
    set_stage(
        "mattersim_relaxation",
        "running",
        "Running 64 MatterSim-5M relaxations with 16 persistent workers.",
        {"workers_per_gpu": 2, "total_concurrency": 16, "tasks": 64},
    )
    core.relax_initialize()
    processes = []
    worker_logs = LOGS / "mattersim_relax_workers"
    worker_logs.mkdir(parents=True, exist_ok=True)
    for gpu in range(8):
        for slot in range(2):
            env = core.task_environment(gpu, f"fn_pra_relax_gpu{gpu}_slot{slot}")
            env["MATTERGEN_FN_PRA_RELAX_WORKER"] = "1"
            stream = (worker_logs / f"gpu{gpu}_slot{slot}.log").open(
                "a", encoding="utf-8"
            )
            process = subprocess.Popen(
                [
                    str(PYTHON),
                    "-m",
                    "research.fn_pra.run_mattersim_relaxation",
                    "relax-worker",
                    "--gpu",
                    str(gpu),
                    "--slot",
                    str(slot),
                ],
                cwd=PROJECT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((process, stream))
    return_codes = []
    for process, stream in processes:
        return_codes.append(process.wait())
        stream.close()
    state = core.relax_initialize()
    valid = [
        task
        for task in state["tasks"]
        if task["status"] == "success" and core.validate_relax(task)
    ]
    if len(valid) != 64 or any(code not in (0,) for code in return_codes):
        set_stage(
            "mattersim_relaxation",
            "failed",
            f"MatterSim relaxation incomplete: valid={len(valid)}/64.",
            {"worker_return_codes": return_codes},
        )
        return 1
    converged = sum(bool(task["converged"]) for task in valid)
    summary = {
        "created_at": now(),
        "success": 64,
        "converged": converged,
        "valid_but_not_converged": 64 - converged,
        "workers_per_gpu": 2,
        "total_concurrency": 16,
        "checkpoint_sha256": MATTERSIM_SHA,
    }
    atomic_json(REPORTS / "mattersim_relaxation_summary.json", summary)
    set_stage(
        "mattersim_relaxation",
        "success",
        f"64/64 finite relax outputs; converged={converged}, valid-at-cap={64-converged}.",
        summary,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "relax-worker"), nargs="?", default="launch")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--slot", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "relax-worker":
        if args.gpu is None or args.slot is None:
            raise ValueError("relax-worker requires --gpu and --slot")
        return worker(args.gpu, args.slot)
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
