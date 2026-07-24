#!/usr/bin/env python3
"""Resume-safe threshold probe and 8-seed budget-aware smoke study."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

import progress as main_progress

ROOT = Path("/data/dxl")
PROJECT = ROOT / "mattergen_v1"
PYTHON = ROOT / "envs/mattergen_py310/bin/python"
SAMPLE = ROOT / "tools/budget_aware_gating/run_budget_sample.py"
RESULT = ROOT / "results/budget_aware_gating"
REPORT = ROOT / "reports/budget_aware_gating"
LOG = ROOT / "logs/budget_aware_gating"
TASK_LOG = LOG / "tasks"
PROBE_ROOT = RESULT / "threshold_probe"
SMOKE_ROOT = RESULT / "eight_seed_smoke"
CONFIG_ROOT = REPORT / "frozen_candidate_configs"
PROBE_STATE = RESULT / "progress/threshold_probe_tasks.json"
SMOKE_STATE = RESULT / "progress/eight_seed_tasks.json"
SMOKE_TIMING = REPORT / "eight_seed_wave_timings.json"
PROBE_SEEDS = list(range(14000, 14004))
SMOKE_SEEDS = list(range(14000, 14008))
REPEAT_SEEDS = list(range(14000, 14004))
G3_SOURCE = REPORT / "frozen_formal_baseline/innovation2_g3_config.json"
G3_EXPECTED_SHA = "33177d98708dc0a3d2a05643f640989071653cca40e8aea4e6578269723e07a1"
LEVEL1_KEYS = (
    "rng_state_hash",
    "initial_atomic_numbers_hash",
    "initial_pos_hash",
    "initial_cell_hash",
    "initial_state_hash",
    "final_structure_hash",
    "extxyz_sha256",
)
FIELDS = ("cell", "pos", "atomic")
TRACE_METRICS = tuple(
    f"{phase}_{kind}_{field}"
    for phase in ("predictor", "corrector")
    for kind in ("residual_change", "update")
    for field in FIELDS
)
PREDICTOR_UPDATES = tuple(f"predictor_update_{field}" for field in FIELDS)
ATOMIC_METRICS = tuple(key for key in TRACE_METRICS if key.endswith("_atomic"))
stop_requested = threading.Event()
children: dict[int, subprocess.Popen] = {}
children_lock = threading.Lock()
state_lock = threading.RLock()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def finite(row: dict, key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def values(row: dict, keys: tuple[str, ...]) -> list[float] | None:
    selected = [finite(row, key) for key in keys]
    if any(value is None for value in selected):
        return None
    return [float(value) for value in selected if value is not None]


def task_environment(gpu: int, task_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "MATTERGEN_INTEROP_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MATTERGEN_BUDGET_TASK": "1",
            "MATTERGEN_BUDGET_TASK_ID": task_id,
        }
    )
    return env


def config_args(config: dict) -> list[str]:
    args = [
        "--warmup", str(config["warmup_frac"]),
        "--min-progress", str(config["min_progress"]),
        "--max-progress", str(config["max_progress"]),
        "--threshold", str(config["convergence_threshold"]),
        "--stable-steps", str(config["consecutive_stable_steps"]),
        "--calibration-interval", str(config["calibration_interval"]),
        "--max-skips", str(config["max_consecutive_skips"]),
        "--fallback-threshold", str(config["fallback_threshold"]),
        "--max-skip-ratio", str(config.get("max_skip_ratio", 1.0)),
        "--atomic-threshold", str(config.get("atomic_stability_threshold", 0.05)),
        "--atomic-stable-steps", str(config.get("atomic_min_stable_steps", 1)),
        "--calibration-min", str(config.get("calibration_interval_min", 4)),
        "--calibration-max", str(config.get("calibration_interval_max", 16)),
        "--field-aggregation", config.get("field_aggregation", "all_fields"),
        "--rescue-enabled" if config.get("rescue_enabled", True) else "--no-rescue-enabled",
        "--atomic-veto-enabled" if config.get("atomic_veto_enabled", False) else "--no-atomic-veto-enabled",
        "--adaptive-calibration-enabled" if config.get("adaptive_calibration_enabled", False) else "--no-adaptive-calibration-enabled",
    ]
    if config.get("gating_enabled", False):
        args.append("--gating-enabled")
    if config.get("budget_aware_enabled", False):
        args.append("--budget-aware-enabled")
    return args


def command(task: dict) -> list[str]:
    return [
        str(PYTHON), str(SAMPLE),
        "--output-dir", task["output_dir"],
        "--seed", str(task["seed"]),
        "--physical-gpu", str(task["gpu"]),
        "--config-id", task["config_id"],
        "--repeat-index", str(task["repeat_index"]),
        "--sampling-steps", "1000",
        "--trace", task["trace"],
        *config_args(task["config"]),
    ]


def validate(task: dict) -> bool:
    output = Path(task["output_dir"])
    try:
        summary = read_json(output / "run_summary.json")
        hashes = read_json(output / "structure_hashes.json")
        corrector = read_json(output / "corrector_summary.json")
        if not summary.get("success") or int(summary["seed"]) != int(task["seed"]):
            return False
        if summary["config_id"] != task["config_id"]:
            return False
        if not summary.get("basic_structure_valid"):
            return False
        if not all(hashes.get(key) for key in LEVEL1_KEYS):
            return False
        if int(corrector["physical_model_forward_count"]) != int(corrector["predictor_forward_count"]) + int(corrector["corrector_forward_count"]):
            return False
        if task["trace"] == "disk":
            with (output / "corrector_trace.csv").open(newline="", encoding="utf-8") as stream:
                if sum(1 for _ in csv.DictReader(stream)) != 1000:
                    return False
        return True
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def quarantine(output: Path) -> None:
    if output.exists():
        os.replace(output, output.with_name(f"{output.name}.incomplete.{int(time.time())}.{os.getpid()}"))


def save_state(path: Path, state: dict) -> None:
    with state_lock:
        state["updated_at"] = now()
        atomic_json(path, state)
        atomic_csv(path.with_suffix(".csv"), state["tasks"])


def load_state(path: Path, tasks: list[dict]) -> dict:
    if path.exists():
        state = read_json(path)
        known = {task["task_id"]: task for task in state["tasks"]}
        for task in tasks:
            if task["task_id"] not in known:
                task.update(status="pending", attempt=0, failure_reason="")
                state["tasks"].append(task)
        for task in state["tasks"]:
            if task["status"] in {"running", "interrupted", "failed", "incomplete", "skipped_resume"}:
                task["status"] = "success" if validate(task) else "interrupted"
            elif task["status"] == "success" and not validate(task):
                task["status"] = "incomplete"
        save_state(path, state)
        return state
    for task in tasks:
        task.update(status="pending", attempt=0, failure_reason="")
    state = {"created_at": now(), "updated_at": now(), "tasks": tasks}
    save_state(path, state)
    return state


def safe_interrupt(pid: int) -> bool:
    try:
        proc = Path(f"/proc/{pid}")
        if proc.stat().st_uid != os.getuid():
            return False
        command_line = (proc / "cmdline").read_bytes()
        environment = (proc / "environ").read_bytes()
        cwd = Path(os.readlink(proc / "cwd")).resolve()
        pgid = os.getpgid(pid)
        if b"run_budget_sample.py" not in command_line:
            return False
        if b"MATTERGEN_BUDGET_TASK=1" not in environment:
            return False
        if cwd != PROJECT.resolve() or pgid != pid:
            return False
        os.killpg(pgid, signal.SIGINT)
        return True
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return False


def on_signal(signum: int, _frame) -> None:
    stop_requested.set()
    main_progress.update(None, None, "stop requested", {"stop_requested": True, "last_signal": signum})
    with children_lock:
        pids = list(children)
    for pid in pids:
        safe_interrupt(pid)


def run_task(task: dict, state: dict, state_path: Path) -> bool:
    with state_lock:
        row = next(item for item in state["tasks"] if item["task_id"] == task["task_id"])
        if row["status"] == "success" and validate(row):
            row["status"] = "skipped_resume"
            save_state(state_path, state)
            row["status"] = "success"
            save_state(state_path, state)
            return True
        if stop_requested.is_set():
            return False
        row.update(status="running", attempt=int(row["attempt"]) + 1, started_at=now(), finished_at=None, failure_reason="")
        save_state(state_path, state)
    output = Path(row["output_dir"])
    quarantine(output)
    TASK_LOG.mkdir(parents=True, exist_ok=True)
    stdout = TASK_LOG / f"{row['task_id']}.stdout.log"
    stderr = TASK_LOG / f"{row['task_id']}.stderr.log"
    started = time.monotonic()
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        process = subprocess.Popen(command(row), cwd=PROJECT, env=task_environment(int(row["gpu"]), row["task_id"]), stdout=out, stderr=err, start_new_session=True)
        with children_lock:
            children[process.pid] = process
        code = process.wait()
        with children_lock:
            children.pop(process.pid, None)
    elapsed = time.monotonic() - started
    if output.exists():
        shutil.copyfile(stdout, output / "stdout.log")
        shutil.copyfile(stderr, output / "stderr.log")
    valid = code == 0 and validate(row)
    with state_lock:
        row.update(status="success" if valid else ("interrupted" if stop_requested.is_set() else "failed"), return_code=code, elapsed_seconds=elapsed, finished_at=now(), failure_reason="" if valid else f"return_code={code}; strict validation failed")
        save_state(state_path, state)
    return valid


def run_wave(tasks: list[dict], state: dict, state_path: Path, workers: int) -> dict:
    pending = [task for task in tasks if not (task.get("status") == "success" and validate(task))]
    started = time.monotonic()
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_task, task, state, state_path): task for task in pending}
            for future in as_completed(futures):
                if not future.result():
                    stop_requested.set()
    elapsed = time.monotonic() - started
    return {"success": all(validate(task) for task in tasks), "elapsed_seconds": elapsed, "executed_tasks": len(pending), "throughput_structures_per_hour": len(tasks) * 3600.0 / elapsed if elapsed > 0 and pending else None}


def probe_config() -> dict:
    return {
        "gating_enabled": True,
        "budget_aware_enabled": False,
        "warmup_frac": 1.0,
        "min_progress": 0.0,
        "max_progress": 1.0,
        "convergence_threshold": 0.05,
        "consecutive_stable_steps": 3,
        "calibration_interval": 10,
        "max_consecutive_skips": 8,
        "fallback_threshold": 1.0e9,
        "rescue_enabled": True,
        "max_skip_ratio": 1.0,
        "atomic_veto_enabled": False,
        "atomic_stability_threshold": 0.05,
        "atomic_min_stable_steps": 1,
        "adaptive_calibration_enabled": False,
        "calibration_interval_min": 4,
        "calibration_interval_max": 16,
        "field_aggregation": "all_fields",
    }


def task(stage: str, config_id: str, config: dict, seed: int, gpu: int, repeat_index: int, trace: str) -> dict:
    task_id = f"{stage}_{config_id}_seed{seed}_rep{repeat_index}"
    return {"task_id": task_id, "stage": stage, "config_id": config_id, "config": config, "seed": seed, "gpu": gpu, "repeat_index": repeat_index, "trace": trace, "output_dir": str((PROBE_ROOT if stage == "threshold_probe" else SMOKE_ROOT) / task_id)}


def read_trace(output: Path) -> list[dict]:
    with (output / "corrector_trace.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def simulate(traces: list[list[dict]], definition: dict, threshold: float, atomic_threshold: float, fallback_threshold: float) -> dict:
    total = scheduled = rescues = fallbacks = calibrations = vetoes = 0
    budget_used_total = 0
    for rows in traces:
        gateable = sum(definition["min_progress"] <= index / max(len(rows) - 1, 1) <= definition["max_progress"] for index in range(len(rows)))
        budget = math.floor(gateable * definition["max_skip_ratio"] + 1e-12)
        budget_used = stable_count = atomic_stable_steps = consecutive = since = 0
        current_interval = definition["calibration_interval_min"]
        previous: dict[str, float | None] = {field: None for field in FIELDS}
        previous_all: float | None = None
        for index, row in enumerate(rows):
            progress = index / max(len(rows) - 1, 1)
            outside = not definition["min_progress"] <= progress <= definition["max_progress"]
            fallback = previous_all is None or previous_all > fallback_threshold
            atomic_veto = bool(previous["cell"] is not None and previous["pos"] is not None and previous["cell"] <= threshold and previous["pos"] <= threshold and atomic_stable_steps < definition["atomic_min_stable_steps"])
            if outside:
                execute = True
            elif fallback:
                execute = True
                fallbacks += 1
                current_interval = definition["calibration_interval_min"]
            elif budget_used >= budget:
                execute = True
            elif atomic_veto:
                execute = True
                vetoes += 1
                current_interval = definition["calibration_interval_min"]
            elif consecutive >= definition["max_consecutive_skips"] or since >= current_interval:
                execute = True
                calibrations += 1
                current_interval = min(definition["calibration_interval_max"], max(current_interval + 1, math.ceil(current_interval * 1.5)))
            elif stable_count >= definition["consecutive_stable_steps"]:
                execute = False
            else:
                execute = True
            if execute:
                consecutive = since = 0
            else:
                scheduled += 1
                budget_used += 1
                consecutive += 1
                since += 1
                pred = values(row, PREDICTOR_UPDATES)
                if pred is None or max(pred) > fallback_threshold:
                    rescues += 1
                    consecutive = since = stable_count = 0
                    current_interval = definition["calibration_interval_min"]
            per_field = {}
            for field in FIELDS:
                selected = values(row, tuple(key for key in TRACE_METRICS if key.endswith(f"_{field}")))
                per_field[field] = max(selected) if selected is not None else None
            all_finite = all(value is not None for value in per_field.values())
            previous_all = max(float(value) for value in per_field.values() if value is not None) if all_finite else None
            stable_count = stable_count + 1 if previous_all is not None and previous_all <= threshold else 0
            atomic_stable_steps = atomic_stable_steps + 1 if per_field["atomic"] is not None and per_field["atomic"] <= atomic_threshold else 0
            previous = per_field
            if definition.get("adaptive_calibration_enabled") and (previous_all is None or previous_all > threshold):
                current_interval = max(definition["calibration_interval_min"], current_interval // 2)
            total += 1
        budget_used_total += budget_used
    net = scheduled - rescues
    return {"scheduled_skips": scheduled, "estimated_rescues": rescues, "estimated_fallbacks": fallbacks, "estimated_calibrations": calibrations, "estimated_atomic_vetoes": vetoes, "budget_used": budget_used_total, "net_corrector_skips": net, "estimated_corrector_skip_rate": net / total, "estimated_physical_forward_reduction": net / (2 * total)}


def derive(probe_tasks: list[dict]) -> dict:
    traces = [read_trace(Path(item["output_dir"])) for item in probe_tasks]
    all_metrics = []
    atomic_metrics = []
    predictor_metrics = []
    for rows in traces:
        for row in rows:
            selected = values(row, TRACE_METRICS)
            atomic = values(row, ATOMIC_METRICS)
            predictor = values(row, PREDICTOR_UPDATES)
            if selected is not None:
                all_metrics.append(max(selected))
            if atomic is not None:
                atomic_metrics.append(max(atomic))
            if predictor is not None:
                predictor_metrics.append(max(predictor))
    if not all_metrics or not atomic_metrics:
        raise RuntimeError("probe produced no finite all-field metrics")
    definitions = {
        "G1": {"target": 0.13, "min_progress": 0.40, "max_progress": 1.0, "max_skip_ratio": 0.45, "consecutive_stable_steps": 4, "calibration_interval": 4, "calibration_interval_min": 4, "calibration_interval_max": 8, "max_consecutive_skips": 5, "atomic_min_stable_steps": 3, "atomic_quantile": 0.75, "adaptive_calibration_enabled": True},
        "G2": {"target": 0.21, "min_progress": 0.30, "max_progress": 1.0, "max_skip_ratio": 0.60, "consecutive_stable_steps": 3, "calibration_interval": 6, "calibration_interval_min": 6, "calibration_interval_max": 12, "max_consecutive_skips": 8, "atomic_min_stable_steps": 2, "atomic_quantile": 0.90, "adaptive_calibration_enabled": True},
    }
    search = sorted({float(np.quantile(all_metrics, q)) for q in np.linspace(0.05, 0.999, 400)})
    candidates = {}
    for config_id, definition in definitions.items():
        best = None
        for threshold in search:
            atomic_threshold = min(threshold, float(np.quantile(atomic_metrics, definition["atomic_quantile"])))
            fallback = max(threshold * 1.5, float(np.quantile(all_metrics, 0.99)), float(np.quantile(predictor_metrics, 0.99)))
            estimate = simulate(traces, definition, threshold, atomic_threshold, fallback)
            score = abs(estimate["estimated_physical_forward_reduction"] - definition["target"])
            if best is None or score < best[0]:
                best = (score, threshold, atomic_threshold, fallback, estimate)
        assert best is not None
        _, threshold, atomic_threshold, fallback, estimate = best
        candidates[config_id] = {
            "config_id": config_id,
            "gating_enabled": True,
            "budget_aware_enabled": True,
            "warmup_frac": definition["min_progress"],
            "min_progress": definition["min_progress"],
            "max_progress": definition["max_progress"],
            "convergence_threshold": threshold,
            "consecutive_stable_steps": definition["consecutive_stable_steps"],
            "calibration_interval": definition["calibration_interval"],
            "max_consecutive_skips": definition["max_consecutive_skips"],
            "fallback_threshold": fallback,
            "rescue_enabled": True,
            "max_skip_ratio": definition["max_skip_ratio"],
            "atomic_veto_enabled": True,
            "atomic_stability_threshold": atomic_threshold,
            "atomic_min_stable_steps": definition["atomic_min_stable_steps"],
            "adaptive_calibration_enabled": definition["adaptive_calibration_enabled"],
            "calibration_interval_min": definition["calibration_interval_min"],
            "calibration_interval_max": definition["calibration_interval_max"],
            "field_aggregation": "all_fields",
            "target_physical_forward_reduction": definition["target"],
            "probe_estimate": estimate,
            "probe_seeds": PROBE_SEEDS,
            "source_commit": subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True).strip(),
        }
    quantiles = {f"q{int(q*100):02d}": float(np.quantile(all_metrics, q)) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)}
    payload = {"method": "full-corrector trace quantiles plus offline budget/veto/calibration simulation", "probe_seeds": PROBE_SEEDS, "aggregation": "all_fields", "metric_quantiles": quantiles, "atomic_metric_quantiles": {f"q{int(q*100):02d}": float(np.quantile(atomic_metrics, q)) for q in (0.50, 0.75, 0.90, 0.95, 0.99)}, "candidates": candidates}
    atomic_json(REPORT / "threshold_probe_statistics.json", payload)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    for config_id, config in candidates.items():
        path = CONFIG_ROOT / f"{config_id}.json"
        atomic_json(path, config)
        atomic_text(CONFIG_ROOT / f"{config_id}.sha256", f"{sha256(path)}  {path.name}\n")
        os.chmod(path, 0o444)
    return payload


def a0_config() -> dict:
    config = probe_config()
    config.update(gating_enabled=False, warmup_frac=0.15, min_progress=0.15, max_progress=0.95, fallback_threshold=0.20)
    return config


def g3_config() -> dict:
    if sha256(G3_SOURCE) != G3_EXPECTED_SHA:
        raise RuntimeError("original G3 frozen SHA mismatch")
    gate = read_json(G3_SOURCE)["corrector_gating"]
    return {
        "config_id": "G3",
        "gating_enabled": True,
        "budget_aware_enabled": False,
        "warmup_frac": gate["corrector_warmup_frac"],
        "min_progress": gate["corrector_min_progress"],
        "max_progress": gate["corrector_max_progress"],
        "convergence_threshold": gate["corrector_convergence_threshold"],
        "consecutive_stable_steps": gate["corrector_consecutive_stable_steps"],
        "calibration_interval": gate["corrector_calibration_interval"],
        "max_consecutive_skips": gate["corrector_max_consecutive_skips"],
        "fallback_threshold": gate["corrector_fallback_threshold"],
        "rescue_enabled": gate["corrector_rescue_enabled"],
        "max_skip_ratio": 1.0,
        "atomic_veto_enabled": False,
        "atomic_stability_threshold": 0.05,
        "atomic_min_stable_steps": 1,
        "adaptive_calibration_enabled": False,
        "calibration_interval_min": gate["corrector_calibration_interval"],
        "calibration_interval_max": gate["corrector_calibration_interval"],
        "field_aggregation": "all_fields",
        "original_g3_config_sha256": G3_EXPECTED_SHA,
    }


def result(task: dict) -> dict:
    output = Path(task["output_dir"])
    summary = read_json(output / "run_summary.json")
    hashes = read_json(output / "structure_hashes.json")
    corrector = read_json(output / "corrector_summary.json")
    return {"task_id": task["task_id"], "config_id": task["config_id"], "seed": task["seed"], "repeat_index": task["repeat_index"], "gpu": task["gpu"], "elapsed_seconds": summary["elapsed_seconds"], "basic_structure_valid": summary["basic_structure_valid"], "composition_valid": summary["composition_valid"], **{key: hashes[key] for key in LEVEL1_KEYS}, **{key: corrector.get(key) for key in ("physical_model_forward_count", "predictor_forward_count", "corrector_forward_count", "corrector_skipped_count", "corrector_skip_rate", "physical_forward_reduction", "corrector_calibration_count", "corrector_fallback_count", "corrector_rescue_count", "corrector_atomic_veto_count", "corrector_skip_budget", "corrector_budget_used", "budget_exhausted", "calibration_interval_mean", "cell_stable_rate", "pos_stable_rate", "atomic_stable_rate", "all_fields_stable_rate")}}


def smoke_analysis(main_tasks: list[dict], repeat_tasks: list[dict], timings: dict, configs: dict) -> dict:
    main_rows = [result(task) for task in main_tasks]
    repeat_rows = [result(task) for task in repeat_tasks]
    atomic_csv(REPORT / "eight_seed_runs.csv", main_rows + repeat_rows)
    by = {(row["config_id"], row["seed"], row["repeat_index"]): row for row in main_rows + repeat_rows}
    determinism = {}
    for config_id in configs:
        determinism[config_id] = all(all(by[(config_id, seed, 1)][key] == by[(config_id, seed, 2)][key] for key in LEVEL1_KEYS) for seed in REPEAT_SEEDS)
    initial_pairing = all(len({by[(config_id, seed, 1)]["initial_state_hash"] for config_id in configs}) == 1 for seed in SMOKE_SEEDS)
    aggregate = {}
    for config_id in configs:
        rows = [row for row in main_rows if row["config_id"] == config_id]
        aggregate[config_id] = {
            "generation_success_rate": len(rows) / 8,
            "basic_structure_validity": statistics.mean(bool(row["basic_structure_valid"]) for row in rows),
            "composition_validity": statistics.mean(bool(row["composition_valid"]) for row in rows),
            "median_elapsed_seconds": statistics.median(float(row["elapsed_seconds"]) for row in rows),
            "physical_forward_reduction": statistics.mean(float(row["physical_forward_reduction"]) for row in rows),
            "corrector_skip_rate": statistics.mean(float(row["corrector_skip_rate"]) for row in rows),
            "atomic_veto_mean": statistics.mean(float(row["corrector_atomic_veto_count"] or 0) for row in rows),
            "fallback_mean": statistics.mean(float(row["corrector_fallback_count"] or 0) for row in rows),
            "calibration_mean": statistics.mean(float(row["corrector_calibration_count"] or 0) for row in rows),
            "rescue_mean": statistics.mean(float(row["corrector_rescue_count"] or 0) for row in rows),
            "wave_elapsed_seconds": timings[config_id]["elapsed_seconds"],
            "throughput_structures_per_hour": timings[config_id]["throughput_structures_per_hour"],
            "determinism_level1": determinism[config_id],
        }
    a0 = aggregate["A0"]
    decisions = {}
    for config_id in ("G1", "G2"):
        item = aggregate[config_id]
        item["speed_multiplier"] = a0["median_elapsed_seconds"] / item["median_elapsed_seconds"]
        item["throughput_gain"] = item["throughput_structures_per_hour"] / a0["throughput_structures_per_hour"] - 1.0
        gates = {
            "physical_forward_reduction_ge_10_percent": item["physical_forward_reduction"] >= 0.10,
            "speed_or_throughput": item["speed_multiplier"] >= 1.10 or item["throughput_gain"] >= 0.12,
            "generation_success_100_percent": item["generation_success_rate"] == 1.0,
            "determinism_level1": item["determinism_level1"],
            "structure_validity_100_percent": item["basic_structure_validity"] == 1.0,
            "composition_validity_not_obviously_lower": item["composition_validity"] >= a0["composition_validity"] - 0.125,
            "cross_config_initial_pairing": initial_pairing,
        }
        decisions[config_id] = {"go": all(gates.values()), "gates": gates}
    payload = {"BUDGET_AWARE_GATING_GO": any(item["go"] for item in decisions.values()), "seeds": [14000, 14007], "repeat_seeds": REPEAT_SEEDS, "aggregate": aggregate, "candidate_decisions": decisions, "initial_hash_pairing": initial_pairing, "original_g3_parameters_modified": False, "formal_30000_seeds_started": False}
    atomic_json(REPORT / "eight_seed_go_no_go.json", payload)
    lines = ["# Budget-aware 8-seed smoke", "", f"- BUDGET_AWARE_GATING_GO={payload['BUDGET_AWARE_GATING_GO']}", f"- Initial hashes paired={initial_pairing}", ""]
    for config_id in configs:
        lines.append(f"- {config_id}: {json.dumps(aggregate[config_id], ensure_ascii=False)}")
    atomic_text(REPORT / "eight_seed_go_no_go.md", "\n".join(lines) + "\n")
    return payload


def main() -> int:
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    main_progress.update(None, None, "probe/smoke launcher started", {"launcher_pid": os.getpid(), "launcher_pgid": os.getpgid(0), "stop_requested": False})
    try:
        main_progress.update("threshold_probe", "running", "Full A0-equivalent Corrector traces on seeds 14000-14003.", {})
        probe_tasks = [task("threshold_probe", "P0_FULL", probe_config(), seed, index, 1, "disk") for index, seed in enumerate(PROBE_SEEDS)]
        probe_state = load_state(PROBE_STATE, probe_tasks)
        probe_rows = [next(item for item in probe_state["tasks"] if item["task_id"] == planned["task_id"]) for planned in probe_tasks]
        probe_wave = run_wave(probe_rows, probe_state, PROBE_STATE, 4)
        if not probe_wave["success"]:
            raise RuntimeError("threshold probe failed strict validation")
        candidates = derive(probe_rows)
        main_progress.update("threshold_probe", "success", "4/4 full traces validated; G1/G2 resolved configs frozen by SHA256.", {"probe_completed": True})

        configs = {"A0": a0_config(), "G1": candidates["candidates"]["G1"], "G2": candidates["candidates"]["G2"], "G3": g3_config()}
        main_progress.update("eight_seed_generation", "running", "A0/G1/G2/G3 seeds 14000-14007 plus four Level-1 repeats per config.", {})
        main_tasks = [task("eight_seed_smoke", config_id, config, seed, index, 1, "off") for config_id, config in configs.items() for index, seed in enumerate(SMOKE_SEEDS)]
        repeat_tasks = [task("eight_seed_smoke", config_id, config, seed, index, 2, "off") for config_id, config in configs.items() for index, seed in enumerate(REPEAT_SEEDS)]
        smoke_state = load_state(SMOKE_STATE, main_tasks + repeat_tasks)
        timings = read_json(SMOKE_TIMING) if SMOKE_TIMING.exists() else {}
        for config_id in configs:
            wave_tasks = [next(item for item in smoke_state["tasks"] if item["task_id"] == planned["task_id"]) for planned in main_tasks if planned["config_id"] == config_id]
            wave = run_wave(wave_tasks, smoke_state, SMOKE_STATE, 8)
            if not wave["success"]:
                raise RuntimeError(f"smoke main wave failed: {config_id}")
            if wave["executed_tasks"]:
                timings[config_id] = wave
                atomic_json(SMOKE_TIMING, timings)
            elif config_id not in timings:
                elapsed = max(float(result(item)["elapsed_seconds"]) for item in wave_tasks)
                timings[config_id] = {"success": True, "elapsed_seconds": elapsed, "executed_tasks": 0, "throughput_structures_per_hour": len(wave_tasks) * 3600.0 / elapsed, "measurement": "resume-fallback-max-task-elapsed"}
                atomic_json(SMOKE_TIMING, timings)
        repeats = [next(item for item in smoke_state["tasks"] if item["task_id"] == planned["task_id"]) for planned in repeat_tasks]
        if not run_wave(repeats, smoke_state, SMOKE_STATE, 8)["success"]:
            raise RuntimeError("smoke determinism repeats failed")
        main_progress.update("eight_seed_generation", "success", "A0/G1/G2/G3 main=32/32 and determinism repeats=16/16.", {})
        main_progress.update("eight_seed_go_no_go", "running", "Evaluating speed, physical-forward, determinism and validity gates.", {})
        decision = smoke_analysis([next(item for item in smoke_state["tasks"] if item["task_id"] == planned["task_id"]) for planned in main_tasks], repeats, timings, configs)
        status = "success"
        main_progress.update("eight_seed_go_no_go", status, f"BUDGET_AWARE_GATING_GO={decision['BUDGET_AWARE_GATING_GO']}", {"budget_aware_gating_go": decision["BUDGET_AWARE_GATING_GO"], "eight_seed_completed": True})
        if not decision["BUDGET_AWARE_GATING_GO"]:
            main_progress.update(None, None, "Both conservative candidates failed 8-seed gates; stopping without retuning.", {"overall_status": "incomplete"})
            return 2
        return 0
    except KeyboardInterrupt:
        main_progress.update(main_progress.read_json(main_progress.MASTER)["current_stage"] if hasattr(main_progress, "read_json") else None, "interrupted", "Validated SIGINT; success tasks preserved.", {})
        return 130
    except BaseException as error:
        current = read_json(main_progress.MASTER)["current_stage"]
        main_progress.update(current, "failed", f"{type(error).__name__}: {error}", {"overall_status": "failed"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
