#!/usr/bin/env python3
"""Resume-safe eight-GPU launcher for CG-TDR A0 feature generation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


STOP_EVENT = threading.Event()
ACTIVE: dict[int, subprocess.Popen] = {}
ACTIVE_LOCK = threading.Lock()
WRITE_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_event(path: Path, payload: dict) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": now(), **payload}, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def successful(seed: int, result_root: Path, teacher_root: Path) -> bool:
    summary_path = result_root / f"seed_{seed}" / "run_summary.json"
    feature_path = teacher_root / f"seed_{seed}.pt"
    if not summary_path.exists() or not feature_path.exists():
        return False
    try:
        return bool(json.loads(summary_path.read_text(encoding="utf-8")).get("success"))
    except (OSError, json.JSONDecodeError):
        return False


def update_progress(
    progress_path: Path,
    seeds: list[int],
    result_root: Path,
    teacher_root: Path,
    status: str,
) -> None:
    completed = sum(successful(seed, result_root, teacher_root) for seed in seeds)
    with ACTIVE_LOCK:
        active = {str(gpu): process.pid for gpu, process in ACTIVE.items()}
    atomic_json(
        progress_path,
        {
            "stage": "teacher_data_generation",
            "overall_status": status,
            "total_tasks": len(seeds),
            "completed_tasks": completed,
            "remaining_tasks": len(seeds) - completed,
            "active_workers": active,
            "gpu_count": 8,
            "teacher_seed_start": seeds[0],
            "teacher_seed_end": seeds[-1],
            "eight_seed_started": False,
            "thirty_two_seed_started": False,
            "sixty_four_seed_started": False,
            "formal_seeds_started": False,
            "updated_at": now(),
        },
    )


def stop_handler(_signum: int, _frame) -> None:
    STOP_EVENT.set()
    with ACTIVE_LOCK:
        children = list(ACTIVE.values())
    for child in children:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)


def worker(
    *,
    gpu: int,
    seeds: list[int],
    repository: Path,
    result_root: Path,
    teacher_root: Path,
    log_root: Path,
    progress_path: Path,
    event_path: Path,
    all_seeds: list[int],
) -> int:
    failures = 0
    for seed in seeds:
        if STOP_EVENT.is_set():
            break
        if successful(seed, result_root, teacher_root):
            continue
        output_dir = result_root / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"seed_{seed}.log"
        command = [
            sys.executable,
            str(repository / "research/cg_tdr/run_teacher_sample.py"),
            "--seed",
            str(seed),
            "--physical-gpu",
            str(gpu),
            "--output-dir",
            str(output_dir),
            "--teacher-dir",
            str(teacher_root),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "NUMEXPR_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        append_event(event_path, {"event": "task_started", "seed": seed, "gpu": gpu})
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=repository,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            with ACTIVE_LOCK:
                ACTIVE[gpu] = process
            return_code = process.wait()
            with ACTIVE_LOCK:
                ACTIVE.pop(gpu, None)
        if return_code != 0 or not successful(seed, result_root, teacher_root):
            failures += 1
            append_event(
                event_path,
                {
                    "event": "task_failed",
                    "seed": seed,
                    "gpu": gpu,
                    "return_code": return_code,
                },
            )
        else:
            append_event(event_path, {"event": "task_completed", "seed": seed, "gpu": gpu})
        with WRITE_LOCK:
            update_progress(
                progress_path,
                all_seeds,
                result_root,
                teacher_root,
                "stopping" if STOP_EVENT.is_set() else "running",
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=30000)
    parser.add_argument("--seed-end", type=int, default=30511)
    parser.add_argument("--repository", default="/data/dxl/mattergen_v1")
    parser.add_argument(
        "--result-root", default="/data/dxl/results/cg_tdr/phase0/teacher_generation"
    )
    parser.add_argument("--teacher-root", default="/data/dxl/data/cg_tdr_teacher/features")
    parser.add_argument("--log-root", default="/data/dxl/logs/cg_tdr/phase0/teacher_generation")
    parser.add_argument(
        "--progress-root", default="/data/dxl/results/cg_tdr/phase0/progress"
    )
    args = parser.parse_args()

    repository = Path(args.repository).resolve()
    result_root = Path(args.result_root).resolve()
    teacher_root = Path(args.teacher_root).resolve()
    log_root = Path(args.log_root).resolve()
    progress_root = Path(args.progress_root).resolve()
    for directory in (result_root, teacher_root, log_root, progress_root):
        directory.mkdir(parents=True, exist_ok=True)
    progress_path = progress_root / "master_progress.json"
    event_path = progress_root / "events.jsonl"
    lock_path = progress_root / "launcher.lock"
    seeds = list(range(args.seed_start, args.seed_end + 1))

    lock_stream = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another CG-TDR teacher launcher holds the lock.", file=sys.stderr)
        return 2
    lock_stream.write(str(os.getpid()) + "\n")
    lock_stream.flush()
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    update_progress(progress_path, seeds, result_root, teacher_root, "running")
    append_event(
        event_path,
        {
            "event": "launcher_started",
            "pid": os.getpid(),
            "seed_start": seeds[0],
            "seed_end": seeds[-1],
        },
    )

    partitions = [seeds[gpu::8] for gpu in range(8)]
    failures = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                worker,
                gpu=gpu,
                seeds=partitions[gpu],
                repository=repository,
                result_root=result_root,
                teacher_root=teacher_root,
                log_root=log_root,
                progress_path=progress_path,
                event_path=event_path,
                all_seeds=seeds,
            )
            for gpu in range(8)
        ]
        for future in as_completed(futures):
            failures += future.result()
    completed = sum(successful(seed, result_root, teacher_root) for seed in seeds)
    status = "stopped" if STOP_EVENT.is_set() else (
        "completed" if completed == len(seeds) and failures == 0 else "failed"
    )
    update_progress(progress_path, seeds, result_root, teacher_root, status)
    append_event(
        event_path,
        {
            "event": "launcher_finished",
            "status": status,
            "completed": completed,
            "failures": failures,
        },
    )
    return 0 if status == "completed" else (130 if status == "stopped" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
