from __future__ import annotations

import os
import subprocess
import sys

from research.fn_pra.phase1_common import CACHE, LOGS, PROJECT, read_json, set_stage


TEACHER_PYTHON = "/data/dxl/envs/fn_pra_teacher/bin/python"


def run_split(split: str) -> None:
    staging = CACHE / "staging" / split
    complete = len(list(staging.glob("shard_*.npz"))) == 8 and len(
        list(staging.glob("shard_*.json"))
    ) == 8
    if complete:
        return
    processes = []
    handles = []
    for shard in range(8):
        log_path = LOGS / f"teacher_cache_{split}_shard_{shard:02d}.log"
        handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        command = [
            TEACHER_PYTHON,
            "-m",
            "research.fn_pra.teacher_cache_worker",
            "--split",
            split,
            "--shard",
            str(shard),
            "--num-shards",
            "8",
            "--batch-size",
            "64",
            "--chunk-size",
            "256",
        ]
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard, process))
        handles.append(handle)
    failures = []
    for shard, process in processes:
        code = process.wait()
        if code != 0:
            failures.append((shard, code))
    for handle in handles:
        handle.close()
    if failures:
        raise RuntimeError(f"{split} teacher cache failures: {failures}")


def main() -> None:
    existing = read_json(CACHE / "cache_integrity_report.json", {})
    if existing.get("passed"):
        set_stage("online_teacher_validation", "success", "Existing validated teacher cache reused.")
        set_stage("teacher_cache", "success", "Existing validated teacher cache reused.")
        return
    try:
        set_stage(
            "online_teacher_validation",
            "running",
            "Validating CHGNet online output before/against cache.",
        )
        set_stage(
            "teacher_cache",
            "running",
            "Building full finite-dft_mag train/val cache on 8 GPUs.",
        )
        run_split("train")
        run_split("val")
        subprocess.run(
            [sys.executable, "-m", "research.fn_pra.merge_teacher_cache"],
            cwd=PROJECT,
            check=True,
        )
    except Exception as exc:
        set_stage("teacher_cache", "failed", repr(exc))
        raise


if __name__ == "__main__":
    main()
