from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from research.fn_pra.phase1_common import LOGS, PROJECT, set_stage


PYTHONS = {
    "chgnet": "/data/dxl/envs/fn_pra_teacher/bin/python",
    "mattersim": "/data/dxl/envs/mattergen_py310/bin/python",
}


def run_candidate(candidate: str) -> None:
    processes = []
    logs = []
    for shard in range(8):
        path = LOGS / f"teacher_probe_{candidate}_shard_{shard:02d}.log"
        handle = path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        command = [
            PYTHONS[candidate],
            "-m",
            "research.fn_pra.teacher_probe_worker",
            "--candidate",
            candidate,
            "--shard",
            str(shard),
            "--num-shards",
            "8",
            "--batch-size",
            "16",
        ]
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((shard, process))
        logs.append(handle)
    failures = []
    for shard, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((shard, return_code))
    for handle in logs:
        handle.close()
    if failures:
        raise RuntimeError(f"{candidate} probe shard failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.parse_args()
    set_stage("teacher_candidate_audit", "success", "MatterSim-5M and CHGNet 0.3.0 are available with fixed checkpoints.")
    set_stage("teacher_probe", "running", "Running CHGNet and MatterSim probes, eight GPU shards per candidate.")
    try:
        for candidate in ("chgnet", "mattersim"):
            expected = list((Path("/data/dxl/results/fn_pra/phase1") / f"teacher_probe/{candidate}").glob("shard_*.npz"))
            if len(expected) == 8:
                continue
            run_candidate(candidate)
        subprocess.run(
            [sys.executable, "-m", "research.fn_pra.merge_teacher_probe"],
            cwd=PROJECT,
            check=True,
        )
    except Exception as exc:
        set_stage("teacher_probe", "failed", repr(exc))
        raise


if __name__ == "__main__":
    main()
