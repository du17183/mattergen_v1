from __future__ import annotations

import subprocess
from pathlib import Path

from research.spg_static_mvp.analyze_shapes import main as analyze_shapes
from research.spg_static_mvp.common import (
    LOGS,
    PROJECT,
    PYTHON,
    RESULTS,
    atomic_json,
    base_environment,
    now,
    set_stage,
)


SEEDS = tuple(range(24500, 24532))


def main() -> int:
    set_stage(
        "shape_distribution",
        "running",
        "Collecting 64000 real C0 predictor/corrector states across 32 seeds.",
        {"seeds": list(SEEDS), "expected_states": 64_000, "gpu_count": 8},
    )
    log_dir = LOGS / "shape_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    for gpu in range(8):
        seeds = SEEDS[gpu::8]
        environment = base_environment(gpu)
        log_path = log_dir / f"gpu{gpu}.log"
        stream = log_path.open("a", encoding="utf-8")
        command = [
            str(PYTHON),
            "-m",
            "research.spg_static_mvp.shape_worker",
            "--physical-gpu",
            str(gpu),
            "--seeds",
            *[str(seed) for seed in seeds],
        ]
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((gpu, process, command, str(log_path)))
        streams.append(stream)
    atomic_json(
        RESULTS / "progress/shape_launch.json",
        {
            "created_at": now(),
            "workers": [
                {
                    "gpu": gpu,
                    "pid": process.pid,
                    "command": command,
                    "log": log,
                }
                for gpu, process, command, log in processes
            ],
        },
    )
    return_codes = []
    try:
        for gpu, process, _, log in processes:
            return_codes.append(
                {"gpu": gpu, "returncode": process.wait(), "log": log}
            )
    finally:
        for stream in streams:
            stream.close()
    failures = [row for row in return_codes if row["returncode"] != 0]
    if failures:
        set_stage(
            "shape_distribution",
            "failed",
            "One or more shape-audit workers failed.",
            {"workers": return_codes},
        )
        raise RuntimeError(f"shape audit workers failed: {failures}")
    bucket = analyze_shapes()
    set_stage(
        "shape_distribution",
        "success",
        (
            f"Collected {bucket['total_states']} states; froze bucket "
            f"{bucket['selected_bucket']} with {bucket['state_coverage']:.2%} coverage."
        ),
        bucket,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
