"""Continue an already-running generation stage through evaluation and speed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from research.corrector_distillation.v3_anchor_protocol import (
    PROJECT_ROOT, methods_for, output_root_for, seeds_for,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stage-b", "stage-c"), required=True)
    parser.add_argument("--generation-pid", type=int, required=True)
    parser.add_argument("--speed-pid", type=int)
    parser.add_argument("--metrics-pid", type=int)
    args = parser.parse_args()
    root = output_root_for(args.stage)
    methods = methods_for(args.stage)
    seeds = seeds_for(args.stage)
    previous = None
    while not (root / "throughput_raw.json").exists():
        counts = {m: sum((root / "generation" / m / str(s) / "run_summary.json").exists() for s in seeds) for m in methods}
        if counts != previous:
            print(json.dumps({"event": "generation_progress", "counts": counts}), flush=True)
            previous = counts
        try:
            os.kill(args.generation_pid, 0)
        except ProcessLookupError:
            raise RuntimeError("generation process exited before completion; inspect logs")
        time.sleep(15)

    def command(module, *extra):
        return [sys.executable, "-m", f"research.corrector_distillation.{module}",
                "--stage", args.stage, *extra]

    def run(cmd):
        print(json.dumps({"event": "command_started", "command": cmd}), flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    if not (root / "generation_per_seed.csv").exists():
        run(command("aggregate_v3_anchor_generation"))
    run(command("run_v3_anchor_quality", "--phase", "relax"))
    if args.metrics_pid is not None:
        while not all((root / "quality" / m / "quality_summary.json").exists() for m in methods):
            os.kill(args.metrics_pid, 0)
            time.sleep(15)
    else:
        run(command("run_v3_anchor_quality", "--phase", "metrics"))
    speed_stage = "single-h20-b" if args.stage == "stage-b" else "single-h20-c"
    speed_complete = output_root_for(speed_stage) / "throughput_raw.json"
    if args.speed_pid is not None:
        while not speed_complete.exists():
            os.kill(args.speed_pid, 0)
            time.sleep(15)
    elif not speed_complete.exists():
        run([sys.executable, "-m", "research.corrector_distillation.run_v3_anchor_generation",
             "--stage", speed_stage])
    run(command("v3_anchor_statistics"))
    print(json.dumps({"event": "stage_complete", "stage": args.stage}), flush=True)


if __name__ == "__main__":
    main()
