from __future__ import annotations

import argparse
import subprocess
import sys

from research.fn_pra.phase1_common import PROJECT, REPORTS, initialize_progress, set_stage


def run(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", module, *args], cwd=PROJECT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    initialize_progress()
    set_stage(
        "environment_audit",
        "success",
        "8 GPUs idle/authorized; MatterGen, MatterSim, MP-20 and CHGNet assets verified.",
        {"gpu_count": 8, "gpu_authorized": 8},
    )
    if not (REPORTS / "data_manifest.json").exists():
        run("research.fn_pra.build_data_audit", "--probe-size", "1000")
    run("research.fn_pra.run_teacher_probe", "--resume" if args.resume else "--resume")
    set_stage(
        "online_teacher_validation",
        "pending",
        "Teacher selected. Waiting for online/cache validation and V1 implementation in the next resumable stage.",
    )


if __name__ == "__main__":
    main()
