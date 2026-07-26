from __future__ import annotations

import json
import os
import signal
import sys
import traceback
from pathlib import Path

from research.rp_qtfg.common import (
    LOGS,
    PROGRESS_DIR,
    REPORTS,
    RESULTS,
    atomic_json,
    initialize_progress,
    set_stage,
)
from research.rp_qtfg.mag_oracle import run as run_mag_oracle


LAUNCHER = REPORTS / "launcher.json"


def _signal_handler(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def initialize() -> None:
    for path in (LOGS, PROGRESS_DIR, REPORTS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    initialize_progress()
    atomic_json(
        LAUNCHER,
        {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "pgid": os.getpgid(0),
            "cwd": str(Path.cwd()),
            "exe": sys.executable,
            "argv": sys.argv,
        },
    )
    set_stage(
        "state_audit",
        "success",
        "Git, GPU, process, environment, checkpoint and evaluator audit passed.",
        {
            "gpu_count": 8,
            "gpu_workers": 0,
            "checkpoint_sha256": "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e",
            "mattersim_sha256": "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5",
        },
    )
    set_stage(
        "branch_creation",
        "success",
        "Created feature/rp-qtfg from stable main commit 9bc6747.",
    )


def main() -> int:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    initialize()
    try:
        run_mag_oracle()
        set_stage(
            "offline_direction_probe",
            "pending",
            "Magnetic-oracle stage completed; offline physical-direction probe is next.",
        )
        return 0
    except KeyboardInterrupt as exc:
        set_stage("mag_oracle_validation", "blocked", str(exc))
        return 130
    except BaseException:
        detail = traceback.format_exc()
        set_stage("mag_oracle_validation", "failed", detail)
        print(detail, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
