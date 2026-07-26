from __future__ import annotations

import subprocess
import sys
import traceback

from research.rp_qtfg.analyze_offline import run as analyze
from research.rp_qtfg.common import set_stage
from research.rp_qtfg.offline_probe import run as probe


MATTERSIM_PYTHON = "/data/dxl/envs/mattergen_py310/bin/python"


def main() -> int:
    try:
        probe()
        subprocess.run(
            [
                MATTERSIM_PYTHON,
                "-m",
                "research.rp_qtfg.offline_relax",
                "launch",
                "--workers",
                "16",
            ],
            cwd="/data/dxl/mattergen_v1",
            check=True,
        )
        result = analyze()
        return 0 if result["PHYSICS_DIRECTION_GO"] else 2
    except KeyboardInterrupt as exc:
        set_stage(
            "offline_direction_probe",
            "blocked",
            str(exc),
        )
        return 130
    except BaseException:
        detail = traceback.format_exc()
        set_stage(
            "offline_direction_probe",
            "failed",
            detail,
        )
        print(detail, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
