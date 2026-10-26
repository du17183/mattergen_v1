"""Run the frozen MatterSim-5M relaxation on the 32 A0 final structures."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
POTENTIAL = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/"
    "mattersim/mattersim-v1.0.0-5M.pth"
)


def main() -> None:
    (PROJECT_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    output = ROOT / "diagnostic_final_relaxation"
    if output.exists():
        raise FileExistsError(output)
    command = [
        sys.executable, "-m", "research.corrector_distillation.relax_many_individual",
        "--structures-path", str(ROOT / "diagnostic_final_generated.extxyz"),
        "--potential-path", str(POTENTIAL), "--output-dir", str(output),
        "--device", "cuda", "--fmax", "0.05", "--max-natoms-per-batch", "512",
        "--cuda-3x3-det-workaround",
    ]
    environment = dict(os.environ)
    environment.update(
        TMPDIR=str(PROJECT_ROOT / ".tmp"),
        XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
        HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
        TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        PYTHONUNBUFFERED="1",
    )
    log = ROOT / "diagnostic_final_relaxation.log"
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
