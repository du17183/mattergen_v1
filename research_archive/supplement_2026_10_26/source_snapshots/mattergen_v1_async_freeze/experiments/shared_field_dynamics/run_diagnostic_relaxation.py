"""Run the frozen MatterSim-5M relaxation for the diagnostic structures."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path("/mnt/datasets-livsyn/dxl/mattergen_v1")
POTENTIAL = ARTIFACT_ROOT / "checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth"


def main() -> None:
    (PROJECT_ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    output = ROOT / "diagnostic_relaxation"
    if output.exists():
        raise FileExistsError(output)
    log_path = ROOT / "diagnostic_relaxation.log"
    command = [
        sys.executable, "-m", "research.corrector_distillation.relax_many_individual",
        "--structures-path", str(ROOT / "generated_diagnostic.extxyz"),
        "--potential-path", str(POTENTIAL),
        "--output-dir", str(output),
        "--device", "cuda", "--fmax", "0.05", "--max-natoms-per-batch", "512",
        "--cuda-3x3-det-workaround",
    ]
    environment = dict(os.environ)
    environment.update(
        TMPDIR=str(PROJECT_ROOT / ".tmp"),
        XDG_CACHE_HOME=str(PROJECT_ROOT / ".cache"),
        HF_HOME=str(PROJECT_ROOT / ".cache/huggingface"),
        TORCH_HOME=str(PROJECT_ROOT / ".cache/torch"),
        PYTHONUNBUFFERED="1",
    )
    with log_path.open("x", encoding="utf-8") as stream:
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    print(f"MatterSim relaxation complete: {output}")


if __name__ == "__main__":
    main()
