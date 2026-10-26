"""Run only historical A1 seed 740000 via the frozen entry, isolated from A1 data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

from experiments.innovation2_finalization import generate_frozen_pair as entry


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "alm_replay_740000"
ALM_PYTHON = Path("/mnt/datasets-livsyn/dxl/alm/.venv/bin/python")


def main() -> None:
    if Path(sys.executable).resolve() != ALM_PYTHON.resolve():
        raise RuntimeError(f"Expected ALM interpreter, got {sys.executable}")
    if OUTPUT.exists():
        raise FileExistsError(f"Isolated replay output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    provenance = {
        "purpose": "historical-seed reproducibility gate only; no Timing",
        "interpreter": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "seed": 740000,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "historical_entry": str(Path(entry.__file__).resolve()),
    }
    (OUTPUT / "replay_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    original_output = entry.output
    entry.output = lambda code: OUTPUT if code == "A1" else original_output(code)
    sys.argv = [str(Path(entry.__file__).resolve()), "--seed", "740000"]
    entry.main()


if __name__ == "__main__":
    main()
