"""Run frozen C0/FT0/TCL Formal256 generation in paired seed blocks."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/tcl_formal256"
P0_ROOT = PROJECT_ROOT / "experiments/tcl_dml_p0"
P1_SCRIPT = PROJECT_ROOT / "experiments/tcl_p1/run_generation.py"
SEEDS = tuple(range(83000, 83256))
EXPECTED_BASE_HASH = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"


def load_frozen_generation():
    spec = importlib.util.spec_from_file_location("frozen_p1_generation", P1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.SEEDS = SEEDS
    return module


def preflight(module) -> dict[str, str]:
    if SEEDS != tuple(range(83000, 83256)) or len(SEEDS) != 256:
        raise RuntimeError("Formal256 frozen seed range changed")
    p0_rows = {
        row["method"]: row
        for row in csv.DictReader((P0_ROOT / "training_summary.csv").open())
    }
    base_path = module.MODEL_ROOT / "checkpoints/last.ckpt"
    base_hash = module.sha256(base_path)
    if base_hash != EXPECTED_BASE_HASH:
        raise RuntimeError(f"C0 base checkpoint changed: {base_hash}")
    actual = {"C0_base": base_hash}
    for method, path in module.CHECKPOINTS.items():
        value = module.sha256(path)
        if value != module.EXPECTED_HASHES[method]:
            raise RuntimeError(f"{method} checkpoint changed: {value}")
        if value != p0_rows[method]["checkpoint_sha256"]:
            raise RuntimeError(f"{method} does not match P0 training summary")
        actual[method] = value
    return actual


def main() -> None:
    frozen = load_frozen_generation()
    print(json.dumps({
        "methods": frozen.METHODS,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "seed_count": len(SEEDS),
        "formal_training_steps": 0,
        "checkpoint_hashes": preflight(frozen),
        "sampling": {"target": 0.1, "cfg": 2.0, "steps": 1000, "corrector": 1},
        "schedule": "paired blocks of eight; C0 then FT0 then TCL",
        "dft_verified": False,
    }), flush=True)
    for start in range(0, len(SEEDS), 8):
        block = SEEDS[start:start + 8]
        block_index = start // 8 + 1
        print(json.dumps({
            "event": "paired_block_started", "block": block_index,
            "seed_start": block[0], "seed_stop": block[-1],
        }), flush=True)
        for method in frozen.METHODS:
            frozen.run_chunk(method, block, block_index)
        print(json.dumps({
            "event": "paired_block_complete", "block": block_index,
            "seed_start": block[0], "seed_stop": block[-1],
        }), flush=True)
    frozen.write_results()


if __name__ == "__main__":
    main()
