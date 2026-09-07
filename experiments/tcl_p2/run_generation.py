"""Run the frozen P1 generator orchestration on 64 new paired P2 seeds."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/tcl_p2"
P0_ROOT = PROJECT_ROOT / "experiments/tcl_dml_p0"
P1_SCRIPT = PROJECT_ROOT / "experiments/tcl_p1/run_generation.py"
SEEDS = tuple(range(82000, 82064))


def load_p1():
    spec = importlib.util.spec_from_file_location("frozen_p1_generation", P1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.SEEDS = SEEDS
    return module


def preflight(module) -> dict[str, str]:
    if SEEDS != tuple(range(82000, 82064)) or len(SEEDS) != 64:
        raise RuntimeError("P2 frozen seed range changed")
    p0_rows = {
        row["method"]: row
        for row in csv.DictReader((P0_ROOT / "training_summary.csv").open())
    }
    actual = {}
    for method, path in module.CHECKPOINTS.items():
        value = module.sha256(path)
        if value != module.EXPECTED_HASHES[method]:
            raise RuntimeError(f"{method} checkpoint changed: {value}")
        if value != p0_rows[method]["checkpoint_sha256"]:
            raise RuntimeError(f"{method} does not match P0 training summary")
        actual[method] = value
    return actual


def main() -> None:
    frozen = load_p1()
    print(json.dumps({
        "methods": frozen.METHODS,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "training_steps": 0,
        "checkpoint_hashes": preflight(frozen),
        "sampling": {"target": 0.1, "cfg": 2.0, "steps": 1000, "corrector": 1},
    }), flush=True)
    for method in frozen.METHODS:
        for start in range(0, len(SEEDS), 8):
            frozen.run_chunk(method, SEEDS[start:start + 8], start // 8 + 1)
    frozen.write_results()


if __name__ == "__main__":
    main()
