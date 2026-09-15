"""Use the frozen one-shot generator for Formal256 only after P0 GO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_fresh_p0_generation as implementation


ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / "formal256"


def assert_authorized():
    p0 = json.loads((ROOT / "p0/decision_summary.json").read_text())
    if p0["RISK_CALIBRATED_CFG_P0"] != "GO" or p0["next"] != "RUN_FRESH_FORMAL256":
        raise RuntimeError("Formal256 prohibited because fresh P0 is not GO")
    offline = json.loads((ROOT / "offline/decision_summary.json").read_text())
    reservation = json.loads((ROOT / "future_seed_reservations.json").read_text())
    seeds = tuple(map(int, reservation["formal256_seeds"]))
    if len(seeds) != 256 or len(set(seeds)) != 256 or reservation["historical_overlap"] != 0:
        raise RuntimeError("Formal256 seed reservation invalid")
    seed_path = FORMAL / "seeds.json"
    if not seed_path.exists():
        seed_path.write_text(json.dumps({"seeds": seeds, "count": 256, "source": "future_seed_reservations.json"}, indent=2) + "\n")
    elif tuple(json.loads(seed_path.read_text())["seeds"]) != seeds:
        raise RuntimeError("Formal256 seeds changed")
    return seeds, offline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--seeds", required=True)
    args = parser.parse_args()
    implementation.P0 = FORMAL
    implementation.assert_authorized = assert_authorized
    implementation.run_worker(tuple(int(value) for value in args.seeds.split(",") if value))
