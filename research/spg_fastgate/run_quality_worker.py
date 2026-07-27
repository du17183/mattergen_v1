"""One-GPU resume-safe native batch generation worker."""

from __future__ import annotations

import argparse
import json
from research.spg_fastgate.common import RESULTS
from research.spg_fastgate.generation import (
    build_generator,
    configure_determinism,
    run_group_guarded,
)


SEED_START = 24064
SEED_STOP = 24128
OUTPUT = RESULTS / "quality_generation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("C0", "A0"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 4), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=range(8), required=True)
    return parser.parse_args()


def status_success(method: str, batch_size: int, seed: int) -> bool:
    path = OUTPUT / method / f"B{batch_size}" / f"seed_{seed}" / "status.json"
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("success") is True


def lane_seeds(gpu: int) -> list[int]:
    return list(range(SEED_START + gpu, SEED_STOP, 8))


def groups_for_gpu(gpu: int, batch_size: int) -> list[tuple[int, ...]]:
    seeds = lane_seeds(gpu)
    if batch_size == 1:
        return [(seed,) for seed in seeds]
    return [tuple(seeds[index : index + batch_size]) for index in range(0, len(seeds), batch_size)]


def main() -> int:
    args = parse_args()
    configure_determinism()
    groups = groups_for_gpu(args.physical_gpu, args.batch_size)
    pending = [
        group
        for group in groups
        if not all(status_success(args.method, args.batch_size, seed) for seed in group)
    ]
    if not pending:
        return 0
    generator = build_generator(
        args.method,
        batch_size=args.batch_size,
        sampling_steps=1000,
    )
    for group in pending:
        if any(status_success(args.method, args.batch_size, seed) for seed in group):
            raise RuntimeError(
                f"partial successful native batch cannot be rerun safely: {args.method} {group}"
            )
        run_group_guarded(
            generator=generator,
            method=args.method,
            seeds=group,
            output_root=OUTPUT,
            physical_gpu=args.physical_gpu,
            save_outputs=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
