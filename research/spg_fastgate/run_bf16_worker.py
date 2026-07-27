"""One-seed FP32 or field-safe-BF16 endpoint generation worker."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext

from research.spg_fastgate.bf16_probe import field_safe_linear_bf16
from research.spg_fastgate.common import RESULTS
from research.spg_fastgate.generation import (
    build_generator,
    configure_determinism,
    run_group_guarded,
)


SEED_START = 24128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("C0", "A0"), required=True)
    parser.add_argument("--precision", choices=("FP32", "FIELD_SAFE_BF16"), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=range(8), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = SEED_START + args.physical_gpu
    output = RESULTS / "bf16_generation" / args.precision
    status = output / args.method / "B1" / f"seed_{seed}" / "status.json"
    if status.is_file() and json.loads(status.read_text(encoding="utf-8")).get("success"):
        return 0
    configure_determinism()
    generator = build_generator(args.method, batch_size=1, sampling_steps=1000)
    context = (
        field_safe_linear_bf16(generator.model.diffusion_module.model)
        if args.precision == "FIELD_SAFE_BF16"
        else nullcontext()
    )
    with context:
        run_group_guarded(
            generator=generator,
            method=args.method,
            seeds=[seed],
            output_root=output,
            physical_gpu=args.physical_gpu,
            save_outputs=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
