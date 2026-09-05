"""Pre-registered protocol for Periodic Exact Anchor V3 experiments."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from research.corrector_distillation.formal256_protocol import (
    ASSET_PATHS,
    EXPECTED_HASHES,
    HISTORICAL_SEED_RANGES,
    sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/corrector_residual_distillation_v3_anchor"
START_HEAD = "0a1a8e2631ec2be8d1bdc092948c2cd2e905be2f"
BRANCH = "experiment/corrector-residual-distillation-v3-anchor"
SMOKE_SEEDS = tuple(range(67900, 67904))
STAGE_B_SEEDS = tuple(range(68000, 68032))
STAGE_C_SEEDS = tuple(range(69000, 69064))
STAGE_B_SINGLE_H20_SEEDS = tuple(range(68000, 68004))
STAGE_C_SINGLE_H20_SEEDS = tuple(range(69000, 69016))
V2_LABEL = "V2_Frozen_Atomic75_Late30"
STAGE_B_METHODS = (
    "C0",
    V2_LABEL,
    "V3_AnchorK4",
    "V3_AnchorK8",
    "V3_AnchorK16",
)
SMOKE_METHODS = STAGE_B_METHODS[1:]
METHOD_TO_K = {
    "V3_AnchorK4": 4,
    "V3_AnchorK8": 8,
    "V3_AnchorK16": 16,
}
V2_CONFIG = PROJECT_ROOT / "experiments/corrector_residual_distillation_v2/v2_frozen_config.yaml"
FROZEN_V3_CONFIG = EXPERIMENT_ROOT / "v3_frozen_config.yaml"


def load_v2_config() -> dict[str, Any]:
    return json.loads(V2_CONFIG.read_text(encoding="utf-8"))


def load_frozen_v3() -> dict[str, Any]:
    return json.loads(FROZEN_V3_CONFIG.read_text(encoding="utf-8"))


def methods_for(stage: str) -> tuple[str, ...]:
    if stage == "smoke":
        return SMOKE_METHODS
    if stage in ("stage-b", "single-h20-b"):
        return STAGE_B_METHODS
    if stage in ("stage-c", "single-h20-c"):
        frozen = load_frozen_v3()
        return ("C0", V2_LABEL, str(frozen["selected_method"]))
    raise ValueError(stage)


def seeds_for(stage: str) -> tuple[int, ...]:
    return {
        "smoke": SMOKE_SEEDS,
        "stage-b": STAGE_B_SEEDS,
        "single-h20-b": STAGE_B_SINGLE_H20_SEEDS,
        "stage-c": STAGE_C_SEEDS,
        "single-h20-c": STAGE_C_SINGLE_H20_SEEDS,
    }[stage]


def output_root_for(stage: str) -> Path:
    if stage == "smoke":
        return EXPERIMENT_ROOT / "smoke"
    if stage == "stage-b":
        return EXPERIMENT_ROOT / "stage_b"
    if stage == "single-h20-b":
        return EXPERIMENT_ROOT / "stage_b/single_h20"
    if stage == "stage-c":
        return EXPERIMENT_ROOT / "stage_c"
    if stage == "single-h20-c":
        return EXPERIMENT_ROOT / "stage_c/single_h20"
    raise ValueError(stage)


def fixed_shards(seeds: tuple[int, ...]) -> dict[int, tuple[int, ...]]:
    if len(seeds) % 8:
        raise ValueError("fixed 8-GPU shards require a seed count divisible by 8")
    per_gpu = len(seeds) // 8
    return {
        gpu: tuple(seeds[gpu * per_gpu : (gpu + 1) * per_gpu])
        for gpu in range(8)
    }


def benchmark_command(*, output_root: Path, label: str, seed: int) -> list[str]:
    allowed = set(SMOKE_SEEDS) | set(STAGE_B_SEEDS) | set(STAGE_C_SEEDS)
    if seed not in allowed:
        raise ValueError(f"seed {seed} is outside the V3 pre-registered intervals")
    base = [
        sys.executable,
        "-m",
        "research.corrector_distillation.benchmark_sampler",
        "--checkpoint-root",
        str(ASSET_PATHS["mattergen_checkpoint"].parents[1].resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--seed",
        str(seed),
    ]
    if label == "C0":
        return [*base, "--method", "C0", "--label", label]
    if label != V2_LABEL and label not in METHOD_TO_K:
        raise ValueError(f"unregistered V3 method {label!r}")
    config = load_v2_config()
    decision = config["decision_rule"]
    command = [
        *base,
        "--method",
        "Adapter+Fallback",
        "--label",
        label,
        "--adapter-checkpoint",
        str(Path(config["adapter"]["checkpoint"]).resolve()),
        "--coverage",
        str(decision["coverage_target"]),
        "--risk-mode",
        str(decision["risk_mode"]),
        "--risk-fields",
        ",".join(decision["risk_fields"]),
        "--late-exact-start",
        str(decision["late_exact_start"]),
    ]
    if label in METHOD_TO_K:
        command.extend(("--periodic-exact-anchor-k", str(METHOD_TO_K[label])))
    return command


def validate_protocol() -> dict[str, Any]:
    failures: list[str] = []
    actual_hashes = {name: sha256(path) for name, path in ASSET_PATHS.items()}
    for name in (
        "v2_frozen_config",
        "adapter_checkpoint",
        "risk_calibration_file",
        "mattergen_checkpoint",
        "mattersim_checkpoint",
        "alex_mp_reference",
    ):
        if actual_hashes[name] != EXPECTED_HASHES[name]:
            failures.append(f"{name} hash changed")
    intervals = {
        **HISTORICAL_SEED_RANGES,
        "formal256_frozen": ((67000, 67255),),
    }
    new_sets = {
        "smoke": set(SMOKE_SEEDS),
        "stage_b": set(STAGE_B_SEEDS),
        "stage_c": set(STAGE_C_SEEDS),
    }
    overlaps: dict[str, list[int]] = {}
    for new_name, new_seeds in new_sets.items():
        for old_name, ranges in intervals.items():
            overlap = sorted(
                new_seeds.intersection(
                    seed for start, end in ranges for seed in range(start, end + 1)
                )
            )
            overlaps[f"{new_name}_vs_{old_name}"] = overlap
            if overlap:
                failures.append(f"seed overlap: {new_name} vs {old_name}")
    for left, right in (("smoke", "stage_b"), ("smoke", "stage_c"), ("stage_b", "stage_c")):
        overlap = sorted(new_sets[left] & new_sets[right])
        overlaps[f"{left}_vs_{right}"] = overlap
        if overlap:
            failures.append(f"new seed overlap: {left} vs {right}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True
    ).strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", START_HEAD, head],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    if branch != BRANCH:
        failures.append(f"branch {branch!r} != {BRANCH!r}")
    if not ancestor:
        failures.append(f"required start HEAD {START_HEAD} is not an ancestor")
    return {
        "passed": not failures,
        "failures": failures,
        "head": head,
        "branch": branch,
        "start_head_is_ancestor": ancestor,
        "asset_hashes": actual_hashes,
        "seed_overlaps": overlaps,
        "seed_ranges": {
            "smoke": [SMOKE_SEEDS[0], SMOKE_SEEDS[-1]],
            "stage_b": [STAGE_B_SEEDS[0], STAGE_B_SEEDS[-1]],
            "stage_c": [STAGE_C_SEEDS[0], STAGE_C_SEEDS[-1]],
        },
    }
