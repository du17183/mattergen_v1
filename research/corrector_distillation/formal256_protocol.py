"""Immutable protocol helpers for the confirmatory V2 formal-256 run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/corrector_residual_distillation_formal256"
V2_ROOT = PROJECT_ROOT / "experiments/corrector_residual_distillation_v2"
V2_FROZEN_HEAD = "1b8b62284f694b08d778895da345c708204ec2db"
FORMAL_BRANCH = "experiment/corrector-residual-distillation-formal256"
FORMAL_SEED_START = 67000
FORMAL_SEED_END = 67255
FORMAL_SEEDS = tuple(range(FORMAL_SEED_START, FORMAL_SEED_END + 1))
SINGLE_H20_SEEDS = tuple(range(67000, 67016))
METHODS = ("C0", "V2_Frozen_Atomic75_Late30")

FROZEN_CONFIG_PATH = V2_ROOT / "v2_frozen_config.yaml"
EXPECTED_HASHES = {
    "v2_frozen_config": "be8fdfa9a1f754c4a3bded1a4a583c042cfce3c72a7152f55b2e2f799a546464",
    "adapter_checkpoint": "a2fd0f17c27b760958b60c1d1386577883635a34fb2101eb08e38fd8472702c7",
    "risk_calibration_file": "dff93d12d3637e260a3cfe0b9f418b39e809a39e13deb6d185ea0b4c31d3fd9f",
    "risk_model_canonical": "43d47bdb9a23f4f3eba40ca670ef1bbd4d9204a194e0b668a1a57ceb6728fbfc",
    "mattergen_checkpoint": "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e",
    "mattersim_checkpoint": "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5",
    "alex_mp_reference": "c722f72c7d0cd398fc382870f0b731425b54bd4492b089ca36e3119e49f469b5",
}
ASSET_PATHS = {
    "v2_frozen_config": FROZEN_CONFIG_PATH,
    "adapter_checkpoint": V2_ROOT / "frozen/residual_adapter.pt",
    "risk_calibration_file": V2_ROOT / "frozen/risk_calibration.json",
    "mattergen_checkpoint": PROJECT_ROOT / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt",
    "mattersim_checkpoint": PROJECT_ROOT / "checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth",
    "alex_mp_reference": PROJECT_ROOT / "data-release/alex-mp/reference_MP2020correction.gz",
}

HISTORICAL_SEED_RANGES = {
    "historical_misc_formal": ((14000, 14031), (32000, 32063), (33000, 33127), (41000, 41063), (50000, 50063)),
    "adaptive_cfg_formal256": ((20000, 20255),),
    "e3_pcr_training": ((20000, 20063),),
    "e3_pcr_formal256": ((40000, 40255),),
    "v1_baseline_recovery": ((60000, 60007),),
    "v1_train": ((61000, 61015),),
    "v1_validation": ((62000, 62007),),
    "v1_stage_c": ((63000, 63031),),
    "v2_dagger_train": ((64000, 64015),),
    "v2_validation_calibration": ((65000, 65015),),
    "v2_stage_c": ((66000, 66063),),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_config() -> dict[str, Any]:
    # The .yaml file is deliberately canonical JSON. Read this exact frozen file;
    # no duplicated formal-run approximation is permitted.
    with FROZEN_CONFIG_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def fixed_shards() -> dict[int, tuple[int, ...]]:
    return {
        gpu: tuple(range(FORMAL_SEED_START + 32 * gpu, FORMAL_SEED_START + 32 * (gpu + 1)))
        for gpu in range(8)
    }


def seed_audit_payload() -> dict[str, Any]:
    formal = set(FORMAL_SEEDS)
    intersections = {
        name: sorted(
            formal.intersection(
                seed for start, end in ranges for seed in range(start, end + 1)
            )
        )
        for name, ranges in HISTORICAL_SEED_RANGES.items()
    }
    shards = fixed_shards()
    assigned = [seed for gpu in range(8) for seed in shards[gpu]]
    return {
        "schema_version": 1,
        "formal_seed_range": [FORMAL_SEED_START, FORMAL_SEED_END],
        "formal_seed_count": len(FORMAL_SEEDS),
        "formal_seed_sha256": hashlib.sha256(
            ",".join(map(str, FORMAL_SEEDS)).encode("ascii")
        ).hexdigest(),
        "historical_seed_ranges": {
            name: [list(pair) for pair in ranges]
            for name, ranges in HISTORICAL_SEED_RANGES.items()
        },
        "intersections": intersections,
        "fixed_gpu_shards": {
            str(gpu): [seeds[0], seeds[-1]] for gpu, seeds in shards.items()
        },
        "shard_assignment_exact": assigned == list(FORMAL_SEEDS),
        "all_intersections_empty": all(not values for values in intersections.values()),
        "passed": assigned == list(FORMAL_SEEDS)
        and all(not values for values in intersections.values()),
        "policy": "If any overlap exists, replace the entire contiguous 256-seed interval; never replace a subset.",
        "sources": {
            "v1": "experiments/corrector_residual_distillation_v1/config/experiment.json",
            "v2": "experiments/corrector_residual_distillation_v2/v2_frozen_config.yaml",
            "adaptive_cfg": "origin/feature/a0-e3g-formal256:reports/a0_e3g_formal256/frozen_manifest.json",
            "e3_pcr": "origin/feature/q3-e3-pcr-formal256:reports/q3_e3_pcr/formal256/formal_seed_audit.json",
        },
    }


def validate_frozen_protocol(*, verify_git: bool = True) -> dict[str, Any]:
    config = load_frozen_config()
    actual_hashes = {name: sha256(path) for name, path in ASSET_PATHS.items()}
    failures: list[str] = []
    for name, expected in EXPECTED_HASHES.items():
        if name == "risk_model_canonical":
            actual = config["adapter"]["risk_model_canonical_sha256"]
        else:
            actual = actual_hashes[name]
        if actual != expected:
            failures.append(f"{name} SHA256 {actual} != {expected}")
    required_values = {
        "selected_method": config.get("selected_method") == "B4a_V2_Atomic75_Late30",
        "adapter_parameters": config["adapter"].get("parameter_count") == 2661,
        "target": config["sampling"].get("target") == {"dft_mag_density": 0.1},
        "cfg": config["sampling"].get("guidance_scale") == 2.0,
        "steps": config["sampling"].get("diffusion_steps") == 1000,
        "corrector": config["sampling"].get("corrector_steps_per_timestep") == 1,
        "batch": config["sampling"].get("batch_size") == 1,
        "deterministic": config["sampling"].get("deterministic") is True,
        "risk_mode": config["decision_rule"].get("risk_mode") == "field",
        "risk_fields": config["decision_rule"].get("risk_fields") == ["atomic_numbers"],
        "coverage": config["decision_rule"].get("coverage_target") == 0.75,
        "atomic_threshold": config["decision_rule"].get("threshold_atomic_numbers") == 0.8289545722625317,
        "late_exact": config["decision_rule"].get("late_exact_start") == 0.7,
        "early_reuse_disabled": config["decision_rule"].get("early_reuse_end") is None,
        "a0_prohibited": config.get("a0_combination_prohibited") is True,
    }
    failures.extend(name for name, passed in required_values.items() if not passed)
    git_state: dict[str, Any] = {}
    if verify_git:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True
        ).strip()
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", V2_FROZEN_HEAD, head],
            cwd=PROJECT_ROOT,
            check=False,
        ).returncode == 0
        git_state = {"head": head, "branch": branch, "v2_frozen_head_is_ancestor": ancestor}
        if branch != FORMAL_BRANCH:
            failures.append(f"branch {branch!r} != {FORMAL_BRANCH!r}")
        if not ancestor:
            failures.append(f"V2 frozen HEAD {V2_FROZEN_HEAD} is not an ancestor of {head}")
    audit = seed_audit_payload()
    if not audit["passed"]:
        failures.append("formal seed audit failed")
    return {
        "passed": not failures,
        "failures": failures,
        "actual_hashes": actual_hashes,
        "required_values": required_values,
        "seed_audit": audit,
        "git": git_state,
        "config": config,
    }


def benchmark_command(
    *, checkpoint_root: Path, output_root: Path, label: str, seed: int
) -> list[str]:
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"seed {seed} is outside immutable formal interval")
    config = load_frozen_config()
    command = [
        str(PROJECT_ROOT / ".venv/bin/python"),
        "-m",
        "research.corrector_distillation.benchmark_sampler",
        "--checkpoint-root",
        str(checkpoint_root.resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--seed",
        str(seed),
    ]
    if label == "C0":
        return [*command, "--method", "C0", "--label", "C0"]
    if label != "V2_Frozen_Atomic75_Late30":
        raise ValueError(f"formal protocol prohibits method {label!r}")
    decision = config["decision_rule"]
    return [
        *command,
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
