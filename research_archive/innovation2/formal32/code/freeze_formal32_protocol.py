"""Freeze Formal32 directly from the audited P0 configuration and code."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

import yaml


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
P0 = PROJECT_ROOT / "experiments/mattersim_late_force_guidance_p0"
SEEDS = tuple(range(730000, 730032))
TEXT_EXTENSIONS = {
    ".csv", ".json", ".jsonl", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"
}
SCOPES = ("experiments", "diagnostics", "research", "thesis")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def worktrees() -> list[Path]:
    output = subprocess.check_output(
        ["git", "worktree", "list", "--porcelain"], cwd=PROJECT_ROOT, text=True
    )
    return [Path(line.split(" ", 1)[1]) for line in output.splitlines() if line.startswith("worktree ")]


def scan_seed_references() -> tuple[dict[int, list[str]], list[str]]:
    patterns = {
        seed: re.compile(rb"(?<![0-9.])" + str(seed).encode() + rb"(?![0-9.])")
        for seed in SEEDS
    }
    found: dict[int, list[str]] = {seed: [] for seed in SEEDS}
    searched: list[str] = []
    for worktree in worktrees():
        for scope_name in SCOPES:
            scope = worktree / scope_name
            if not scope.exists():
                continue
            searched.append(str(scope))
            for path in scope.rglob("*"):
                try:
                    if path == ROOT or ROOT in path.parents:
                        continue
                    if path.is_dir():
                        for seed in SEEDS:
                            if path.name == str(seed):
                                found[seed].append(f"path:{path}")
                        continue
                    if path.suffix.lower() not in TEXT_EXTENSIONS or path.stat().st_size > 64 * 1024 * 1024:
                        continue
                    content = path.read_bytes()
                    for seed, pattern in patterns.items():
                        if pattern.search(content):
                            found[seed].append(f"text:{path}")
                except (OSError, UnicodeError):
                    continue
    return found, searched


def read_trace_times() -> list[float]:
    with (P0 / "guidance_trace.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 320:
        raise RuntimeError(f"expected 320 P0 guidance rows, got {len(rows)}")
    return sorted({float(row["t_norm"]) for row in rows}, reverse=True)


def main() -> None:
    if ROOT.exists() and any(path.name in {"formal32_config.yaml", "formal32_seeds.json"} for path in ROOT.iterdir()):
        raise FileExistsError("Formal32 protocol is already frozen")
    required = {
        "guidance_config": P0 / "guidance_config.yaml",
        "decision_summary": P0 / "decision_summary.json",
        "final_report": P0 / "final_report.md",
        "guidance_implementation": P0 / "late_force_sampler.py",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    p0_config = yaml.safe_load(required["guidance_config"].read_text(encoding="utf-8"))
    p0_decision = json.loads(required["decision_summary"].read_text(encoding="utf-8"))
    if p0_decision.get("MATTERSIM_FORCE_GUIDANCE_P0") != "GO":
        raise RuntimeError("P0 is not GO")
    if p0_config["F0"].get("guidance_lambda_sweep") is not False:
        raise RuntimeError("P0 guidance is not a single frozen setting")
    references, searched = scan_seed_references()
    overlap = {seed: hits for seed, hits in references.items() if hits}
    if overlap:
        raise RuntimeError(f"candidate Formal32 seeds overlap history: {overlap}")
    trigger_times = read_trace_times()
    if len(trigger_times) != 20 or max(trigger_times) > float(p0_config["F0"]["guidance_t_max"]) + 1e-8:
        raise RuntimeError(f"unexpected P0 trigger grid: {trigger_times}")

    source_hashes = {name: sha256(path) for name, path in required.items()}
    formal = {
        "experiment": "mattersim_late_force_guidance_formal32",
        "frozen": True,
        "source": {
            "p0_directory": str(P0),
            "files": {name: str(path) for name, path in required.items()},
            "sha256": source_hashes,
            "p0_decision": "GO",
        },
        "common": p0_config["common"],
        "F0": p0_config["F0"],
        "actual_trigger_model_t": trigger_times,
        "expected_guidance_events_per_sample": len(trigger_times),
        "paired_seeds": list(SEEDS),
        "primary": "MatterSim-5M pre-relaxation mean MaxF",
        "statistics": {
            "bootstrap_resamples": 20_000,
            "bootstrap_seed": 20260914,
            "formal_relative_maxF_reduction": True,
            "also_report_absolute_maxF_change": True,
        },
        "decision": {
            "strong_confirmed": {
                "mean_relative_maxF_reduction_min": 0.15,
                "bootstrap_relative_reduction_ci95_low_min": 0.10,
                "paired_wins_min": 24,
            },
            "confirmed": {
                "mean_relative_maxF_reduction_min": 0.15,
                "bootstrap_relative_reduction_ci95_low_strictly_positive": True,
                "paired_wins_min": 20,
            },
            "borderline": {
                "mean_relative_maxF_reduction_interval": [0.08, 0.15],
                "paired_wins_interval": [17, 19],
            },
            "fail": {
                "mean_relative_maxF_reduction_below": 0.08,
                "paired_wins_max": 16,
            },
        },
        "guardrails": {
            "mag_mae_worsening_max_fraction": 0.10,
            "nus_drop_max_percentage_points": 5.0,
            "validity_drop_max_percentage_points": 5.0,
            "e_hull_mean_increase_max_ev_per_atom": 0.01,
            "runtime_ratio_max": 1.5,
        },
        "prohibited": [
            "parameter_changes", "timestep_sweep", "lambda_sweep", "force_cap_sweep",
            "stress_guidance", "cell_guidance", "energy_guidance", "atomic_guidance",
            "adaptive_cfg", "A_plus_E3", "formal256", "weight_updates", "training",
        ],
    }
    seed_payload = {
        "schema_version": 1,
        "paired_seeds": [
            {"seed": seed, "historical_search_status": "NO_REFERENCE_FOUND_BEFORE_REGISTRATION"}
            for seed in SEEDS
        ],
        "continuous_range": [SEEDS[0], SEEDS[-1]],
        "searched_scopes": searched,
        "excluded_output_directory": str(ROOT),
        "historical_overlap_count": 0,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "formal32_config.yaml").write_text(
        yaml.safe_dump(formal, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (ROOT / "formal32_seeds.json").write_text(
        json.dumps(seed_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "formal32_frozen", "seeds": [SEEDS[0], SEEDS[-1]],
                      "source_hashes": source_hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
