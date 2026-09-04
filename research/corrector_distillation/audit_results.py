"""Audit the committed residual-distillation experiment tables and local assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


METHODS = {
    "C0",
    "A0",
    "Skip",
    "Reuse",
    "Adapter",
    "Adapter+Fallback@25",
    "Adapter+Fallback@50",
    "Adapter+Fallback@75",
    "Adapter+Fallback@90",
    "A0+Adapter+Fallback@75",
}
TEST_SEEDS = set(range(63000, 63032))
SKIP_FAILURE_SEEDS = {63001, 63006, 63007, 63009, 63017, 63019, 63022, 63027}
MATTERGEN_SHA256 = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
MATTERSIM_SHA256 = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
REFERENCE_SHA256 = "c722f72c7d0cd398fc382870f0b731425b54bd4492b089ca36e3119e49f469b5"
ADAPTER_SHA256 = "fae136412448be0b82267dd1a40d0168cf4847130d41cfa4295c8def715108c4"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_finite(rows: list[dict[str, str]], columns: list[str], table: str) -> None:
    for row_number, row in enumerate(rows, start=2):
        for column in columns:
            value = row[column]
            require(value != "", f"{table}:{row_number}: empty {column}")
            require(math.isfinite(float(value)), f"{table}:{row_number}: non-finite {column}")


def audit(project_root: Path) -> dict[str, object]:
    experiment = project_root / "experiments/corrector_residual_distillation_v1"
    benchmark = read_csv(experiment / "benchmark.csv")
    ablation = read_csv(experiment / "ablation.csv")
    quality = read_csv(experiment / "quality_metrics.csv")
    teacher = json.loads((experiment / "teacher_data_manifest.json").read_text())

    require(len(benchmark) == 320, "benchmark.csv must contain 10 methods x 32 attempts")
    by_method = Counter(row["method"] for row in benchmark)
    require(set(by_method) == METHODS, "benchmark.csv method set differs from the protocol")
    require(all(count == 32 for count in by_method.values()), "every method needs 32 attempts")
    for method in METHODS:
        seeds = {int(row["seed"]) for row in benchmark if row["method"] == method}
        require(seeds == TEST_SEEDS, f"{method}: Stage-C seeds are incomplete")

    failures = {
        int(row["seed"])
        for row in benchmark
        if row["method"] == "Skip" and row["success"] != "True"
    }
    require(failures == SKIP_FAILURE_SEEDS, "Skip failure seeds changed")
    require(
        all(row["success"] == "True" for row in benchmark if row["method"] != "Skip"),
        "a non-Skip Stage-C attempt failed",
    )
    successful = [row for row in benchmark if row["success"] == "True"]
    require(
        all(row["checkpoint_sha256"] == MATTERGEN_SHA256 for row in successful),
        "successful benchmark rows do not share the audited MatterGen checkpoint",
    )
    failed = [row for row in benchmark if row["success"] != "True"]
    require(
        all(row["failure_log_path"] and int(row["failure_return_code"]) != 0 for row in failed),
        "failed attempts must retain a log path and nonzero return code",
    )
    require(
        all(row["timing_includes_teacher_recording"] == "False" for row in benchmark),
        "benchmark timing includes teacher-recorder I/O",
    )
    require_finite(
        successful,
        [
            "elapsed_seconds",
            "mattergen_score_calls",
            "saved_score_calls",
            "adapter_coverage",
            "forward_reduction",
            "peak_allocated_bytes",
        ],
        "benchmark.csv",
    )
    adapter_rows = [row for row in benchmark if "Adapter" in row["method"]]
    require(
        all(row["adapter_checkpoint_sha256"] == ADAPTER_SHA256 for row in adapter_rows),
        "adapter benchmark rows do not share the audited adapter checkpoint",
    )

    require(len(ablation) == 10 and {row["method"] for row in ablation} == METHODS, "bad ablation rows")
    require(all(int(row["n"]) == 32 for row in ablation), "ablation must retain all attempts")
    for row in ablation:
        expected = 24 if row["method"] == "Skip" else 32
        require(int(row["successful_n"]) == expected, f"{row['method']}: bad successful_n")
        require(int(row["quality_n"]) == expected, f"{row['method']}: bad quality_n")
    require_finite(
        ablation,
        [
            "generation_success_rate",
            "paired_speedup_mean",
            "mattergen_score_calls_mean",
            "forward_reduction_mean",
            "avg_energy_above_hull_per_atom",
            "avg_rmsd_from_relaxation",
            "frac_novel_unique_stable_structures",
        ],
        "ablation.csv",
    )

    require(len(quality) == 10 and {row["method"] for row in quality} == METHODS, "bad quality rows")
    for row in quality:
        expected = 24 if row["method"] == "Skip" else 32
        require(int(row["n_input"]) == expected, f"{row['method']}: bad quality n_input")
        require(int(row["n"]) == expected, f"{row['method']}: bad relaxed quality n")
        require(float(row["relaxation_success_rate"]) == 1.0, f"{row['method']}: relaxation failed")
    require_finite(
        quality,
        [
            "pre_relaxation_max_force_mean",
            "avg_energy_above_hull_per_atom",
            "avg_rmsd_from_relaxation",
            "frac_novel_structures",
            "frac_novel_unique_stable_structures",
            "frac_stable_structures",
            "frac_unique_structures",
        ],
        "quality_metrics.csv",
    )

    require(teacher["total_records"] == 56000, "teacher record count changed")
    require(teacher["total_bytes"] == 360067968, "teacher byte count changed")
    require(len(teacher["runs"]) == 56, "teacher run count changed")
    require(teacher["cross_split_overlap"] == [], "teacher splits overlap")
    require(teacher["historical_seed_overlap"] == [], "teacher seeds overlap historical seeds")
    require(all(run["completed"] for run in teacher["runs"]), "teacher run is incomplete")
    require(all(run["records"] == 1000 for run in teacher["runs"]), "teacher run is incomplete")
    require(all(run["storage_dtype"] == "bfloat16" for run in teacher["runs"]), "wrong teacher dtype")
    for run in teacher["runs"]:
        manifest = experiment / run["manifest"]
        require(manifest.is_file(), f"missing {manifest}")
        require(sha256(manifest) == run["manifest_sha256"], f"hash mismatch: {manifest}")

    assets = {
        "mattergen": (
            project_root / "checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt",
            MATTERGEN_SHA256,
        ),
        "mattersim": (
            project_root / "checkpoints/official/mattersim/mattersim-v1.0.0-5M.pth",
            MATTERSIM_SHA256,
        ),
        "reference": (project_root / "data-release/alex-mp/reference_MP2020correction.gz", REFERENCE_SHA256),
        "adapter": (experiment / "adapter_final/residual_adapter.pt", ADAPTER_SHA256),
    }
    asset_sizes: dict[str, int] = {}
    for name, (path, expected_hash) in assets.items():
        require(path.is_file(), f"missing local asset: {path}")
        require(sha256(path) == expected_hash, f"hash mismatch: {path}")
        asset_sizes[name] = path.stat().st_size

    return {
        "status": "PASS",
        "benchmark_attempts": len(benchmark),
        "methods": len(METHODS),
        "successful_generations": sum(row["success"] == "True" for row in benchmark),
        "retained_failed_generations": len(failures),
        "quality_rows": len(quality),
        "teacher_records": teacher["total_records"],
        "teacher_runs": len(teacher["runs"]),
        "asset_sizes": asset_sizes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.project_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
