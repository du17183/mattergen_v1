"""Paired quality, force-tail, speed and streak statistics for V3."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from ase.io import read
from scipy.stats import binomtest, spearmanr, wilcoxon

from research.corrector_distillation.v3_anchor_protocol import (
    EXPERIMENT_ROOT, V2_LABEL, methods_for, output_root_for, seeds_for,
)


N_BOOT = 20_000
BOOT_SEED = 20260905
METRICS = {
    "e_hull": "energy_above_hull_per_atom", "stable": "stable",
    "nus": "novel_unique_stable", "novel": "novel", "unique": "unique",
    "rmsd": "rmsd_from_relaxation",
}
BINARY = {"stable", "nus", "novel", "unique"}


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replace_stage(path: Path, stage: str, rows: list[dict]) -> None:
    previous = csv_rows(path) if path.exists() else []
    write_csv(path, [r for r in previous if r["stage"] != stage] + rows)


def boot_ci(left: np.ndarray, right: np.ndarray, statistic) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    estimates = []
    for offset in range(0, N_BOOT, 1000):
        indices = rng.integers(0, len(left), (min(1000, N_BOOT-offset), len(left)))
        estimates.append(statistic(left[indices], right[indices]))
    return tuple(float(x) for x in np.quantile(np.concatenate(estimates), [.025, .975]))


def load_quality(root: Path, method: str, seeds: tuple[int, ...]) -> list[dict]:
    qroot = root / "quality" / method
    rroot = root / "relaxation/precomputed" / method
    per_structure = csv_rows(qroot / "per_structure.csv")
    with gzip.open(qroot / "official_detailed.json.gz", "rt") as stream:
        detailed = json.load(stream)
    atoms = read(rroot / "initial_with_properties.extxyz", index=":")
    relax = json.loads((rroot / "relaxation_summary.json").read_text())
    assert [int(r["sample_seed"]) for r in per_structure] == list(seeds)
    assert relax["sample_seeds"] == list(seeds)
    assert len(atoms) == len(seeds)
    assert all(len(detailed[name]) == len(seeds) for name in METRICS.values())
    rows = []
    for i, (seed, structure, source) in enumerate(zip(seeds, atoms, per_structure)):
        forces = np.linalg.norm(structure.get_forces(), axis=1)
        assert np.isfinite(forces).all()
        assert np.isclose(forces.max(), float(source["pre_relaxation_max_force"]))
        row = {"method": method, "seed": seed, "formula": source["formula"],
               "num_atoms": int(source["num_atoms"]),
               "relaxation_steps": int(source["relaxation_steps"]),
               **{k: float(detailed[v][i]) for k, v in METRICS.items()},
               "force_mean": float(forces.mean()), "force_max": float(forces.max())}
        assert all(np.isfinite(row[k]) for k in (*METRICS, "force_mean", "force_max"))
        rows.append(row)
    return rows


def collect_speed(stage: str) -> tuple[list[dict], dict[str, dict]]:
    speed_stage = "single-h20-b" if stage == "stage-b" else "single-h20-c"
    root = output_root_for(speed_stage)
    seeds = seeds_for(speed_stage)
    methods = methods_for(stage)
    runs = {}
    rows = []
    summary = {}
    for seed in seeds:
        for method in methods:
            directory = root / "generation" / method / str(seed)
            run = json.loads((directory / "run_summary.json").read_text())
            assert run["success"] and run["seed"] == seed and run["method"] == method
            assert run["peak_allocated_bytes"] > 0
            runs[method, seed] = run
    for method in methods:
        current = []
        for seed in seeds:
            run = runs[method, seed]
            c0 = runs["C0", seed]
            row = {"stage": stage, "method": method, "seed": seed,
                   "gpu": "physical GPU0 NVIDIA H20", "batch_size": 1,
                   "time_per_sample": run["elapsed_seconds"],
                   "samples_per_hour": run["samples_per_hour"],
                   "speedup_c0_over_method": c0["elapsed_seconds"] / run["elapsed_seconds"],
                   "score_calls": run["mattergen_score_calls"],
                   "second_forward_calls": run["mattergen_score_calls"] - 1000,
                   "periodic_exact_calls": run.get("periodic_exact_calls", 0),
                   "atomic_fallback_calls": run.get("field_risk_fallback_calls", 0),
                   "late_exact_calls": run.get("late_exact_calls", 0),
                   "adapter_calls": run.get("adapter_calls", 0),
                   "coverage": run.get("adapter_acceptance_overall", 0),
                   "forward_reduction": run["forward_reduction"],
                   "peak_memory_bytes": run["peak_allocated_bytes"]}
            current.append(row)
        a = np.array([runs["C0", s]["elapsed_seconds"] for s in seeds])
        b = np.array([runs[method, s]["elapsed_seconds"] for s in seeds])
        low, high = boot_ci(a, b, lambda x, y: (x/y).mean(axis=1))
        summary[method] = {
            "single_h20_n": len(seeds), "time_per_sample": float(b.mean()),
            "samples_per_hour": float(3600/b.mean()),
            "speedup": float((a/b).mean()), "speedup_ci_low": low, "speedup_ci_high": high,
            **{k: float(np.mean([r[k] for r in current])) for k in (
                "score_calls", "second_forward_calls", "periodic_exact_calls",
                "atomic_fallback_calls", "late_exact_calls", "adapter_calls",
                "coverage", "forward_reduction", "peak_memory_bytes")},
        }
        rows.extend(current)
    return rows, summary


def streaks_from_trace(path: Path) -> list[int]:
    if not path.exists():
        return []
    streaks = []
    current = 0
    for row in csv_rows(path):
        if row["used_adapter"].lower() == "true":
            current += 1
        else:
            if current:
                streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    return streaks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stage-b", "stage-c"), required=True)
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()
    stage = args.stage
    root = output_root_for(stage)
    methods, seeds = methods_for(stage), seeds_for(stage)
    data = {m: load_quality(root, m, seeds) for m in methods}
    if args.quality_only:
        speed = {m: {} for m in methods}
    else:
        speed_rows, speed = collect_speed(stage)
        replace_stage(EXPERIMENT_ROOT / "single_h20_benchmark.csv", stage, speed_rows)
    metric_names = (*METRICS, "force_mean", "force_max")
    vectors = {(m, k): np.array([r[k] for r in data[m]]) for m in methods for k in metric_names}
    results, tails, mechanisms, per_seed = [], [], [], []
    for method in methods:
        generation = [json.loads((root / "generation" / method / str(s) / "run_summary.json").read_text()) for s in seeds]
        result = {"method": method, "n": len(seeds),
                  **{k: float(vectors[method, k].mean()) for k in metric_names},
                  **speed[method],
                  "main_coverage": float(np.mean([r.get("adapter_acceptance_overall", 0) for r in generation])),
                  "main_forward_reduction": float(np.mean([r["forward_reduction"] for r in generation])),
                  "main_periodic_calls": float(np.mean([r.get("periodic_exact_calls", 0) for r in generation]))}
        results.append(result)
        force = vectors[method, "force_max"]
        tails.append({"method": method, "n": len(seeds),
                      **{f"p{int(q*100)}": float(np.quantile(force, q)) for q in (.5, .75, .9, .95, .99)},
                      "max": float(force.max()),
                      "force_gt1_count": int((force>1).sum()), "force_gt1": float((force>1).mean()),
                      "force_gt2_count": int((force>2).sum()), "force_gt2": float((force>2).mean())})
        pooled = []
        longest = []
        for i, seed in enumerate(seeds):
            streak = streaks_from_trace(root / "generation" / method / str(seed) / "residual_trace.csv")
            pooled.extend(streak)
            longest.append(max(streak, default=0))
            assert max(streak, default=0) == generation[i].get("adapter_streak_max", 0)
            per_seed.append({**data[method][i],
                             "streak_mean": float(np.mean(streak)) if streak else 0.,
                             "streak_p95": float(np.quantile(streak, .95, method="inverted_cdf")) if streak else 0.,
                             "streak_max": max(streak, default=0),
                             "periodic_exact_calls": generation[i].get("periodic_exact_calls", 0)})
        longest = np.array(longest)
        rho, p = spearmanr(longest, force) if np.ptp(longest)>0 else (float("nan"), float("nan"))
        mechanisms.append({"stage": stage, "method": method, "n": len(seeds),
                           "streak_count": len(pooled),
                           "pooled_streak_mean": float(np.mean(pooled)) if pooled else 0.,
                           "pooled_streak_p95": float(np.quantile(pooled, .95, method="inverted_cdf")) if pooled else 0.,
                           "streak_max": max(pooled, default=0),
                           "seed_mean_of_max_streak": float(longest.mean()),
                           "longest_streak_force_spearman": float(rho), "spearman_p": float(p),
                           "force_gt1_n": int((force>1).sum()),
                           "force_gt1_mean_longest_streak": float(longest[force>1].mean()) if (force>1).any() else float("nan"),
                           "force_le1_mean_longest_streak": float(longest[force<=1].mean()) if (force<=1).any() else float("nan")})
    comparisons = []
    for baseline in ("C0", V2_LABEL):
        for method in methods:
            if method == baseline or method == "C0":
                continue
            for metric in (*metric_names, "force_p95", "force_p99", "force_gt1", "force_gt2"):
                tail = metric.startswith("force_p")
                threshold = metric in ("force_gt1", "force_gt2")
                source = "force_max" if tail or threshold else metric
                a, b = vectors[baseline, source], vectors[method, source]
                if threshold:
                    cutoff = 1 if metric == "force_gt1" else 2
                    a, b = (a>cutoff).astype(float), (b>cutoff).astype(float)
                if tail:
                    q = .95 if metric == "force_p95" else .99
                    av, bv = float(np.quantile(a,q)), float(np.quantile(b,q))
                    low, high = boot_ci(a,b,lambda x,y: np.quantile(y,q,axis=1)-np.quantile(x,q,axis=1))
                else:
                    av, bv = float(a.mean()), float(b.mean())
                    low, high = boot_ci(a,b,lambda x,y: (y-x).mean(axis=1))
                binary = metric in BINARY or threshold
                gain, loss = int(((a==0)&(b==1)).sum()), int(((a==1)&(b==0)).sum())
                discordant = gain+loss
                mc = float(binomtest(min(gain,loss),discordant,.5).pvalue) if binary and discordant else 1.0
                wp = float(wilcoxon(b,a,zero_method="pratt").pvalue) if not tail and np.any(a!=b) else 1.0
                comparisons.append({"baseline": baseline, "method": method, "metric": metric,
                                    "n_paired": len(seeds), "baseline_value": av, "method_value": bv,
                                    "delta_method_minus_baseline": bv-av,
                                    "paired_bootstrap95_low": low, "paired_bootstrap95_high": high,
                                    "wilcoxon_pratt_p": wp if not tail else "",
                                    "mcnemar_exact_p": mc if binary else "",
                                    "bootstrap_samples": N_BOOT, "bootstrap_seed": BOOT_SEED})
    prefix = stage.replace("-", "_")
    write_csv(EXPERIMENT_ROOT / f"{prefix}_results.csv", results)
    write_csv(EXPERIMENT_ROOT / f"{prefix}_force_tail.csv", tails)
    write_csv(root / "quality_per_seed.csv", per_seed)
    write_csv(root / "paired_statistics.csv", comparisons)
    replace_stage(EXPERIMENT_ROOT / "anchor_streak_analysis.csv", stage, mechanisms)
    print(json.dumps({"results": results, "force_tail": tails, "streak": mechanisms}, indent=2))


if __name__ == "__main__":
    main()
