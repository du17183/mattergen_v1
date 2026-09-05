"""Complete paired quality, tail, and non-inferiority analysis for formal256."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from ase.io import read
from scipy.stats import binomtest, wilcoxon

from research.corrector_distillation.formal256_protocol import (
    EXPERIMENT_ROOT,
    FORMAL_SEEDS,
    METHODS,
    sha256,
    validate_frozen_protocol,
)


BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260905
BASELINE = "C0"
CANDIDATE = "V2_Frozen_Atomic75_Late30"
METRICS = {
    "e_hull": ("energy_above_hull_per_atom", "lower"),
    "stable": ("stable", "higher"),
    "nus": ("novel_unique_stable", "higher"),
    "novel": ("novel", "higher"),
    "unique": ("unique", "higher"),
    "rmsd": ("rmsd_from_relaxation", "lower"),
}
CONTINUOUS = {"e_hull", "rmsd", "force_mean", "force_max"}
BINARY = {"stable", "nus", "novel", "unique"}
QUANTILES = {"p5": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_statistic(
    baseline: np.ndarray,
    candidate: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    rng: np.random.Generator,
) -> tuple[float, float]:
    estimates = []
    remaining = BOOTSTRAP_SAMPLES
    while remaining:
        current = min(remaining, 2_000)
        indices = rng.integers(0, baseline.size, size=(current, baseline.size))
        estimates.append(statistic(baseline[indices], candidate[indices]))
        remaining -= current
    low, high = np.quantile(np.concatenate(estimates), (0.025, 0.975))
    return float(low), float(high)


def mean_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (right - left).mean(axis=1)


def mean_ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return right.mean(axis=1) / left.mean(axis=1)


def sample_std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def load_method(method: str) -> dict[int, dict[str, float | int | str | bool]]:
    quality_dir = EXPERIMENT_ROOT / "quality" / method
    precomputed_dir = EXPERIMENT_ROOT / "relaxation/precomputed" / method
    summary = json.loads((quality_dir / "quality_summary.json").read_text(encoding="utf-8"))
    relax_summary = json.loads(
        (precomputed_dir / "relaxation_summary.json").read_text(encoding="utf-8")
    )
    with gzip.open(quality_dir / "official_detailed.json.gz", "rt", encoding="utf-8") as stream:
        detailed = json.load(stream)
    with (quality_dir / "per_structure.csv").open(newline="", encoding="utf-8") as stream:
        per_structure = list(csv.DictReader(stream))
    initial_atoms = read(precomputed_dir / "initial_with_properties.extxyz", index=":")
    expected = len(FORMAL_SEEDS)
    sizes = {
        len(per_structure),
        len(initial_atoms),
        len(relax_summary["pre_relaxation_max_forces"]),
        len(relax_summary["relaxation_steps"]),
        *(len(detailed[source]) for source, _direction in METRICS.values()),
    }
    if sizes != {expected} or int(summary["n"]) != expected or int(relax_summary["n"]) != expected:
        raise ValueError(f"{method} quality arrays are incomplete or misaligned: {sizes}")
    result = {}
    for index, (row, atoms) in enumerate(zip(per_structure, initial_atoms)):
        seed = int(row["sample_seed"])
        if seed in result:
            raise ValueError(f"duplicate {method} seed {seed}")
        force_norms = np.linalg.norm(np.asarray(atoms.get_forces(), dtype=float), axis=1)
        record: dict[str, float | int | str | bool] = {
            "method": method,
            "seed": seed,
            "success": True,
            "formula": row["formula"],
            "num_atoms": int(row["num_atoms"]),
            "relaxation_steps": int(row["relaxation_steps"]),
            "final_energy_ev": float(row["final_energy_ev"]),
            "force_mean": float(force_norms.mean()),
            "force_max": float(force_norms.max()),
        }
        if not math.isclose(
            float(record["force_max"]), float(row["pre_relaxation_max_force"]), rel_tol=1e-5, abs_tol=1e-6
        ):
            raise ValueError(f"force provenance mismatch for {method} seed {seed}")
        for output_name, (source_name, _direction) in METRICS.items():
            value = detailed[source_name][index]
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"missing {output_name} for {method} seed {seed}")
            record[output_name] = float(value)
        result[seed] = record
    if sorted(result) != list(FORMAL_SEEDS):
        raise ValueError(f"{method} seed set does not match immutable formal seeds")
    return result


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": sample_std(values),
        "p5": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def values(data: dict[int, dict[str, object]], metric: str) -> np.ndarray:
    return np.asarray([float(data[seed][metric]) for seed in FORMAL_SEEDS], dtype=float)


def main() -> None:
    preflight = validate_frozen_protocol()
    if not preflight["passed"]:
        raise RuntimeError(preflight["failures"])
    data = {method: load_method(method) for method in METHODS}
    quality_rows = [data[method][seed] for seed in FORMAL_SEEDS for method in METHODS]
    write_csv(EXPERIMENT_ROOT / "quality_per_seed.csv", quality_rows)

    audit = {
        "schema_version": 1,
        "surrogate": "MatterSim-5M",
        "dft_verified": False,
        "expected_per_method": len(FORMAL_SEEDS),
        "methods": {
            method: {
                "success": len(data[method]),
                "seed_set_exact": sorted(data[method]) == list(FORMAL_SEEDS),
                "quality_summary_sha256": sha256(EXPERIMENT_ROOT / "quality" / method / "quality_summary.json"),
                "relaxation_summary_sha256": sha256(
                    EXPERIMENT_ROOT / "relaxation/precomputed" / method / "relaxation_summary.json"
                ),
            }
            for method in METHODS
        },
    }
    audit["passed"] = all(
        item["success"] == len(FORMAL_SEEDS) and item["seed_set_exact"]
        for item in audit["methods"].values()
    )
    (EXPERIMENT_ROOT / "formal256_quality_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    paired_rows = []
    metric_names = [*METRICS, "force_mean", "force_max"]
    for seed in FORMAL_SEEDS:
        row: dict[str, object] = {"seed": seed}
        for metric in metric_names:
            left = float(data[BASELINE][seed][metric])
            right = float(data[CANDIDATE][seed][metric])
            row[f"c0_{metric}"] = left
            row[f"v2_{metric}"] = right
            row[f"delta_{metric}"] = right - left
        paired_rows.append(row)
    write_csv(EXPERIMENT_ROOT / "paired_per_seed.csv", paired_rows)

    summary_rows = []
    statistics_rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for metric in metric_names:
        baseline = values(data[BASELINE], metric)
        candidate = values(data[CANDIDATE], metric)
        baseline_desc = describe(baseline)
        candidate_desc = describe(candidate)
        summary_rows.append(
            {
                "metric": metric,
                "direction": METRICS.get(metric, (None, "lower"))[1],
                "n": len(FORMAL_SEEDS),
                **{f"c0_{name}": value for name, value in baseline_desc.items()},
                **{f"v2_{name}": value for name, value in candidate_desc.items()},
                "delta_mean_v2_minus_c0": candidate_desc["mean"] - baseline_desc["mean"],
                "relative_change_mean": (
                    candidate_desc["mean"] / baseline_desc["mean"] - 1.0
                    if baseline_desc["mean"] != 0 else float("nan")
                ),
            }
        )
        deltas = candidate - baseline
        ci_low, ci_high = bootstrap_statistic(baseline, candidate, mean_delta, rng=rng)
        try:
            wilcoxon_result = wilcoxon(
                candidate, baseline, zero_method="pratt", alternative="two-sided", method="auto"
            )
            wilcoxon_stat = float(wilcoxon_result.statistic)
            wilcoxon_p = float(wilcoxon_result.pvalue)
        except ValueError:
            wilcoxon_stat = float("nan")
            wilcoxon_p = 1.0 if np.all(deltas == 0) else float("nan")
        discordant_01 = int(np.sum((baseline == 0) & (candidate == 1))) if metric in BINARY else 0
        discordant_10 = int(np.sum((baseline == 1) & (candidate == 0))) if metric in BINARY else 0
        discordant = discordant_01 + discordant_10
        mcnemar_p = (
            float(binomtest(min(discordant_01, discordant_10), discordant, 0.5).pvalue)
            if discordant else 1.0
        ) if metric in BINARY else float("nan")
        statistics_rows.append(
            {
                "metric": metric,
                "metric_type": "binary" if metric in BINARY else "continuous",
                "n_paired": len(FORMAL_SEEDS),
                "c0_mean": float(baseline.mean()),
                "v2_mean": float(candidate.mean()),
                "mean_paired_difference_v2_minus_c0": float(deltas.mean()),
                "paired_difference_median": float(np.median(deltas)),
                "paired_difference_std": sample_std(deltas),
                "paired_bootstrap95_low": ci_low,
                "paired_bootstrap95_high": ci_high,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "wilcoxon_pratt_statistic": wilcoxon_stat,
                "wilcoxon_pratt_p_two_sided": wilcoxon_p,
                "mcnemar_discordant_c0_0_v2_1": discordant_01 if metric in BINARY else "",
                "mcnemar_discordant_c0_1_v2_0": discordant_10 if metric in BINARY else "",
                "mcnemar_exact_p_two_sided": mcnemar_p if metric in BINARY else "",
                "equivalence_claim_from_p_value_prohibited": True,
            }
        )
    write_csv(EXPERIMENT_ROOT / "quality_summary.csv", summary_rows)
    write_csv(EXPERIMENT_ROOT / "paired_statistics.csv", statistics_rows)

    tail_outputs = {
        "force_max": EXPERIMENT_ROOT / "force_tail_analysis.csv",
        "rmsd": EXPERIMENT_ROOT / "rmsd_tail_analysis.csv",
    }
    tail_cache: dict[str, dict[str, dict[str, float]]] = {}
    for metric, output in tail_outputs.items():
        baseline = values(data[BASELINE], metric)
        candidate = values(data[CANDIDATE], metric)
        rows = []
        tail_cache[metric] = {}
        for label, quantile in {k: v for k, v in QUANTILES.items() if k not in ("p5", "p25")}.items():
            c0_value = float(np.quantile(baseline, quantile))
            v2_value = float(np.quantile(candidate, quantile))
            statistic = lambda left, right, q=quantile: np.quantile(right, q, axis=1) - np.quantile(left, q, axis=1)
            ci_low, ci_high = bootstrap_statistic(baseline, candidate, statistic, rng=rng)
            ratio_statistic = lambda left, right, q=quantile: np.quantile(right, q, axis=1) / np.quantile(left, q, axis=1)
            ratio_ci_low, ratio_ci_high = bootstrap_statistic(
                baseline, candidate, ratio_statistic, rng=rng
            )
            rows.append(
                {
                    "metric": metric,
                    "tail_statistic": label,
                    "c0": c0_value,
                    "v2": v2_value,
                    "delta_v2_minus_c0": v2_value - c0_value,
                    "ratio_v2_over_c0": v2_value / c0_value if c0_value else float("nan"),
                    "paired_bootstrap_delta95_low": ci_low,
                    "paired_bootstrap_delta95_high": ci_high,
                    "paired_bootstrap_ratio95_low": ratio_ci_low,
                    "paired_bootstrap_ratio95_high": ratio_ci_high,
                    "bootstrap_samples": BOOTSTRAP_SAMPLES,
                }
            )
            tail_cache[metric][label] = rows[-1]
        rows.append(
            {
                "metric": metric,
                "tail_statistic": "max",
                "c0": float(baseline.max()),
                "v2": float(candidate.max()),
                "delta_v2_minus_c0": float(candidate.max() - baseline.max()),
                "ratio_v2_over_c0": float(candidate.max() / baseline.max()) if baseline.max() else float("nan"),
                "paired_bootstrap_delta95_low": "",
                "paired_bootstrap_delta95_high": "",
                "paired_bootstrap_ratio95_low": "",
                "paired_bootstrap_ratio95_high": "",
                "bootstrap_samples": 0,
            }
        )
        if metric == "force_max":
            for threshold in (1.0, 2.0):
                left = (baseline > threshold).astype(float)
                right = (candidate > threshold).astype(float)
                ci_low, ci_high = bootstrap_statistic(left, right, mean_delta, rng=rng)
                rows.append(
                    {
                        "metric": metric,
                        "tail_statistic": f"proportion_gt_{threshold:g}_ev_per_angstrom",
                        "c0": float(left.mean()),
                        "v2": float(right.mean()),
                        "delta_v2_minus_c0": float((right - left).mean()),
                        "ratio_v2_over_c0": float(right.mean() / left.mean()) if left.mean() else float("nan"),
                        "paired_bootstrap_delta95_low": ci_low,
                        "paired_bootstrap_delta95_high": ci_high,
                        "paired_bootstrap_ratio95_low": "",
                        "paired_bootstrap_ratio95_high": "",
                        "bootstrap_samples": BOOTSTRAP_SAMPLES,
                    }
                )
                tail_cache[metric][f"gt_{threshold:g}"] = rows[-1]
        write_csv(output, rows)

    margins = json.loads((EXPERIMENT_ROOT / "noninferiority_margins.yaml").read_text(encoding="utf-8"))
    ni_rows = []
    stats_by_metric = {row["metric"]: row for row in statistics_rows}
    for item in margins["primary_endpoints"]:
        metric = item["metric"]
        baseline = values(data[BASELINE], metric)
        candidate = values(data[CANDIDATE], metric)
        if item["scale"] == "absolute":
            estimate = float((candidate - baseline).mean())
            ci_low = float(stats_by_metric[metric]["paired_bootstrap95_low"])
            ci_high = float(stats_by_metric[metric]["paired_bootstrap95_high"])
            if item["direction"] == "lower":
                passed = ci_high <= float(item["maximum_harm"])
            else:
                passed = ci_low >= -float(item["maximum_harm"])
        else:
            estimate = float(candidate.mean() / baseline.mean())
            ci_low, ci_high = bootstrap_statistic(baseline, candidate, mean_ratio, rng=rng)
            passed = ci_high <= float(item["maximum_ratio"])
        ni_rows.append(
            {
                "endpoint": metric,
                "scale": item["scale"],
                "direction": item["direction"],
                "estimate": estimate,
                "bootstrap95_low": ci_low,
                "bootstrap95_high": ci_high,
                "margin": item.get("maximum_harm", item.get("maximum_ratio")),
                "noninferiority_pass": passed,
                "two_sided_95_interval_used_conservatively": True,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
            }
        )
    for guard in margins["force_tail_guardrails"]:
        row = tail_cache["force_max"][guard["statistic"]]
        if guard["scale"] == "ratio":
            estimate = float(row["ratio_v2_over_c0"])
            passed = float(row["paired_bootstrap_ratio95_high"]) <= float(
                guard["maximum_ratio"]
            )
            margin = guard["maximum_ratio"]
            guard_ci_low = row["paired_bootstrap_ratio95_low"]
            guard_ci_high = row["paired_bootstrap_ratio95_high"]
        else:
            estimate = float(row["delta_v2_minus_c0"])
            passed = float(row["paired_bootstrap_delta95_high"]) <= float(guard["maximum_harm"])
            margin = guard["maximum_harm"]
            guard_ci_low = row["paired_bootstrap_delta95_low"]
            guard_ci_high = row["paired_bootstrap_delta95_high"]
        ni_rows.append(
            {
                "endpoint": f"force_tail_{guard['statistic']}",
                "scale": guard["scale"],
                "direction": "lower",
                "estimate": estimate,
                "bootstrap95_low": guard_ci_low,
                "bootstrap95_high": guard_ci_high,
                "margin": margin,
                "noninferiority_pass": passed,
                "two_sided_95_interval_used_conservatively": True,
                "bootstrap_samples": row["bootstrap_samples"],
            }
        )
    write_csv(EXPERIMENT_ROOT / "noninferiority_results.csv", ni_rows)
    print(json.dumps({"quality_audit": audit, "noninferiority": ni_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
