"""Aggregate the paired 16-seed smoke, apply frozen fail-fast/expansion gates."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
METHODS = ("B0", "C0")
SEEDS = tuple(range(110000, 110016))
LOWER_METRICS = (
    "mag_absolute_error", "rmsd_a", "atomic_force_mean_ev_a",
    "max_force_ev_a", "e_hull_ev_atom", "runtime_seconds", "peak_vram_mb",
    "score_calls",
)
RATE_METRICS = ("mag_hit", "valid", "stable", "novel", "unique", "nus")


def paired_bootstrap(frame: pd.DataFrame, metric: str, lower_better: bool) -> dict:
    base = frame[f"{metric}_B0"].to_numpy(dtype=float)
    candidate = frame[f"{metric}_C0"].to_numpy(dtype=float)
    finite = np.isfinite(base) & np.isfinite(candidate)
    base, candidate = base[finite], candidate[finite]
    if not len(base):
        return {"n": 0, "effect": None, "ci95": [None, None]}
    if lower_better:
        effect = (base.mean() - candidate.mean()) / base.mean()
    else:
        effect = candidate.mean() - base.mean()
    rng = np.random.default_rng(20260912)
    indices = rng.integers(0, len(base), size=(20000, len(base)))
    base_means = base[indices].mean(axis=1)
    candidate_means = candidate[indices].mean(axis=1)
    if lower_better:
        draws = np.divide(
            base_means - candidate_means, base_means,
            out=np.full_like(base_means, np.nan), where=base_means != 0,
        )
    else:
        draws = candidate_means - base_means
    finite_draws = draws[np.isfinite(draws)]
    return {
        "n": int(len(base)), "effect": float(effect),
        "ci95": [float(np.quantile(finite_draws, 0.025)),
                 float(np.quantile(finite_draws, 0.975))],
        "resamples": 20000,
    }


def collect_method(method: str, generation: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    generated = generation[generation.method == method].sort_values("seed").reset_index(drop=True)
    prop = properties[properties.method == method].sort_values("seed").reset_index(drop=True)
    if generated.seed.astype(int).tolist() != list(SEEDS) or prop.seed.astype(int).tolist() != list(SEEDS):
        raise RuntimeError(f"seed order mismatch for {method}")
    with gzip.open(ROOT / "quality" / method / "official_detailed.json.gz", "rt") as stream:
        detail = json.load(stream)
    initial = read(
        ROOT / "relaxation" / method / "initial_with_properties.extxyz", index=":"
    )
    relaxation = json.loads(
        (ROOT / "relaxation" / method / "relaxation_summary.json").read_text()
    )
    if len(initial) != len(SEEDS) or len(detail["energy_above_hull_per_atom"]) != len(SEEDS):
        raise RuntimeError(f"quality row count mismatch for {method}")
    rows = []
    for index, seed in enumerate(SEEDS):
        forces = np.linalg.norm(initial[index].get_forces(), axis=1)
        rows.append({
            "method": method, "seed": seed, "formula": generated.loc[index, "formula"],
            "num_atoms": int(generated.loc[index, "num_atoms"]),
            "mag_density": float(prop.loc[index, "chgnet_mag_density"]),
            "mag_absolute_error": float(prop.loc[index, "mag_absolute_error"]),
            "mag_hit": bool(prop.loc[index, "mag_hit"]),
            "valid": bool(prop.loc[index, "structure_valid"]),
            "rmsd_a": float(detail["rmsd_from_relaxation"][index]),
            "atomic_force_mean_ev_a": float(forces.mean()),
            "max_force_ev_a": float(forces.max()),
            "e_hull_ev_atom": float(detail["energy_above_hull_per_atom"][index]),
            "stable": bool(detail["stable"][index]),
            "novel": bool(detail["novel"][index]),
            "unique": bool(detail["unique"][index]),
            "nus": bool(detail["novel_unique_stable"][index]),
            "relaxation_steps": int(relaxation["relaxation_steps"][index]),
            "runtime_seconds": float(generated.loc[index, "elapsed_seconds"]),
            "peak_vram_mb": float(generated.loc[index, "peak_allocated_bytes"]) / 1024**2,
            "score_calls": int(generated.loc[index, "mattergen_score_calls"]),
            "surrogate_property_eval": True, "dft_verified": False,
        })
    return pd.DataFrame(rows)


def main() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    properties = pd.read_csv(ROOT / "property_metrics.csv")
    per_sample = pd.concat(
        [collect_method(method, generation, properties) for method in METHODS],
        ignore_index=True,
    )
    per_sample.to_csv(ROOT / "per_sample_metrics.csv", index=False)
    aggregate_rows = []
    for method in METHODS:
        selected = per_sample[per_sample.method == method]
        row = {"method": method, "n": len(selected)}
        for metric in LOWER_METRICS:
            row[f"{metric}_mean"] = float(selected[metric].mean())
        for metric in RATE_METRICS:
            row[f"{metric}_rate"] = float(selected[metric].mean())
        row["mag_density_mean"] = float(selected.mag_density.mean())
        row["relaxation_steps_mean"] = float(selected.relaxation_steps.mean())
        aggregate_rows.append(row)
    aggregates = pd.DataFrame(aggregate_rows)
    aggregates.to_csv(ROOT / "smoke_results.csv", index=False)

    paired = per_sample.pivot(index="seed", columns="method")
    paired.columns = [f"{metric}_{method}" for metric, method in paired.columns]
    paired = paired.reset_index()
    bootstrap = {
        metric: paired_bootstrap(paired, metric, lower_better=True)
        for metric in LOWER_METRICS
    }
    bootstrap.update({
        metric: paired_bootstrap(paired, metric, lower_better=False)
        for metric in RATE_METRICS
    })
    base = aggregates[aggregates.method == "B0"].iloc[0]
    candidate = aggregates[aggregates.method == "C0"].iloc[0]
    lower_improvement = {
        metric: float(
            (base[f"{metric}_mean"] - candidate[f"{metric}_mean"])
            / base[f"{metric}_mean"]
        )
        for metric in LOWER_METRICS
    }
    rate_delta = {
        metric: float(candidate[f"{metric}_rate"] - base[f"{metric}_rate"])
        for metric in RATE_METRICS
    }
    fail_fast = {
        "mag_mae_worsening_gt_15pct": bool(lower_improvement["mag_absolute_error"] < -0.15),
        "max_force_worsening_gt_20pct": bool(lower_improvement["max_force_ev_a"] < -0.20),
        "e_hull_increase_gt_0p02": bool(
            candidate["e_hull_ev_atom_mean"] - base["e_hull_ev_atom_mean"] > 0.02
        ),
        "nus_drop_gt_10pp": bool(rate_delta["nus"] < -0.10),
        "validity_drop_gt_10pp": bool(rate_delta["valid"] < -0.10),
    }
    fail_early = any(fail_fast.values())
    generation_signal = {
        "property_mag_mae_improvement_ge_8pct": bool(lower_improvement["mag_absolute_error"] >= 0.08),
        "geometry_max_force_improvement_ge_8pct": bool(lower_improvement["max_force_ev_a"] >= 0.08),
        "geometry_rmsd_improvement_ge_8pct": bool(lower_improvement["rmsd_a"] >= 0.08),
        "quality_nus_gain_ge_5pp": bool(rate_delta["nus"] >= 0.05),
    }
    training = json.loads((ROOT / "checkpoints/training_summary.json").read_text())
    formal32_eligible = bool(
        not fail_early
        and any(generation_signal.values())
        and training["validation_signal"]["validation_go_signal"]
        and not training["fusion_unused"]
    )
    weak_primary = max(
        lower_improvement["mag_absolute_error"],
        lower_improvement["max_force_ev_a"],
        lower_improvement["rmsd_a"], rate_delta["nus"],
    )
    if fail_early:
        verdict = "FAIL_EARLY"
        next_step = "STOP"
    elif formal32_eligible:
        verdict = "PASS_SMOKE"
        next_step = "EXPAND_TO_FORMAL32"
    elif weak_primary >= 0.05 and not training["fusion_unused"]:
        verdict = "BORDERLINE"
        next_step = "STOP_NO_FORMAL32"
    else:
        verdict = "FAIL"
        next_step = "STOP"
    decision = {
        "experiment": "Global Cell-Atom Bidirectional Fusion 16-seed Smoke P0",
        "seeds": list(SEEDS), "n_per_method": 16,
        "baseline": "Official dft_mag_density Full FT",
        "target_dft_mag_density": 0.2, "guidance_scale": 2.0,
        "sampling_steps": 1000, "corrector_steps": 1, "adaptive_cfg": False,
        "lower_is_better_relative_improvement": lower_improvement,
        "rate_absolute_delta": rate_delta,
        "paired_bootstrap": bootstrap,
        "fail_fast_checks": fail_fast, "generation_expansion_signals": generation_signal,
        "validation_signal": training["validation_signal"],
        "training_signal_weak": training["training_signal_weak"],
        "fusion_unused": training["fusion_unused"],
        "formal32_eligible": formal32_eligible,
        "cell_atom_fusion_p0": verdict, "next": next_step,
        "surrogate_property_eval": True, "dft_verified": False,
    }
    (ROOT / "smoke_decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = aggregates.to_markdown(index=False, floatfmt=".6f")
    report = f"""# Global Cell--Atom Bidirectional Fusion: 16-seed Smoke P0

## Result

`CELL_ATOM_FUSION_P0={verdict}`; `NEXT={next_step}`.

This is a paired, fresh-seed smoke over seeds 110000--110015. B0 is the frozen official Full FT model; C0 changes only the trained fusion module. Both use target 0.2, constant CFG 2, 1000 diffusion steps, one corrector step, and Adaptive CFG off.

## Aggregate metrics

{table}

Lower-is-better relative improvements (positive favors C0):

```json
{json.dumps(lower_improvement, indent=2, sort_keys=True)}
```

Rate changes are absolute fractions (0.05 = 5 percentage points):

```json
{json.dumps(rate_delta, indent=2, sort_keys=True)}
```

## Gates

Fail-fast checks: `{json.dumps(fail_fast, sort_keys=True)}`.

Expansion signals: `{json.dumps(generation_signal, sort_keys=True)}`.

Formal32 eligibility additionally requires a validation GO signal and a used fusion mechanism; result: `{formal32_eligible}`.

## Scope

Magnetic density is evaluated by the frozen CHGNet 0.3.0 surrogate (held-out MAE 0.001449, Pearson 0.9499, hit tolerance 0.01). `SURROGATE_PROPERTY_EVAL=True`; `DFT_VERIFIED=False`. Energy/force/relaxation metrics use MatterSim-5M. Paired bootstrap uses 20,000 resamples and is descriptive at n=16.
"""
    (ROOT / "smoke_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
