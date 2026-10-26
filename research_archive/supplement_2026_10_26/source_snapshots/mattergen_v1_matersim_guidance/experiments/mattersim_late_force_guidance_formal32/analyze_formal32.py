"""Analyze the frozen 32-pair MatterSim late-force Formal32 experiment."""
from __future__ import annotations

import csv
from datetime import datetime
import gzip
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import yaml

from experiments.mattersim_late_force_guidance_p0 import analyze_p0 as p0


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
P0_ROOT = PROJECT_ROOT / "experiments/mattersim_late_force_guidance_p0"
METHODS = ("C0", "F0")
SEEDS = tuple(range(730000, 730032))
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260914
MODEL = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/"
    "checkpoints/dft_mag_density/checkpoints/last.ckpt"
)
MATTERSIM = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/"
    "mattersim-v1.0.0-5M.pth"
)
CHGNET = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/.cache/models/chgnet/0.3.0/"
    "chgnet_0.3.0_e29f68s314m37.pth.tar"
)

p0.ROOT = ROOT
p0.PROJECT_ROOT = PROJECT_ROOT
p0.SEEDS = SEEDS
p0.METHODS = METHODS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty output table: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_trace() -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        source = ROOT / "generation" / "F0" / str(seed) / "late_force_trace.csv"
        seed_rows = read_csv(source)
        if len(seed_rows) != 20 or {int(row["seed"]) for row in seed_rows} != {seed}:
            raise RuntimeError(f"F0/{seed}: expected exactly 20 guidance rows")
        for row in seed_rows:
            accepted = truth(row["accepted"])
            fallback = truth(row["fallback"])
            proposed = float(row["max_cart_correction_a"])
            rows.append(row | {
                "guidance_attempted": True,
                "guidance_rejected": not accepted,
                "raw_force_mean_norm_ev_a": float(row["mean_force_ev_a"]),
                "raw_force_max_norm_ev_a": float(row["max_force_ev_a"]),
                "bounded_transform_max_displacement_a": proposed,
                "proposed_max_displacement_a": proposed,
                "accepted_max_displacement_a": proposed if accepted else 0.0,
                "hard_cap_reached": bool(
                    abs(proposed - float(row["hard_cart_cap_a"])) <= 1e-12
                ),
                "safety_fallback": fallback,
            })
    write_csv(ROOT / "guidance_trace.csv", rows)
    return rows


def load_data() -> dict[str, list[dict]]:
    generation = read_csv(ROOT / "generation_manifest.csv")
    properties = read_csv(ROOT / "property_metrics.csv")
    data = {
        method: p0.load_method(method, generation, properties)
        for method in METHODS
    }
    generation_by_key = {
        (row["method"], int(row["seed"])): row for row in generation
    }
    for method, rows in data.items():
        for row in rows:
            source = generation_by_key[(method, int(row["seed"]))]
            row["peak_allocated_bytes"] = int(source["peak_allocated_bytes"])
    return data


def arr(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=float)


def tail(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def paired_metric(
    baseline: np.ndarray, candidate: np.ndarray, *, direction: str,
    indices: np.ndarray,
) -> dict:
    delta = candidate - baseline
    bootstrap = delta[indices].mean(axis=1)
    tolerance = 1e-12
    favorable = delta < -tolerance if direction == "lower" else delta > tolerance
    unfavorable = delta > tolerance if direction == "lower" else delta < -tolerance
    ties = ~(favorable | unfavorable)
    return {
        "n_pairs": len(delta), "preferred_direction": direction,
        "C0_mean": float(baseline.mean()), "F0_mean": float(candidate.mean()),
        "mean_delta_F0_minus_C0": float(delta.mean()),
        "median_delta_F0_minus_C0": float(np.median(delta)),
        "paired_delta_bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "F0_wins": int(favorable.sum()), "ties": int(ties.sum()),
        "F0_losses": int(unfavorable.sum()),
    }


def statistics(data: dict[str, list[dict]]) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(SEEDS), size=(N_BOOTSTRAP, len(SEEDS)))
    specs = {
        "maxF_ev_per_a": "lower",
        "rmsd_a": "lower",
        "atomic_force_mean_ev_per_a": "lower",
        "e_hull_ev_per_atom": "lower",
        "mag_absolute_error_a3": "lower",
    }
    results = {
        key: paired_metric(arr(data["C0"], key), arr(data["F0"], key),
                           direction=direction, indices=indices)
        for key, direction in specs.items()
    }
    c0_maxf = arr(data["C0"], "maxF_ev_per_a")
    f0_maxf = arr(data["F0"], "maxF_ev_per_a")
    absolute_improvement = c0_maxf - f0_maxf
    boot_abs = absolute_improvement[indices].mean(axis=1)
    boot_rel = (
        c0_maxf[indices].mean(axis=1) - f0_maxf[indices].mean(axis=1)
    ) / c0_maxf[indices].mean(axis=1)
    results["formal_primary_maxF"] = {
        "criterion": "relative reduction of paired-cohort mean MaxF",
        "absolute_improvement_C0_minus_F0_ev_per_a": float(absolute_improvement.mean()),
        "absolute_improvement_bootstrap_ci95_ev_per_a": [
            float(np.quantile(boot_abs, 0.025)), float(np.quantile(boot_abs, 0.975))
        ],
        "relative_reduction": float((c0_maxf.mean() - f0_maxf.mean()) / c0_maxf.mean()),
        "relative_reduction_bootstrap_ci95": [
            float(np.quantile(boot_rel, 0.025)), float(np.quantile(boot_rel, 0.975))
        ],
        "central_estimate_definition": "(mean(C0)-mean(F0))/mean(C0)",
    }
    results["bootstrap"] = {"resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED}
    (ROOT / "bootstrap_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paired_rows = []
    keys = (
        "maxF_ev_per_a", "atomic_force_mean_ev_per_a", "rmsd_a",
        "e_hull_ev_per_atom", "stable", "nus", "novel", "unique",
        "mag_density_a3", "mag_absolute_error_a3", "mag_hit", "valid",
        "end_to_end_generation_seconds", "peak_allocated_bytes",
    )
    for index, seed in enumerate(SEEDS):
        c0 = data["C0"][index]
        f0 = data["F0"][index]
        row = {
            "seed": seed, "C0_formula": c0["formula"], "F0_formula": f0["formula"],
            "maxF_pair_outcome_for_F0": (
                "win" if f0["maxF_ev_per_a"] < c0["maxF_ev_per_a"] - 1e-12
                else "loss" if f0["maxF_ev_per_a"] > c0["maxF_ev_per_a"] + 1e-12
                else "tie"
            ),
        }
        for key in keys:
            row[f"C0_{key}"] = c0[key]
            row[f"F0_{key}"] = f0[key]
            row[f"delta_F0_minus_C0_{key}"] = float(f0[key]) - float(c0[key])
        paired_rows.append(row)
    write_csv(ROOT / "paired_results.csv", paired_rows)
    return results, paired_rows


def summaries(data: dict[str, list[dict]]) -> tuple[list[dict], list[dict], list[dict]]:
    physics_rows = []
    quality_rows = []
    efficiency_rows = []
    baseline_runtime = float(arr(data["C0"], "end_to_end_generation_seconds").mean())
    for method in METHODS:
        rows = data[method]
        maxf_tail = tail(arr(rows, "maxF_ev_per_a"))
        quality_rows.append({
            "method": method, "n": len(rows),
            "maxF_mean_ev_per_a": maxf_tail["mean"],
            "maxF_median_ev_per_a": maxf_tail["median"],
            "maxF_p75_ev_per_a": maxf_tail["p75"],
            "maxF_p90_ev_per_a": maxf_tail["p90"],
            "maxF_p95_ev_per_a": maxf_tail["p95"],
            "maxF_max_ev_per_a": maxf_tail["max"],
            "atomic_force_mean_ev_per_a": float(arr(rows, "atomic_force_mean_ev_per_a").mean()),
            "rmsd_mean_a": float(arr(rows, "rmsd_a").mean()),
            "e_hull_mean_ev_per_atom": float(arr(rows, "e_hull_ev_per_atom").mean()),
            "stable_fraction": float(arr(rows, "stable").mean()),
            "novel_fraction": float(arr(rows, "novel").mean()),
            "unique_fraction": float(arr(rows, "unique").mean()),
            "nus_fraction": float(arr(rows, "nus").mean()),
            "validity_fraction": float(arr(rows, "valid").mean()),
            "mag_density_mean_a3": float(arr(rows, "mag_density_a3").mean()),
            "mag_mae_a3": float(arr(rows, "mag_absolute_error_a3").mean()),
            "mag_hit_fraction": float(arr(rows, "mag_hit").mean()),
            "surrogate_property_eval": True, "dft_verified": False,
        })
        runtime = float(arr(rows, "end_to_end_generation_seconds").mean())
        efficiency_rows.append({
            "method": method, "n": len(rows),
            "generation_seconds_mean": float(arr(rows, "generation_seconds").mean()),
            "end_to_end_generation_seconds_mean": runtime,
            "runtime_ratio_vs_C0": runtime / baseline_runtime,
            "guidance_overhead_seconds_mean": float(arr(rows, "guidance_seconds").mean()),
            "peak_vram_bytes_mean": float(arr(rows, "peak_allocated_bytes").mean()),
            "peak_vram_bytes_max": int(arr(rows, "peak_allocated_bytes").max()),
            "mattergen_score_calls_mean": float(arr(rows, "mattergen_score_calls").mean()),
            "guidance_calls_total": int(arr(rows, "guidance_calls").sum()),
            "guidance_accepted_total": int(arr(rows, "guidance_accepted").sum()),
            "guidance_fallbacks_total": int(arr(rows, "guidance_fallbacks").sum()),
        })
        for row in rows:
            physics_rows.append({key: row[key] for key in (
                "method", "seed", "formula", "num_atoms", "maxF_ev_per_a",
                "atomic_force_mean_ev_per_a", "rmsd_a", "relaxation_steps",
                "minimum_periodic_distance_a", "surrogate_property_eval", "dft_verified",
            )})
    write_csv(ROOT / "physics_metrics.csv", physics_rows)
    write_csv(ROOT / "quality_metrics.csv", quality_rows)
    write_csv(ROOT / "efficiency_metrics.csv", efficiency_rows)
    return physics_rows, quality_rows, efficiency_rows


def decide(
    data: dict[str, list[dict]], quality: list[dict], efficiency: list[dict],
    bootstrap: dict,
) -> tuple[dict, dict]:
    q = {row["method"]: row for row in quality}
    e = {row["method"]: row for row in efficiency}
    primary = bootstrap["formal_primary_maxF"]
    relative = float(primary["relative_reduction"])
    ci_low, ci_high = map(float, primary["relative_reduction_bootstrap_ci95"])
    wins = int(bootstrap["maxF_ev_per_a"]["F0_wins"])
    ties = int(bootstrap["maxF_ev_per_a"]["ties"])
    losses = int(bootstrap["maxF_ev_per_a"]["F0_losses"])
    mag_worsening = (q["F0"]["mag_mae_a3"] - q["C0"]["mag_mae_a3"]) / q["C0"]["mag_mae_a3"]
    nus_drop = (q["C0"]["nus_fraction"] - q["F0"]["nus_fraction"]) * 100.0
    validity_drop = (q["C0"]["validity_fraction"] - q["F0"]["validity_fraction"]) * 100.0
    ehull_increase = q["F0"]["e_hull_mean_ev_per_atom"] - q["C0"]["e_hull_mean_ev_per_atom"]
    runtime_ratio = e["F0"]["runtime_ratio_vs_C0"]
    guardrail_pass = {
        "mag_mae_worsening_le_10pct": bool(mag_worsening <= 0.10 + 1e-12),
        "nus_drop_le_5pp": bool(nus_drop <= 5.0 + 1e-12),
        "validity_drop_le_5pp": bool(validity_drop <= 5.0 + 1e-12),
        "e_hull_increase_le_0.01_ev_atom": bool(ehull_increase <= 0.01 + 1e-12),
        "runtime_ratio_le_1.5": bool(runtime_ratio <= 1.5 + 1e-12),
    }
    all_guardrails = all(guardrail_pass.values())
    if all_guardrails and relative >= 0.15 and ci_low >= 0.10 and wins >= 24:
        verdict, next_action = "STRONG_CONFIRMED", "FORMAL256"
        recommendation = "MATTERSIM_FORCE_GUIDED_DIFFUSION"
    elif all_guardrails and relative >= 0.15 and ci_low > 0.0 and wins >= 20:
        verdict, next_action = "CONFIRMED", "FORMAL64_OR_FORMAL256"
        recommendation = "MATTERSIM_FORCE_GUIDED_DIFFUSION"
    elif (not all_guardrails) or relative < 0.08 or wins <= 16 or relative <= 0.0:
        verdict, next_action = "FAIL", "STOP_MATTERSIM_GUIDANCE"
        recommendation = "E3_PCR_FALLBACK"
    else:
        verdict, next_action = "BORDERLINE", "KEEP_E3_PCR"
        recommendation = "E3_PCR_FALLBACK"
    criteria = {
        "primary_maxF_criterion_passed": bool(relative >= 0.15),
        "strong_paired_criterion_passed": bool(wins >= 24),
        "confirmed_paired_criterion_passed": bool(wins >= 20),
        "strong_bootstrap_criterion_passed": bool(ci_low >= 0.10),
        "confirmed_bootstrap_criterion_passed": bool(ci_low > 0.0),
        "all_guardrails_passed": all_guardrails,
    }
    guardrails = {
        "schema_version": 1,
        "values": {
            "mag_mae_worsening_fraction": mag_worsening,
            "nus_drop_percentage_points": nus_drop,
            "validity_drop_percentage_points": validity_drop,
            "e_hull_mean_increase_ev_per_atom": ehull_increase,
            "runtime_ratio": runtime_ratio,
        },
        "pass": guardrail_pass,
        "all_pass": all_guardrails,
    }
    (ROOT / "guardrail_results.json").write_text(
        json.dumps(guardrails, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    decision = {
        "schema_version": 1,
        "MATTERSIM_FORCE_GUIDANCE_FORMAL32": verdict,
        "Recommended_Innovation2": recommendation,
        "NEXT": next_action,
        "primary": "MatterSim MaxF on final generated structure before relaxation",
        "formal_primary_criterion": "relative reduction of paired-cohort mean MaxF",
        "n_paired": len(SEEDS),
        "C0_mean_maxF_ev_per_a": q["C0"]["maxF_mean_ev_per_a"],
        "F0_mean_maxF_ev_per_a": q["F0"]["maxF_mean_ev_per_a"],
        "absolute_maxF_improvement_ev_per_a": primary["absolute_improvement_C0_minus_F0_ev_per_a"],
        "relative_maxF_reduction": relative,
        "relative_maxF_reduction_bootstrap_ci95": [ci_low, ci_high],
        "maxF_wins_ties_losses": [wins, ties, losses],
        "criteria": criteria,
        "guardrails": guardrails,
        "SURROGATE_PROPERTY_EVAL": True,
        "DFT_VERIFIED": False,
        "Adaptive_CFG_used": "NO",
        "formal256_run": False,
    }
    (ROOT / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return decision, guardrails


def guidance_behavior(trace_rows: list[dict]) -> dict:
    correction = np.asarray([float(row["max_cart_correction_a"]) for row in trace_rows])
    raw_mean_force = np.asarray([float(row["mean_force_ev_a"]) for row in trace_rows])
    raw_max_force = np.asarray([float(row["max_force_ev_a"]) for row in trace_rows])
    accepted = np.asarray([truth(row["accepted"]) for row in trace_rows])
    fallback = np.asarray([truth(row["fallback"]) for row in trace_rows])
    return {
        "total_possible_events": len(SEEDS) * 20,
        "attempted_events": len(trace_rows),
        "accepted_events": int(accepted.sum()),
        "rejected_events": int((~accepted).sum()),
        "fallback_events": int(fallback.sum()),
        "correction_statistic_definition": "per-event maximum per-atom Cartesian displacement",
        "correction_mean_a": float(correction.mean()),
        "correction_median_a": float(np.median(correction)),
        "correction_p95_a": float(np.quantile(correction, 0.95)),
        "correction_max_a": float(correction.max()),
        "raw_force_mean_norm_event_mean_ev_a": float(raw_mean_force.mean()),
        "raw_force_max_norm_event_mean_ev_a": float(raw_max_force.mean()),
        "post_normalize_clip_representation": "bounded Cartesian displacement",
        "bounded_transform_event_max_displacement_mean_a": float(correction.mean()),
        "minimum_candidate_distance_a": float(min(float(row["min_distance_after_a"]) for row in trace_rows)),
    }


def make_plots(data: dict[str, list[dict]], quality: list[dict], paired: list[dict]) -> None:
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=False)
    c0 = arr(data["C0"], "maxF_ev_per_a")
    f0 = arr(data["F0"], "maxF_ev_per_a")
    improvement = c0 - f0

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(c0, f0, color="#1769aa", alpha=0.85)
    limit = max(c0.max(), f0.max()) * 1.05
    ax.plot([0, limit], [0, limit], "--", color="gray", linewidth=1)
    ax.set(xlabel="C0 MaxF (eV/Å)", ylabel="F0 MaxF (eV/Å)",
           title="Formal32 paired MaxF")
    fig.tight_layout(); fig.savefig(plots / "maxf_paired_scatter.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(improvement, bins=12, color="#2e7d32", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="C0 - F0 MaxF (eV/Å)", ylabel="Count",
           title="Paired MaxF improvement distribution")
    fig.tight_layout(); fig.savefig(plots / "maxf_delta_distribution.png", dpi=180); plt.close(fig)

    names = ["Median", "P75", "P90", "P95", "Max"]
    qmap = {row["method"]: row for row in quality}
    cvals = [qmap["C0"][f"maxF_{name.lower()}_ev_per_a"] for name in names]
    fvals = [qmap["F0"][f"maxF_{name.lower()}_ev_per_a"] for name in names]
    x = np.arange(len(names)); width = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(x - width / 2, cvals, width, label="C0", color="#9e9e9e")
    ax.bar(x + width / 2, fvals, width, label="F0", color="#1769aa")
    ax.set_xticks(x, names); ax.set_ylabel("MaxF (eV/Å)"); ax.legend()
    ax.set_title("MaxF bad-tail comparison")
    fig.tight_layout(); fig.savefig(plots / "maxf_tail_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(c0, improvement, color="#6a1b9a", alpha=0.85)
    slope, intercept = np.polyfit(c0, improvement, 1)
    xx = np.linspace(c0.min(), c0.max(), 100)
    ax.plot(xx, slope * xx + intercept, color="#6a1b9a", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="C0 MaxF (eV/Å)", ylabel="C0 - F0 MaxF (eV/Å)",
           title="Base MaxF vs paired improvement")
    fig.tight_layout(); fig.savefig(plots / "base_maxf_vs_improvement.png", dpi=180); plt.close(fig)

    metrics = ["atomic_force_mean_ev_per_a", "rmsd_mean_a", "e_hull_mean_ev_per_atom", "mag_mae_a3"]
    labels = ["Atomic force", "RMSD", "E-hull", "Mag MAE"]
    csec = np.asarray([qmap["C0"][key] for key in metrics])
    fsec = np.asarray([qmap["F0"][key] for key in metrics])
    relative = (fsec - csec) / np.maximum(np.abs(csec), 1e-12) * 100.0
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(np.arange(len(labels)), relative, color="#ef6c00")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    ax.set_ylabel("F0 vs C0 mean change (%)")
    ax.set_title("Secondary metrics")
    fig.tight_layout(); fig.savefig(plots / "secondary_metrics.png", dpi=180); plt.close(fig)


def write_report(
    decision: dict, quality: list[dict], efficiency: list[dict],
    bootstrap: dict, behavior: dict, data: dict[str, list[dict]],
) -> None:
    q = {row["method"]: row for row in quality}
    e = {row["method"]: row for row in efficiency}
    primary = bootstrap["formal_primary_maxF"]
    c0 = arr(data["C0"], "maxF_ev_per_a")
    improvement = c0 - arr(data["F0"], "maxF_ev_per_a")
    rho = spearmanr(c0, improvement)
    pearson = pearsonr(c0, improvement)
    lines = [
        "# MatterSim Late Force-Guided Diffusion", "",
        "## Formal32 FINAL REPORT", "",
        "========== REPRODUCIBILITY ==========", "",
        f"1. MatterGen HEAD: `{git('rev-parse', 'HEAD')}`.",
        f"2. Branch: `{git('branch', '--show-current')}`.",
        f"3. Base checkpoint path: `{MODEL}`.",
        f"4. Base checkpoint SHA256: `{sha256(MODEL)}`.",
        f"5. MatterSim version: `{metadata.version('mattersim')}`; checkpoint SHA256 `{sha256(MATTERSIM)}`.",
        f"6. CHGNet version: `0.3.0`; checkpoint SHA256 `{sha256(CHGNET)}`.", "",
        "========== FROZEN METHOD ==========", "",
        "7. Guidance window: model time `t<=0.02`; actual 20-point trigger grid `0.020...0.001`.",
        "8. Position-only? **YES**; cell/stress/atomic/energy guidance are OFF.",
        "9. Force source: MatterSim-5M on MatterGen predicted clean x0; raw noisy xt is not evaluated.",
        "10. Force transform: remove translation; force becomes frozen bounded Cartesian displacement; `delta_frac=delta_cart@inverse(clean_cell)` and exact predictor coefficient maps it to score.",
        "11. Trust-region cap: nominal 0.005 Å/atom; hard cap 0.01 Å/atom; protocol absolute cap 0.02 Å.",
        "12. Safety guard: finite clean state, positive clean-cell determinant, candidate periodic minimum distance >=0.5 Å.",
        "13. Fallback rule: use the original unmodified MatterGen score when the safety guard fails.",
        "14. CFG: constant 2.0.", "15. Diffusion steps: 1000.",
        "16. Corrector steps: 1 per timestep.", "17. Adaptive CFG used? **NO**.", "",
        "========== SEEDS ==========", "",
        "18. Seed range: paired fresh `730000-730031`.",
        "19. Historical overlap scan: no exact seed reference in experiments/diagnostics/research/thesis across all worktrees before registration.",
        "20. Number generated: C0 32/32 and F0 32/32; every registered pair is analyzed.", "",
        "========== PRIMARY ==========", "",
        f"21. C0 Mean MaxF: **{q['C0']['maxF_mean_ev_per_a']:.9f} eV/Å**.",
        f"22. F0 Mean MaxF: **{q['F0']['maxF_mean_ev_per_a']:.9f} eV/Å**.",
        f"23. Absolute improvement C0-F0: **{primary['absolute_improvement_C0_minus_F0_ev_per_a']:.9f} eV/Å**; 95% CI `{primary['absolute_improvement_bootstrap_ci95_ev_per_a']}`.",
        f"24. Formal relative reduction `(mean(C0)-mean(F0))/mean(C0)`: **{primary['relative_reduction']*100:.3f}%**.",
        f"25. Wins/ties/losses for F0: **{decision['maxF_wins_ties_losses'][0]}/{decision['maxF_wins_ties_losses'][1]}/{decision['maxF_wins_ties_losses'][2]}**.",
        f"26. Relative-reduction 20,000 paired bootstrap 95% CI: **[{primary['relative_reduction_bootstrap_ci95'][0]*100:.3f}%, {primary['relative_reduction_bootstrap_ci95'][1]*100:.3f}%]**.",
        f"27. Median MaxF C0/F0: `{q['C0']['maxF_median_ev_per_a']:.6f}/{q['F0']['maxF_median_ev_per_a']:.6f}` eV/Å.",
        f"28. P90 MaxF C0/F0: `{q['C0']['maxF_p90_ev_per_a']:.6f}/{q['F0']['maxF_p90_ev_per_a']:.6f}` eV/Å.",
        f"29. P75/P95/max MaxF C0: `{q['C0']['maxF_p75_ev_per_a']:.6f}/{q['C0']['maxF_p95_ev_per_a']:.6f}/{q['C0']['maxF_max_ev_per_a']:.6f}` eV/Å; F0: `{q['F0']['maxF_p75_ev_per_a']:.6f}/{q['F0']['maxF_p95_ev_per_a']:.6f}/{q['F0']['maxF_max_ev_per_a']:.6f}` eV/Å.", "",
        "========== SECONDARY ==========", "",
        f"30. RMSD mean C0/F0: `{q['C0']['rmsd_mean_a']:.6f}/{q['F0']['rmsd_mean_a']:.6f}` Å.",
        f"31. Atomic force mean C0/F0: `{q['C0']['atomic_force_mean_ev_per_a']:.6f}/{q['F0']['atomic_force_mean_ev_per_a']:.6f}` eV/Å.",
        f"32. E-hull mean C0/F0: `{q['C0']['e_hull_mean_ev_per_atom']:.6f}/{q['F0']['e_hull_mean_ev_per_atom']:.6f}` eV/atom.",
        f"33. Stable C0/F0: `{q['C0']['stable_fraction']:.2%}/{q['F0']['stable_fraction']:.2%}`.",
        f"34. Novel C0/F0: `{q['C0']['novel_fraction']:.2%}/{q['F0']['novel_fraction']:.2%}`.",
        f"35. Unique C0/F0: `{q['C0']['unique_fraction']:.2%}/{q['F0']['unique_fraction']:.2%}`.",
        f"36. NUS C0/F0: `{q['C0']['nus_fraction']:.2%}/{q['F0']['nus_fraction']:.2%}`.",
        f"37. Validity C0/F0: `{q['C0']['validity_fraction']:.2%}/{q['F0']['validity_fraction']:.2%}`.",
        f"38. Mean CHGNet mag density C0/F0: `{q['C0']['mag_density_mean_a3']:.6f}/{q['F0']['mag_density_mean_a3']:.6f}` Å^-3.",
        f"39. Mag MAE C0/F0: `{q['C0']['mag_mae_a3']:.6f}/{q['F0']['mag_mae_a3']:.6f}` Å^-3.",
        f"40. Mag hit C0/F0: `{q['C0']['mag_hit_fraction']:.2%}/{q['F0']['mag_hit_fraction']:.2%}` at tau=0.01.", "",
        "========== GUIDANCE BEHAVIOR ==========", "",
        f"41. Guidance attempts: {behavior['attempted_events']} / {behavior['total_possible_events']} possible.",
        f"42. Accepted: {behavior['accepted_events']}.",
        f"43. Rejected: {behavior['rejected_events']}.",
        f"44. Fallback: {behavior['fallback_events']}.",
        f"45. Mean event-max correction magnitude after the bounded transform: `{behavior['correction_mean_a']:.9f}` Å; median `{behavior['correction_median_a']:.9f}` Å; pre-guidance raw mean-force-norm event mean `{behavior['raw_force_mean_norm_event_mean_ev_a']:.9f}` eV/Å.",
        f"46. P95 event-max correction: `{behavior['correction_p95_a']:.9f}` Å; max `{behavior['correction_max_a']:.9f}` Å; pre-guidance raw max-force-norm event mean `{behavior['raw_force_max_norm_event_mean_ev_a']:.9f}` eV/Å. Event-level raw force and transformed displacement columns are in `guidance_trace.csv`.", "",
        "========== EFFICIENCY ==========", "",
        f"47. C0 mean end-to-end generation runtime: `{e['C0']['end_to_end_generation_seconds_mean']:.3f}` s.",
        f"48. F0 mean end-to-end generation runtime: `{e['F0']['end_to_end_generation_seconds_mean']:.3f}` s.",
        f"49. Runtime ratio F0/C0: `{e['F0']['runtime_ratio_vs_C0']:.5f}x`; no acceleration claim is made.",
        f"50. Peak VRAM max C0/F0: `{e['C0']['peak_vram_bytes_max']/2**20:.1f}/{e['F0']['peak_vram_bytes_max']/2**20:.1f}` MiB.",
        f"51. MatterGen score calls mean C0/F0: `{e['C0']['mattergen_score_calls_mean']:.0f}/{e['F0']['mattergen_score_calls_mean']:.0f}`; MatterSim guidance overhead mean `{e['F0']['guidance_overhead_seconds_mean']:.3f}` s.", "",
        "========== GUARDRAILS ==========", "",
        f"52. Mag MAE guardrail: `{decision['guardrails']['values']['mag_mae_worsening_fraction']*100:.3f}%` worsening; PASS={decision['guardrails']['pass']['mag_mae_worsening_le_10pct']}.",
        f"53. NUS guardrail: drop `{decision['guardrails']['values']['nus_drop_percentage_points']:.3f}` pp; PASS={decision['guardrails']['pass']['nus_drop_le_5pp']}.",
        f"54. Validity guardrail: drop `{decision['guardrails']['values']['validity_drop_percentage_points']:.3f}` pp; PASS={decision['guardrails']['pass']['validity_drop_le_5pp']}.",
        f"55. E-hull guardrail: increase `{decision['guardrails']['values']['e_hull_mean_increase_ev_per_atom']:.6f}` eV/atom; PASS={decision['guardrails']['pass']['e_hull_increase_le_0.01_ev_atom']}.",
        f"56. Runtime guardrail: ratio `{decision['guardrails']['values']['runtime_ratio']:.5f}x`; PASS={decision['guardrails']['pass']['runtime_ratio_le_1.5']}.", "",
        "========== DECISION ==========", "",
        f"57. MATTERSIM_FORCE_GUIDANCE_FORMAL32 = **{decision['MATTERSIM_FORCE_GUIDANCE_FORMAL32']}**.",
        f"58. Primary MaxF criterion passed? **{'YES' if decision['criteria']['primary_maxF_criterion_passed'] else 'NO'}**.",
        f"59. Paired criterion passed? **{'YES' if decision['criteria']['confirmed_paired_criterion_passed'] else 'NO'}** (strong threshold separately recorded).",
        f"60. Bootstrap criterion passed? **{'YES' if decision['criteria']['confirmed_bootstrap_criterion_passed'] else 'NO'}** (strong threshold separately recorded).",
        f"61. All guardrails passed? **{'YES' if decision['criteria']['all_guardrails_passed'] else 'NO'}**.",
        "62. SURROGATE_PROPERTY_EVAL=True.", "63. DFT_VERIFIED=False.",
        f"64. NEXT = **{decision['NEXT']}**; Recommended Innovation2 = **{decision['Recommended_Innovation2']}**.",
        f"65. Scientific conclusion: on 32 completely fresh paired seeds, the frozen position-only method is `{decision['MATTERSIM_FORCE_GUIDANCE_FORMAL32']}` under the preregistered MaxF and guardrail rules; all claims are surrogate, not DFT.", "",
        "## Bad-tail and mechanism analysis (non-gating)", "",
        f"Base MaxF versus paired improvement: Spearman rho `{float(rho.statistic):.6f}` (p `{float(rho.pvalue):.4g}`); Pearson r `{float(pearson.statistic):.6f}` (p `{float(pearson.pvalue):.4g}`). Positive correlation means initially worse-force samples benefit more. This analysis does not affect the gate.",
    ]
    (ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(trace: list[dict], decision: dict) -> None:
    config = yaml.safe_load((ROOT / "formal32_config.yaml").read_text(encoding="utf-8"))
    required = [
        "formal32_config.yaml", "formal32_seeds.json", "generation_manifest.csv",
        "guidance_trace.csv", "property_metrics.csv", "quality_metrics.csv",
        "physics_metrics.csv", "efficiency_metrics.csv", "paired_results.csv",
        "bootstrap_results.json", "guardrail_results.json", "decision_summary.json",
        "audit_results.json", "final_report.md",
    ]
    plots = [
        "plots/maxf_paired_scatter.png", "plots/maxf_delta_distribution.png",
        "plots/maxf_tail_comparison.png", "plots/base_maxf_vs_improvement.png",
        "plots/secondary_metrics.png",
    ]
    generation = read_csv(ROOT / "generation_manifest.csv")
    seeds_payload = json.loads((ROOT / "formal32_seeds.json").read_text(encoding="utf-8"))
    summaries = {
        (method, seed): json.loads(
            (ROOT / "generation" / method / str(seed) / "run_summary.json").read_text()
        )
        for method in METHODS for seed in SEEDS
    }
    expected_times = np.asarray(config["actual_trigger_model_t"], dtype=float)
    observed_times = {
        seed: np.asarray([
            float(row["t_norm"]) for row in trace if int(row["seed"]) == seed
        ], dtype=float)
        for seed in SEEDS
    }
    source_hashes_unchanged = all(
        sha256(Path(config["source"]["files"][name])) == expected
        for name, expected in config["source"]["sha256"].items()
    )
    preflight_log = (ROOT / "logs" / "formal32_pair_730000_gpu_5.log").read_text(
        encoding="utf-8", errors="replace"
    )
    checks = {
        # audit_results.json is atomically created at the end of this function.
        "required_files_present": all(
            (ROOT / name).is_file()
            for name in required + plots if name != "audit_results.json"
        ),
        "generation_rows_64": len(generation) == 64,
        "all_generation_success": all(truth(row["success"]) for row in generation),
        "all_adaptive_cfg_false": all(not truth(row["adaptive_cfg_used"]) for row in generation),
        "paired_rows_32": len(read_csv(ROOT / "paired_results.csv")) == 32,
        "property_rows_64": len(read_csv(ROOT / "property_metrics.csv")) == 64,
        "physics_rows_64": len(read_csv(ROOT / "physics_metrics.csv")) == 64,
        "guidance_rows_640": len(trace) == 640,
        "all_f0_trace_grids_exact": all(
            len(observed_times[seed]) == len(expected_times)
            and np.allclose(observed_times[seed], expected_times, rtol=0.0, atol=1e-12)
            for seed in SEEDS
        ),
        "all_p0_source_hashes_unchanged": source_hashes_unchanged,
        "all_score_calls_2000": all(int(row["mattergen_score_calls"]) == 2000 for row in generation),
        "all_checkpoint_hashes_exact": all(
            summary["checkpoint_sha256"]
            == "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
            for summary in summaries.values()
        ) and all(
            summaries[("F0", seed)]["mattersim_checkpoint_sha256"]
            == "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
            for seed in SEEDS
        ),
        "all_common_sampling_contract_exact": all(
            float(summary["target_dft_mag_density"]) == 0.2
            and float(summary["guidance_scale"]) == 2.0
            and summary["guidance_schedule"] == "constant"
            and int(summary["sampling_steps"]) == 1000
            and int(summary["corrector_steps_per_timestep"]) == 1
            and str(summary["physical_gpu"]) in {"5", "6", "7"}
            for summary in summaries.values()
        ),
        "paired_formula_and_atom_count_match": all(
            summaries[("C0", seed)]["formula"] == summaries[("F0", seed)]["formula"]
            and int(summaries[("C0", seed)]["num_atoms"])
            == int(summaries[("F0", seed)]["num_atoms"])
            for seed in SEEDS
        ),
        "all_f0_guidance_events_20": all(
            int(summaries[("F0", seed)]["mattersim_guidance_eligible"]) == 20
            for seed in SEEDS
        ),
        "all_f0_guidance_accepted_or_fallback_accounted": all(
            int(summaries[("F0", seed)]["mattersim_guidance_accepted"])
            + int(summaries[("F0", seed)]["mattersim_guidance_fallbacks"]) == 20
            for seed in SEEDS
        ),
        "all_f0_position_only": all(
            summaries[("F0", seed)]["position_force_guidance"]
            and not summaries[("F0", seed)]["stress_guidance"]
            and not summaries[("F0", seed)]["cell_guidance"]
            and not summaries[("F0", seed)]["atomic_guidance"]
            and not summaries[("F0", seed)]["energy_guidance"]
            for seed in SEEDS
        ),
        "fresh_seed_registration_exact": (
            seeds_payload["historical_overlap_count"] == 0
            and [int(item["seed"]) for item in seeds_payload["paired_seeds"]] == list(SEEDS)
            and all(
                item["historical_search_status"]
                == "NO_REFERENCE_FOUND_BEFORE_REGISTRATION"
                for item in seeds_payload["paired_seeds"]
            )
        ),
        "no_partial_generation_directories": not any(
            path.name.endswith(".partial") for path in (ROOT / "generation").rglob("*.partial")
        ),
        "stash_count_5": len(git("stash", "list").splitlines()) == 5,
        "formal256_not_run": not (ROOT / "formal256").exists(),
        "surrogate_true_dft_false": decision["SURROGATE_PROPERTY_EVAL"] is True
        and decision["DFT_VERIFIED"] is False,
    }
    payload = {
        "AUDIT": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "required_file_count": len(required), "plot_count": len(plots),
        "mattergen_head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "stash_count_preserved": len(git("stash", "list").splitlines()),
        "verdict": decision["MATTERSIM_FORCE_GUIDANCE_FORMAL32"],
        "pre_scientific_import_preflight": {
            "status": "CORRECTED_BEFORE_SAMPLING",
            "wrong_environment_detected": (
                "MODELS_PROJECT_ROOT: /mnt/datasets-livsyn/dxl/alm/external/mattergen/mattergen"
                in preflight_log
            ),
            "formal_worktree_import_confirmed": (
                "MODELS_PROJECT_ROOT: /mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/mattergen"
                in preflight_log
            ),
            "failed_attempt_produced_scientific_output": False,
            "registered_seed_replaced": False,
        },
        "property_wrapper_preflight": {
            "status": "CORRECTED_BEFORE_INFERENCE",
            "cause": "concurrent relaxation had already created the same p0_structures symlink",
            "failed_attempt_produced_property_output": False,
            "evaluator_or_data_changed": False,
        },
    }
    (ROOT / "audit_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if payload["AUDIT"] != "PASS":
        raise RuntimeError(f"Formal32 audit failed: {checks}")


def main() -> None:
    for output in (
        "guidance_trace.csv", "physics_metrics.csv", "quality_metrics.csv",
        "efficiency_metrics.csv", "paired_results.csv", "bootstrap_results.json",
        "guardrail_results.json", "decision_summary.json", "final_report.md",
        "audit_results.json", "plots",
    ):
        if (ROOT / output).exists():
            raise FileExistsError(ROOT / output)
    trace = aggregate_trace()
    data = load_data()
    bootstrap, paired = statistics(data)
    _, quality, efficiency = summaries(data)
    decision, _ = decide(data, quality, efficiency, bootstrap)
    behavior = guidance_behavior(trace)
    make_plots(data, quality, paired)
    write_report(decision, quality, efficiency, bootstrap, behavior, data)
    audit(trace, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
