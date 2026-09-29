"""Reproducible, read-only post-analysis and publication plots for the frozen study.

This script never regenerates samples or changes the frozen selector. It reads the
completed search_baseline/budget_scaling outputs and writes only derived files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DERIVED = RESULTS / "trajectory_search"
FIGURES = ROOT / "figures" / "search_framework"
SKILL_SCRIPTS = Path("/root/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


N = 128
N_BOOT = 20_000
BOOT_SEED = 2026092801
EPS = 1e-12
COLORS = {"C0": "#68768A", "Independent_Best_of_2": "#326D83", "Fixed_K2": "#AA6240"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.transparent": False,
    }
)


def read_cohort(name: str, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    source = RESULTS / name
    table_name = "table_best_of_n.csv" if name == "search_baseline" else "table_budget_scaling.csv"
    table = pd.read_csv(source / table_name).set_index("Method").loc[methods].reset_index()
    paired = pd.read_csv(source / "paired_results.csv").sort_values("seed")
    intervals = json.loads((source / "bootstrap_20k.json").read_text())
    compute = json.loads((source / "compute_accounting.json").read_text())
    assert len(paired) == N and paired.seed.nunique() == N
    assert set(methods) == set(paired.columns) - {"seed"}
    assert all(paired[m].notna().all() for m in methods)
    for _, row in table.iterrows():
        method = row["Method"]
        assert np.isclose(float(row["Budget"]), compute["per_method_deployment_multiplier"][method])
        assert np.isclose(float(row["Property MAE"]), paired[method].mean(), atol=1e-12)
    outcomes = pd.read_csv(source / "selected_outcomes.csv")
    assert len(outcomes) == N * len(methods)
    assert all(outcomes.groupby("result_method").seed.nunique().loc[m] == N for m in methods)
    for run_seed in paired.seed:
        run = json.loads((source / "generation" / str(run_seed) / "run_summary.json").read_text())
        assert run["success"] and run["reference_reproduction"]["success"]
    return table, paired, intervals, compute


def bootstrap(values: np.ndarray, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    assert len(values) == N and np.isfinite(values).all()
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, N, size=(N_BOOT, N))].mean(axis=1)
    return {
        "n": N,
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "wins": int((values > EPS).sum()),
        "ties": int((np.abs(values) <= EPS).sum()),
        "losses": int((values < -EPS).sum()),
        "resamples": N_BOOT,
    }


def save_pub(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(path) + ".alignment.json",
        overlay_svg=str(path) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(str(path) + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(str(path) + ".svg", bbox_inches="tight", facecolor="white")
    fig.savefig(str(path) + ".png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mean_intervals(paired: pd.DataFrame, methods: list[str], cohort_seed: int) -> pd.DataFrame:
    rows = []
    for i, method in enumerate(methods):
        stats = bootstrap(paired[method].to_numpy(float), cohort_seed + i)
        rows.append({"Method": method, "Mean": stats["mean"], "CI95_low": stats["ci95_low"], "CI95_high": stats["ci95_high"]})
    return pd.DataFrame(rows)


def draw_compute_curve(table: pd.DataFrame, means: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.50, 2.65), constrained_layout=True)
    x = table["Budget"].to_numpy(float)
    y = means.Mean.to_numpy(float)
    lo = means.CI95_low.to_numpy(float)
    hi = means.CI95_high.to_numpy(float)
    ax.plot(x, y, color="#326D83", lw=1.5, zorder=2)
    ax.errorbar(x, y, yerr=np.vstack((y - lo, hi - y)), fmt="none", ecolor="#326D83", capsize=2, lw=1, zorder=1)
    ax.scatter(x, y, s=24, c="#326D83", edgecolors="white", lw=0.5, zorder=3)
    ax.set(xlabel="Relative MatterGen score calls (× C0)")
    ax.set_title("Property MAE (CHGNet)", loc="left", pad=5)
    ax.set_xticks(x, [f"{budget:.1f}×\n{method}" for budget, method in zip(x, table.Method, strict=True)])
    ax.set_xlim(0.83, 3.58)
    ax.set_ylim(0.018, 0.045)
    ax.grid(axis="y", color="#E5E9ED", lw=0.5)
    save_pub(fig, "compute_performance_curve")


def draw_width_curve(margins: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.50, 2.65), constrained_layout=True)
    x = margins.K.to_numpy(int)
    y = margins["Paired gain"].to_numpy(float)
    lo = margins.CI95_low.to_numpy(float)
    hi = margins.CI95_high.to_numpy(float)
    ax.axhline(0, color="#89939F", lw=0.7, zorder=0)
    ax.errorbar(x, y, yerr=np.vstack((y - lo, hi - y)), fmt="o", color="#326D83", markersize=4.5, capsize=2.5, lw=1, zorder=3)
    ax.set(xlabel="Additional searched suffix (K)")
    ax.set_title("Incremental paired MAE reduction", loc="left", pad=5)
    ax.set_xticks(x)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(-0.0007, max(0.0053, hi.max() + 0.00035))
    ax.grid(axis="y", color="#E5E9ED", lw=0.5)
    save_pub(fig, "search_width_curve")


def draw_method_comparison(table: pd.DataFrame, means: pd.DataFrame) -> None:
    methods = ["C0", "Independent_Best_of_2", "Fixed_K2"]
    labels = ["C0\n1.0×", "Independent\nBest-of-2\n2.0×", "Fixed-K2\n2.2×"]
    x = np.arange(3)
    y = means.set_index("Method").loc[methods, "Mean"].to_numpy(float)
    lo = means.set_index("Method").loc[methods, "CI95_low"].to_numpy(float)
    hi = means.set_index("Method").loc[methods, "CI95_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(3.50, 2.70), constrained_layout=True)
    ax.errorbar(x, y, yerr=np.vstack((y - lo, hi - y)), fmt="none", ecolor="#596575", capsize=3, lw=1, zorder=1)
    ax.scatter(x, y, s=70, marker="D", c=[COLORS[m] for m in methods], edgecolors="white", lw=0.7, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_title("Property MAE (CHGNet)", loc="left", pad=5)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0.019, 0.046)
    ax.grid(axis="y", color="#E5E9ED", lw=0.5)
    save_pub(fig, "method_comparison")


def main() -> None:
    baseline_methods = ["C0", "Independent_Best_of_2", "Fixed_K2"]
    width_methods = [f"K{k}" for k in range(5)]
    base, base_paired, base_stats, base_compute = read_cohort("search_baseline", baseline_methods)
    width, width_paired, width_stats, width_compute = read_cohort("budget_scaling", width_methods)
    assert set(base_paired.seed).isdisjoint(width_paired.seed)
    assert list(base_paired.seed) == list(range(2100000000, 2100000000 + N))
    assert list(width_paired.seed) == list(range(2100010000, 2100010000 + N))
    assert base_compute["total_study_acquisition_score_calls"] == N * 8400
    assert width_compute["total_study_acquisition_score_calls"] == N * 8800

    best_dir = DERIVED / "best_of_2"
    width_dir = DERIVED / "budget_scaling"
    best_dir.mkdir(parents=True, exist_ok=True)
    width_dir.mkdir(parents=True, exist_ok=True)
    base_out = base[["Method", "Budget", "Property MAE", "E-hull", "Stable", "NUS", "Validity", "Novel", "Unique", "95% CI paired gain vs C0", "W", "T", "L", "quality_guardrails_pass"]].copy()
    base_out["Score calls/sample"] = base_out.Budget.mul(2000).astype(int)
    base_out["CHGNet selector calls/sample"] = base_out.Method.map(base_compute["per_method_chgnet_and_mattersim_selector_calls"])
    base_out["MatterSim selector calls/sample"] = base_out["CHGNet selector calls/sample"]
    base_out.to_csv(best_dir / "best_of_2_summary.csv", index=False)
    width_out = width[["Method", "Budget", "Property MAE", "E-hull", "Stable", "NUS", "Validity", "Novel", "Unique", "95% CI paired gain vs C0", "W", "T", "L", "quality_guardrails_pass"]].copy()
    width_out["Score calls/sample"] = width_out.Budget.mul(2000).astype(int)
    width_out["CHGNet selector calls/sample"] = width_out.Method.map(width_compute["per_method_chgnet_and_mattersim_selector_calls"])
    width_out["MatterSim selector calls/sample"] = width_out["CHGNet selector calls/sample"]
    width_out.to_csv(width_dir / "budget_scaling_summary.csv", index=False)

    margin_rows = []
    for k in range(1, 5):
        gain = width_paired[f"K{k-1}"].to_numpy(float) - width_paired[f"K{k}"].to_numpy(float)
        stats = bootstrap(gain, BOOT_SEED + 200 + k)
        margin_rows.append({"K": k, "Paired gain": stats["mean"], "CI95_low": stats["ci95_low"], "CI95_high": stats["ci95_high"], "W": stats["wins"], "T": stats["ties"], "L": stats["losses"], "N": N, "Resamples": N_BOOT, "Status": "posthoc_descriptive"})
    margins = pd.DataFrame(margin_rows)
    margins.to_csv(width_dir / "incremental_paired_gains.csv", index=False)

    base_means = mean_intervals(base_paired, baseline_methods, BOOT_SEED + 300)
    width_means = mean_intervals(width_paired, width_methods, BOOT_SEED + 400)
    base_means.to_csv(best_dir / "mean_mae_bootstrap_20k.csv", index=False)
    width_means.to_csv(width_dir / "mean_mae_bootstrap_20k.csv", index=False)

    runs = [json.loads((RESULTS / "search_baseline" / "generation" / str(seed) / "run_summary.json").read_text()) for seed in base_paired.seed]
    primary = np.array([row["primary_elapsed_seconds"] for row in runs], dtype=float)
    independent = np.array([row["independent_elapsed_seconds"] for row in runs], dtype=float)
    # These allocation proxies are not isolated measured deployment timings.
    wall = pd.DataFrame(
        [
            {"Measure": "primary acquisition", "Mean seconds/sample": primary.mean(), "Median seconds/sample": np.median(primary), "P95 seconds/sample": np.percentile(primary, 95), "Basis": "measured, includes C0 replay and two suffixes"},
            {"Measure": "independent second full sample", "Mean seconds/sample": independent.mean(), "Median seconds/sample": np.median(independent), "P95 seconds/sample": np.percentile(independent, 95), "Basis": "measured independent generation"},
            {"Measure": "total study acquisition", "Mean seconds/sample": (primary + independent).mean(), "Median seconds/sample": np.median(primary + independent), "P95 seconds/sample": np.percentile(primary + independent, 95), "Basis": "measured generation, all candidates plus replay"},
            {"Measure": "C0 deployment", "Mean seconds/sample": (primary * 2000 / 6400).mean(), "Median seconds/sample": np.median(primary * 2000 / 6400), "P95 seconds/sample": np.percentile(primary * 2000 / 6400, 95), "Basis": "score-proportional estimate only; excludes verifier"},
            {"Measure": "Independent Best-of-2 deployment", "Mean seconds/sample": (primary * 2000 / 6400 + independent).mean(), "Median seconds/sample": np.median(primary * 2000 / 6400 + independent), "P95 seconds/sample": np.percentile(primary * 2000 / 6400 + independent, 95), "Basis": "score-proportional primary allocation plus measured second sample; excludes verifier"},
            {"Measure": "Fixed-K2 deployment", "Mean seconds/sample": (primary * 4400 / 6400).mean(), "Median seconds/sample": np.median(primary * 4400 / 6400), "P95 seconds/sample": np.percentile(primary * 4400 / 6400, 95), "Basis": "score-proportional estimate only; excludes verifier"},
        ]
    )
    wall.to_csv(best_dir / "generation_wall_time_accounting.csv", index=False)

    best_diff = base_stats["Independent_Best_of_2_minus_Fixed_K2"]
    gain_bo2 = base_stats["C0_minus_Independent_Best_of_2"]["mean"]
    gain_fixed = base_stats["C0_minus_Fixed_K2"]["mean"]
    normalized = {
        "method": "descriptive C0-minus-method gain per additional C0-equivalent score-call unit",
        "Independent_Best_of_2": gain_bo2 / (2.0 - 1.0),
        "Fixed_K2": gain_fixed / (2.2 - 1.0),
        "warning": "Not an exact-cost, verifier-call, terminal-count, or wall-time matched causal comparison.",
        "direct_paired_Best_of_2_minus_Fixed_K2": best_diff,
    }
    (best_dir / "compute_normalized_comparison.json").write_text(json.dumps(normalized, indent=2) + "\n")

    draw_compute_curve(width, width_means)
    draw_width_curve(margins)
    draw_method_comparison(base, base_means)
    print(json.dumps({"baseline_n": len(base_paired), "budget_n": len(width_paired), "baseline_direct_difference": best_diff, "marginal_gains": margin_rows, "wall_time": wall.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
