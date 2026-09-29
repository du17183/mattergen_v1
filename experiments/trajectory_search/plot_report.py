"""Single-panel vector figures and an evidence-bounded thesis report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.trajectory_search.evaluate import RESULTS, PROJECT
from experiments.trajectory_search.analyze import BOOT_SEED, N_BOOT


plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.transparent": False,
})


def bootstrap_mean_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def draw(cohort: str) -> None:
    output = RESULTS / cohort
    paired = pd.read_csv(output / "paired_results.csv").sort_values("seed")
    order = ["C0", "Independent_Best_of_2", "Fixed_K2"] if cohort == "search_baseline" else [f"K{k}" for k in range(5)]
    names = ["C0", "Independent\nBest-of-2", "Fixed-K2"] if cohort == "search_baseline" else [f"K{k}" for k in range(5)]
    colors = ["#555b65", "#4476a6", "#bd6848"] if cohort == "search_baseline" else ["#555b65", "#8ca5be", "#4476a6", "#ca947a", "#bd6848"]
    means = np.asarray([paired[name].mean() for name in order])
    intervals = np.asarray([bootstrap_mean_interval(paired[name].to_numpy(float), BOOT_SEED + 500 + index) for index, name in enumerate(order)])
    fig, ax = plt.subplots(figsize=(3.5, 2.8), layout="constrained")
    x = np.arange(len(order), dtype=float)
    for index, color in enumerate(colors):
        ax.errorbar(x[index], means[index], yerr=[[means[index] - intervals[index, 0]], [intervals[index, 1] - means[index]]], fmt="o", ms=5.2, lw=1.35, capsize=2.5, color=color, mec="white", mew=.5)
    ax.set_xticks(x, names)
    ax.set_ylabel("Property MAE (CHGNet proxy)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#dde1e5", linewidth=.6)
    ax.set_axisbelow(True)
    if cohort == "budget_scaling":
        budgets = [1.0, 1.6, 2.2, 2.8, 3.4]
        for index, budget in enumerate(budgets):
            ax.text(x[index], means[index] + .002, f"{budget:.1f}×", ha="center", va="bottom", fontsize=6, color="#4f5964")
    destination = output / ("fig_best_of_n" if cohort == "search_baseline" else "fig_budget_scaling")
    fig.savefig(destination.with_suffix(".pdf"))
    fig.savefig(destination.with_suffix(".svg"))
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    plt.close(fig)
    (output / "figure_legend.md").write_text(
        "Points show mean absolute error of CHGNet-estimated magnetic-density target 0.1; whiskers are 20,000-resample percentile 95% intervals of the mean over 128 independently generated primary seeds. Methods are paired by primary seed for inferential comparisons in bootstrap_20k.json. This panel's whiskers are not paired-effect intervals. Fixed-K2 is 2.2× and Independent Best-of-2 2.0× MatterGen score-call budget; verifier costs are recorded separately. All outcomes are surrogate-only, not DFT validated.\n"
    )


def report() -> None:
    baseline = RESULTS / "search_baseline"
    scaling = RESULTS / "budget_scaling"
    b_table = pd.read_csv(baseline / "table_best_of_n.csv")
    k_table = pd.read_csv(scaling / "table_budget_scaling.csv")
    b_stats = json.loads((baseline / "bootstrap_20k.json").read_text())
    comparison = b_stats["Independent_Best_of_2_minus_Fixed_K2"]
    quality = bool(b_table.set_index("Method").loc["Fixed_K2", "quality_guardrails_pass"])
    positive = comparison["ci95_low"] > 0 and quality
    interpretation = (
        "Fixed-K2 shows a positive paired proxy-MAE contrast against Independent Best-of-2 with the prespecified CI above zero and passes quality guardrails. This is a positive contrast at 10% higher generation budget, but does **not** establish cost-efficiency or a trajectory-specific causal benefit: Fixed-K2 uses 2.2× score calls and three terminals, whereas Best-of-2 uses 2.0× and two. An exactly budget/candidate-matched control plus independent verification is still needed for that stronger claim."
        if positive else
        "The prespecified evidence does **not** support claiming that Fixed-K2 beats Independent Best-of-2 under the current asymmetric comparison. Fixed-K2 remains a valid historical C1 method, but its distinction from extra-budget Best-of-N is unproven."
    )
    def table_md(frame: pd.DataFrame) -> str:
        fields = ["Method", "Budget", "Property MAE", "95% CI paired gain vs C0", "Stable", "NUS", "E-hull", "Validity", "W", "T", "L"]
        lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
        for _, row in frame.iterrows():
            lines.append("|" + "|".join(str(row[field]) for field in fields) + "|")
        return "\n".join(lines)
    lines = [
        "# Innovation 1 trajectory-search evidence report", "", "Status: both fresh 128-seed cohorts complete; surrogate-only, DFT_VERIFIED=false.", "",
        "## Method positioning", "", "Fixed-K2 is a frozen-model, one-checkpoint, width-two, verifier-guided suffix search with exact C0 fallback. This is a search-framework instance, not a newly trained model. Historical C1-128 and the negative Linear-K2 finding remain unchanged.", "",
        "## Independent-sampling comparison", "", table_md(b_table), "",
        f"Paired Independent Best-of-2 minus Fixed-K2 mean = {comparison['mean']:.8f}, 95% CI [{comparison['ci95_low']:.8f}, {comparison['ci95_high']:.8f}], W/T/L={comparison['wins']}/{comparison['ties']}/{comparison['losses']}.", "", interpretation, "",
        "## Budget scaling", "", table_md(k_table), "",
        "The K=0–4 curve is a prespecified nested-prefix diagnostic. It may motivate K=2 as a pragmatic trade-off, but it does not retrospectively tune or change historical C1. Marginal comparisons remain exploratory unless separately preregistered/confirmed.", "",
        "## Thesis and publication recommendation", "", "For a master's thesis, the title ‘Budget-Constrained Trajectory Search with Verifier-Guided Selection for Frozen Diffusion Models’ is acceptable only with the above evidence limits stated. For a paper, add an exact-cost/terminal-count comparator and independent verifier or DFT validation before asserting a trajectory-search advantage beyond Best-of-N. Preserve the Linear-K2 null result. Do not implement Adaptive Budget Search in this stage; if evidence remains favorable, design it as a separate preregistered study.", "",
        "## Evidence locations", "", "Audit: `docs/search_framework_audit.md`; preregistration: `docs/trajectory_search_design.md`; fresh seed registry: `experiments/trajectory_search/protocol/seed_manifest.json`; per-seed generation, proxy evaluation, 20k bootstrap, tables and figures: `results/search_baseline/` and `results/budget_scaling/`.", "",
    ]
    target = PROJECT / "docs/trajectory_search_report.md"
    if target.exists():
        raise FileExistsError("final report already frozen")
    target.write_text("\n".join(lines))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("search_baseline", "budget_scaling"))
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report()
    elif args.cohort:
        draw(args.cohort)
    else:
        parser.error("select --cohort or --report")
