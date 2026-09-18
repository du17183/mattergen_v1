#!/usr/bin/env python3
"""Generate the final thesis figures from compact frozen result files only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
I1 = ROOT / "innovation1"
I2 = ROOT / "innovation2"
BOOT_SEED = 20_260_915
N_BOOT = 20_000
COLORS = {"C0": "#6B7280", "Fixed-K2": "#2563EB", "Linear-K2": "#D97706", "Random-K2": "#8B5CF6", "RC-NFGD": "#059669"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "svg.hashsalt": "mattergen-thesis-final-2026",
})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        target = directory / f"{stem}.{suffix}"
        metadata = {"Creator": "MatterGen thesis release"}
        if suffix == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        elif suffix == "svg":
            metadata.update({"Date": None})
        elif suffix == "png":
            metadata = {"Software": "MatterGen thesis release"}
        fig.savefig(
            target,
            bbox_inches="tight",
            dpi=300 if suffix == "png" else None,
            metadata=metadata,
        )
        # Matplotlib's SVG path formatter leaves insignificant spaces at EOL;
        # normalise them so the repository passes `git diff --check`.
        if suffix == "svg":
            lines = target.read_text(encoding="utf-8").splitlines()
            target.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


def i1_figures() -> None:
    out = I1 / "figures"
    metrics = {row["method"]: row for row in read_csv(I1 / "results/c1_metrics.csv")}
    methods = ["C0", "Fixed_K2", "Linear_K2", "Random_K2"]
    labels = ["C0", "Fixed-K2", "Linear-K2", "Random-K2"]

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    values = [float(metrics[name]["property_mae"]) for name in methods]
    bars = ax.bar(labels, values, color=[COLORS[label] for label in labels])
    ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
    ax.set_ylabel("Property MAE (lower is better)")
    ax.set_ylim(0, max(values) * 1.22)
    fig.tight_layout()
    save(fig, out, "I1_F1_c1_property_mae")

    paired = read_csv(I1 / "results/c1_paired_results.csv")
    c0 = np.asarray([float(row["C0"]) for row in paired])
    fixed = np.asarray([float(row["Fixed_K2"]) for row in paired])
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    for left, right in zip(c0, fixed):
        color = "#059669" if right < left - 1e-12 else "#BFC5CC"
        ax.plot([0, 1], [left, right], color=color, alpha=0.45, linewidth=0.65)
    ax.scatter(np.zeros_like(c0), c0, s=9, color=COLORS["C0"], zorder=3)
    ax.scatter(np.ones_like(fixed), fixed, s=9, color=COLORS["Fixed-K2"], zorder=3)
    ax.set_xticks([0, 1], ["C0", "Fixed-K2"])
    ax.set_ylabel("Per-seed property absolute error")
    ax.text(0.02, 0.98, "53 improved / 75 tie-fallback / 0 worse", transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    save(fig, out, "I1_F2_fixed_vs_c0_paired")

    gains = c0 - fixed
    rng = np.random.default_rng(BOOT_SEED)
    draws = gains[rng.integers(0, len(gains), size=(N_BOOT, len(gains)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(draws, bins=55, color=COLORS["Fixed-K2"], alpha=0.82)
    ax.axvline(0, color="#991B1B", linestyle="--", linewidth=1)
    ax.axvline(gains.mean(), color="black", linewidth=1.2, label=f"mean={gains.mean():.4f}")
    ax.axvspan(low, high, color="#93C5FD", alpha=0.35, label=f"95% CI [{low:.4f}, {high:.4f}]")
    ax.set_xlabel("Mean absolute property improvement (C0 - Fixed-K2)")
    ax.set_ylabel("Bootstrap count")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "I1_F3_fixed_gain_bootstrap")

    quality = [("E-hull", "e_hull", "eV/atom"), ("Stable", "stable", "fraction"), ("NUS", "nus", "fraction"), ("Validity", "validity", "fraction")]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0))
    for ax, (label, key, unit) in zip(axes.flat, quality):
        values = [float(metrics["C0"][key]), float(metrics["Fixed_K2"][key])]
        bars = ax.bar(["C0", "Fixed-K2"], values, color=[COLORS["C0"], COLORS["Fixed-K2"]])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_title(label)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values + [0.01]) * 1.25)
    fig.tight_layout()
    save(fig, out, "I1_F4_quality_guardrails")

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    labels = ["C0\ndeployment", "Fixed-K2\ndeployment", "Phase B\ndata construction"]
    values = [1.0, 2.2, 3.4]
    bars = ax.bar(labels, values, color=[COLORS["C0"], COLORS["Fixed-K2"], "#9CA3AF"])
    ax.bar_label(bars, labels=["1.0x", "2.2x", "3.4x"], padding=2)
    ax.set_ylabel("MatterGen generation compute relative to C0")
    ax.text(2, 0.12, "dataset construction,\nnot deployment", ha="center", va="bottom", fontsize=8, color="#7F1D1D")
    ax.set_ylim(0, 4.05)
    fig.tight_layout()
    save(fig, out, "I1_F5_compute_comparison")

    status_rows = read_csv(ROOT / "combined_summary/experiment_status.csv")
    wanted = ["Adaptive CFG V1", "Robust V2", "Counterfactual Oracle V3", "Risk-Calibrated V4", "Safe Selection V5", "Stage-Calibrated CFG", "Field-Decoupled CFG", "Branch-Compatible Oracle", "Phase B Linear-K2", "C1 Fixed-K2", "C1 Linear-K2"]
    status = {row["Experiment"]: row["Status"] for row in status_rows}
    color_by_status = {"MIXED": "#F59E0B", "FAIL": "#DC2626", "SUPPORTED_MECHANISM": "#0EA5E9", "SUPPORTED": "#059669", "NOT_SUPPORTED": "#DC2626"}
    fig, ax = plt.subplots(figsize=(11.0, 2.8))
    ax.set_xlim(-0.6, len(wanted) - 0.4); ax.set_ylim(-1.1, 1.1); ax.axis("off")
    for idx, name in enumerate(wanted):
        if idx:
            ax.annotate("", xy=(idx - 0.33, 0), xytext=(idx - 0.67, 0), arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1))
        label = name.replace("Counterfactual ", "Counterfactual\n").replace("Risk-Calibrated ", "Risk\n").replace("Safe Selection ", "Safe\n").replace("Stage-Calibrated ", "Stage\n").replace("Field-Decoupled ", "Field\n").replace("Branch-Compatible ", "Branch\n").replace("Phase B ", "Phase B\n").replace("C1 ", "C1\n")
        ax.text(idx, 0, f"{label}\n[{status[name]}]", ha="center", va="center", fontsize=7.2, color="white", bbox=dict(boxstyle="round,pad=0.35", fc=color_by_status.get(status[name], "#6B7280"), ec="none"))
    ax.text(0.5, -0.92, "Evidence progression; boxes are study outcomes, not a monotonic performance curve.", ha="left", fontsize=8, color="#374151")
    fig.tight_layout()
    save(fig, out, "I1_F6_evidence_progression")


def i2_figures() -> None:
    out = I2 / "figures"
    fig, ax = plt.subplots(figsize=(9.2, 2.2))
    ax.axis("off")
    steps = ["MatterGen\nreverse diffusion", "late clean x0", "neural-force\nevaluation", "Cartesian to\nfractional map", "bounded position\ncorrection", "continue\ndiffusion"]
    for idx, label in enumerate(steps):
        x = idx / (len(steps) - 1)
        if idx:
            ax.annotate("", xy=(x - 0.035, 0.5), xytext=((idx - 1) / (len(steps) - 1) + 0.075, 0.5), arrowprops=dict(arrowstyle="->", lw=1.3, color="#475569"))
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.45", fc="#ECFDF5" if idx else "#F3F4F6", ec="#059669" if idx else "#6B7280", lw=1.2))
    ax.text(0.5, 0.08, "Predictor only, t <= 0.02; atomic and cell scores are unchanged", ha="center", fontsize=8, color="#374151")
    fig.tight_layout()
    save(fig, out, "I2_F1_rc_nfgd_pipeline")

    table = {row["Metric"]: row for row in read_csv(I2 / "tables/formal256_main_results.csv")}
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2))
    for ax, metric in zip(axes, ["MaxF", "Mean Force", "RMSD"]):
        row = table[metric]
        values = [float(row["Baseline value"]), float(row["Method value"])]
        bars = ax.bar(["C0", "RC-NFGD"], values, color=[COLORS["C0"], COLORS["RC-NFGD"]])
        ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)
        ax.set_title(metric)
        ax.set_ylabel(row["Unit"])
        ax.set_ylim(0, max(values) * 1.24)
    fig.tight_layout()
    save(fig, out, "I2_F2_formal256_primary_metrics")

    paired = read_csv(I2 / "results/mattersim_late_force_guidance_formal256/paired_results.csv")
    metrics = [("MaxF", "maxF_ev_per_a", "eV/angstrom"), ("Mean Force", "atomic_force_mean_ev_per_a", "eV/angstrom"), ("RMSD", "rmsd_a", "angstrom")]
    fig, axes = plt.subplots(1, 3, figsize=(8.7, 3.0))
    for ax, (label, column, unit) in zip(axes, metrics):
        delta = np.asarray([float(row[f"C0_{column}"]) - float(row[f"F0_{column}"]) for row in paired])
        ax.hist(delta, bins=35, color=COLORS["RC-NFGD"], alpha=0.82)
        ax.axvline(0, color="#991B1B", linestyle="--", linewidth=1)
        ax.axvline(delta.mean(), color="black", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel(f"paired improvement ({unit})")
        ax.set_ylabel("count")
    fig.tight_layout()
    save(fig, out, "I2_F3_paired_improvement_distributions")

    scale = read_csv(I2 / "tables/effect_scale_consistency.csv")
    labels = [row["Cohort"] for row in scale]
    means = np.asarray([float(row["Relative MaxF reduction"]) for row in scale])
    lows = np.asarray([float(row["CI low"]) for row in scale])
    highs = np.asarray([float(row["CI high"]) for row in scale])
    x = np.arange(len(scale))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.errorbar(x, means * 100, yerr=np.vstack([(means - lows) * 100, (highs - means) * 100]), fmt="o", color=COLORS["RC-NFGD"], ecolor="#6EE7B7", capsize=4, markersize=6)
    ax.axhline(0, color="#991B1B", linestyle="--", linewidth=1)
    ax.set_xticks(x, [f"{label}\n(n={scale[idx]['N']})" for idx, label in enumerate(labels)])
    ax.set_ylabel("Relative MaxF reduction (%)")
    ax.text(0.02, 0.03, "Independent cohorts; not a time series", transform=ax.transAxes, fontsize=8, color="#4B5563")
    fig.tight_layout()
    save(fig, out, "I2_F4_effect_consistency")

    independent = read_csv(I2 / "results/chgnet_formal256/independent_per_structure.csv")
    chg: dict[str, dict[int, float]] = {"C0": {}, "F0": {}}
    for row in independent:
        chg[row["method"]][int(row["seed"])] = float(row["maxF_ev_a"])
    matter = {int(row["seed"]): float(row["C0_maxF_ev_per_a"]) - float(row["F0_maxF_ev_per_a"]) for row in paired}
    seeds = sorted(set(matter) & set(chg["C0"]) & set(chg["F0"]))
    x_values = np.asarray([matter[seed] for seed in seeds])
    y_values = np.asarray([chg["C0"][seed] - chg["F0"][seed] for seed in seeds])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(x_values, y_values, s=15, alpha=0.62, color="#0F766E", edgecolors="none")
    ax.axhline(0, color="#9CA3AF", linewidth=0.8); ax.axvline(0, color="#9CA3AF", linewidth=0.8)
    ax.set_xlabel("MatterSim paired MaxF improvement (eV/angstrom)")
    ax.set_ylabel("CHGNet paired MaxF improvement (eV/angstrom)")
    ax.text(0.02, 0.98, f"paired n={len(seeds)}", transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    save(fig, out, "I2_F5_mattersim_vs_chgnet")

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.1))
    for ax, metric in zip(axes, ["Stable", "Validity", "Property MAE"]):
        row = table[metric]
        values = [float(row["Baseline value"]), float(row["Method value"])]
        bars = ax.bar(["C0", "RC-NFGD"], values, color=[COLORS["C0"], COLORS["RC-NFGD"]])
        fmt = "%.6f" if metric == "Property MAE" else "%.3f"
        ax.bar_label(bars, fmt=fmt, fontsize=7, padding=2)
        ax.set_title(metric)
        ax.set_ylim(0, max(values + [0.001]) * 1.23)
        if metric == "Property MAE":
            ax.text(0.5, 0.05, "slight worsening", transform=ax.transAxes, ha="center", color="#991B1B", fontsize=8)
    fig.tight_layout()
    save(fig, out, "I2_F6_quality_guardrails")


def main() -> None:
    i1_figures()
    i2_figures()
    print("FIGURE_REPRODUCTION=PASS")


if __name__ == "__main__":
    main()
