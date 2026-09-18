"""Build the counterfactual dataset and apply the frozen Oracle Headroom Gate."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from ase.io import read
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/counterfactual_adaptive_cfg_v3"
ORACLE = ROOT / "oracle"
PLOTS = ROOT / "plots"
DEADBAND = 0.05
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260917
ACTIONS = ("DOWN", "KEEP", "UP")
TIE_ORDER = {"KEEP": 0, "DOWN": 1, "UP": 2}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def enrich_with_quality(properties: pd.DataFrame) -> pd.DataFrame:
    values = properties.copy()
    for column in (
        "e_hull", "stable", "nus", "novel", "unique", "rmsd",
        "pre_relaxation_max_force", "relaxation_steps", "quality_success",
    ):
        values[column] = np.nan if column != "quality_success" else False
    for kind in ("SHORT", "FULL"):
        for action in ACTIONS:
            group = f"{kind}_{action}"
            structure_path = ORACLE / "evaluation_structures" / f"{group}.extxyz"
            quality_root = ORACLE / "quality" / group
            if not structure_path.exists() or not (quality_root / "official_detailed.json.gz").exists():
                continue
            atoms = read(structure_path, index=":")
            per_structure = pd.read_csv(quality_root / "per_structure.csv")
            with gzip.open(quality_root / "official_detailed.json.gz", "rt") as stream:
                detailed = json.load(stream)
            if len(per_structure) != len(detailed["energy_above_hull_per_atom"]):
                raise RuntimeError(f"quality length mismatch: {group}")
            for output_index, row in per_structure.reset_index(drop=True).iterrows():
                input_index = int(row["index"])
                identifier = str(atoms[input_index].info["branch_id"])
                selected = values.index[values["branch_id"] == identifier]
                if len(selected) != 1:
                    raise RuntimeError(f"branch id lookup failed: {identifier}")
                target = selected[0]
                values.loc[target, "e_hull"] = float(detailed["energy_above_hull_per_atom"][output_index])
                values.loc[target, "stable"] = bool(detailed["stable"][output_index])
                values.loc[target, "nus"] = bool(detailed["novel_unique_stable"][output_index])
                values.loc[target, "novel"] = bool(detailed["novel"][output_index])
                values.loc[target, "unique"] = bool(detailed["unique"][output_index])
                values.loc[target, "rmsd"] = float(detailed["rmsd_from_relaxation"][output_index])
                values.loc[target, "pre_relaxation_max_force"] = float(row["pre_relaxation_max_force"])
                values.loc[target, "relaxation_steps"] = int(row["relaxation_steps"])
                values.loc[target, "quality_success"] = True
    for column in ("stable", "nus", "novel", "unique"):
        values[column] = values[column].fillna(False).astype(bool)
    values["evaluation_valid"] = values["structure_valid"].astype(bool) & values["quality_success"].astype(bool)
    return values


def labels_for(frame: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_kind = frame[frame["outcome_kind"] == kind].copy()
    if selected_kind.empty:
        raise RuntimeError(f"no {kind} counterfactual rows")
    state_rows: list[dict] = []
    annotated: list[pd.DataFrame] = []
    for (seed, step), state in selected_kind.groupby(["seed", "sampling_step"], sort=True):
        if set(state["candidate_action"]) != set(ACTIONS) or len(state) != 3:
            raise RuntimeError(f"candidate set mismatch: {kind}/{seed}/{step}")
        base = state[state["candidate_action"] == "KEEP"].iloc[0]
        ranked = state.assign(
            tie_order=state["candidate_action"].map(TIE_ORDER)
        ).sort_values(["oracle_objective", "tie_order"])
        raw_best = ranked.iloc[0]
        base_objective = float(base["oracle_objective"])
        raw_gain = (base_objective - float(raw_best["oracle_objective"])) / max(abs(base_objective), 1e-12)
        label = str(raw_best["candidate_action"]) if raw_gain >= DEADBAND else "KEEP"
        chosen = state[state["candidate_action"] == label].iloc[0]
        chosen_gain = (base_objective - float(chosen["oracle_objective"])) / max(abs(base_objective), 1e-12)
        state_rows.append(
            {
                "outcome_kind": kind,
                "seed": int(seed),
                "sampling_step": int(step),
                "decision_ordinal": int(base["decision_ordinal"]),
                "decision_t_norm": float(base["decision_t_norm"]),
                "oracle_label": label,
                "cfg2_objective": base_objective,
                "oracle_objective": float(chosen["oracle_objective"]),
                "oracle_relative_gain": chosen_gain,
                "raw_best_action": str(raw_best["candidate_action"]),
                "raw_best_relative_gain": raw_gain,
            }
        )
        state = state.copy()
        state["oracle_label"] = label
        state["oracle_relative_gain"] = chosen_gain
        for metric in (
            "property_absolute_error", "oracle_objective", "e_hull", "stable",
            "nus", "novel", "unique", "evaluation_valid",
        ):
            base_value = float(base[metric]) if pd.notna(base[metric]) else np.nan
            state[f"delta_{metric}_vs_cfg2"] = pd.to_numeric(state[metric], errors="coerce") - base_value
        annotated.append(state)
    return pd.DataFrame(state_rows), pd.concat(annotated, ignore_index=True)


def chosen_rows(frame: pd.DataFrame, labels: pd.DataFrame, label_column: str = "oracle_label") -> pd.DataFrame:
    merged = frame.merge(
        labels[["seed", "sampling_step", label_column]],
        on=["seed", "sampling_step"],
        how="inner",
    )
    return merged[merged["candidate_action"] == merged[label_column]].copy()


def metric_summary(name: str, base: pd.DataFrame, selected: pd.DataFrame) -> dict:
    def mean(column: str, values: pd.DataFrame) -> float:
        return float(pd.to_numeric(values[column], errors="coerce").mean())

    base_objective = mean("oracle_objective", base)
    selected_objective = mean("oracle_objective", selected)
    return {
        "selection": name,
        "n": len(base),
        "cfg2_property_mae": mean("property_absolute_error", base),
        "selected_property_mae": mean("property_absolute_error", selected),
        "cfg2_normalized_objective": base_objective,
        "selected_normalized_objective": selected_objective,
        "property_objective_improvement_fraction": (
            (base_objective - selected_objective) / max(abs(base_objective), 1e-12)
        ),
        "cfg2_e_hull": mean("e_hull", base),
        "selected_e_hull": mean("e_hull", selected),
        "cfg2_stable": mean("stable", base),
        "selected_stable": mean("stable", selected),
        "cfg2_nus": mean("nus", base),
        "selected_nus": mean("nus", selected),
        "cfg2_validity": mean("evaluation_valid", base),
        "selected_validity": mean("evaluation_valid", selected),
    }


def action_distribution(labels: pd.DataFrame, label_type: str) -> list[dict]:
    rows: list[dict] = []

    def add(stratum: str, key: str, values: pd.Series) -> None:
        counts = values.value_counts()
        n = len(values)
        rows.append(
            {
                "label_type": label_type,
                "stratum": stratum,
                "key": key,
                "n": n,
                **{f"{action.lower()}_fraction": float(counts.get(action, 0) / n) for action in ACTIONS},
            }
        )

    add("overall", "all", labels["oracle_label"])
    for step, values in labels.groupby("sampling_step"):
        add("timestep", str(int(step)), values["oracle_label"])
    for seed, values in labels.groupby("seed"):
        add("sample", str(int(seed)), values["oracle_label"])
    stages = {0: "early", 1: "early", 2: "early_middle", 3: "middle", 4: "middle", 5: "middle_late", 6: "late", 7: "late"}
    staged = labels.assign(stage=labels["decision_ordinal"].map(stages))
    for stage, values in staged.groupby("stage"):
        add("stage", str(stage), values["oracle_label"])
    return rows


def cluster_bootstrap(base: pd.DataFrame, selected: pd.DataFrame) -> dict:
    paired = base[["seed", "sampling_step", "oracle_objective"]].rename(
        columns={"oracle_objective": "base"}
    ).merge(
        selected[["seed", "sampling_step", "oracle_objective"]].rename(
            columns={"oracle_objective": "selected"}
        ),
        on=["seed", "sampling_step"],
        how="inner",
    )
    by_seed = paired.assign(delta=paired["selected"] - paired["base"]).groupby("seed")["delta"].mean()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    array = by_seed.to_numpy(dtype=float)
    draws = rng.choice(array, size=(BOOTSTRAP_RESAMPLES, len(array)), replace=True).mean(axis=1)
    return {
        "metric": "selected_minus_cfg2_normalized_property_objective",
        "cluster_unit": "seed",
        "n_seeds": len(array),
        "n_states": len(paired),
        "mean_delta": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def make_plots(short_labels: pd.DataFrame, full_labels: pd.DataFrame, full_base: pd.DataFrame, full_selected: pd.DataFrame) -> None:
    PLOTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(short_labels["oracle_relative_gain"], bins=30, alpha=0.65, label="short all states")
    ax.hist(full_labels["oracle_relative_gain"], bins=18, alpha=0.65, label="full subset")
    ax.axvline(DEADBAND, color="black", linestyle="--", label="5% deadband")
    ax.set(xlabel="Oracle relative objective gain vs CFG2", ylabel="states")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "oracle_gain_distribution.png", dpi=180)
    plt.close(fig)

    pivot = short_labels.pivot_table(
        index="sampling_step", columns="oracle_label", values="seed", aggfunc="count", fill_value=0
    ).reindex(columns=ACTIONS, fill_value=0)
    fraction = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stackplot(fraction.index, *[fraction[action] for action in ACTIONS], labels=ACTIONS)
    ax.set(xlabel="Reverse sampling index", ylabel="Action fraction", ylim=(0, 1))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "oracle_action_by_timestep.png", dpi=180)
    plt.close(fig)

    encoded = short_labels.assign(code=short_labels["oracle_label"].map({"DOWN": -1, "KEEP": 0, "UP": 1})).pivot(
        index="seed", columns="sampling_step", values="code"
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(encoded.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(encoded.columns)), labels=encoded.columns, rotation=45)
    ax.set_yticks(range(len(encoded.index)), labels=encoded.index, fontsize=6)
    ax.set(xlabel="Reverse sampling index", ylabel="Oracle seed")
    colorbar = fig.colorbar(image, ax=ax, ticks=[-1, 0, 1])
    colorbar.ax.set_yticklabels(["DOWN", "KEEP", "UP"])
    fig.tight_layout()
    fig.savefig(PLOTS / "oracle_action_by_sample.png", dpi=180)
    plt.close(fig)

    paired = full_base[["seed", "sampling_step", "oracle_objective"]].merge(
        full_selected[["seed", "sampling_step", "oracle_objective"]],
        on=["seed", "sampling_step"], suffixes=("_cfg2", "_oracle")
    )
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(paired["oracle_objective_cfg2"], paired["oracle_objective_oracle"], alpha=0.7)
    limit = float(max(paired["oracle_objective_cfg2"].max(), paired["oracle_objective_oracle"].max()))
    ax.plot([0, limit], [0, limit], "k--")
    ax.set(xlabel="CFG2 normalized property objective", ylabel="Full oracle objective")
    fig.tight_layout()
    fig.savefig(PLOTS / "cfg2_vs_oracle.png", dpi=180)
    plt.close(fig)


def main() -> None:
    properties = pd.read_csv(ORACLE / "property_metrics.csv")
    enriched = enrich_with_quality(properties)
    enriched.to_csv(ORACLE / "all_outcome_metrics.csv", index=False)
    short_labels, short_candidates = labels_for(enriched, "SHORT")
    full_labels, full_candidates = labels_for(enriched, "FULL")
    short_labels.to_csv(ORACLE / "short_oracle_labels.csv", index=False)
    full_labels.to_csv(ORACLE / "full_oracle_labels.csv", index=False)

    full_key = ["seed", "sampling_step", "candidate_action"]
    full_metrics = full_candidates.copy()
    short_dataset = short_candidates.merge(
        full_metrics[
            full_key + [
                "property_absolute_error", "oracle_objective", "e_hull", "stable", "nus",
                "novel", "unique", "evaluation_valid",
            ]
        ],
        on=full_key,
        how="left",
        suffixes=("_short", "_final"),
    ).merge(
        full_labels[["seed", "sampling_step", "oracle_label", "oracle_relative_gain"]].rename(
            columns={"oracle_label": "full_oracle_label", "oracle_relative_gain": "full_oracle_relative_gain"}
        ),
        on=["seed", "sampling_step"],
        how="left",
    )
    short_dataset.to_csv(ORACLE / "oracle_dataset.csv", index=False)

    short_base = short_candidates[short_candidates["candidate_action"] == "KEEP"]
    short_selected = chosen_rows(short_candidates, short_labels)
    full_base = full_candidates[full_candidates["candidate_action"] == "KEEP"]
    full_selected = chosen_rows(full_candidates, full_labels)
    short_choices = short_labels[["seed", "sampling_step", "oracle_label"]].rename(
        columns={"oracle_label": "short_label"}
    )
    full_with_short = full_candidates.merge(short_choices, on=["seed", "sampling_step"], how="inner")
    short_selected_full = full_with_short[
        full_with_short["candidate_action"] == full_with_short["short_label"]
    ].copy()
    summaries = [
        metric_summary("short_oracle_vs_cfg2", short_base, short_selected),
        metric_summary("full_oracle_vs_cfg2", full_base, full_selected),
        metric_summary("short_selected_full_vs_cfg2", full_base, short_selected_full),
    ]
    write_csv(ORACLE / "oracle_headroom.csv", summaries)
    distribution_rows = action_distribution(short_labels, "short")
    distribution_rows += action_distribution(full_labels, "full")
    write_csv(ORACLE / "action_distribution.csv", distribution_rows)

    fidelity = short_labels.merge(
        full_labels,
        on=["seed", "sampling_step"],
        suffixes=("_short", "_full"),
    )
    exact_agreement = float((fidelity["oracle_label_short"] == fidelity["oracle_label_full"]).mean())
    short_summary, full_summary, short_full_summary = summaries
    full_counts = full_labels["oracle_label"].value_counts(normalize=True)
    short_counts = short_labels["oracle_label"].value_counts(normalize=True)
    full_property_improvement = float(full_summary["property_objective_improvement_fraction"])
    short_property_improvement = float(short_summary["property_objective_improvement_fraction"])
    short_selected_full_improvement = float(short_full_summary["property_objective_improvement_fraction"])
    guardrails = {
        "mean_ehull_worsening_le_0.02": (
            full_summary["selected_e_hull"] - full_summary["cfg2_e_hull"] <= 0.02
        ),
        "stable_drop_le_5pp": (
            full_summary["cfg2_stable"] - full_summary["selected_stable"] <= 0.05
        ),
        "nus_drop_le_5pp": (
            full_summary["cfg2_nus"] - full_summary["selected_nus"] <= 0.05
        ),
        "validity_drop_le_5pp": (
            full_summary["cfg2_validity"] - full_summary["selected_validity"] <= 0.05
        ),
    }
    checks = {
        "short_property_improvement_ge_10pct": short_property_improvement >= 0.10,
        "full_property_improvement_ge_10pct": full_property_improvement >= 0.10,
        "short_non_keep_ge_20pct": 1.0 - float(short_counts.get("KEEP", 0.0)) >= 0.20,
        "full_non_keep_ge_20pct": 1.0 - float(full_counts.get("KEEP", 0.0)) >= 0.20,
        "full_max_action_share_le_80pct": float(full_counts.max()) <= 0.80,
        "full_down_share_ge_5pct": float(full_counts.get("DOWN", 0.0)) >= 0.05,
        "full_up_share_ge_5pct": float(full_counts.get("UP", 0.0)) >= 0.05,
        "short_full_label_agreement_ge_50pct": exact_agreement >= 0.50,
        "short_selected_full_improvement_ge_5pct": short_selected_full_improvement >= 0.05,
        **guardrails,
    }
    decision = "GO" if all(checks.values()) else "FAIL"

    def harm(selected: pd.DataFrame, base: pd.DataFrame) -> dict:
        paired = base[[
            "seed", "sampling_step", "property_absolute_error", "e_hull", "stable", "evaluation_valid"
        ]].merge(
            selected[[
                "seed", "sampling_step", "property_absolute_error", "e_hull", "stable", "evaluation_valid"
            ]],
            on=["seed", "sampling_step"], suffixes=("_cfg2", "_selected")
        )
        return {
            "property_harm_rate": float((
                paired["property_absolute_error_selected"]
                > paired["property_absolute_error_cfg2"] * 1.05
            ).mean()),
            "ehull_harm_rate": float((
                paired["e_hull_selected"] > paired["e_hull_cfg2"] + 0.02
            ).fillna(False).mean()),
            "stable_harm_rate": float((
                paired["stable_cfg2"].astype(bool) & ~paired["stable_selected"].astype(bool)
            ).mean()),
            "validity_harm_rate": float((
                paired["evaluation_valid_cfg2"].astype(bool)
                & ~paired["evaluation_valid_selected"].astype(bool)
            ).mean()),
        }

    bootstrap = cluster_bootstrap(full_base, full_selected)
    (ORACLE / "oracle_bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    summary = {
        "ORACLE_HEADROOM": decision,
        "next": "FEATURE_AUDIT" if decision == "GO" else "STOP_ADAPTIVE_CFG_RESEARCH",
        "n_oracle_seeds": int(short_labels["seed"].nunique()),
        "n_short_states": len(short_labels),
        "n_full_states": len(full_labels),
        "n_short_candidate_outcomes": len(short_candidates),
        "n_full_candidate_outcomes": len(full_candidates),
        "short_property_objective_improvement_fraction": short_property_improvement,
        "full_property_objective_improvement_fraction": full_property_improvement,
        "short_selected_full_property_improvement_fraction": short_selected_full_improvement,
        "short_full_exact_label_agreement": exact_agreement,
        "short_action_distribution": {action: float(short_counts.get(action, 0.0)) for action in ACTIONS},
        "full_action_distribution": {action: float(full_counts.get(action, 0.0)) for action in ACTIONS},
        "full_oracle_harm": harm(full_selected, full_base),
        "short_selected_full_harm": harm(short_selected_full, full_base),
        "gate_checks": checks,
        "guardrails": guardrails,
        "bootstrap": bootstrap,
        "parameters_retuned_after_results": False,
        "surrogate_property_eval": True,
        "dft_verified": False,
    }
    (ORACLE / "oracle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plots(short_labels, full_labels, full_base, full_selected)
    report = f"""# Counterfactual Adaptive CFG V3 — Oracle Diagnostic

## Frozen design

32 fresh calibration-only seeds; CFG candidates 1.75/2.00/2.25; eight pre-registered reverse timesteps; H=15; exactly 25% of states receive full-horizon validation. The scalar objective is normalized CHGNet magnetic-density target error plus a frozen invalid penalty. A 5% deadband selects KEEP.

## Headroom result

```text
ORACLE_HEADROOM = {decision}
NEXT = {summary['next']}
```

- Short-state Oracle objective improvement: {short_property_improvement:.6%}
- Full-outcome Oracle objective improvement: {full_property_improvement:.6%}
- Full improvement obtained by short labels: {short_selected_full_improvement:.6%}
- Short/full exact action agreement: {exact_agreement:.6%}
- Short action distribution: {summary['short_action_distribution']}
- Full action distribution: {summary['full_action_distribution']}
- Full Oracle harm: {summary['full_oracle_harm']}
- Gate checks: {checks}

## Scientific boundary

This is a diagnostic oracle, not a deployable controller. Candidate selection is retrospective. All property and energy/stability values are frozen surrogate evaluations (`SURROGATE_PROPERTY_EVAL=True`, `DFT_VERIFIED=False`).
"""
    (ORACLE / "final_report.md").write_text(report)
    if decision == "FAIL":
        not_run = {
            "ORACLE_PREDICTABILITY": "NOT_RUN",
            "V3_P0": "NOT_RUN",
            "V3_FORMAL256": "NOT_RUN",
            "reason": "ORACLE_HEADROOM=FAIL; frozen stop rule applied",
            "rescue_tuning": False,
        }
        (ROOT / "controller/decision_summary.json").write_text(json.dumps(not_run, indent=2) + "\n")
        (ROOT / "p0/decision_summary.json").write_text(json.dumps(not_run, indent=2) + "\n")
        final = {
            "ORACLE_HEADROOM": "FAIL",
            "ORACLE_PREDICTABILITY": "NOT_RUN",
            "V3_P0": "NOT_RUN",
            "V3_FORMAL256": "NOT_RUN",
            "INNOVATION1_FINAL_STATUS": "MIXED",
            "STOP_ADAPTIVE_CFG_RESEARCH": True,
            "parameters_retuned_after_results": False,
            "all_data_retained": True,
            "surrogate_property_eval": True,
            "dft_verified": False,
        }
        (ROOT / "final_status.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
