"""Frozen terminal selection, continuation gate, and confirmatory inference."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Mapping

from ase.io import read, write
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]; ROOT = PROJECT / "experiments/frozen_linear_k2_confirmation"; PROTOCOL_ROOT = ROOT / "protocol"; PLOTS = ROOT / "plots"
METHODS = ("C0", "Fixed_K2", "Random_K2", "Linear_K2"); N_BOOT = 20_000; BOOT_SEED = 2026091701; EPS = 1e-12; PROPERTY_TOL = 1e-6


def json_write(path: Path, payload: object) -> None: path.write_text(json.dumps(payload, indent=2, default=float) + "\n")


def branch_quality(phase: str) -> pd.DataFrame:
    phase_root = ROOT / phase; values = pd.read_csv(phase_root / "property_metrics_raw.csv")
    for column, default in (("e_hull", np.nan), ("stable", False), ("nus", False), ("novel", False), ("unique", False), ("quality_success", False), ("max_force", np.nan), ("rmsd", np.nan)): values[column] = default
    for structure_path in sorted((phase_root / "evaluation_structures").glob("*.extxyz")):
        group = structure_path.stem; quality_root = phase_root / "quality_branches" / group
        if not (quality_root / "official_detailed.json.gz").exists(): raise FileNotFoundError(f"missing branch quality {phase}/{group}")
        atoms = read(structure_path, index=":"); per = pd.read_csv(quality_root / "per_structure.csv")
        with gzip.open(quality_root / "official_detailed.json.gz", "rt") as stream: detailed = json.load(stream)
        if len(per) != len(detailed["energy_above_hull_per_atom"]): raise RuntimeError(f"quality length mismatch {group}")
        for output_index, row in per.reset_index(drop=True).iterrows():
            identifier = str(atoms[int(row["index"])].info["branch_id"]); match = values.index[values.branch_id == identifier]
            if len(match) != 1: raise RuntimeError(f"branch lookup failed {identifier}")
            target = match[0]; values.loc[target, "e_hull"] = float(detailed["energy_above_hull_per_atom"][output_index]); values.loc[target, "stable"] = bool(detailed["stable"][output_index]); values.loc[target, "nus"] = bool(detailed["novel_unique_stable"][output_index]); values.loc[target, "novel"] = bool(detailed["novel"][output_index]); values.loc[target, "unique"] = bool(detailed["unique"][output_index]); values.loc[target, "rmsd"] = float(detailed["rmsd_from_relaxation"][output_index]); values.loc[target, "max_force"] = float(row["pre_relaxation_max_force"]); values.loc[target, "quality_success"] = True
    for column in ("structure_valid", "stable", "nus", "novel", "unique", "quality_success"): values[column] = values[column].fillna(False).astype(bool)
    values["evaluation_valid"] = values.structure_valid & values.quality_success; values.to_csv(phase_root / "evaluated_branches.csv", index=False); return values


def select_outcomes(phase: str, values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = values[values.method == "C0"].set_index("seed")
    if len(base) != values.seed.nunique(): raise RuntimeError("C0 incomplete")
    selected_rows, candidate_rows = [], []
    for seed in sorted(map(int, values.seed.unique())):
        c0 = base.loc[seed]
        selected_rows.append({**c0.to_dict(), "seed": seed, "result_method": "C0", "terminal_policy": "C0", "safe_candidate_count": 0, "fallback_to_c0": True})
        for method in METHODS[1:]:
            candidates = values[(values.seed == seed) & (values.method == method)].copy()
            if len(candidates) != 2: raise RuntimeError(f"{seed}/{method} candidate count != 2")
            candidates["property_gain_vs_c0"] = float(c0.property_absolute_error) - candidates.property_absolute_error
            candidates["safe_a"] = (
                (candidates.property_gain_vs_c0 > PROPERTY_TOL)
                & (candidates.evaluation_valid.astype(int) >= int(bool(c0.evaluation_valid)))
                & (candidates.stable.astype(int) >= int(bool(c0.stable)))
                & np.isfinite(candidates.e_hull) & np.isfinite(float(c0.e_hull))
                & (candidates.e_hull <= float(c0.e_hull) + 0.01 + EPS)
            )
            candidate_rows.extend(candidates.to_dict("records")); eligible = candidates[candidates.safe_a]
            if eligible.empty:
                chosen = c0; terminal = "C0"; fallback = True
            else:
                chosen = eligible.sort_values(["property_absolute_error", "policy_id", "allocation_rank"]).iloc[0]; terminal = str(chosen.policy_id); fallback = False
            selected_rows.append({**chosen.to_dict(), "seed": seed, "result_method": method, "terminal_policy": terminal, "safe_candidate_count": int(candidates.safe_a.sum()), "fallback_to_c0": fallback})
    return pd.DataFrame(selected_rows), pd.DataFrame(candidate_rows)


def provisional_metrics(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        frame = outcomes[outcomes.result_method == method]
        rows.append({"method": method, "n": len(frame), "property_mae": float(frame.property_absolute_error.mean()), "e_hull": float(frame.e_hull.mean()), "stable": float(frame.stable.mean()), "novel": float(frame.novel.mean()), "unique": float(frame.unique.mean()), "nus": float(frame.nus.mean()), "validity": float(frame.evaluation_valid.mean()), "max_force": float(frame.max_force.mean()), "rmsd": float(frame.rmsd.mean()), "acceptance_rate": float((~frame.fallback_to_c0).mean()), "fallback_rate": float(frame.fallback_to_c0.mean())})
    return pd.DataFrame(rows)


def write_mixed(phase: str, outcomes: pd.DataFrame) -> None:
    phase_root = ROOT / phase; destination = phase_root / "mixed_evaluation_structures"; destination.mkdir(exist_ok=False)
    atoms = read(phase_root / "deployment_final_all.extxyz", index=":"); lookup = {str(atom.info["branch_id"]): atom for atom in atoms}
    c0_lookup = {int(atom.info["seed"]): atom for atom in atoms if str(atom.info["method"]) == "C0"}
    for method in METHODS:
        selected = []
        for _, row in outcomes[outcomes.result_method == method].sort_values("seed").iterrows():
            if not bool(row.structure_valid): continue
            atom = c0_lookup[int(row.seed)].copy() if str(row.terminal_policy) == "C0" else lookup[str(row.branch_id)].copy()
            atom.info.update(sample_seed=int(row.seed), selected_method=method, terminal_policy=str(row.terminal_policy), branch_id=f"{method}:{int(row.seed)}:{row.terminal_policy}"); selected.append(atom)
        if selected: write(destination / f"{method}.extxyz", selected)


def selection_stage(phase: str) -> None:
    phase_root = ROOT / phase
    if (phase_root / "selected_outcomes.csv").exists(): raise FileExistsError("selection already frozen")
    values = branch_quality(phase); outcomes, candidates = select_outcomes(phase, values); outcomes.to_csv(phase_root / "selected_outcomes.csv", index=False); candidates.to_csv(phase_root / "selected_candidates.csv", index=False); provisional_metrics(outcomes).to_csv(phase_root / "metrics_provisional.csv", index=False)
    paired = outcomes.pivot(index="seed", columns="result_method", values="property_absolute_error").reset_index(); paired["linear_vs_fixed_gain"] = paired.Fixed_K2 - paired.Linear_K2; paired["linear_vs_random_gain"] = paired.Random_K2 - paired.Linear_K2; paired["linear_vs_c0_gain"] = paired.C0 - paired.Linear_K2; paired.to_csv(phase_root / "paired_results.csv", index=False)
    summaries = [json.loads(path.read_text()) for path in sorted((phase_root / "generation").glob("*/run_summary.json"))]
    pd.DataFrame([
        {"method": "C0", "candidate_count": 0, "mattergen_score_calls_per_seed": 2000, "compute_multiplier": 1.0, "chgnet_calls_per_seed": 1, "mattersim_selector_calls_per_seed": 1},
        *[{"method": method, "candidate_count": 2, "mattergen_score_calls_per_seed": 4400, "compute_multiplier": 2.2, "chgnet_calls_per_seed": 3, "mattersim_selector_calls_per_seed": 3} for method in METHODS[1:]],
    ]).to_csv(phase_root / "compute_accounting.csv", index=False)
    json_write(phase_root / "acquisition_accounting.json", {"total_gpu_seconds": sum(row["elapsed_seconds"] for row in summaries), "total_acquisition_score_calls": sum(row["mattergen_score_calls_acquisition"] for row in summaries), "mean_wall_seconds_per_seed": float(np.mean([row["elapsed_seconds"] for row in summaries])), "study_acquisition_multiplier_per_seed": 5.6, "reference_reproduction_all_pass": all(row["reference_reproduction"]["success"] for row in summaries)})
    write_mixed(phase, outcomes)


def mixed_quality(phase: str, method: str, expected_n: int) -> dict[str, float]:
    phase_root = ROOT / phase; structure_path = phase_root / "mixed_evaluation_structures" / f"{method}.extxyz"; quality_root = phase_root / "quality_mixed" / method
    atoms = read(structure_path, index=":"); per = pd.read_csv(quality_root / "per_structure.csv")
    with gzip.open(quality_root / "official_detailed.json.gz", "rt") as stream: detailed = json.load(stream)
    if len(per) != len(atoms): raise RuntimeError(f"mixed quality mismatch {phase}/{method}")
    return {"e_hull": float(np.mean(detailed["energy_above_hull_per_atom"])), "stable": float(np.mean(detailed["stable"])), "novel": float(np.mean(detailed["novel"])), "unique": float(np.mean(detailed["unique"])), "nus": float(np.mean(detailed["novel_unique_stable"])), "validity": float(len(atoms) / expected_n), "max_force": float(per.pre_relaxation_max_force.mean()), "rmsd": float(np.mean(detailed["rmsd_from_relaxation"]))}


def bootstrap(values: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, float); rng = np.random.default_rng(seed); draws = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "ci95_low": float(np.quantile(draws, .025)), "ci95_high": float(np.quantile(draws, .975)), "wins": int((values > EPS).sum()), "ties": int((np.abs(values) <= EPS).sum()), "losses": int((values < -EPS).sum()), "resamples": N_BOOT}


def final_metrics(phase: str, expected_n: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    phase_root = ROOT / phase; outcomes = pd.read_csv(phase_root / "selected_outcomes.csv"); metrics = provisional_metrics(outcomes).set_index("method")
    for method in METHODS:
        for key, value in mixed_quality(phase, method, expected_n).items(): metrics.loc[method, key] = value
    metrics = metrics.reset_index(); paired = pd.read_csv(phase_root / "paired_results.csv"); boot = {"linear_vs_fixed": bootstrap(paired.linear_vs_fixed_gain.to_numpy(float), BOOT_SEED), "linear_vs_random": bootstrap(paired.linear_vs_random_gain.to_numpy(float), BOOT_SEED + 1), "linear_vs_c0": bootstrap(paired.linear_vs_c0_gain.to_numpy(float), BOOT_SEED + 2)}
    metrics.to_csv(phase_root / "metrics.csv", index=False); json_write(phase_root / "bootstrap_20k.json", boot); return metrics, outcomes, boot


def quality_gate(metrics: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    values = metrics.set_index("method"); linear, fixed = values.loc["Linear_K2"], values.loc["Fixed_K2"]
    checks = {"e_hull_worsening": float(linear.e_hull - fixed.e_hull), "stable_drop": float(fixed.stable - linear.stable), "nus_drop": float(fixed.nus - linear.nus), "validity_drop": float(fixed.validity - linear.validity)}
    passed = checks["e_hull_worsening"] <= .01 + EPS and checks["stable_drop"] <= .05 + EPS and checks["nus_drop"] <= .05 + EPS and checks["validity_drop"] <= .05 + EPS
    return passed, {**checks, "thresholds": {"e_hull": .01, "stable": .05, "nus": .05, "validity": .05}, "pass": passed}


def candidate_frequencies(outcomes: pd.DataFrame) -> dict[str, Any]:
    linear = outcomes[outcomes.result_method == "Linear_K2"]; accepted = linear[~linear.fallback_to_c0]
    return {"acceptance_rate": float((~linear.fallback_to_c0).mean()), "fallback_rate": float(linear.fallback_to_c0.mean()), "terminal_policy_rates": {policy: float((linear.terminal_policy == policy).mean()) for policy in ("C0", "GPulse", "APulse", "PPulse", "CPulse")}}


def report_phase(phase: str, metrics: pd.DataFrame, boot: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    table = metrics.set_index("method"); lines = [f"# {phase} confirmation report", "", "Surrogate-only; DFT_VERIFIED=false.", ""]
    for key, value in decision.items(): lines.append(f"{key} = {value}")
    lines.extend(["", "| Method | MAE | E-hull | Stable | NUS | Validity | Compute x |", "|---|---:|---:|---:|---:|---:|---:|"])
    for method in METHODS: lines.append(f"| {method} | {table.loc[method].property_mae:.8f} | {table.loc[method].e_hull:.6f} | {table.loc[method].stable:.4f} | {table.loc[method].nus:.4f} | {table.loc[method].validity:.4f} | {1.0 if method == 'C0' else 2.2:.1f} |")
    lines.extend(["", f"Linear-vs-Fixed paired gain CI: [{boot['linear_vs_fixed']['ci95_low']:.8f}, {boot['linear_vs_fixed']['ci95_high']:.8f}]"])
    (ROOT / phase / "final_report.md").write_text("\n".join(lines) + "\n")


def finalize_cohort(phase: str) -> dict[str, Any]:
    expected_n = 128; metrics, outcomes, boot = final_metrics(phase, expected_n); values = metrics.set_index("method"); linear, fixed, random = values.loc["Linear_K2"], values.loc["Fixed_K2"], values.loc["Random_K2"]
    relative = float((fixed.property_mae - linear.property_mae) / max(float(fixed.property_mae), EPS)); quality_pass, quality = quality_gate(metrics); json_write(ROOT / phase / "quality_guardrails.json", quality); frequencies = candidate_frequencies(outcomes); json_write(ROOT / phase / "acceptance_frequencies.json", frequencies)
    if phase == "c1_128":
        direction = bool(linear.property_mae < fixed.property_mae - EPS); floor = bool(relative >= .01 - EPS); random_pass = bool(linear.property_mae <= random.property_mae + EPS); continuation = direction and floor and random_pass and quality_pass
        decision = {"C1_CONFIRMATION": "CONTINUE" if continuation else "FAIL", "C2_RUN": "YES" if continuation else "NO", "N_CONFIRMATORY": 128, "LINEAR_K2_MAE": float(linear.property_mae), "FIXED_K2_MAE": float(fixed.property_mae), "RANDOM_K2_MAE": float(random.property_mae), "C0_MAE": float(values.loc['C0'].property_mae), "LINEAR_VS_FIXED_RELATIVE_GAIN": relative, "LINEAR_VS_FIXED_BOOTSTRAP_CI": [boot['linear_vs_fixed']['ci95_low'], boot['linear_vs_fixed']['ci95_high']], "QUALITY_GUARDRAILS": "PASS" if quality_pass else "FAIL", "COMPUTE_MULTIPLIER_LINEAR": 2.2, "COMPUTE_MULTIPLIER_FIXED": 2.2, "continuation_checks": {"direction": direction, "effect_floor_1pct": floor, "random": random_pass, "quality": quality_pass}, "FROZEN_LINEAR_K2_CONFIRMATION": "PENDING" if continuation else "NOT_SUPPORTED", "INNOVATION1_FINAL_STATUS": "MIXED", "DFT_VERIFIED": False}
        json_write(ROOT / phase / "continuation_decision.json", decision)
    else:
        decision = {"C2_COMPLETE": True, "N": 128, "LINEAR_VS_FIXED_RELATIVE_GAIN": relative, "QUALITY_GUARDRAILS": "PASS" if quality_pass else "FAIL", "DFT_VERIFIED": False}
        json_write(ROOT / phase / "decision_summary.json", decision)
    report_phase(phase, metrics, boot, decision); make_basic_plots(phase, pd.read_csv(ROOT / phase / "paired_results.csv"), metrics); return decision


def pool_stage() -> None:
    destination = ROOT / "pooled256"; destination.mkdir(exist_ok=False); outcomes = pd.concat([pd.read_csv(ROOT / phase / "selected_outcomes.csv") for phase in ("c1_128", "c2_128")], ignore_index=True); outcomes.to_csv(destination / "selected_outcomes.csv", index=False)
    paired = outcomes.pivot(index="seed", columns="result_method", values="property_absolute_error").reset_index(); paired["linear_vs_fixed_gain"] = paired.Fixed_K2 - paired.Linear_K2; paired["linear_vs_random_gain"] = paired.Random_K2 - paired.Linear_K2; paired["linear_vs_c0_gain"] = paired.C0 - paired.Linear_K2; paired.to_csv(destination / "paired_results.csv", index=False)
    mixed = destination / "mixed_evaluation_structures"; mixed.mkdir()
    for method in METHODS:
        atoms = []
        for phase in ("c1_128", "c2_128"): atoms.extend(read(ROOT / phase / "mixed_evaluation_structures" / f"{method}.extxyz", index=":"))
        write(mixed / f"{method}.extxyz", atoms)
    predictions = pd.concat([pd.read_csv(ROOT / phase / "allocator_predictions.csv") for phase in ("c1_128", "c2_128")], ignore_index=True); predictions.to_csv(destination / "allocator_predictions.csv", index=False)


def final_pooled() -> None:
    phase = "pooled256"; metrics, outcomes, boot = final_metrics(phase, 256); values = metrics.set_index("method"); linear, fixed, random = values.loc["Linear_K2"], values.loc["Fixed_K2"], values.loc["Random_K2"]; relative = float((fixed.property_mae - linear.property_mae) / max(float(fixed.property_mae), EPS)); quality_pass, quality = quality_gate(metrics); json_write(ROOT / phase / "quality_guardrails.json", quality)
    c1 = pd.read_csv(ROOT / "c1_128/paired_results.csv"); c2 = pd.read_csv(ROOT / "c2_128/paired_results.csv"); consistency = {"c1_mean_gain": float(c1.linear_vs_fixed_gain.mean()), "c2_mean_gain": float(c2.linear_vs_fixed_gain.mean()), "pooled_mean_gain": float(pd.read_csv(ROOT / phase / "paired_results.csv").linear_vs_fixed_gain.mean())}; consistency["status"] = "PASS" if consistency["c1_mean_gain"] > 0 and consistency["c2_mean_gain"] > 0 else "CONCERN"; json_write(ROOT / phase / "cohort_consistency.json", consistency)
    s1 = boot["linear_vs_fixed"]["mean"] > 0 and boot["linear_vs_fixed"]["ci95_low"] > 0; s2 = boot["linear_vs_random"]["mean"] > 0; s3 = quality_pass; s4 = True; s5 = consistency["status"] == "PASS"; supported = s1 and s2 and s3 and s4 and s5
    effect = "STRONG" if relative >= .05 else "MODERATE" if relative >= .02 else "SMALL" if relative > 0 else "NON_POSITIVE"
    decision = {"C1_CONFIRMATION": "CONTINUE", "C2_RUN": "YES", "N_CONFIRMATORY": 256, "LINEAR_K2_MAE": float(linear.property_mae), "FIXED_K2_MAE": float(fixed.property_mae), "RANDOM_K2_MAE": float(random.property_mae), "C0_MAE": float(values.loc['C0'].property_mae), "LINEAR_VS_FIXED_RELATIVE_GAIN": relative, "LINEAR_VS_FIXED_BOOTSTRAP_CI": [boot['linear_vs_fixed']['ci95_low'], boot['linear_vs_fixed']['ci95_high']], "LINEAR_VS_RANDOM_RELATIVE_GAIN": float((random.property_mae-linear.property_mae)/max(float(random.property_mae),EPS)), "QUALITY_GUARDRAILS": "PASS" if quality_pass else "FAIL", "COMPUTE_MULTIPLIER_LINEAR": 2.2, "COMPUTE_MULTIPLIER_FIXED": 2.2, "COHORT_CONSISTENCY": consistency["status"], "PRACTICAL_EFFECT": effect, "SUCCESS_GATES": {"S1_primary_ci": s1, "S2_random_direction": s2, "S3_quality": s3, "S4_compute": s4, "S5_consistency": s5}, "FROZEN_LINEAR_K2_CONFIRMATION": "SUPPORTED" if supported else "DIRECTIONAL_ONLY" if boot['linear_vs_fixed']['mean'] > 0 else "NOT_SUPPORTED", "ADAPTIVE_ALLOCATION_VALUE_CONFIRMATORY": "SUPPORTED" if supported else "NOT_SUPPORTED", "INNOVATION1_FINAL_STATUS": "SUPPORTED" if supported else "MIXED", "DFT_VERIFIED": False}
    json_write(ROOT / phase / "decision_summary.json", decision); json_write(ROOT / phase / "acceptance_frequencies.json", candidate_frequencies(outcomes)); report_phase(phase, metrics, boot, decision); make_basic_plots(phase, pd.read_csv(ROOT / phase / "paired_results.csv"), metrics); consistency_plots(c1, c2); allocator_stability(outcomes)


def make_basic_plots(phase: str, paired: pd.DataFrame, metrics: pd.DataFrame) -> None:
    PLOTS.mkdir(exist_ok=True); suffix = phase.replace("_128", "").replace("256", "")
    for column, filename, labels in (("linear_vs_fixed_gain", "linear_vs_fixed_paired.png", ("Fixed", "Linear")), ("linear_vs_random_gain", "linear_vs_random_paired.png", ("Random", "Linear")), ("linear_vs_c0_gain", "linear_vs_c0_paired.png", ("C0", "Linear"))):
        if phase != "pooled256": continue
        left_name = {"linear_vs_fixed_gain":"Fixed_K2","linear_vs_random_gain":"Random_K2","linear_vs_c0_gain":"C0"}[column]; fig, ax = plt.subplots(figsize=(6,5))
        for _, row in paired.iterrows(): ax.plot([0,1],[row[left_name],row.Linear_K2],color=".75",linewidth=.4)
        ax.set_xticks([0,1],labels); ax.set_ylabel("Property absolute error"); fig.tight_layout(); fig.savefig(PLOTS / filename,dpi=180); plt.close(fig)
    if phase == "pooled256":
        values = paired.linear_vs_fixed_gain.to_numpy(float); rng=np.random.default_rng(BOOT_SEED); draws=values[rng.integers(0,len(values),size=(N_BOOT,len(values)))].mean(axis=1); fig,ax=plt.subplots(figsize=(6,4)); ax.hist(draws,bins=60); ax.axvline(0,color="red"); fig.tight_layout(); fig.savefig(PLOTS/"bootstrap_linear_vs_fixed.png",dpi=180); plt.close(fig)
        view=metrics.set_index("method"); fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(view.property_mae,view.e_hull); [ax.annotate(i,(r.property_mae,r.e_hull)) for i,r in view.iterrows()]; ax.set(xlabel="Property MAE",ylabel="E-hull"); fig.tight_layout(); fig.savefig(PLOTS/"property_quality_pareto.png",dpi=180); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4)); ax.bar(range(4),[1,2.2,2.2,2.2]); ax.set_xticks(range(4),METHODS,rotation=25,ha="right"); ax.set_ylabel("Compute x C0"); fig.tight_layout(); fig.savefig(PLOTS/"compute_efficiency.png",dpi=180); plt.close(fig)


def consistency_plots(c1: pd.DataFrame, c2: pd.DataFrame) -> None:
    fig,ax=plt.subplots(figsize=(6,4)); means=[c1.linear_vs_fixed_gain.mean(),c2.linear_vs_fixed_gain.mean()]; ax.bar([0,1],means); ax.axhline(0,color="black",linewidth=.8); ax.set_xticks([0,1],["C1","C2"]); ax.set_ylabel("Mean Linear vs Fixed gain"); fig.tight_layout(); fig.savefig(PLOTS/"c1_c2_effect_consistency.png",dpi=180); plt.close(fig)


def allocator_stability(outcomes: pd.DataFrame) -> None:
    predictions = pd.read_csv(ROOT / "pooled256/allocator_predictions.csv"); rows=[]
    phase_b = pd.read_csv("/mnt/datasets-livsyn/dxl/mattergen_v1/experiments/reference_preserved_budgeted_cfg/feature_study/allocator_test_selections.csv"); phase_b=phase_b[phase_b.method=="Linear_K2"]
    for cohort, seeds in (("phase_b_test",set(map(int,phase_b.seed))), ("c1",set(json.loads((PROTOCOL_ROOT/"seed_manifest_256.json").read_text())["c1_128"])), ("c2",set(json.loads((PROTOCOL_ROOT/"seed_manifest_256.json").read_text())["c2_128"])), ("pooled",set(map(int,predictions.seed)))):
        if cohort=="phase_b_test":
            choices=pd.concat([phase_b[["seed","allocated_policy_1"]].rename(columns={"allocated_policy_1":"policy_id"}),phase_b[["seed","allocated_policy_2"]].rename(columns={"allocated_policy_2":"policy_id"})])
        else:
            selected = predictions.selected_top2.map(
                lambda value: value if isinstance(value, (bool, np.bool_))
                else str(value).strip().lower() in {"true", "1", "yes"}
            )
            choices=predictions[(predictions.seed.isin(seeds)) & selected][["seed","policy_id"]]
        for policy in ("GPulse","APulse","PPulse","CPulse"): rows.append({"cohort":cohort,"policy_id":policy,"top2_selection_rate":float((choices.policy_id==policy).sum()/max(len(seeds),1))})
    frame=pd.DataFrame(rows); frame.to_csv(ROOT/"pooled256/allocator_stability.csv",index=False); pivot=frame.pivot(index="cohort",columns="policy_id",values="top2_selection_rate"); fig,ax=plt.subplots(figsize=(8,4)); pivot.plot(kind="bar",ax=ax); fig.tight_layout(); fig.savefig(PLOTS/"candidate_selection_frequency.png",dpi=180); plt.close(fig)
    freqs=json.loads((ROOT/"pooled256/acceptance_frequencies.json").read_text()); fig,ax=plt.subplots(figsize=(5,4)); ax.bar(["accept","fallback"],[freqs["acceptance_rate"],freqs["fallback_rate"]]); fig.tight_layout(); fig.savefig(PLOTS/"fallback_rate.png",dpi=180); plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=("c1_128","c2_128","pooled256"),required=True); parser.add_argument("--stage",choices=("select","finalize","pool"),required=True); args=parser.parse_args()
    if args.stage=="select": selection_stage(args.phase)
    elif args.stage=="pool": pool_stage()
    elif args.phase=="pooled256": final_pooled()
    else: finalize_cohort(args.phase)
