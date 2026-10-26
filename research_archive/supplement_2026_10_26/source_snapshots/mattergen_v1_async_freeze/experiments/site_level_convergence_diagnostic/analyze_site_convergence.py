"""Calibrate and evaluate per-site clean-position convergence heterogeneity."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.geometry import find_mic
from ase.io import read
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / "shared_field_dynamics"
CALIBRATION_SEEDS = tuple(range(99000, 99032))
VALIDATION_SEEDS = tuple(range(99032, 99064))
ALL_SEEDS = CALIBRATION_SEEDS + VALIDATION_SEEDS
K = 3
MAX_CONVERGENCE_T = 0.5
EARLY_T = 0.10
LATE_T = 0.05
SITE_SPREAD_GATE = 0.10
ANALYSIS_OBSERVATION_STEPS = (
    *range(0, 850, 20), 850, 860, 880, 900, 920, 940, 950, 960, 970, 980, 990, 999,
)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(quantile, cumulative, values))


def _load_observations(seeds: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        path = ROOT / "rerun" / str(seed) / "site_clean_estimates.npz"
        summary_path = path.with_name("run_summary.json")
        if not path.is_file() or not summary_path.is_file():
            raise RuntimeError(f"missing site rerun for seed {seed}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("exact_reproduction") is not True
                or int(summary["mattergen_score_calls"]) != 2000
                or int(summary["site_diagnostic_observation_count"]) != 1000):
            raise RuntimeError(f"site rerun contract failure for seed {seed}")
        with np.load(path, allow_pickle=False) as data:
            steps = data["sampling_step"].astype(int)
            model_t = data["model_t"].astype(float)
            positions = data["clean_frac_position"].astype(np.float64)
            cells = data["clean_cell"].astype(np.float64)
        if positions.shape[0] != 1000 or positions.shape[2] != 3 or cells.shape != (1000, 3, 3):
            raise RuntimeError(f"bad tensor shapes for seed {seed}")
        selected_indices = np.flatnonzero(np.isin(steps, ANALYSIS_OBSERVATION_STEPS))
        if tuple(steps[selected_indices]) != ANALYSIS_OBSERVATION_STEPS:
            raise RuntimeError(f"frozen 55-point observation schedule missing for seed {seed}")
        steps = steps[selected_indices]
        model_t = model_t[selected_indices]
        positions = positions[selected_indices]
        cells = cells[selected_indices]
        num_atoms = positions.shape[1]
        distances = np.full((len(steps), num_atoms), np.nan, dtype=float)
        for index in range(1, len(steps)):
            raw_vectors = (positions[index] - positions[index - 1]) @ cells[index - 1]
            _, lengths = find_mic(raw_vectors, cell=cells[index - 1], pbc=True)
            distances[index] = np.asarray(lengths, dtype=float)
        if not np.isfinite(distances[1:]).all():
            raise FloatingPointError(f"non-finite MIC distances for seed {seed}")
        for atom_index in range(num_atoms):
            rows.append(pd.DataFrame({
                "seed": seed,
                "subset": summary["subset"],
                "sampling_step": steps,
                "model_t": model_t,
                "reverse_progress": 1.0 - model_t,
                "num_atoms": num_atoms,
                "atom_index": atom_index,
                "site_displacement_Angstrom": distances[:, atom_index],
            }))
    return pd.concat(rows, ignore_index=True)


def _site_times(observations: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (seed, atom_index), group in observations.groupby(["seed", "atom_index"], sort=True):
        group = group.sort_values("sampling_step").reset_index(drop=True)
        values = group["site_displacement_Angstrom"].to_numpy(float)
        times = group["model_t"].to_numpy(float)
        stable = np.isfinite(values) & (values < threshold) & (times <= MAX_CONVERGENCE_T + 1e-8)
        run = 0
        onset = None
        for index, item in enumerate(stable):
            run = run + 1 if item else 0
            if run >= K:
                onset = index - K + 1
                break
        convergence_t = float(times[onset]) if onset is not None else float("nan")
        post_onset = stable[onset + K:] if onset is not None else np.asarray([], dtype=bool)
        post_stable_fraction = (
            float(post_onset.mean()) if len(post_onset) else (1.0 if onset is not None else np.nan)
        )
        post_violation_count = int((~post_onset).sum()) if onset is not None else -1
        late = values[times <= EARLY_T + 1e-8]
        finite_late = late[np.isfinite(late)]
        rows.append({
            "seed": int(seed),
            "subset": str(group.iloc[0]["subset"]),
            "atom_index": int(atom_index),
            "num_atoms": int(group.iloc[0]["num_atoms"]),
            "T_site": convergence_t,
            "T_site_endpoint_censored": convergence_t if np.isfinite(convergence_t) else 0.0,
            "converged": bool(np.isfinite(convergence_t)),
            "convergence_sampling_step": int(group.iloc[onset]["sampling_step"]) if onset is not None else -1,
            "post_convergence_below_threshold_fraction": post_stable_fraction,
            "post_convergence_violation_count": post_violation_count,
            "transient_convergence": bool(onset is not None and post_stable_fraction < 0.80),
            "final_site_displacement": float(values[-1]),
            "max_late_displacement": float(finite_late.max()),
        })
    return pd.DataFrame(rows)


def _candidate_diagnostic(observations: pd.DataFrame, quantile: float) -> dict:
    tail = observations[observations["model_t"] <= LATE_T + 1e-8].copy()
    tail["weight"] = 1.0 / tail["num_atoms"].astype(float)
    threshold = _weighted_quantile(
        tail["site_displacement_Angstrom"].to_numpy(float),
        tail["weight"].to_numpy(float), quantile,
    )
    sites = _site_times(observations, threshold)
    convergence_rate = float(sites["converged"].mean())
    early_fraction = float(
        (sites["converged"] & (sites["T_site"] >= EARLY_T - 1e-8)).mean()
    )
    return {
        "quantile": quantile,
        "threshold_Angstrom": threshold,
        "calibration_site_count": len(sites),
        "calibration_convergence_rate": convergence_rate,
        "calibration_early_fraction": early_fraction,
        "admissible": bool(convergence_rate >= 0.40 and early_fraction <= 0.90),
    }


def _freeze_threshold(observations: pd.DataFrame) -> dict:
    candidates = [_candidate_diagnostic(observations, q)
                  for q in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)]
    primary = next(item for item in candidates if item["quantile"] == 0.90)
    if primary["admissible"]:
        selected = primary
        reason = "trajectory-balanced q90 late-tail rule was calibration-admissible"
    else:
        admissible = [item for item in candidates if item["admissible"]]
        if not admissible:
            raise RuntimeError("no calibration-only threshold avoids degenerate site convergence")
        selected = min(
            admissible,
            key=lambda item: (
                abs(item["calibration_early_fraction"] - 0.50),
                abs(item["quantile"] - 0.90),
            ),
        )
        reason = "q90 was degenerate; used preregistered calibration-only fallback"
    return {
        "frozen": True,
        "calibration_seeds": [CALIBRATION_SEEDS[0], CALIBRATION_SEEDS[-1]],
        "validation_used_for_selection": False,
        "analysis_observation_count_per_trajectory": len(ANALYSIS_OBSERVATION_STEPS),
        "recorded_clean_estimate_count_per_trajectory": 1000,
        "analysis_observation_steps": list(ANALYSIS_OBSERVATION_STEPS),
        "clean_estimate": "wrap(x_t + sigma_t^2 * guided_score_pos)",
        "minimum_image": "ASE find_mic float64 using previous clean cell",
        "late_tail_model_t_max": LATE_T,
        "trajectory_balanced": True,
        "consecutive_observations": K,
        "convergence_forbidden_while_model_t_gt": MAX_CONVERGENCE_T,
        "threshold_site_Angstrom": selected["threshold_Angstrom"],
        "selected_quantile": selected["quantile"],
        "selection_reason": reason,
        "candidate_diagnostics": candidates,
    }


def _attach_final_labels(sites: pd.DataFrame) -> pd.DataFrame:
    generated = read(SHARED / "generated_diagnostic.extxyz", index=":")
    force_atoms = read(SHARED / "diagnostic_relaxation" / "initial_with_properties.extxyz", index=":")
    if len(generated) != 64 or len(force_atoms) != 64:
        raise RuntimeError("shared final structure/force count changed")
    rows = []
    crystal_nn = CrystalNN()
    adaptor = AseAtomsAdaptor()
    for offset, seed in enumerate(ALL_SEEDS):
        atoms = generated[offset]
        evaluated = force_atoms[offset]
        if not np.array_equal(atoms.numbers, evaluated.numbers):
            raise RuntimeError(f"force atom order mismatch at seed {seed}")
        forces = np.asarray(evaluated.get_forces(), dtype=float)
        structure = adaptor.get_structure(atoms)
        for atom_index, atomic_number in enumerate(atoms.numbers):
            neighbors = structure.get_neighbors(structure[atom_index], 6.0)
            nearest = float(min(item.nn_distance for item in neighbors)) if neighbors else np.nan
            local_count = int(sum(item.nn_distance <= 3.0 for item in neighbors))
            try:
                coordination = float(crystal_nn.get_cn(structure, atom_index, on_disorder="take_majority_strict"))
            except Exception:
                coordination = float("nan")
            rows.append({
                "seed": seed,
                "atom_index": atom_index,
                "atomic_number": int(atomic_number),
                "species": structure[atom_index].specie.symbol,
                "coordination_number": coordination,
                "nearest_neighbor_distance_Angstrom": nearest,
                "local_neighbor_count_3A": local_count,
                "final_unrelaxed_force_Angstrom_eV": float(np.linalg.norm(forces[atom_index])),
            })
    labels = pd.DataFrame(rows)
    return sites.merge(labels, on=["seed", "atom_index"], how="left", validate="one_to_one")


def _structure_summary(sites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in sites.groupby("seed", sort=True):
        values = group["T_site_endpoint_censored"].to_numpy(float)
        rows.append({
            "seed": int(seed), "subset": str(group.iloc[0]["subset"]),
            "num_atoms": len(group),
            "median_T_site": float(np.median(values)),
            "mean_T_site": float(np.mean(values)),
            "std_T_site": float(np.std(values, ddof=0)),
            "p10_T_site": float(np.quantile(values, 0.10)),
            "p25_T_site": float(np.quantile(values, 0.25)),
            "p75_T_site": float(np.quantile(values, 0.75)),
            "p90_T_site": float(np.quantile(values, 0.90)),
            "site_spread_p90_minus_p10": float(np.quantile(values, 0.90) - np.quantile(values, 0.10)),
            "site_range_max_minus_min": float(values.max() - values.min()),
            "fraction_early_converged": float(
                (group["converged"] & (group["T_site"] >= EARLY_T - 1e-8)).mean()
            ),
            "fraction_late_converged": float(
                (group["converged"] & (group["T_site"] <= LATE_T + 1e-8)).mean()
            ),
            "fraction_not_converged": float((~group["converged"]).mean()),
            "contains_early_and_late_converged": bool(
                (group["converged"] & (group["T_site"] >= EARLY_T - 1e-8)).any()
                and (group["converged"] & (group["T_site"] <= LATE_T + 1e-8)).any()
            ),
        })
    return pd.DataFrame(rows)


def _species_summary(sites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, number, species), group in sites.groupby(["subset", "atomic_number", "species"]):
        values = group["T_site_endpoint_censored"].to_numpy(float)
        rows.append({
            "subset": subset, "atomic_number": int(number), "species": species,
            "n_sites": len(group), "n_structures": group["seed"].nunique(),
            "convergence_rate": float(group["converged"].mean()),
            "mean_T_site_censored": float(values.mean()),
            "median_T_site_censored": float(np.median(values)),
            "std_T_site_censored": float(np.std(values, ddof=0)),
            "within_species_variance": float(np.var(values, ddof=0)),
        })
    return pd.DataFrame(rows).sort_values(["subset", "atomic_number"])


def _species_eta_squared(group: pd.DataFrame) -> dict:
    values = group["T_site_endpoint_censored"].to_numpy(float)
    overall = values.mean()
    total = float(np.sum((values - overall) ** 2))
    within = 0.0
    eligible = 0
    within_stds = []
    for _, species_group in group.groupby("atomic_number"):
        local = species_group["T_site_endpoint_censored"].to_numpy(float)
        within += float(np.sum((local - local.mean()) ** 2))
        if len(local) >= 2:
            eligible += 1
            within_stds.append(float(np.std(local, ddof=0)))
    eta = 1.0 - within / total if total > 0 else 0.0
    return {
        "eta_squared_species": eta,
        "pooled_within_species_std": math.sqrt(within / max(len(values), 1)),
        "median_within_species_std": float(np.median(within_stds)) if within_stds else np.nan,
        "eligible_species_count": eligible,
    }


def _cluster_bootstrap_spearman(group: pd.DataFrame, x_name: str, *, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(20260912)
    seeds = group["seed"].unique()
    if len(seeds) == 0:
        return float("nan"), float("nan")
    values = []
    for _ in range(n):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        pieces = [group[group["seed"] == seed] for seed in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        keep = np.isfinite(boot[x_name]) & np.isfinite(boot["T_site_endpoint_censored"])
        if keep.sum() >= 3:
            statistic = float(spearmanr(
                boot.loc[keep, x_name], boot.loc[keep, "T_site_endpoint_censored"]
            ).statistic)
            if np.isfinite(statistic):
                values.append(statistic)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(item) for item in np.quantile(values, [0.025, 0.975]))


def _environment_correlations(sites: pd.DataFrame) -> pd.DataFrame:
    variables = (
        "atomic_number", "coordination_number", "nearest_neighbor_distance_Angstrom",
        "local_neighbor_count_3A", "final_unrelaxed_force_Angstrom_eV",
    )
    rows = []
    for subset in ("calibration", "validation", "all"):
        group = sites if subset == "all" else sites[sites["subset"] == subset]
        for variable in variables:
            keep = np.isfinite(group[variable]) & np.isfinite(group["T_site_endpoint_censored"])
            rho = float(spearmanr(
                group.loc[keep, variable], group.loc[keep, "T_site_endpoint_censored"]
            ).statistic) if keep.sum() >= 3 else np.nan
            low, high = _cluster_bootstrap_spearman(group.loc[keep], variable)
            rows.append({
                "subset": subset, "variable": variable, "n_sites": int(keep.sum()),
                "n_structures": int(group.loc[keep, "seed"].nunique()),
                "spearman_rho": rho, "cluster_bootstrap_ci_low": low,
                "cluster_bootstrap_ci_high": high,
            })
    return pd.DataFrame(rows)


def _plots(sites: pd.DataFrame, structures: pd.DataFrame) -> None:
    plot_root = ROOT / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.6))
    for subset, color in (("calibration", "#4477AA"), ("validation", "#EE6677")):
        axis.hist(sites.loc[sites["subset"] == subset, "T_site_endpoint_censored"],
                  bins=np.linspace(0, 0.51, 27), alpha=0.5, label=subset, color=color)
    axis.set(xlabel="site convergence T (endpoint-censored at 0)", ylabel="site count")
    axis.legend(); axis.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(plot_root / "site_convergence_distribution.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.6))
    values = structures[structures["subset"] == "validation"]["site_spread_p90_minus_p10"]
    axis.hist(values, bins=18, color="#4477AA", alpha=0.8)
    axis.axvline(SITE_SPREAD_GATE, color="#CC3311", ls="--", label="gate=0.10")
    axis.set(xlabel="SiteSpread = P90(T)-P10(T)", ylabel="validation structures")
    axis.legend(); axis.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(plot_root / "site_spread_distribution.png", dpi=180); plt.close(fig)

    common = sites[sites["subset"] == "validation"]["species"].value_counts().head(15).index
    data = [sites[(sites["subset"] == "validation") & (sites["species"] == name)]
            ["T_site_endpoint_censored"].to_numpy(float) for name in common]
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.boxplot(data, tick_labels=list(common), showfliers=False)
    axis.set(xlabel="species (15 most frequent)", ylabel="site convergence T")
    axis.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(plot_root / "site_convergence_vs_species.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.8))
    validation = sites[sites["subset"] == "validation"]
    axis.scatter(validation["final_unrelaxed_force_Angstrom_eV"],
                 validation["T_site_endpoint_censored"], s=16, alpha=0.55)
    axis.set(xlabel="MatterSim final-generated-site force (eV/A)", ylabel="site convergence T")
    axis.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(plot_root / "site_convergence_vs_force.png", dpi=180); plt.close(fig)

    validation_seeds = structures[structures["subset"] == "validation"]["seed"].tolist()
    max_atoms = int(sites[sites["subset"] == "validation"]["num_atoms"].max())
    matrix = np.full((len(validation_seeds), max_atoms), np.nan)
    for row, seed in enumerate(validation_seeds):
        local = sites[sites["seed"] == seed].sort_values("atom_index")
        matrix[row, :len(local)] = local["T_site_endpoint_censored"]
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=0.5, cmap="viridis")
    axis.set(xlabel="site index", ylabel="validation structure (seed order)")
    fig.colorbar(image, ax=axis, label="site convergence T")
    fig.tight_layout(); fig.savefig(plot_root / "per_structure_heatmap.png", dpi=180); plt.close(fig)


def _decision(observations: pd.DataFrame, threshold: float, sites: pd.DataFrame,
              structures: pd.DataFrame, correlations: pd.DataFrame) -> dict:
    validation = structures[structures["subset"] == "validation"]
    spread_fraction = float((validation["site_spread_p90_minus_p10"] >= SITE_SPREAD_GATE - 1e-8).mean())
    range_fraction = float((validation["site_range_max_minus_min"] >= 0.15 - 1e-8).mean())
    mixed_fraction = float(validation["contains_early_and_late_converged"].mean())
    stable_relations = []
    for variable in correlations["variable"].unique():
        cal = correlations[(correlations["subset"] == "calibration") & (correlations["variable"] == variable)].iloc[0]
        val = correlations[(correlations["subset"] == "validation") & (correlations["variable"] == variable)].iloc[0]
        if (np.isfinite(cal["spearman_rho"]) and np.isfinite(val["spearman_rho"])
                and abs(cal["spearman_rho"]) >= 0.30 and abs(val["spearman_rho"]) >= 0.30
                and np.sign(cal["spearman_rho"]) == np.sign(val["spearman_rho"])):
            stable_relations.append(variable)
    eta_cal = _species_eta_squared(sites[sites["subset"] == "calibration"])
    eta_val = _species_eta_squared(sites[sites["subset"] == "validation"])
    species_driven = bool(
        eta_val["eta_squared_species"] >= 0.75
        and eta_val["pooled_within_species_std"] < 0.03
    )
    sensitivity = {}
    for factor in (0.9, 1.1):
        sensitivity_sites = _site_times(observations, threshold * factor)
        sensitivity_structures = _structure_summary(sensitivity_sites)
        sensitivity_validation = sensitivity_structures[
            sensitivity_structures["subset"] == "validation"
        ]
        sensitivity[str(factor)] = {
            "threshold_Angstrom": threshold * factor,
            "fraction_site_spread_ge_0p10": float(
                (sensitivity_validation["site_spread_p90_minus_p10"]
                 >= SITE_SPREAD_GATE - 1e-8).mean()
            ),
            "site_convergence_rate": float(
                sensitivity_sites[sensitivity_sites["subset"] == "validation"]["converged"].mean()
            ),
        }
    validation_sites = sites[sites["subset"] == "validation"]
    transient_fraction = float(
        validation_sites.loc[validation_sites["converged"], "transient_convergence"].mean()
    )
    measurement_noise_driven = bool(
        transient_fraction >= 0.50
        and sensitivity["0.9"]["fraction_site_spread_ge_0p10"] < 0.40 - 1e-8
    )
    heterogeneity = spread_fraction >= 0.60 - 1e-8
    auxiliary = {
        "range_fraction_ge_0p15_pass": range_fraction >= 0.50 - 1e-8,
        "early_and_late_fraction_pass": mixed_fraction >= 0.50 - 1e-8,
        "stable_environment_relation_pass": bool(stable_relations),
    }
    mechanism = any(auxiliary.values())
    if heterogeneity and mechanism and not species_driven and not measurement_noise_driven:
        verdict, next_step = "GO", "SITE_LEVEL_ADAPTIVE_DIFFUSION"
    elif spread_fraction < 0.40 - 1e-8 or species_driven or measurement_noise_driven:
        verdict, next_step = "FAIL", "STOP_ASYNC_FAMILY"
    else:
        verdict, next_step = "BORDERLINE", "STOP_ASYNC_FAMILY"
    return {
        "SITE_LEVEL_HETEROGENEITY": heterogeneity,
        "SITE_LEVEL_DIAGNOSTIC": verdict,
        "NEXT": next_step,
        "REUSED_EXISTING_TRAJECTORIES": False,
        "REUSED_EXISTING_FINAL_QUALITY_AND_FORCES": True,
        "validation_structure_count": len(validation),
        "fraction_site_spread_ge_0p10": spread_fraction,
        "fraction_site_range_ge_0p15": range_fraction,
        "fraction_contains_early_and_late": mixed_fraction,
        "auxiliary_evidence": auxiliary,
        "stable_environment_relations": stable_relations,
        "species_confound_calibration": eta_cal,
        "species_confound_validation": eta_val,
        "SPECIES_DRIVEN_HETEROGENEITY": species_driven,
        "measurement_noise_audit": {
            "definition_frozen_before_validation_analysis": True,
            "transient_definition": "less than 80% of later observations remain below threshold",
            "fraction_converged_validation_sites_transient": transient_fraction,
            "threshold_sensitivity": sensitivity,
        },
        "MEASUREMENT_NOISE_DRIVEN": measurement_noise_driven,
        "surrogate_property_eval": True,
        "dft_verified": False,
    }


def _report(threshold: dict, sites: pd.DataFrame, structures: pd.DataFrame,
            correlations: pd.DataFrame, decision: dict) -> None:
    val = structures[structures["subset"] == "validation"]
    spread = val["site_spread_p90_minus_p10"].to_numpy(float)
    corr_lines = []
    for _, row in correlations[correlations["subset"] == "validation"].iterrows():
        corr_lines.append(
            f"- {row['variable']}: rho={row['spearman_rho']:.4f}, "
            f"structure-cluster bootstrap 95% CI [{row['cluster_bootstrap_ci_low']:.4f}, "
            f"{row['cluster_bootstrap_ci_high']:.4f}]"
        )
    text = f"""# Site-Level Position Convergence Diagnostic FINAL

1. trajectories reused/generated：`REUSED_EXISTING_TRAJECTORIES=False`。原 64 条仅有 field aggregate，故用相同 seeds 作只读重放；最终结构和 MatterSim per-atom force 复用。
2. sample count：64 trajectories（calibration 32 / validation 32），共 576 atomic sites（calibration 276 / validation 300）。
3. observation count：正式分析使用与既有 field diagnostic 完全相同的 55 个 predictor points / trajectory；复跑被动保存全部 1000 点作审计。每条仍为 2000 score calls，无额外 GemNet forward。
4. clean-estimate：`wrap(x_t + sigma_t^2 * CFG-guided score_pos)`，predictor 前、官方 corrector 后。
5. site displacement：相邻 clean estimate，用前一 clean cell 和 ASE `find_mic` float64 求严格 triclinic minimum-image Cartesian distance。
6. threshold：`{threshold['threshold_site_Angstrom']:.8f} Å`；calibration t<=0.05 trajectory-balanced q{int(threshold['selected_quantile']*100)}，validation 未参与。
7. K=3 consecutive global predictor observations；t>0.5 不判 convergence。未收敛 site 保留 `T_site=NaN`，结构 spread 使用 endpoint-censored `T=0` 并明确标记。
8. median SiteSpread：`{np.median(spread):.6f}`。
9. mean SiteSpread：`{np.mean(spread):.6f}`。
10. P25/P75/P90 SiteSpread：`{np.quantile(spread,.25):.6f} / {np.quantile(spread,.75):.6f} / {np.quantile(spread,.90):.6f}`。
11. fraction SiteSpread>=0.10：`{decision['fraction_site_spread_ge_0p10']:.2%}`。
12. fraction max-min>=0.15：`{decision['fraction_site_range_ge_0p15']:.2%}`；同晶体同时有 early/late converged site：`{decision['fraction_contains_early_and_late']:.2%}`。
13. within-species heterogeneity：validation pooled within-species std=`{decision['species_confound_validation']['pooled_within_species_std']:.6f}`，median species std=`{decision['species_confound_validation']['median_within_species_std']:.6f}`。
14. species-level differences：validation eta²=`{decision['species_confound_validation']['eta_squared_species']:.4f}`；`SPECIES_DRIVEN_HETEROGENEITY={str(decision['SPECIES_DRIVEN_HETEROGENEITY']).upper()}`。
Measurement-noise audit：converged validation sites 中 transient 比例=`{decision['measurement_noise_audit']['fraction_converged_validation_sites_transient']:.2%}`；0.9× threshold 下 SiteSpread gate fraction=`{decision['measurement_noise_audit']['threshold_sensitivity']['0.9']['fraction_site_spread_ge_0p10']:.2%}`；`MEASUREMENT_NOISE_DRIVEN={str(decision['MEASUREMENT_NOISE_DRIVEN']).upper()}`。
15–17. 与 coordination / local geometry / final-generated-site MatterSim force 的 validation Spearman：

{chr(10).join(corr_lines)}

18. **SITE_LEVEL_HETEROGENEITY={str(decision['SITE_LEVEL_HETEROGENEITY']).upper()}**。
19. **SITE_LEVEL_DIAGNOSTIC={decision['SITE_LEVEL_DIAGNOSTIC']}**。
20. **NEXT={decision['NEXT']}**。

结论只针对当前 official Full FT、target=0.2、CFG=2 和冻结定义。MatterSim force 是最终生成但松弛前结构的 surrogate site force；`SURROGATE_PROPERTY_EVAL=True`，`DFT_VERIFIED=False`。
"""
    (ROOT / "final_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()
    threshold_path = ROOT / "site_convergence_threshold.json"
    if args.calibrate_only:
        if threshold_path.exists():
            raise FileExistsError(threshold_path)
        observations = _load_observations(CALIBRATION_SEEDS)
        threshold = _freeze_threshold(observations)
        threshold_path.write_text(json.dumps(threshold, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(threshold, indent=2, sort_keys=True))
        return
    if not threshold_path.is_file():
        raise FileNotFoundError("freeze calibration threshold before validation analysis")
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    observations = _load_observations(ALL_SEEDS)
    observations.to_csv(ROOT / "site_displacement_observations.csv.gz", index=False,
                        compression="gzip", quoting=csv.QUOTE_MINIMAL)
    sites = _attach_final_labels(_site_times(observations, threshold["threshold_site_Angstrom"]))
    sites.to_csv(ROOT / "site_convergence_metrics.csv", index=False)
    structures = _structure_summary(sites)
    structures.to_csv(ROOT / "structure_site_summary.csv", index=False)
    species = _species_summary(sites)
    species.to_csv(ROOT / "species_summary.csv", index=False)
    correlations = _environment_correlations(sites)
    correlations.to_csv(ROOT / "environment_correlations.csv", index=False)
    decision = _decision(
        observations, threshold["threshold_site_Angstrom"], sites, structures, correlations
    )
    (ROOT / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plots(sites, structures)
    _report(threshold, sites, structures, correlations, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
