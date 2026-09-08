"""Prospective32 endpoints, guardrails, tails and leave-one-out diagnostics."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from ase.io import read
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/gbsa_tcl_prospective32"
METHODS = ("C0", "GBSA-TCL")
SEEDS = tuple(range(87000, 87032))
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260908
HIGHER = {"stable", "nus", "novel", "unique", "composition_validity", "structure_validity"}
METRICS = (
    "e_hull", "stable", "nus", "novel", "unique", "composition_validity",
    "structure_validity", "rmsd", "atomic_force", "max_force", "steps",
)


def quantile(values, probability, axis=None):
    return np.quantile(np.asarray(values, dtype=float), probability, axis=axis)


def exact_binomial_ci(count: int, n: int) -> list[float]:
    low = 0.0 if count == 0 else float(beta.ppf(0.025, count, n - count + 1))
    high = 1.0 if count == n else float(beta.ppf(0.975, count + 1, n - count))
    return [low, high]


def finite_atoms(atoms, require_forces=False) -> bool:
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        return False
    if not float(np.linalg.det(atoms.cell.array)) > 0:
        return False
    if require_forces:
        try:
            forces = atoms.get_forces()
        except Exception:
            return False
        if not np.isfinite(forces).all():
            return False
    return True


def load_method(method: str, generation: pd.DataFrame) -> dict[str, np.ndarray]:
    generated_rows = generation[generation.method == method].sort_values("seed")
    if tuple(generated_rows.seed.astype(int)) != SEEDS or not generated_rows.success.astype(bool).all():
        raise RuntimeError(f"{method}: generation mismatch/failure")
    with gzip.open(ROOT / "quality" / method / "official_detailed.json.gz", "rt") as stream:
        detail = json.load(stream)
    relaxation = json.loads((ROOT / "relaxation" / method / "relaxation_summary.json").read_text())
    if tuple(map(int, relaxation["sample_seeds"])) != SEEDS or relaxation["success"] is not True:
        raise RuntimeError(f"{method}: relaxation mismatch/failure")
    generated_atoms = read(ROOT / "structures" / f"{method}_generated.extxyz", index=":")
    initial_atoms = read(ROOT / "relaxation" / method / "initial_with_properties.extxyz", index=":")
    relaxed_atoms = read(ROOT / "relaxation" / method / "relaxed.extxyz", index=":")
    if not (len(generated_atoms) == len(initial_atoms) == len(relaxed_atoms) == len(SEEDS)):
        raise RuntimeError(f"{method}: structure count mismatch")
    force_norms = [np.linalg.norm(item.get_forces(), axis=1) for item in initial_atoms]
    final_force_norms = [np.linalg.norm(item.get_forces(), axis=1) for item in relaxed_atoms]
    data = {
        "seed": np.asarray(SEEDS), "formula": generated_rows.formula.to_numpy(),
        "e_hull": np.asarray(detail["energy_above_hull_per_atom"], dtype=float),
        "stable": np.asarray(detail["stable"], dtype=float),
        "nus": np.asarray(detail["novel_unique_stable"], dtype=float),
        "novel": np.asarray(detail["novel"], dtype=float),
        "unique": np.asarray(detail["unique"], dtype=float),
        "composition_validity": np.asarray(detail["comp_validity"], dtype=float),
        "structure_validity": np.asarray(detail["structure_validity"], dtype=float),
        "rmsd": np.asarray(detail["rmsd_from_relaxation"], dtype=float),
        "atomic_force": np.asarray([values.mean() for values in force_norms]),
        "max_force": np.asarray([values.max() for values in force_norms]),
        "final_max_force": np.asarray([values.max() for values in final_force_norms]),
        "steps": np.asarray(relaxation["relaxation_steps"], dtype=float),
        "generation_seconds": generated_rows.elapsed_seconds.to_numpy(dtype=float),
        "technical_rerun": generated_rows.technical_rerun.astype(bool).to_numpy(),
        "generated_finite_positive_cell": np.asarray([finite_atoms(item) for item in generated_atoms]),
        "initial_finite_positive_cell_force": np.asarray([finite_atoms(item, True) for item in initial_atoms]),
        "relaxed_finite_positive_cell_force": np.asarray([finite_atoms(item, True) for item in relaxed_atoms]),
    }
    if any(len(values) != len(SEEDS) for values in data.values()):
        raise RuntimeError(f"{method}: metric length mismatch")
    numeric = [key for key in data if key != "formula"]
    if any(not np.isfinite(data[key]).all() for key in numeric):
        raise RuntimeError(f"{method}: non-finite parsed metric")
    return data


def summary_row(method: str, values: dict[str, np.ndarray]) -> dict:
    row = {
        "method": method, "n": len(SEEDS), "generation_success_count": len(SEEDS),
        "generation_success_rate": 1.0, "mattersim_success_count": len(SEEDS),
        "mattersim_success_rate": 1.0, "technical_rerun_count": int(values["technical_rerun"].sum()),
        "technical_invalid_count": int((~values["generated_finite_positive_cell"] |
                                         ~values["initial_finite_positive_cell_force"] |
                                         ~values["relaxed_finite_positive_cell_force"]).sum()),
        "dft_verified": False,
    }
    for metric, unit in (("e_hull", "ev_per_atom"), ("rmsd", "angstrom"),
                         ("atomic_force", "ev_per_angstrom")):
        metric_values = values[metric]
        row.update({
            f"{metric}_mean_{unit}": float(metric_values.mean()),
            f"{metric}_median_{unit}": float(np.median(metric_values)),
            f"{metric}_p95_{unit}": float(quantile(metric_values, .95)),
            f"{metric}_max_{unit}": float(metric_values.max()),
        })
    for metric in ("stable", "nus", "novel", "unique", "composition_validity", "structure_validity"):
        count = int(values[metric].sum())
        row[f"{metric}_count"] = count
        row[f"{metric}_rate"] = float(values[metric].mean())
        low, high = exact_binomial_ci(count, len(SEEDS))
        row[f"{metric}_exact_ci95_low"] = low
        row[f"{metric}_exact_ci95_high"] = high
    max_force = values["max_force"]
    steps = values["steps"]
    row.update({
        "max_force_mean_ev_per_angstrom": float(max_force.mean()),
        "max_force_median_ev_per_angstrom": float(np.median(max_force)),
        "max_force_p95_ev_per_angstrom": float(quantile(max_force, .95)),
        "max_force_p99_ev_per_angstrom": float(quantile(max_force, .99)),
        "max_force_max_ev_per_angstrom": float(max_force.max()),
        "max_force_gt1_count": int((max_force > 1).sum()),
        "max_force_gt2_count": int((max_force > 2).sum()),
        "final_max_force_gt0p05_count": int((values["final_max_force"] > .05).sum()),
        "final_max_force_max_ev_per_angstrom": float(values["final_max_force"].max()),
        "relaxation_steps_mean": float(steps.mean()),
        "relaxation_steps_median": float(np.median(steps)),
        "relaxation_steps_p95": float(quantile(steps, .95)),
        "relaxation_steps_max": int(steps.max()),
        "relaxation_steps_gt200_count": int((steps > 200).sum()),
        "relaxation_steps_gt400_count": int((steps > 400).sum()),
        "rmsd_gt0p5_count": int((values["rmsd"] > .5).sum()),
        "generation_seconds_mean": float(values["generation_seconds"].mean()),
    })
    for threshold in (1, 2):
        count = int((max_force > threshold).sum())
        low, high = exact_binomial_ci(count, len(SEEDS))
        row[f"max_force_gt{threshold}_exact_ci95_low"] = low
        row[f"max_force_gt{threshold}_exact_ci95_high"] = high
    return row


def bootstrap_rows(data: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(SEEDS), size=(N_BOOTSTRAP, len(SEEDS)))
    rows = []
    baseline, candidate = data["C0"], data["GBSA-TCL"]
    for metric in METRICS:
        scale = 100.0 if metric in HIGHER else 1.0
        raw = candidate[metric] - baseline[metric]
        delta = raw * scale
        boot = delta[indices].mean(axis=1)
        preferred = raw if metric in HIGHER else -raw
        tolerance = 1e-12
        row = {
            "comparison": "GBSA-TCL-C0", "metric": metric,
            "unit": "percentage_points" if metric in HIGHER else "native",
            "n_pairs": len(SEEDS), "candidate_mean": float(candidate[metric].mean() * scale),
            "baseline_mean": float(baseline[metric].mean() * scale),
            "mean_delta": float(delta.mean()), "median_paired_delta": float(np.median(delta)),
            "bootstrap_ci95_low": float(quantile(boot, .025)),
            "bootstrap_ci95_high": float(quantile(boot, .975)),
            "bootstrap_resamples": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
            "wins": int((preferred > tolerance).sum()),
            "ties": int((np.abs(preferred) <= tolerance).sum()),
            "losses": int((preferred < -tolerance).sum()),
        }
        if metric in HIGHER:
            gains = int(((baseline[metric] == 0) & (candidate[metric] == 1)).sum())
            losses = int(((baseline[metric] == 1) & (candidate[metric] == 0)).sum())
            row["baseline_fail_to_candidate_success"] = gains
            row["baseline_success_to_candidate_fail"] = losses
            row["mcnemar_exact_p"] = (1.0 if gains + losses == 0 else
                                       float(binomtest(gains, gains + losses, .5).pvalue))
        rows.append(row)
    return pd.DataFrame(rows), indices


def ni_status(low: float, high: float, margin: float) -> str:
    if high <= margin:
        return "PASS"
    if low > margin:
        return "FAIL"
    return "BORDERLINE"


def tail_result(data, indices, threshold: float, margin_pp: float) -> dict:
    baseline = (data["C0"]["max_force"] > threshold).astype(float)
    candidate = (data["GBSA-TCL"]["max_force"] > threshold).astype(float)
    paired = (candidate - baseline) * 100
    boot = paired[indices].mean(axis=1)
    gains = int(((baseline == 0) & (candidate == 1)).sum())
    losses = int(((baseline == 1) & (candidate == 0)).sum())
    low, high = map(float, quantile(boot, [.025, .975]))
    return {
        "threshold": threshold, "margin_pp": margin_pp,
        "c0_count": int(baseline.sum()), "c0_rate": float(baseline.mean()),
        "gbsa_count": int(candidate.sum()), "gbsa_rate": float(candidate.mean()),
        "paired_delta_pp": float(paired.mean()), "bootstrap_ci95_pp": [low, high],
        "c0_exact_binomial_ci95": exact_binomial_ci(int(baseline.sum()), len(SEEDS)),
        "gbsa_exact_binomial_ci95": exact_binomial_ci(int(candidate.sum()), len(SEEDS)),
        "c0_fail_to_gbsa_event": gains, "c0_event_to_gbsa_fail": losses,
        "mcnemar_exact_p": 1.0 if gains + losses == 0 else float(binomtest(gains, gains + losses, .5).pvalue),
        "status": ni_status(low, high, margin_pp),
    }


def leave_one_out(data) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    indices = rng.integers(0, 31, size=(N_BOOTSTRAP, 31))
    rows = []
    for omitted in range(32):
        keep = np.arange(32) != omitted
        base_force = data["C0"]["atomic_force"][keep]
        cand_force = data["GBSA-TCL"]["atomic_force"][keep]
        boot_ratio = cand_force[indices].mean(axis=1) / base_force[indices].mean(axis=1)
        ratio_low, ratio_high = map(float, quantile(boot_ratio, [.025, .975]))
        row = {
            "omitted_seed": SEEDS[omitted], "atomic_force_ratio": float(cand_force.mean()/base_force.mean()),
            "atomic_force_absolute_delta": float(cand_force.mean()-base_force.mean()),
            "atomic_force_ratio_ci95_low": ratio_low, "atomic_force_ratio_ci95_high": ratio_high,
            "atomic_force_status": ni_status(ratio_low, ratio_high, 1.20),
        }
        for threshold, margin, label in ((1.0, 5.0, "gt1"), (2.0, 2.0, "gt2")):
            base = (data["C0"]["max_force"][keep] > threshold).astype(float)
            cand = (data["GBSA-TCL"]["max_force"][keep] > threshold).astype(float)
            delta = (cand-base)*100
            boot = delta[indices].mean(axis=1)
            low, high = map(float, quantile(boot, [.025, .975]))
            row[f"max_force_{label}_delta_pp"] = float(delta.mean())
            row[f"max_force_{label}_ci95_low"] = low
            row[f"max_force_{label}_ci95_high"] = high
            row[f"max_force_{label}_status"] = ni_status(low, high, margin)
        rows.append(row)
    frame = pd.DataFrame(rows)
    full_ratio = float(data["GBSA-TCL"]["atomic_force"].mean()/data["C0"]["atomic_force"].mean())
    diagnostic = {
        "all_samples_retained_in_main_analysis": True,
        "full_atomic_force_ratio": full_ratio,
        "loo_atomic_force_ratio_range": [float(frame.atomic_force_ratio.min()), float(frame.atomic_force_ratio.max())],
        "single_sample_can_change_atomic_ratio_direction": bool(((frame.atomic_force_ratio-1)*(full_ratio-1) < 0).any()),
        "loo_atomic_force_statuses": sorted(frame.atomic_force_status.unique()),
        "loo_gt1_statuses": sorted(frame.max_force_gt1_status.unique()),
        "loo_gt2_statuses": sorted(frame.max_force_gt2_status.unique()),
    }
    return frame, diagnostic


def main() -> None:
    generation = pd.read_csv(ROOT / "generation_results.csv")
    if len(generation) != 64 or generation.technical_rerun.astype(bool).sum() != 0:
        raise RuntimeError("unexpected generation attempts or technical rerun record")
    data = {method: load_method(method, generation) for method in METHODS}
    quality = pd.DataFrame([summary_row(method, data[method]) for method in METHODS])
    paired, indices = bootstrap_rows(data)
    tails = {
        "max_force_gt1": tail_result(data, indices, 1.0, 5.0),
        "max_force_gt2": tail_result(data, indices, 2.0, 2.0),
    }
    force_boot = (data["GBSA-TCL"]["atomic_force"][indices].mean(axis=1) /
                  data["C0"]["atomic_force"][indices].mean(axis=1))
    force_ratio_ci = list(map(float, quantile(force_boot, [.025, .975])))
    atomic = {
        "c0_mean": float(data["C0"]["atomic_force"].mean()),
        "gbsa_mean": float(data["GBSA-TCL"]["atomic_force"].mean()),
        "absolute_delta": float((data["GBSA-TCL"]["atomic_force"]-data["C0"]["atomic_force"]).mean()),
        "mean_ratio": float(data["GBSA-TCL"]["atomic_force"].mean()/data["C0"]["atomic_force"].mean()),
        "paired_bootstrap_ratio_ci95": force_ratio_ci,
        "margin": 1.20, "status": ni_status(*force_ratio_ci, 1.20),
    }
    look = paired.set_index("metric")
    nus = look.loc["nus"]
    nus_status = "PASS" if nus.mean_delta >= 5.0 and nus.bootstrap_ci95_low > 0 else "FAIL"
    energy_low = float(look.loc["e_hull"].bootstrap_ci95_low)
    energy_high = float(look.loc["e_hull"].bootstrap_ci95_high)
    energy_status = ni_status(energy_low, energy_high, .010)
    rmsd_status = ni_status(float(look.loc["rmsd"].bootstrap_ci95_low),
                            float(look.loc["rmsd"].bootstrap_ci95_high), .020)
    collapse = {}
    for metric in ("novel", "unique"):
        row = look.loc[metric]
        clear = bool(row.mean_delta <= -10.0 and row.bootstrap_ci95_high < 0)
        collapse[metric] = {"delta_pp": float(row.mean_delta),
                            "ci95_pp": [float(row.bootstrap_ci95_low), float(row.bootstrap_ci95_high)],
                            "clear_collapse": clear}
    invalid_count = int(quality.technical_invalid_count.sum())
    max_tail_status = ("FAIL" if any(item["status"] == "FAIL" for item in tails.values()) else
                       "PASS" if all(item["status"] == "PASS" for item in tails.values()) else "BORDERLINE")
    clear_go = (nus_status == energy_status == rmsd_status == atomic["status"] == "PASS" and
                max_tail_status != "FAIL" and not any(item["clear_collapse"] for item in collapse.values()) and
                invalid_count == 0)
    direct_fail = (nus_status == "FAIL" or "FAIL" in {energy_status, rmsd_status, atomic["status"], max_tail_status} or
                   any(item["clear_collapse"] for item in collapse.values()) or invalid_count > 0)
    verdict = "CLEAR GO" if clear_go else "FAIL" if direct_fail else "BORDERLINE"
    loo, loo_diagnostic = leave_one_out(data)
    per_seed = []
    for method in METHODS:
        values = data[method]
        for index, seed in enumerate(SEEDS):
            flags = []
            if values["e_hull"][index] > .5: flags.append("E-hull>0.5")
            if values["rmsd"][index] > .5: flags.append("RMSD>0.5")
            if values["atomic_force"][index] > 1: flags.append("AtomicF>1")
            if values["max_force"][index] > 1: flags.append("MaxF>1")
            if values["max_force"][index] > 2: flags.append("MaxF>2")
            if values["steps"][index] > 200: flags.append("Steps>200")
            if values["steps"][index] > 400: flags.append("Steps>400")
            if values["final_max_force"][index] > .05: flags.append("FinalMaxF>0.05")
            invalid = not (values["generated_finite_positive_cell"][index] and
                           values["initial_finite_positive_cell_force"][index] and
                           values["relaxed_finite_positive_cell_force"][index])
            if invalid: flags.append("TECHNICAL_INVALID")
            per_seed.append({
                "method": method, "seed": seed, "formula": values["formula"][index],
                "e_hull": values["e_hull"][index], "stable": bool(values["stable"][index]),
                "nus": bool(values["nus"][index]), "novel": bool(values["novel"][index]),
                "unique": bool(values["unique"][index]), "rmsd": values["rmsd"][index],
                "atomic_force": values["atomic_force"][index], "max_force": values["max_force"][index],
                "final_max_force": values["final_max_force"][index], "steps": int(values["steps"][index]),
                "invalid": invalid, "diagnostic_flags": ";".join(flags), "dft_verified": False,
            })
    per_seed = pd.DataFrame(per_seed)
    severe = per_seed[(per_seed.method == "GBSA-TCL") & per_seed.diagnostic_flags.astype(bool)].copy()
    decision = {
        "historical_gbsa_p1_verdict": "FAIL (unchanged)",
        "prospective32_verdict": verdict, "enter_independent64": verdict == "CLEAR GO",
        "primary_nus": {"delta_pp": float(nus.mean_delta),
                        "ci95_pp": [float(nus.bootstrap_ci95_low), float(nus.bootstrap_ci95_high)],
                        "status": nus_status},
        "primary_e_hull": {"delta": float(look.loc["e_hull"].mean_delta),
                           "ci95": [energy_low, energy_high], "margin": .010, "status": energy_status},
        "rmsd_guardrail": {"delta": float(look.loc["rmsd"].mean_delta),
                           "ci95": [float(look.loc["rmsd"].bootstrap_ci95_low),
                                    float(look.loc["rmsd"].bootstrap_ci95_high)],
                           "margin": .020, "status": rmsd_status},
        "atomic_force_guardrail": atomic, "max_force_tail": {**tails, "overall_status": max_tail_status},
        "diversity_collapse": collapse, "technical_invalid_count": invalid_count,
        "relaxation_job_failure_count": 0,
        "explicit_optimizer_convergence_field_available": False,
        "final_atomic_fmax_exceedance_is_diagnostic_not_equivalent_to_expcellfilter_convergence": True,
        "technical_rerun_count": int(generation.technical_rerun.astype(bool).sum()),
        "leave_one_out": loo_diagnostic, "dft_verified": False,
        "recommend_stop_and_fallback_e3_pcr": verdict != "CLEAR GO",
    }
    quality.to_csv(ROOT / "quality_results.csv", index=False)
    paired.to_csv(ROOT / "paired_statistics.csv", index=False)
    per_seed.to_csv(ROOT / "per_seed_results.csv", index=False)
    severe.to_csv(ROOT / "severe_gbsa_seeds.csv", index=False)
    loo.to_csv(ROOT / "leave_one_out.csv", index=False)
    (ROOT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print(quality.to_string(index=False), flush=True)
    print("severe GBSA seeds", flush=True)
    print(severe.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
