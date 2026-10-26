"""Frozen 16-pair P0 analysis, bootstrap, decision, and required artifacts."""
from __future__ import annotations

import csv
from datetime import datetime
import gzip
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess

from ase.io import read
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
METHODS = ("B0", "F0")
SEEDS = tuple(range(710032, 710048))
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260913
MODEL = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/hf_mattergen/"
    "checkpoints/dft_mag_density/checkpoints/last.ckpt"
)
MATTERSIM = Path(
    "/mnt/datasets-livsyn/dxl/mattergen_v1/checkpoints/official/mattersim/"
    "mattersim-v1.0.0-5M.pth"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


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


def aggregate_guidance_trace() -> list[dict[str, str]]:
    rows = []
    for seed in SEEDS:
        path = ROOT / "p0_generation" / "F0" / str(seed) / "late_force_trace.csv"
        seed_rows = read_csv(path)
        if len(seed_rows) != 20 or {int(row["seed"]) for row in seed_rows} != {seed}:
            raise RuntimeError(f"F0/{seed}: expected exactly 20 frozen trace rows")
        rows.extend(seed_rows)
    write_csv(ROOT / "guidance_trace.csv", rows)
    return rows


def load_method(method: str, generation_rows: list[dict[str, str]],
                property_rows: list[dict[str, str]]) -> list[dict]:
    generated = sorted(
        (row for row in generation_rows if row["method"] == method),
        key=lambda row: int(row["seed"]),
    )
    properties = sorted(
        (row for row in property_rows if row["method"] == method),
        key=lambda row: int(row["seed"]),
    )
    with gzip.open(ROOT / "p0_quality" / method / "official_detailed.json.gz", "rt") as stream:
        detail = json.load(stream)
    relaxation = json.loads(
        (ROOT / "p0_relaxation" / method / "relaxation_summary.json").read_text()
    )
    initial = read(ROOT / "p0_relaxation" / method / "initial_with_properties.extxyz", index=":")
    expected = list(SEEDS)
    if [int(row["seed"]) for row in generated] != expected:
        raise RuntimeError(f"{method}: generation seed mismatch")
    if [int(row["seed"]) for row in properties] != expected:
        raise RuntimeError(f"{method}: property seed mismatch")
    if list(map(int, relaxation["sample_seeds"])) != expected or relaxation["success"] is not True:
        raise RuntimeError(f"{method}: relaxation seed/success mismatch")
    lengths = {len(generated), len(properties), len(initial)}
    lengths.update(len(detail[key]) for key in (
        "energy_above_hull_per_atom", "stable", "novel_unique_stable",
        "novel", "unique", "rmsd_from_relaxation",
    ))
    if lengths != {len(SEEDS)}:
        raise RuntimeError(f"{method}: metric count mismatch {sorted(lengths)}")
    rows = []
    for index, seed in enumerate(SEEDS):
        force_norms = np.linalg.norm(np.asarray(initial[index].get_forces()), axis=1)
        row = {
            "method": method,
            "seed": seed,
            "formula": generated[index]["formula"],
            "num_atoms": int(generated[index]["num_atoms"]),
            "maxF_ev_per_a": float(force_norms.max()),
            "atomic_force_mean_ev_per_a": float(force_norms.mean()),
            "rmsd_a": float(detail["rmsd_from_relaxation"][index]),
            "e_hull_ev_per_atom": float(detail["energy_above_hull_per_atom"][index]),
            "stable": bool(detail["stable"][index]),
            "nus": bool(detail["novel_unique_stable"][index]),
            "novel": bool(detail["novel"][index]),
            "unique": bool(detail["unique"][index]),
            "mag_density_a3": float(properties[index]["chgnet_mag_density"]),
            "mag_absolute_error_a3": float(properties[index]["mag_absolute_error"]),
            "mag_hit": truth(properties[index]["mag_hit"]),
            "valid": truth(properties[index]["structure_valid"]),
            "minimum_periodic_distance_a": float(properties[index]["minimum_periodic_distance_angstrom"]),
            "generation_seconds": float(generated[index]["elapsed_seconds"]),
            "end_to_end_generation_seconds": float(generated[index]["end_to_end_seconds"]),
            "mattergen_score_calls": int(generated[index]["mattergen_score_calls"]),
            "guidance_calls": int(generated[index]["guidance_eligible"]),
            "guidance_accepted": int(generated[index]["guidance_accepted"]),
            "guidance_fallbacks": int(generated[index]["guidance_fallbacks"]),
            "guidance_seconds": float(generated[index]["guidance_seconds"]),
            "relaxation_steps": int(relaxation["relaxation_steps"][index]),
            "surrogate_property_eval": True,
            "dft_verified": False,
        }
        for key, value in row.items():
            if (isinstance(value, float) and not np.isfinite(value)
                    and not (key == "minimum_periodic_distance_a" and len(initial[index]) == 1)):
                raise RuntimeError(f"{method}/{seed}: nonfinite {key}")
        rows.append(row)
    return rows


def array(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=float)


def summarize(method: str, rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(array(rows, key).mean())

    return {
        "method": method, "n": len(rows),
        "maxF_mean_ev_per_a": mean("maxF_ev_per_a"),
        "maxF_median_ev_per_a": float(np.median(array(rows, "maxF_ev_per_a"))),
        "maxF_p95_ev_per_a": float(np.quantile(array(rows, "maxF_ev_per_a"), 0.95)),
        "atomic_force_mean_ev_per_a": mean("atomic_force_mean_ev_per_a"),
        "rmsd_mean_a": mean("rmsd_a"),
        "e_hull_mean_ev_per_atom": mean("e_hull_ev_per_atom"),
        "stable_fraction": mean("stable"), "nus_fraction": mean("nus"),
        "novel_fraction": mean("novel"), "unique_fraction": mean("unique"),
        "mag_mae_a3": mean("mag_absolute_error_a3"),
        "mag_hit_fraction": mean("mag_hit"), "validity_fraction": mean("valid"),
        "minimum_distance_min_a": float(array(rows, "minimum_periodic_distance_a").min()),
        "relaxation_steps_mean": mean("relaxation_steps"),
        "surrogate_property_eval": True, "dft_verified": False,
    }


def efficiency(method: str, rows: list[dict], baseline_end_to_end: float) -> dict:
    end_to_end = float(array(rows, "end_to_end_generation_seconds").mean())
    return {
        "method": method, "n": len(rows),
        "generation_seconds_mean": float(array(rows, "generation_seconds").mean()),
        "end_to_end_generation_seconds_mean": end_to_end,
        "runtime_ratio_vs_B0": end_to_end / baseline_end_to_end,
        "mattergen_score_calls_mean": float(array(rows, "mattergen_score_calls").mean()),
        "guidance_calls_total": int(array(rows, "guidance_calls").sum()),
        "guidance_accepted_total": int(array(rows, "guidance_accepted").sum()),
        "guidance_fallbacks_total": int(array(rows, "guidance_fallbacks").sum()),
        "guidance_seconds_mean": float(array(rows, "guidance_seconds").mean()),
    }


def paired_statistics(data: dict[str, list[dict]]) -> tuple[list[dict], np.ndarray]:
    specifications = (
        ("maxF", "maxF_ev_per_a", "lower", "eV/angstrom", 1.0),
        ("atomic_force_mean", "atomic_force_mean_ev_per_a", "lower", "eV/angstrom", 1.0),
        ("rmsd", "rmsd_a", "lower", "angstrom", 1.0),
        ("e_hull", "e_hull_ev_per_atom", "lower", "eV/atom", 1.0),
        ("stable", "stable", "higher", "percentage_points", 100.0),
        ("nus", "nus", "higher", "percentage_points", 100.0),
        ("novel", "novel", "higher", "percentage_points", 100.0),
        ("unique", "unique", "higher", "percentage_points", 100.0),
        ("mag_absolute_error", "mag_absolute_error_a3", "lower", "angstrom^-3", 1.0),
        ("mag_hit", "mag_hit", "higher", "percentage_points", 100.0),
        ("validity", "valid", "higher", "percentage_points", 100.0),
        ("end_to_end_runtime", "end_to_end_generation_seconds", "lower", "seconds", 1.0),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(SEEDS), size=(N_BOOTSTRAP, len(SEEDS)))
    rows = []
    for metric, key, direction, unit, scale in specifications:
        baseline = array(data["B0"], key)
        candidate = array(data["F0"], key)
        raw_delta = candidate - baseline
        delta = raw_delta * scale
        bootstrap = delta[indices].mean(axis=1)
        tolerance = 1e-12
        favorable = raw_delta < -tolerance if direction == "lower" else raw_delta > tolerance
        unfavorable = raw_delta > tolerance if direction == "lower" else raw_delta < -tolerance
        ties = ~(favorable | unfavorable)
        rows.append({
            "comparison": "F0-B0", "metric": metric,
            "preferred_direction": direction, "unit": unit, "n_pairs": len(SEEDS),
            "F0_mean": float(candidate.mean() * scale), "B0_mean": float(baseline.mean() * scale),
            "mean_delta_F0_minus_B0": float(delta.mean()),
            "median_paired_delta": float(np.median(delta)),
            "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
            "bootstrap_resamples": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
            "F0_favorable_pairs": int(favorable.sum()),
            "B0_favorable_pairs": int(unfavorable.sum()), "ties": int(ties.sum()),
        })
    baseline_maxf = array(data["B0"], "maxF_ev_per_a")
    candidate_maxf = array(data["F0"], "maxF_ev_per_a")
    bootstrap_improvement = (
        baseline_maxf[indices].mean(axis=1) - candidate_maxf[indices].mean(axis=1)
    ) / baseline_maxf[indices].mean(axis=1)
    return rows, bootstrap_improvement


def paired_per_seed(data: dict[str, list[dict]]) -> list[dict]:
    rows = []
    keys = (
        "maxF_ev_per_a", "atomic_force_mean_ev_per_a", "rmsd_a",
        "e_hull_ev_per_atom", "stable", "nus", "novel", "unique",
        "mag_absolute_error_a3", "mag_hit", "valid",
        "end_to_end_generation_seconds",
    )
    for index, seed in enumerate(SEEDS):
        baseline = data["B0"][index]
        candidate = data["F0"][index]
        row = {
            "seed": seed, "B0_formula": baseline["formula"],
            "F0_formula": candidate["formula"],
            "maxF_pair_outcome_for_F0": (
                "win" if candidate["maxF_ev_per_a"] < baseline["maxF_ev_per_a"] - 1e-12
                else "loss" if candidate["maxF_ev_per_a"] > baseline["maxF_ev_per_a"] + 1e-12
                else "tie"
            ),
        }
        for key in keys:
            row[f"B0_{key}"] = baseline[key]
            row[f"F0_{key}"] = candidate[key]
            row[f"delta_F0_minus_B0_{key}"] = float(candidate[key]) - float(baseline[key])
        rows.append(row)
    return rows


def main() -> None:
    trace_rows = aggregate_guidance_trace()
    generation = read_csv(ROOT / "p0_generation_results.csv")
    properties = read_csv(ROOT / "p0_property_metrics.csv")
    data = {method: load_method(method, generation, properties) for method in METHODS}
    heldout = data["B0"] + data["F0"]
    write_csv(ROOT / "heldout_metrics.csv", heldout)
    quality = [summarize(method, data[method]) for method in METHODS]
    write_csv(ROOT / "quality_metrics.csv", quality)
    baseline_runtime = float(array(data["B0"], "end_to_end_generation_seconds").mean())
    efficiencies = [efficiency(method, data[method], baseline_runtime) for method in METHODS]
    write_csv(ROOT / "efficiency_metrics.csv", efficiencies)
    paired, bootstrap_improvement = paired_statistics(data)
    write_csv(ROOT / "p0_bootstrap.csv", paired)
    per_seed = paired_per_seed(data)
    write_csv(ROOT / "paired_results.csv", per_seed)
    write_csv(ROOT / "p0_paired_per_seed.csv", per_seed)

    b0 = quality[0]
    f0 = quality[1]
    maxf_improvement = (b0["maxF_mean_ev_per_a"] - f0["maxF_mean_ev_per_a"]) / b0["maxF_mean_ev_per_a"]
    maxf_delta = array(data["F0"], "maxF_ev_per_a") - array(data["B0"], "maxF_ev_per_a")
    wins = int((maxf_delta < -1e-12).sum())
    losses = int((maxf_delta > 1e-12).sum())
    ties = len(SEEDS) - wins - losses
    mag_worsening = (f0["mag_mae_a3"] - b0["mag_mae_a3"]) / b0["mag_mae_a3"]
    nus_drop_pp = (b0["nus_fraction"] - f0["nus_fraction"]) * 100.0
    validity_drop_pp = (b0["validity_fraction"] - f0["validity_fraction"]) * 100.0
    ehull_increase = f0["e_hull_mean_ev_per_atom"] - b0["e_hull_mean_ev_per_atom"]
    runtime_ratio = efficiencies[1]["runtime_ratio_vs_B0"]
    guardrails = {
        "mag_mae_worsening_le_10pct": bool(mag_worsening <= 0.10 + 1e-12),
        "nus_drop_le_5pp": bool(nus_drop_pp <= 5.0 + 1e-12),
        "validity_drop_le_5pp": bool(validity_drop_pp <= 5.0 + 1e-12),
        "e_hull_increase_le_0.01_ev_atom": bool(ehull_increase <= 0.01 + 1e-12),
        "runtime_ratio_le_1.5": bool(runtime_ratio <= 1.5 + 1e-12),
    }
    all_guardrails = all(guardrails.values())
    if maxf_improvement >= 0.15 and wins >= 11 and all_guardrails:
        verdict = "GO"
        next_action = "FORMAL32"
    elif maxf_improvement < 0.08 or wins <= 8 or not all_guardrails:
        verdict = "FAIL"
        next_action = "STOP_A"
    else:
        verdict = "BORDERLINE"
        next_action = "STOP_A_NO_RESCUE_SWEEP"

    trace_max_correction = max(float(row["max_cart_correction_a"]) for row in trace_rows)
    decision = {
        "LATE_PHYSICS_SIGNAL_RELIABLE": "YES",
        "MATTERSIM_FORCE_GUIDANCE_P0": verdict,
        "NEXT": next_action,
        "Adaptive_CFG_used": "NO",
        "SURROGATE_PROPERTY_EVAL": True,
        "DFT_VERIFIED": False,
        "n_paired": len(SEEDS),
        "primary": "MatterSim pre-relaxation MaxF",
        "maxF_B0_mean_ev_per_a": b0["maxF_mean_ev_per_a"],
        "maxF_F0_mean_ev_per_a": f0["maxF_mean_ev_per_a"],
        "mean_maxF_improvement_fraction": maxf_improvement,
        "mean_maxF_improvement_percent": maxf_improvement * 100.0,
        "maxF_improvement_bootstrap_ci95_percent": [
            float(np.quantile(bootstrap_improvement, 0.025) * 100.0),
            float(np.quantile(bootstrap_improvement, 0.975) * 100.0),
        ],
        "maxF_paired_wins_ties_losses": [wins, ties, losses],
        "guardrail_values": {
            "mag_mae_worsening_fraction": mag_worsening,
            "nus_drop_pp": nus_drop_pp,
            "validity_drop_pp": validity_drop_pp,
            "e_hull_increase_ev_atom": ehull_increase,
            "runtime_ratio": runtime_ratio,
        },
        "guardrail_pass": guardrails,
        "all_guardrails_pass": all_guardrails,
        "guidance": {
            "eligible_calls": len(trace_rows),
            "accepted_calls": sum(truth(row["accepted"]) for row in trace_rows),
            "fallback_calls": sum(truth(row["fallback"]) for row in trace_rows),
            "max_observed_cartesian_correction_a": trace_max_correction,
            "hard_cartesian_cap_a": 0.01,
            "absolute_protocol_cap_a": 0.02,
            "guidance_lambda": 1.0,
            "lambda_sweep": False,
        },
        "formal32_run": False,
        "formal256_run": False,
    }
    (ROOT / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "p0_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(ROOT / "p0_summary.csv", [{
        "verdict": verdict, "next": next_action, "n_paired": len(SEEDS),
        "B0_mean_maxF_ev_per_a": b0["maxF_mean_ev_per_a"],
        "F0_mean_maxF_ev_per_a": f0["maxF_mean_ev_per_a"],
        "mean_maxF_improvement_percent": maxf_improvement * 100.0,
        "F0_maxF_wins": wins, "ties": ties, "F0_maxF_losses": losses,
        "mag_mae_worsening_percent": mag_worsening * 100.0,
        "nus_drop_pp": nus_drop_pp, "validity_drop_pp": validity_drop_pp,
        "e_hull_increase_ev_atom": ehull_increase, "runtime_ratio": runtime_ratio,
        "all_guardrails_pass": all_guardrails,
        "surrogate_property_eval": True, "dft_verified": False,
    }])

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "worktree": str(PROJECT_ROOT), "git_head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "git_status_short": git("status", "--short").splitlines(),
        "stash_list_preserved": git("stash", "list").splitlines(),
        "mattergen_checkpoint": str(MODEL), "mattergen_checkpoint_sha256": sha256(MODEL),
        "mattersim_checkpoint": str(MATTERSIM), "mattersim_checkpoint_sha256": sha256(MATTERSIM),
        "mattersim_package_version": metadata.version("mattersim"),
        "chgnet_package_version": "0.3.0",
        "gpu_pool_physical": sorted({int(row["physical_gpu"]) for row in generation}),
        "gpu_2_excluded": True,
        "runtime_contention_note": (
            "Seeds 710033 on GPU0 and 710034 on GPU4 completed under documented external "
            "contention from PIDs 3760852-3760855. No further A work was dispatched to "
            "GPU0/4; remaining seeds migrated uniquely to GPUs5/6/7 without reruns."
        ),
        "diagnostic_seeds": [710000, 710031], "p0_paired_seeds": list(SEEDS),
        "seed_freshness_audit": (
            "Before registration, all /mnt/datasets-livsyn/dxl/mattergen_v1* paths and text "
            "contained no historical 710000-710099 or 720000-720099 references."
        ),
        "technical_failed_attempts_retained": [
            {
                "stage": "diagnostic_seed_710000_smoke",
                "cause": "frozen MatterGen dependency used removed numpy.math under NumPy 2",
                "repair": "local compatibility alias np.math=math; no scientific change",
                "path": "failed_attempts/710000_numpy2_smoke",
            },
            {
                "stage": "p0_relaxation_launcher",
                "cause": "independent worktree lacked the untracked worktree-relative MatterSim checkpoint",
                "repair": "frozen main-project absolute path plus pre-run SHA256 validation",
                "path": "failed_attempts/relaxation_bad_worktree_checkpoint_path",
                "structures_relaxed_before_failure": 0,
            },
        ],
        "baseline": {"target_dft_mag_density": 0.2, "cfg": 2.0,
                     "diffusion_steps": 1000, "corrector_steps": 1,
                     "adaptive_cfg": False},
        "forbidden_components_used": [],
        "stress_guidance": False, "cell_guidance": False,
        "atomic_guidance": False, "retraining": False,
    }
    (ROOT / "guidance_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = f"""# Late-stage MatterSim position-force guidance P0

## Decision

**LATE_PHYSICS_SIGNAL_RELIABLE=YES**  
**MATTERSIM_FORCE_GUIDANCE_P0={verdict}**  
**NEXT={next_action}**

Adaptive CFG used? **NO**. `SURROGATE_PROPERTY_EVAL=True` and
`DFT_VERIFIED=False`. Formal32 and Formal256 were not run.

## Diagnostic

The frozen 32-seed clean-estimate diagnostic passed: the best late point was
`t_norm=0.02`, where clean-estimate MaxF predicted final MaxF with Spearman
rho=0.925587 and 100% finite/valid MatterSim evaluations. The final row marked
`requested_t_norm=0.00` is actually the state after the epsilon predictor at
`actual_model_t=0.001`, not a mathematical t=0 model evaluation.

## Frozen guidance mechanism

Only the position predictor is guided for `t<=0.02`. MatterSim force is
evaluated on complete clean `x0_hat(t)`, centered to remove translation, and
converted using the row-vector identity `delta_frac = delta_cart @ inv(cell)`.
The single frozen lambda is 1.0; there was no sweep. Nominal per-atom Cartesian
contribution is 0.005 A, hard-capped at 0.01 A (below the protocol's absolute
0.02 A limit). A candidate with minimum periodic distance below 0.5 A falls
back to the untouched official score. Cell, atomic-type, and stress scores are
never guided.

Across {len(trace_rows)} eligible calls, {decision['guidance']['accepted_calls']} were accepted and
{decision['guidance']['fallback_calls']} fell back. The largest observed correction was
{trace_max_correction:.6g} A.

## Paired P0 effect (16 fresh seeds)

| Metric | B0 | F0 | F0-B0 / improvement |
|---|---:|---:|---:|
| Mean MaxF (eV/A) | {b0['maxF_mean_ev_per_a']:.6g} | {f0['maxF_mean_ev_per_a']:.6g} | {maxf_improvement*100:.3f}% improvement |
| Mean atomic force (eV/A) | {b0['atomic_force_mean_ev_per_a']:.6g} | {f0['atomic_force_mean_ev_per_a']:.6g} | {f0['atomic_force_mean_ev_per_a']-b0['atomic_force_mean_ev_per_a']:+.6g} |
| Mean RMSD (A) | {b0['rmsd_mean_a']:.6g} | {f0['rmsd_mean_a']:.6g} | {f0['rmsd_mean_a']-b0['rmsd_mean_a']:+.6g} |
| Mean E-hull (eV/atom) | {b0['e_hull_mean_ev_per_atom']:.6g} | {f0['e_hull_mean_ev_per_atom']:.6g} | {ehull_increase:+.6g} |
| Stable | {b0['stable_fraction']*100:.2f}% | {f0['stable_fraction']*100:.2f}% | {(f0['stable_fraction']-b0['stable_fraction'])*100:+.2f} pp |
| NUS | {b0['nus_fraction']*100:.2f}% | {f0['nus_fraction']*100:.2f}% | {(f0['nus_fraction']-b0['nus_fraction'])*100:+.2f} pp |
| Mag MAE (A^-3) | {b0['mag_mae_a3']:.6g} | {f0['mag_mae_a3']:.6g} | {mag_worsening*100:+.3f}% |
| Validity | {b0['validity_fraction']*100:.2f}% | {f0['validity_fraction']*100:.2f}% | {-validity_drop_pp:+.2f} pp |
| End-to-end generation | {baseline_runtime:.3f}s | {efficiencies[1]['end_to_end_generation_seconds_mean']:.3f}s | {runtime_ratio:.4f}x |

MaxF paired wins/ties/losses for F0 are **{wins}/{ties}/{losses}**. The paired
20,000-resample bootstrap 95% CI for relative mean MaxF improvement is
[{decision['maxF_improvement_bootstrap_ci95_percent'][0]:.3f}%,
 {decision['maxF_improvement_bootstrap_ci95_percent'][1]:.3f}%]. Full per-seed,
quality, efficiency, and bootstrap results are in the required CSV artifacts.

Seeds 710033 (GPU0) and 710034 (GPU4) completed under logged external GPU
contention from PIDs 3760852-3760855. After those pairs completed, no further
A work was dispatched to GPU0/4; the remaining unique seeds used GPUs5/6/7.
The runtime gate uses every preregistered pair, including the contended runs.

## Gate application

GO requires mean MaxF improvement >=15%, >=11/16 paired wins, and all five
guardrails. The frozen gate produced `{verdict}`. This stage performs no rescue
sweep and does not automatically start Formal32.

## Execution audit

Two technical, pre-scientific failures are retained under `failed_attempts/`:
the first diagnostic smoke exposed a NumPy-2 `np.math` compatibility issue,
and the first relaxation launcher used a checkpoint path relative to the
independent worktree. The fixes were respectively a compatibility alias and
the already frozen main-project absolute checkpoint path with SHA256
verification. Neither changed a seed, structure, model, threshold, lambda, or
scientific protocol; the relaxation path failure occurred before any structure
was relaxed. All 32 diagnostic trajectories and all 16 P0 pairs completed.
"""
    (ROOT / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
