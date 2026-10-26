"""Emit explicit NOT_RUN artifacts after a hard B0 diagnostic failure."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def write_csv(name: str, fieldnames: list[str], row: dict[str, object]) -> None:
    with (ROOT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def fmt(value: float | None, digits: int = 6) -> str:
    return "NOT_ESTIMABLE" if value is None else f"{value:.{digits}f}"


def main() -> None:
    diagnostic = json.loads((ROOT / "diagnostic_summary.json").read_text(encoding="utf-8"))
    if diagnostic["diagnostic"] != "FAIL" or diagnostic["FP_PC_MOTIVATED"] != "NO":
        raise RuntimeError("stop finalizer is legal only after diagnostic FAIL")
    reason = diagnostic["gate_reason"]
    status = "NOT_RUN_DIAGNOSTIC_FAIL"
    region = diagnostic["thresholds"]

    (ROOT / "fp_pc_config.yaml").write_text(
        f"""experiment: fp_pc_p0
stage: P0
status: {status}
reason: {reason}
executed: false
baseline:
  model: official_dft_mag_density_full_ft
  target_dft_mag_density: 0.2
  cfg: 2.0
  guidance_schedule: constant
  adaptive_cfg: false
  diffusion_steps: 1000
  corrector_steps_per_timestep: 1
planned_method_not_implemented:
  fields: [pos, cell]
  atomic_numbers_changed: false
  same_check_for_predictor_and_corrector: true
  d_min_floor_angstrom: {region['d_min_floor_angstrom']}
  VN_low_angstrom3_per_atom: {region['VN_low_angstrom3_per_atom']}
  VN_high_angstrom3_per_atom: {region['VN_high_angstrom3_per_atom']}
  initial_alpha: 0.5
  alpha_multiplier: 0.5
  maximum_backtracks: 4
  fallback_after_failure: current_state
  feasible_proposal_behavior: accept_unmodified
p0_seeds_reserved_but_not_used: [720000, 720015]
formal32: prohibited
formal256: prohibited
""", encoding="utf-8"
    )
    trace_fields = [
        "status", "reason", "seed", "timestep", "stage", "min_distance_before",
        "min_distance_candidate", "min_distance_after", "VN_before", "VN_candidate",
        "VN_after", "violated_distance", "violated_volume", "alpha_final",
        "num_backtracks", "fallback",
    ]
    write_csv("fp_pc_trace.csv", trace_fields, {"status": status, "reason": reason})
    write_csv(
        "heldout_metrics.csv",
        ["status", "reason", "method", "n", "mean_maxF", "mean_RMSD", "mean_Ehull", "stable_rate"],
        {"status": status, "reason": reason, "method": "P0", "n": 0},
    )
    write_csv(
        "quality_metrics.csv",
        ["status", "reason", "method", "n", "mag_MAE", "hit_rate", "validity", "NUS", "novel", "unique"],
        {"status": status, "reason": reason, "method": "P0", "n": 0},
    )
    write_csv(
        "efficiency_metrics.csv",
        ["status", "reason", "method", "n", "runtime_seconds", "runtime_ratio", "backtracking_count", "fallback_count"],
        {"status": status, "reason": reason, "method": "P0", "n": 0},
    )
    write_csv(
        "paired_results.csv",
        ["status", "reason", "metric", "n_pairs", "mean_delta", "median_delta", "ci95_low", "ci95_high", "wins", "ties", "losses"],
        {"status": status, "reason": reason, "metric": "NOT_RUN", "n_pairs": 0},
    )

    decision = {
        "schema_version": 1,
        "candidate": "FP-PC",
        "diagnostic": "FAIL",
        "FP_PC_MOTIVATED": "NO",
        "FP_PC_P0": status,
        "reason": reason,
        "p0_executed": False,
        "fresh_p0_seeds_consumed": [],
        "next": "STOP_FP_PC_NO_RESCUE",
        "adaptive_cfg_used": False,
        "surrogate_property_eval": True,
        "dft_verified": False,
        "formal32_run": False,
        "formal256_run": False,
    }
    (ROOT / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    all_results = diagnostic["trajectory_results"]["all"]
    comparison = all_results["comparison"]
    timing = diagnostic["timing"]
    stage_totals = {
        period: sum(timing.get(f"{stage}_{period}", {}).get("event_count", 0)
                    for stage in ("corrector", "predictor"))
        for period in ("early", "mid", "late")
    }
    metrics = comparison["metrics"]
    burden = diagnostic["secondary_burden_correlations"]
    report = f"""# FP-PC P0 final report

## Decision

```text
FP_PC_MOTIVATED = NO
FP_PC_P0 = {status}
Adaptive CFG used? NO
SURROGATE_PROPERTY_EVAL=True
DFT_VERIFIED=False
NEXT=STOP_FP_PC_NO_RESCUE
```

The frozen B0 gate failed: **{reason}**.  Per protocol, FP-PC was not
implemented, the reserved fresh seeds `720000–720015` were not consumed, no
16-seed generation/evaluation was run, and no rescue threshold or sweep was
attempted.

## Frozen setup

- Worktree/branch: `/mnt/datasets-livsyn/dxl/mattergen_v1_fppc` /
  `experiment/fp-pc-p0`.
- MatterGen HEAD: `1af301c5e534ff6b9b9f96ea5dab48036d9c46af`.
- Official target-0.2 checkpoint SHA256:
  `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`.
- Constant CFG 2.0, 1000 steps, one Corrector step/time; Adaptive CFG OFF.
- Reference-only bounds: `d_min >= {region['d_min_floor_angstrom']:.6f} Å`,
  `{region['VN_low_angstrom3_per_atom']:.6f} <= V/N <= {region['VN_high_angstrom3_per_atom']:.6f} Å³/atom`.
- Threshold source: fixed 4096-structure reservoir from the official Alex-MP
  training split, q0.5%/q99.5%; 4096/4096 parsed, no held-out outcome used.

## B0 execution integrity

- 64 historical target-0.2 seeds (`99000–99063`) were deterministically replayed.
- Every trajectory produced 1000 Corrector plus 1000 Predictor observations:
  128,000 proposal rows total.
- Diagnostic accepted every official proposal unchanged; it did not implement
  FP-PC during data collection.
- Replay equivalence: atomic numbers/formula exact; maximum cell difference
  `{diagnostic['replay_equivalence']['max_abs_cell_difference_angstrom']:.3e} Å`,
  maximum periodic fractional-position difference
  `{diagnostic['replay_equivalence']['max_abs_periodic_fractional_position_difference']:.3e}`.
- Uncontended diagnostic sampling runtime had median `88.245 s/trajectory`
  (maximum `93.730 s`).  Seed `99014` took `424.270 s` after an unrelated
  external process occupied GPU1; it still completed with the exact 2000-call
  contract.  The next seed's interrupted empty attempt was retained in the
  audit log and then executed exactly once to completion on GPU6.  This resource
  event changes neither the stochastic seed nor any scientific measurement.

## Diagnostic questions

**Q1 — Predictor violation trajectories.**
`{int(round(all_results['predictor_violation_rate'] * all_results['n']))}/{all_results['n']}`
(`{100 * all_results['predictor_violation_rate']:.2f}%`) had at least one actionable
Predictor entry violation.

**Q2 — Corrector violation trajectories.**
`{int(round(all_results['corrector_violation_rate'] * all_results['n']))}/{all_results['n']}`
(`{100 * all_results['corrector_violation_rate']:.2f}%`) had at least one actionable
Corrector entry violation.

**Combined trajectory rate.**
`{comparison['violation_n']}/{all_results['n']}`
(`{100 * all_results['any_violation_rate']:.2f}%`) had either stage.  The GO rate
threshold was 20% and the hard low-rate failure threshold was 10%.

**Q3 — final-quality association.**  Group sizes were violation
`n={comparison['violation_n']}` and no-violation `n={comparison['no_violation_n']}`;
estimable=`{comparison['comparison_estimable']}`.  Mean relative/absolute worse
effects (violation minus no-violation; positive means worse) were:

- MaxF relative: `{fmt(metrics['final_maxF']['relative_mean_worse'])}`;
- RMSD relative: `{fmt(metrics['final_RMSD']['relative_mean_worse'])}`;
- E-hull absolute: `{fmt(metrics['final_Ehull']['absolute_mean_difference'])} eV/atom`.

None met the preregistered degradation test of +25%, +25%, or +0.02 eV/atom.
As a secondary, non-gating analysis, total entry count was *negatively*
associated with MaxF rather than positively associated with harm: Spearman was
`{burden['all']['violation_count']['final_maxF']['spearman']:.3f}` overall,
`{burden['calibration']['violation_count']['final_maxF']['spearman']:.3f}` in
calibration and
`{burden['validation']['violation_count']['final_maxF']['spearman']:.3f}` in
validation.  This supplies no rescue evidence for suppressing these proposals.

**Q4 — timing.** Actionable entry event counts were early/mid/late =
`{stage_totals['early']}/{stage_totals['mid']}/{stage_totals['late']}`.
Predictor total=`{diagnostic['proposal_entry_events']['predictor']}` and Corrector
total=`{diagnostic['proposal_entry_events']['corrector']}`.  Timing is descriptive;
it was not used to redefine the threshold or rescue the route.

Absolute candidate infeasibility occurred in
`{100 * diagnostic['absolute_candidate_infeasibility']['row_fraction']:.2f}%` of
proposal rows, but this includes intentionally noisy states already outside the
clean-reference region.  It is not the preregistered actionable gate and cannot
motivate backtracking from an infeasible current state.

## Interpretation

The method required both a sufficiently common **preventable** proposal crossing
and a materially worse final outcome for affected trajectories.  The observed
data failed that joint requirement.  Implementing a trust region now would be a
post-hoc intervention without the required causal/mechanistic motivation.  FP-PC
therefore stops at diagnostic FAIL.  Empty P0 tables are explicit NOT_RUN
artifacts, not missing or zero-valued experimental results.

No MatterGen training, force/energy optimization, atomic-type modification,
A+B combination, Formal32 or Formal256 was performed.  The five historical
stashes and all historical checkpoints/experiments remain untouched.
"""
    (ROOT / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
