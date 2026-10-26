# Robust Adaptive CFG V2 final report

## Frozen method

Multi-field conditional residual → timestep/phase calibration → median field consensus → phase-specific online EMA → magnitude/agreement confidence → fallback to fixed CFG=2 when uncertain → bounded adaptive guidance [1.5, 2.5] → 0.05 slew-rate stabilization. Stage gating is OFF.

## 1. Why V1 was unstable

Stage 0 linked 144,000 controller decisions from 72 historical paired seeds. Raw residuals were strongly timestep-dependent, field scales differed by orders of magnitude, and mean log-residual scale shifted across observed cohorts. Harmful samples had greater average field disagreement, while extreme CFG events alone did not consistently distinguish harm. Formal256 endpoints were retained, but its controller trajectories were never archived.

## 2. Calibration

The calibration used 64 historical fixed-CFG=2 deterministic replays and 128,000 score decisions. Every replayed final structure numerically matched the archived C0 structure. No quality endpoint entered calibration and no hyperparameter sweep was used.

## 3. Fresh P0 decision

```json
{
  "ROBUST_ADAPTIVE_CFG_P0": "FAIL",
  "formal256_allowed": false,
  "next": "STOP_NO_RESCUE_TUNING",
  "harm_relative_reduction_vs_A_OLD": {
    "ehull": 0.0,
    "stable": 0.0,
    "property": 0.0
  },
  "at_least_one_harm_relative_reduction_ge_30pct": false,
  "no_other_harm_increase_gt_5pp": true,
  "mean_quality_changes_A_ROBUST_minus_C0": {
    "ehull": 0.0061503838013679535,
    "stable": 0.0,
    "nus": 0.03125
  },
  "at_least_one_mean_quality_direction_favorable": true,
  "guardrails": {
    "mag_mae_worsening_le_10pct": false,
    "validity_drop_le_5pp": true,
    "nus_drop_le_5pp": true
  },
  "no_severe_stable_collapse_gt_20pp": true,
  "cohort": "p0",
  "n_paired": 32,
  "all_registered_seeds_retained": true,
  "bootstrap_resamples": 20000,
  "bootstrap_seed": 20260916,
  "SURROGATE_PROPERTY_EVAL": true,
  "DFT_VERIFIED": false,
  "parameters_retuned_after_results": false
}
```

## 4. Fresh Formal256 decision

```json
{
  "ROBUST_ADAPTIVE_CFG_FORMAL256": "NOT_RUN",
  "reason": "P0=FAIL; frozen stop rule applied",
  "generation_started": false,
  "rescue_tuning": false
}
```

## Final status

```text
ROBUST_ADAPTIVE_CFG_P0 = FAIL
ROBUST_ADAPTIVE_CFG_FORMAL256 = NOT_RUN
INNOVATION1_FINAL_STATUS = MIXED
```

All registered samples were retained. Parameters and seeds were not modified after observing P0 or Formal256. Energy/stability and magnetic-property evaluations use frozen MatterSim/CHGNet surrogates (`SURROGATE_PROPERTY_EVAL=True`, `DFT_VERIFIED=False`).
