# Robust Adaptive CFG V2 — Fresh P0

All 32 preregistered paired seeds were retained. The methods, controller, calibration, confidence rule, bounds, slew cap, EMA, metrics, bootstrap count, and decision thresholds were frozen before this cohort was generated. No rescue tuning was performed.

## Quality metrics

| method   |   n |   e_hull_mean_ev_per_atom |   e_hull_median_ev_per_atom |   stable_fraction |   novel_fraction |   unique_fraction |   nus_fraction |   validity_fraction |   mag_mae_a3 |   mag_hit_fraction |   maxF_mean_ev_per_a |   atomic_force_mean_ev_per_a |   rmsd_mean_a |   runtime_mean_seconds |   mattergen_score_calls_mean | surrogate_property_eval   | dft_verified   |
|:---------|----:|--------------------------:|----------------------------:|------------------:|-----------------:|------------------:|---------------:|--------------------:|-------------:|-------------------:|---------------------:|-----------------------------:|--------------:|-----------------------:|-----------------------------:|:--------------------------|:---------------|
| C0       |  32 |                 0.085572  |                   0.0792552 |           0.625   |          0.6875  |                 1 |        0.34375 |                   1 |    0.0232102 |            0.3125  |             0.218584 |                    0.106244  |     0.0560879 |                85.4491 |                         2000 | True                      | False          |
| A_OLD    |  32 |                 0.0999509 |                   0.0928659 |           0.5625  |          0.78125 |                 1 |        0.375   |                   1 |    0.0319091 |            0.1875  |             0.249955 |                    0.118641  |     0.053328  |                85.3079 |                         2000 | True                      | False          |
| A_NORM   |  32 |                 0.0933219 |                   0.0870604 |           0.59375 |          0.65625 |                 1 |        0.28125 |                   1 |    0.0278576 |            0.21875 |             0.208845 |                    0.106431  |     0.0725272 |                85.5134 |                         2000 | True                      | False          |
| A_ROBUST |  32 |                 0.0917224 |                   0.0702964 |           0.625   |          0.71875 |                 1 |        0.375   |                   1 |    0.0298703 |            0.1875  |             0.189954 |                    0.0963742 |     0.0414119 |                85.5566 |                         2000 | True                      | False          |

## Harm and tail metrics

| method   | baseline   |   n |   harm_rate_ehull |   harm_rate_stable |   harm_rate_property |   ehull_degradation_p75 |   ehull_degradation_p90 |   worst_quartile_ehull_degradation_mean |   property_degradation_p75 |   property_degradation_p90 |
|:---------|:-----------|----:|------------------:|-------------------:|---------------------:|------------------------:|------------------------:|----------------------------------------:|---------------------------:|---------------------------:|
| A_OLD    | C0         |  32 |           0.4375  |            0.1875  |              0.53125 |               0.0450951 |                0.143842 |                                0.129688 |                  0.0304495 |                  0.0435061 |
| A_NORM   | C0         |  32 |           0.34375 |            0.21875 |              0.46875 |               0.0460387 |                0.12236  |                                0.101814 |                  0.0220856 |                  0.0416604 |
| A_ROBUST | C0         |  32 |           0.4375  |            0.1875  |              0.53125 |               0.0471202 |                0.126682 |                                0.108847 |                  0.0265992 |                  0.0439791 |

## Controller behavior

| method   |   events |   cfg_mean |   cfg_median |   cfg_p05 |   cfg_p95 |   cfg_min |   cfg_max |   max_cfg_excursion |   time_away_from_baseline_rate |   confidence_active_rate |   fallback_to_g0_rate |   field_disagreement_mean |   field_disagreement_p95 |   slew_limit_trigger_count |
|:---------|---------:|-----------:|-------------:|----------:|----------:|----------:|----------:|--------------------:|-------------------------------:|-------------------------:|----------------------:|--------------------------:|-------------------------:|---------------------------:|
| C0       |    64000 |    2       |      2       |   2       |   2       |   2       |   2       |            0        |                       0        |               nan        |            nan        |                 nan       |                nan       |                        nan |
| A_OLD    |    64000 |    2.01851 |      1.99469 |   1.67658 |   2.40511 |   1.1918  |   5       |            3        |                       0.938969 |               nan        |            nan        |                 nan       |                nan       |                        nan |
| A_NORM   |    64000 |    1.61207 |      1.53804 |   1.50247 |   2.017   |   1.50247 |   2.49753 |            0.497527 |                       0.996078 |                 1        |            nan        |                   1.05368 |                  1.69103 |                        nan |
| A_ROBUST |    64000 |    1.87877 |      1.88234 |   1.73717 |   2       |   1.50275 |   2.21983 |            0.497248 |                       0.875016 |                 0.897266 |              0.102734 |                   1.02633 |                  1.68356 |                       7772 |

## Decision

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

All energy/stability and magnetic-property measurements are frozen surrogate evaluations. `SURROGATE_PROPERTY_EVAL=True`; `DFT_VERIFIED=False`. A P0 GO is only permission to run Formal256, not confirmatory evidence. Formal256 confirmation requires a favorable headline bootstrap CI excluding zero, no significant reverse headline, reduced harm, and no severe Stable tail collapse.
