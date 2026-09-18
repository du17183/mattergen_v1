# mattersim_equal_budget_post_generation

Status: COMPLETE
Paired seeds: 32; no seeds dropped.

| Method | Mean MaxF | P95 MaxF | Mag MAE | NUS | Validity | Runtime s | Actual force calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.214179 | 0.755020 | 0.007900 | 34.38% | 100.00% | 84.56 | 0.00 |
| F0 | 0.163650 | 0.673935 | 0.007886 | 34.38% | 100.00% | 85.38 | 20.00 |
| POST | 0.043275 | 0.207183 | 0.007933 | 34.38% | 100.00% | 85.38 | 20.00 |

## Decision and limitations

```json
{
  "track": "A6",
  "n_paired": 32,
  "primary": {
    "baseline": "C0",
    "method": "F0",
    "metric": "maxF_ev_per_a",
    "n_pairs": 32,
    "baseline_mean": 0.214179111061563,
    "method_mean": 0.16364968049584014,
    "absolute_improvement": 0.05052943056572285,
    "absolute_ci95": [
      0.04000252326666669,
      0.061067821424636286
    ],
    "relative_improvement": 0.2359213758768419,
    "relative_ci95": [
      0.1644270408678703,
      0.3423000963410307
    ],
    "wins": 32,
    "ties": 0,
    "losses": 0,
    "win_binomial_one_sided_p": 2.3283064365386963e-10,
    "bootstrap_resamples": 20000,
    "bootstrap_seed": 20260915
  },
  "guardrails": {
    "values": {
      "mag_mae_worsening_fraction": -0.0018098620151902488,
      "nus_drop_pp": 0.0,
      "validity_drop_pp": 0.0,
      "e_hull_increase_ev_atom": -2.2690494855287513e-05,
      "runtime_ratio": 1.0097244432676893
    },
    "pass": {
      "mag_mae_worsening_fraction": true,
      "nus_drop_pp": true,
      "validity_drop_pp": true,
      "e_hull_increase_ev_atom": true,
      "runtime_ratio": true
    },
    "all_pass": true
  },
  "SURROGATE_PROPERTY_EVAL": true,
  "DFT_VERIFIED": false,
  "audit": "PASS",
  "status": "COMPLETE",
  "force_vs_post": {
    "baseline": "POST",
    "method": "F0",
    "metric": "maxF_ev_per_a",
    "n_pairs": 32,
    "baseline_mean": 0.04327507908902397,
    "method_mean": 0.16364968049584014,
    "absolute_improvement": -0.12037460140681618,
    "absolute_ci95": [
      -0.17233593658652244,
      -0.07700278579991995
    ],
    "relative_improvement": -2.7816148217588643,
    "relative_ci95": [
      -12.390563706286486,
      -1.2295922027863562
    ],
    "wins": 0,
    "ties": 0,
    "losses": 32,
    "win_binomial_one_sided_p": 1.0,
    "bootstrap_resamples": 20000,
    "bootstrap_seed": 20260915
  },
  "post_vs_base": {
    "baseline": "C0",
    "method": "POST",
    "metric": "maxF_ev_per_a",
    "n_pairs": 32,
    "baseline_mean": 0.214179111061563,
    "method_mean": 0.04327507908902397,
    "absolute_improvement": 0.17090403197253906,
    "absolute_ci95": [
      0.12168883494630317,
      0.22792739777828877
    ],
    "relative_improvement": 0.7979491142972149,
    "relative_ci95": [
      0.6347305135088329,
      0.9490560399279031
    ],
    "wins": 32,
    "ties": 0,
    "losses": 0,
    "win_binomial_one_sided_p": 2.3283064365386963e-10,
    "bootstrap_resamples": 20000,
    "bootstrap_seed": 20260915
  },
  "budget_includes_failed_actual_calculator_calls": true,
  "evaluation_calls_separate_and_equal_for_all": true,
  "independent_mlip": {
    "DFT_VERIFIED": false,
    "absolute_bootstrap_ci95": [
      0.0021693088726699753,
      0.02120243130832238
    ],
    "absolute_improvement": 0.011381158184092962,
    "baseline": "C0",
    "baseline_mean": 0.2202614958356853,
    "bootstrap_resamples": 20000,
    "ci_excludes_zero_positive": true,
    "cohort": "A6",
    "coverage_fraction": 1.0,
    "elapsed_seconds": 1.3798615009873174,
    "elements_Z": [
      1,
      6,
      7,
      8,
      25,
      26,
      27,
      63,
      64
    ],
    "formal256_evaluation_trigger": true,
    "losses": 9,
    "mean_relative_reduction": 0.05167112000630068,
    "method": "F0",
    "method_mean": 0.2088803376515923,
    "n": 32,
    "relative_bootstrap_ci95": [
      0.011862902541023913,
      0.10400341245055647
    ],
    "status": "PASS",
    "ties": 0,
    "wins": 23
  }
}
```

All structures are assessed by frozen MatterSim/CHGNet surrogates. DFT_VERIFIED=False. Independent CHGNet forces are independent of generation but CHGNet also supplies magnetic guardrails; training data may overlap. P0 tail estimates and interpretive retention thresholds are not confirmatory significance tests. Direction controls reference the paired F0 trajectory. No claim of first-in-literature novelty is made.
