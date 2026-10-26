# closed_loop_force_guidance_p0

Status: FAIL
Paired seeds: 16; no seeds dropped.

| Method | Mean MaxF | P95 MaxF | Mag MAE | NUS | Validity | Runtime s | Actual force calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 | 0.051036 | 0.122384 | 0.010932 | 18.75% | 100.00% | 85.72 | 20.00 |
| F1 | 0.051037 | 0.122381 | 0.010932 | 18.75% | 100.00% | 85.49 | 40.00 |

## Decision and limitations

```json
{
  "track": "B",
  "n_paired": 16,
  "primary": {
    "baseline": "F0",
    "method": "F1",
    "metric": "maxF_ev_per_a",
    "n_pairs": 16,
    "baseline_mean": 0.0510362336733348,
    "method_mean": 0.0510373296222081,
    "absolute_improvement": -1.0959488732979706e-06,
    "absolute_ci95": [
      -1.4589469236158029e-05,
      1.2319027055616755e-05
    ],
    "relative_improvement": -2.1473937130896737e-05,
    "relative_ci95": [
      -0.00033345059650520667,
      0.00025717068014646284
    ],
    "wins": 7,
    "ties": 0,
    "losses": 9,
    "win_binomial_one_sided_p": 0.7727508544921875,
    "bootstrap_resamples": 20000,
    "bootstrap_seed": 20260915
  },
  "guardrails": {
    "values": {
      "mag_mae_worsening_fraction": -3.768020163519375e-05,
      "nus_drop_pp": 0.0,
      "validity_drop_pp": 0.0,
      "e_hull_increase_ev_atom": 1.5629662403204847e-07,
      "runtime_ratio": 0.9972489592514333
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
  "status": "FAIL",
  "p95_relative_reduction": 2.6379478119290283e-05,
  "p90_relative_reduction": -5.511518648100091e-05,
  "high_maxF_threshold_ev_a": 0.2,
  "high_maxF_rate_relative_reduction": null,
  "mean_maxF_worsening": 2.1473937130922494e-05,
  "actual_calls_ratio": 2.0,
  "all_F1_guardrails_pass": true,
  "next_action": "KEEP_F0_NO_TUNING",
  "F0_remains_formal": true,
  "automatic_formal32": false,
  "independent_mlip": {
    "DFT_VERIFIED": false,
    "absolute_bootstrap_ci95": [
      -4.735386984161931e-05,
      6.0498107520485105e-06
    ],
    "absolute_improvement": -1.3440424320576214e-05,
    "baseline": "F0",
    "baseline_mean": 0.09177261917390242,
    "bootstrap_resamples": 20000,
    "ci_excludes_zero_positive": false,
    "cohort": "B",
    "coverage_fraction": 1.0,
    "elapsed_seconds": 0.7527246589888819,
    "elements_Z": [
      7,
      8,
      14,
      26,
      27,
      63,
      64
    ],
    "formal256_evaluation_trigger": false,
    "losses": 8,
    "mean_relative_reduction": -0.00014645353310788154,
    "method": "F1",
    "method_mean": 0.091786059598223,
    "n": 16,
    "relative_bootstrap_ci95": [
      -0.0006209485051095986,
      6.490425930299187e-05
    ],
    "status": "INCONCLUSIVE",
    "ties": 0,
    "wins": 8
  }
}
```

All structures are assessed by frozen MatterSim/CHGNet surrogates. DFT_VERIFIED=False. Independent CHGNet forces are independent of generation but CHGNet also supplies magnetic guardrails; training data may overlap. P0 tail estimates and interpretive retention thresholds are not confirmatory significance tests. Direction controls reference the paired F0 trajectory. No claim of first-in-literature novelty is made.
