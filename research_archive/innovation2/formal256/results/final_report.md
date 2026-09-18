# mattersim_late_force_guidance_formal256

Status: STRONG_CONFIRMED
Paired seeds: 256; no seeds dropped.

| Method | Mean MaxF | P95 MaxF | Mag MAE | NUS | Validity | Runtime s | Actual force calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.226408 | 0.744050 | 0.009757 | 21.09% | 100.00% | 92.85 | 0.00 |
| F0 | 0.158911 | 0.677620 | 0.009868 | 21.48% | 100.00% | 92.66 | 20.00 |

## Decision and limitations

```json
{
  "track": "A1",
  "n_paired": 256,
  "primary": {
    "baseline": "C0",
    "method": "F0",
    "metric": "maxF_ev_per_a",
    "n_pairs": 256,
    "baseline_mean": 0.22640836794160712,
    "method_mean": 0.15891148914495132,
    "absolute_improvement": 0.0674968787966558,
    "absolute_ci95": [
      0.04662706147198513,
      0.10505693295754244
    ],
    "relative_improvement": 0.2981200713131941,
    "relative_ci95": [
      0.21416622087171452,
      0.4087580889652716
    ],
    "wins": 254,
    "ties": 0,
    "losses": 2,
    "win_binomial_one_sided_p": 2.8410403695694194e-73,
    "bootstrap_resamples": 20000,
    "bootstrap_seed": 20260915
  },
  "guardrails": {
    "values": {
      "mag_mae_worsening_fraction": 0.011405300238634642,
      "nus_drop_pp": -0.390625,
      "validity_drop_pp": 0.0,
      "e_hull_increase_ev_atom": 0.00046941984324047614,
      "runtime_ratio": 0.9980005010595485
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
  "status": "STRONG_CONFIRMED",
  "next_action": "COMPLETE_EVIDENCE_CHAIN",
  "independent_mlip": {
    "DFT_VERIFIED": false,
    "absolute_bootstrap_ci95": [
      0.0070520603370439645,
      0.06486799915701037
    ],
    "absolute_improvement": 0.027833333049582146,
    "baseline": "C0",
    "baseline_mean": 0.1938906372800398,
    "bootstrap_resamples": 20000,
    "ci_excludes_zero_positive": true,
    "cohort": "A1",
    "coverage_fraction": 1.0,
    "elapsed_seconds": 4.605878620001022,
    "elements_Z": [
      1,
      3,
      6,
      7,
      8,
      9,
      17,
      25,
      26,
      27,
      28,
      30,
      31,
      32,
      47,
      50,
      63,
      64,
      77,
      81
    ],
    "formal256_evaluation_trigger": true,
    "losses": 84,
    "mean_relative_reduction": 0.14355171265635672,
    "method": "F0",
    "method_mean": 0.16605730423045764,
    "n": 256,
    "relative_bootstrap_ci95": [
      0.043186154812837114,
      0.2772880938390704
    ],
    "status": "PASS",
    "ties": 0,
    "wins": 172
  }
}
```

All structures are assessed by frozen MatterSim/CHGNet surrogates. DFT_VERIFIED=False. Independent CHGNet forces are independent of generation but CHGNet also supplies magnetic guardrails; training data may overlap. P0 tail estimates and interpretive retention thresholds are not confirmatory significance tests. Direction controls reference the paired F0 trajectory. No claim of first-in-literature novelty is made.
