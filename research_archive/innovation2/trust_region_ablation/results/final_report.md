# mattersim_trust_region_ablation

Status: NOT_SUPPORTED
Paired seeds: 32; no seeds dropped.

| Method | Mean MaxF | P95 MaxF | Mag MAE | NUS | Validity | Runtime s | Actual force calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 | 0.277138 | 0.612675 | 0.009208 | 12.50% | 100.00% | 84.49 | 0.00 |
| T1 | 0.231502 | 0.564431 | 0.009234 | 12.50% | 100.00% | 85.31 | 20.00 |
| T2 | 0.060834 | 0.301881 | 0.009426 | 12.50% | 100.00% | 84.71 | 20.00 |

## Decision and limitations

```json
{
  "track": "A4",
  "n_paired": 32,
  "primary": {
    "baseline": "T0",
    "method": "T1",
    "metric": "maxF_ev_per_a",
    "n_pairs": 32,
    "baseline_mean": 0.2771384390608932,
    "method_mean": 0.23150218242651582,
    "absolute_improvement": 0.04563625663437739,
    "absolute_ci95": [
      0.034195606410041365,
      0.06125353271818586
    ],
    "relative_improvement": 0.16466953046650493,
    "relative_ci95": [
      0.10129894658544734,
      0.3688559332814224
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
      "mag_mae_worsening_fraction": 0.002849219196052132,
      "nus_drop_pp": 0.0,
      "validity_drop_pp": 0.0,
      "e_hull_increase_ev_atom": -2.1791754410166453e-05,
      "runtime_ratio": 1.0096908339962045
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
  "status": "NOT_SUPPORTED",
  "conclusion": "TRUST_REGION_NOT_SUPPORTED",
  "bounded_guardrails": {
    "values": {
      "mag_mae_worsening_fraction": 0.002849219196052132,
      "nus_drop_pp": 0.0,
      "validity_drop_pp": 0.0,
      "e_hull_increase_ev_atom": -2.1791754410166453e-05,
      "runtime_ratio": 1.0096908339962045
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
  "unbounded_guardrails": {
    "values": {
      "mag_mae_worsening_fraction": 0.023674027780089788,
      "nus_drop_pp": 0.0,
      "validity_drop_pp": 0.0,
      "e_hull_increase_ev_atom": -3.104212887602986e-05,
      "runtime_ratio": 1.0026280336599995
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
  "interpretation": "Necessary refers only to this frozen linear unbounded comparator and these guardrails, not all uncapped algorithms.",
  "independent_mlip": {
    "DFT_VERIFIED": false,
    "absolute_bootstrap_ci95": [
      0.00023075750488216955,
      0.012770494405770437
    ],
    "absolute_improvement": 0.0063451522851788985,
    "baseline": "T0",
    "baseline_mean": 0.14767950664294283,
    "bootstrap_resamples": 20000,
    "ci_excludes_zero_positive": true,
    "cohort": "A4",
    "coverage_fraction": 1.0,
    "elapsed_seconds": 1.4269982519908808,
    "elements_Z": [
      1,
      6,
      7,
      8,
      9,
      26,
      27,
      64
    ],
    "formal256_evaluation_trigger": true,
    "losses": 10,
    "mean_relative_reduction": 0.042965692596198264,
    "method": "T1",
    "method_mean": 0.14133435435776392,
    "n": 32,
    "relative_bootstrap_ci95": [
      0.001630828510662514,
      0.0931523948537848
    ],
    "status": "PASS",
    "ties": 0,
    "wins": 22
  }
}
```

All structures are assessed by frozen MatterSim/CHGNet surrogates. DFT_VERIFIED=False. Independent CHGNet forces are independent of generation but CHGNet also supplies magnetic guardrails; training data may overlap. P0 tail estimates and interpretive retention thresholds are not confirmatory significance tests. Direction controls reference the paired F0 trajectory. No claim of first-in-literature novelty is made.
