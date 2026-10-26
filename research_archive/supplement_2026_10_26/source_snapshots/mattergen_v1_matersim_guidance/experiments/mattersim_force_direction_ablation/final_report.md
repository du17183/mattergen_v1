# mattersim_force_direction_ablation

Status: SUPPORTED
Paired seeds: 32; no seeds dropped.

| Method | Mean MaxF | P95 MaxF | Mag MAE | NUS | Validity | Runtime s | Actual force calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.141690 | 0.376479 | 0.010471 | 18.75% | 100.00% | 100.41 | 0.00 |
| G1 | 0.098500 | 0.304940 | 0.010502 | 18.75% | 100.00% | 101.99 | 20.00 |
| G2 | 0.153744 | 0.379922 | 0.010473 | 18.75% | 100.00% | 91.64 | 20.00 |
| G3 | 0.151428 | 0.386740 | 0.010472 | 18.75% | 100.00% | 90.03 | 20.00 |
| G4 | 0.188137 | 0.439113 | 0.010444 | 18.75% | 100.00% | 92.85 | 20.00 |

## Decision and limitations

```json
{
  "track": "A3",
  "n_paired": 32,
  "primary": {
    "baseline": "G0",
    "method": "G1",
    "metric": "maxF_ev_per_a",
    "n_pairs": 32,
    "baseline_mean": 0.14169022843555268,
    "method_mean": 0.09850048281560217,
    "absolute_improvement": 0.04318974561995051,
    "absolute_ci95": [
      0.03332395194658162,
      0.05316752924167372
    ],
    "relative_improvement": 0.3048180957630061,
    "relative_ci95": [
      0.2465969373574472,
      0.3886519509599739
    ],
    "wins": 31,
    "ties": 0,
    "losses": 1,
    "win_binomial_one_sided_p": 7.683411240577698e-09,
    "bootstrap_resamples": 20000,
    "bootstrap_seed": 20260915
  },
  "guardrails": {
    "values": {
      "mag_mae_worsening_fraction": 0.0029270355304999345,
      "nus_drop_pp": 0.0,
      "validity_drop_pp": 0.0,
      "e_hull_increase_ev_atom": 2.8439631244436825e-05,
      "runtime_ratio": 1.0157133881596863
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
  "status": "SUPPORTED",
  "mechanism_trend": true,
  "interpretation": "Descriptive prespecified ordering, not a requirement that every contrast be significant.",
  "direction_reference": "G2/G3/G4 replay paired G1 trajectory magnitudes; G4 reverses paired G1 MatterSim direction, not its own current-state force. This limits the causal interpretation.",
  "independent_mlip": {
    "DFT_VERIFIED": false,
    "absolute_bootstrap_ci95": [
      0.002044636337982118,
      0.018415200284925607
    ],
    "absolute_improvement": 0.010295576965296135,
    "baseline": "G0",
    "baseline_mean": 0.12862659032737078,
    "bootstrap_resamples": 20000,
    "ci_excludes_zero_positive": true,
    "cohort": "A3",
    "coverage_fraction": 1.0,
    "elapsed_seconds": 2.026722051028628,
    "elements_Z": [
      1,
      7,
      8,
      25,
      26,
      27,
      28,
      63,
      64
    ],
    "formal256_evaluation_trigger": true,
    "losses": 7,
    "mean_relative_reduction": 0.08004236868203225,
    "method": "G1",
    "method_mean": 0.11833101336207465,
    "n": 32,
    "relative_bootstrap_ci95": [
      0.01643113346090434,
      0.13601545588130082
    ],
    "status": "PASS",
    "ties": 0,
    "wins": 25
  }
}
```

All structures are assessed by frozen MatterSim/CHGNet surrogates. DFT_VERIFIED=False. Independent CHGNet forces are independent of generation but CHGNet also supplies magnetic guardrails; training data may overlap. P0 tail estimates and interpretive retention thresholds are not confirmatory significance tests. Direction controls reference the paired F0 trajectory. No claim of first-in-literature novelty is made.
