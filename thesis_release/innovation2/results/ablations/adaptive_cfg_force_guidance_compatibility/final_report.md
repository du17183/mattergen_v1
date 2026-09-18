# adaptive_cfg_force_guidance_compatibility

Status: FAIL
Paired seeds: 64; no seeds dropped.

| Method | Mean MaxF | P95 MaxF | Mag MAE | NUS | Validity | Runtime s | Actual force calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.218922 | 0.880174 | 0.010561 | 21.88% | 100.00% | 113.59 | 0.00 |
| A0 | 0.239629 | 0.764540 | 0.009706 | 26.56% | 100.00% | 107.01 | 0.00 |
| B0 | 0.171833 | 0.828043 | 0.010151 | 21.88% | 100.00% | 107.94 | 20.00 |
| AB | 0.192794 | 0.747394 | 0.009728 | 26.56% | 100.00% | 118.10 | 20.00 |

## Decision and limitations

```json
{
  "track": "A5",
  "n_paired": 64,
  "primary": {
    "baseline": "C0",
    "method": "A0",
    "metric": "maxF_ev_per_a",
    "n_pairs": 64,
    "baseline_mean": 0.21892229194678153,
    "method_mean": 0.2396292507373313,
    "absolute_improvement": -0.02070695879054976,
    "absolute_ci95": [
      -0.10492229050208261,
      0.054617695426215244
    ],
    "relative_improvement": -0.0945858852765139,
    "relative_ci95": [
      -0.5548637558098206,
      0.21473588818705738
    ],
    "wins": 30,
    "ties": 0,
    "losses": 34,
    "win_binomial_one_sided_p": 0.7338456119860789,
    "bootstrap_resamples": 20000,
    "bootstrap_seed": 20260915
  },
  "guardrails": {
    "values": {
      "mag_mae_worsening_fraction": -0.08101630924018922,
      "nus_drop_pp": -4.6875,
      "validity_drop_pp": 0.0,
      "e_hull_increase_ev_atom": -0.02948893451227501,
      "runtime_ratio": 0.9420623727658788
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
  "retention_CFG": 0.9744789248058638,
  "retention_force": 0.5548701126866487,
  "retention_denominators": {
    "CFG": 0.0008556410434899381,
    "force": 0.047089171783060935
  },
  "AB_guardrails": {
    "values": {
      "mag_mae_worsening_fraction": -0.07894868592011896,
      "nus_drop_pp": -4.6875,
      "validity_drop_pp": 0.0,
      "e_hull_increase_ev_atom": -0.029495948656907148,
      "runtime_ratio": 1.039711114701181
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
  "NUS_preserved_vs_A0": true,
  "threshold_interpretive_not_significance": true,
  "FINAL_MODEL": "F0 standalone; combination not confirmed",
  "independent_mlip": {
    "DFT_VERIFIED": false,
    "absolute_bootstrap_ci95": [
      -0.04159862133155402,
      0.13019150234166127
    ],
    "absolute_improvement": 0.037940596123142316,
    "baseline": "C0",
    "baseline_mean": 0.2367597178280486,
    "bootstrap_resamples": 20000,
    "ci_excludes_zero_positive": false,
    "cohort": "A5",
    "coverage_fraction": 1.0,
    "elapsed_seconds": 3.900191777967848,
    "elements_Z": [
      1,
      3,
      6,
      7,
      8,
      14,
      25,
      26,
      27,
      28,
      31,
      62,
      64,
      74,
      80,
      81,
      83
    ],
    "formal256_evaluation_trigger": true,
    "losses": 26,
    "mean_relative_reduction": 0.16024937211108436,
    "method": "A0",
    "method_mean": 0.19881912170490632,
    "n": 64,
    "relative_bootstrap_ci95": [
      -0.2305870054132817,
      0.43439812513962495
    ],
    "status": "PASS",
    "ties": 0,
    "wins": 38
  }
}
```

All structures are assessed by frozen MatterSim/CHGNet surrogates. DFT_VERIFIED=False. Independent CHGNet forces are independent of generation but CHGNet also supplies magnetic guardrails; training data may overlap. P0 tail estimates and interpretive retention thresholds are not confirmatory significance tests. Direction controls reference the paired F0 trajectory. No claim of first-in-literature novelty is made.
