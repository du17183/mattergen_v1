# Q3 E3-PCR frozen 64-seed validation

- Final state: `Q3_FROZEN_64_GO`
- Effect gate: `True`
- Quality gate: `True`
- Mechanism safety: `True`
- Gate mechanism supported: `False`
- Evaluation seeds: `32000–32063`
- MatterGen sampling was run once per seed and shared by every method.
- Atomic numbers and cells were unchanged.
- Formal 256, A0 compatibility, and DFT were not started.

## Aggregate metrics

| method    |   pre_relax_max_force |   pre_relax_max_force_median |   convergence_rate |   relaxation_steps_mean |   relaxation_steps_median |      rmsd |    ehull |   ehull_median |   stable |   metastable |    nus |     msun |    novel |   unique |   composition_validity |   structure_validity |
|:----------|----------------------:|-----------------------------:|-------------------:|------------------------:|--------------------------:|----------:|---------:|---------------:|---------:|-------------:|-------:|---------:|---------:|---------:|-----------------------:|---------------------:|
| C0        |              0.397573 |                     0.213582 |           0.984375 |                 61.75   |                      34   | 0.0690161 | 0.167429 |       0.123119 |  0.34375 |     0.703125 | 0.1875 | 0.453125 | 0.734375 | 0.984375 |               0.796875 |                    1 |
| Q3_E3_PCR |              0.264137 |                     0.186204 |           1        |                 61.1406 |                      33   | 0.0673763 | 0.167429 |       0.123103 |  0.34375 |     0.703125 | 0.1875 | 0.453125 | 0.734375 | 0.984375 |               0.796875 |                    1 |
| ALWAYS_ON |              0.253888 |                     0.189905 |           1        |                 60.4375 |                      32.5 | 0.0672094 | 0.167414 |       0.123108 |  0.34375 |     0.703125 | 0.1875 | 0.453125 | 0.734375 | 0.984375 |               0.796875 |                    1 |

## Q3 changes versus C0

|   pre_relax_max_force |   pre_relax_max_force_median |   convergence_rate |   relaxation_steps_mean |   relaxation_steps_median |        rmsd |       ehull |   ehull_median |   stable |   metastable |   nus |   msun |   novel |   unique |   composition_validity |   structure_validity |   pre_relax_max_force_relative |   rmsd_relative |
|----------------------:|-----------------------------:|-------------------:|------------------------:|--------------------------:|------------:|------------:|---------------:|---------:|-------------:|------:|-------:|--------:|---------:|-----------------------:|---------------------:|-------------------------------:|----------------:|
|             -0.133436 |                   -0.0273786 |           0.015625 |               -0.609375 |                        -1 | -0.00163979 | 2.11182e-07 |   -1.52811e-05 |        0 |            0 |     0 |      0 |       0 |        0 |                      0 |                    0 |                      -0.335625 |      -0.0237596 |

## Force robustness

```json
{
  "baseline_mean": 0.3975729843787045,
  "bootstrap_95_ci": [
    -0.26132433367138214,
    -0.048601048478058635
  ],
  "leave_one_out_mean_difference_range": [
    -0.13612202475581753,
    -0.08198099005272477
  ],
  "losses": 12,
  "maximum_single_sample_absolute_contribution_rate": 0.39165326123603406,
  "mean_difference": -0.13343563068420478,
  "median_difference": -0.01050434554747965,
  "relative_change": -0.3356254975240015,
  "remove_most_favorable_relative_change": -0.24020871699702884,
  "remove_most_favorable_sample_index": 14,
  "selected_mean": 0.2641373536944997,
  "ties": 0,
  "wilcoxon_p": 1.4196309367659205e-08,
  "wins": 52
}
```

## Ablation

```json
{
  "always_on_force_relative_change": -0.36140561779496005,
  "always_on_mean_force": 0.2538878743407328,
  "always_on_worsening_rate": 0.09375,
  "baseline_mean_force": 0.3975729843787045,
  "gate_mechanism_supported": false,
  "learned_gate_force_relative_change": -0.3356254975240015,
  "learned_gate_mean_force": 0.2641373536944997,
  "learned_gate_worsening_rate": 0.1875,
  "learned_vs_always_on": 0.01024947935376691,
  "learned_vs_random_gate_mean_relative_change": -0.12142578961978795,
  "random_gate": {
    "force_relative_change_range": [
      -0.29997554644828095,
      -0.13046118143146657
    ],
    "mean_force_relative_change": -0.21419970790421355,
    "mean_losses": 3.6,
    "mean_wins": 38.4,
    "random_gate_runs": 5
  }
}
```

## Limits

- MatterSim-5M is the independent evaluator; no DFT was run.
- This is an independent frozen 64-seed validation, not a 256-seed formal confirmation.
