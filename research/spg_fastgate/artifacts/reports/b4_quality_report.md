# SPG Fast Gate B4 quality equivalence

- C0 B4 equivalent: `False`
- A0 B4 equivalent: `False`
- Native batching GO: `False`
- Field-safe BF16 GO: `False`

## Method summary

| config   |   n |   generation_success |   generation_elapsed_median |   generation_composition_validity |   generation_structure_validity |   average_ehull |   median_ehull |   stable_rate |   metastable_le_0_2_rate |   nus_rate |   msun_le_0_2_rate |   novel_rate |   unique_rate |   rmsd_mean |   rmsd_median |   pre_relax_max_force_mean |   pre_relax_max_force_median |   force_convergence_rate |   relaxation_failure_rate |   hit_0_01 |   hit_0_02 |   hit_0_05 |   target_error_mean |   severe_short_bond_count |
|:---------|----:|---------------------:|----------------------------:|----------------------------------:|--------------------------------:|----------------:|---------------:|--------------:|-------------------------:|-----------:|-------------------:|-------------:|--------------:|------------:|--------------:|---------------------------:|-----------------------------:|-------------------------:|--------------------------:|-----------:|-----------:|-----------:|--------------------:|--------------------------:|
| C0_B1    |  64 |                    1 |                     62.1347 |                          0.84375  |                               1 |        0.166456 |      0.140976  |       0.34375 |                 0.671875 |   0.15625  |           0.4375   |      0.75    |      1        |   0.0411442 |     0.0256491 |                   0.377325 |                     0.209226 |                 1        |                         0 |   0.15625  |   0.359375 |   0.703125 |           0.0379391 |                         0 |
| C0_B4    |  64 |                    1 |                     21.3352 |                          0.796875 |                               1 |        0.12915  |      0.0778059 |       0.65625 |                 0.859375 |   0.359375 |           0.546875 |      0.6875  |      1        |   0.0965953 |     0.0282567 |                   0.534568 |                     0.146556 |                 0.953125 |                         0 |   0        |   0.015625 |   0.046875 |           0.0944211 |                         0 |
| A0_B1    |  64 |                    1 |                     61.4247 |                          0.828125 |                               1 |        0.162936 |      0.139328  |       0.28125 |                 0.734375 |   0.15625  |           0.5      |      0.78125 |      0.984375 |   0.0562414 |     0.0214071 |                   0.354619 |                     0.243974 |                 0.984375 |                         0 |   0.171875 |   0.4375   |   0.765625 |           0.0323533 |                         0 |
| A0_B4    |  64 |                    1 |                     21.8003 |                          0.796875 |                               1 |        0.129173 |      0.0778062 |       0.65625 |                 0.859375 |   0.359375 |           0.546875 |      0.6875  |      1        |   0.09637   |     0.0282565 |                   0.534568 |                     0.146556 |                 0.953125 |                         0 |   0        |   0.015625 |   0.046875 |           0.0944211 |                         0 |

## B4 decisions

```json
{
  "C0": {
    "baseline": "C0_B1",
    "candidate": "C0_B4",
    "throughput_speedup": 3.003737098803732,
    "ehull_change": -0.03730578358445896,
    "stable_change": 0.3125,
    "nus_change": 0.203125,
    "composition_change": -0.046875,
    "structure_change": 0.0,
    "rmsd_mean_change_angstrom": 0.05545118231590334,
    "rmsd_bootstrap_95_ci": [
      0.01432286617236533,
      0.11016115146622325
    ],
    "hit_0_02_change": -0.34375,
    "gates": {
      "initial_random_tape_match": true,
      "generation_success_not_lower": true,
      "structure_validity_not_lower": true,
      "composition_decline_le_1_of_64": false,
      "ehull_degradation_le_0_002": true,
      "stable_decline_le_1_of_64": true,
      "nus_decline_le_1_of_64": true,
      "rmsd_no_systematic_worsening": false,
      "relaxation_failure_not_increased": true,
      "hit_0_02_decline_le_1_of_64": false,
      "throughput_ge_2x": true
    },
    "quality_equivalent": false
  },
  "A0": {
    "baseline": "A0_B1",
    "candidate": "A0_B4",
    "throughput_speedup": 2.9371761254901054,
    "ehull_change": -0.033763405967144416,
    "stable_change": 0.375,
    "nus_change": 0.203125,
    "composition_change": -0.03125,
    "structure_change": 0.0,
    "rmsd_mean_change_angstrom": 0.04012854885113369,
    "rmsd_bootstrap_95_ci": [
      -0.0204212513252325,
      0.10392900259802972
    ],
    "hit_0_02_change": -0.421875,
    "gates": {
      "initial_random_tape_match": true,
      "generation_success_not_lower": true,
      "structure_validity_not_lower": true,
      "composition_decline_le_1_of_64": false,
      "ehull_degradation_le_0_002": true,
      "stable_decline_le_1_of_64": true,
      "nus_decline_le_1_of_64": true,
      "rmsd_no_systematic_worsening": false,
      "relaxation_failure_not_increased": true,
      "hit_0_02_decline_le_1_of_64": false,
      "throughput_ge_2x": true
    },
    "quality_equivalent": false
  }
}
```

MatterSim-5M and CHGNet magnetic density are surrogate evaluations; no DFT verification was performed.
