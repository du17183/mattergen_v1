# SPG-MatterGen Fast Gate final report

- SPG_MATTERGEN_FASTGATE_GO: `True`
- NATIVE_BATCHING_GO: `False`
- FIELD_SAFE_BF16_GO: `False`
- PARTIAL_COMPILE_WORKS: `False`
- STATIC_GRAPH_REQUIRED: `True`
- STATIC_PERIODIC_GRAPH_WORTH_BUILDING: `True`
- Estimated single-bucket static-graph speedup: 1.0985×

## Performance

| method   |   batch_size |   fixed8_samples_per_hour |   speedup_vs_b1 |   median_sample_latency_seconds |   mean_gpu_utilization_percent |
|:---------|-------------:|--------------------------:|----------------:|--------------------------------:|-------------------------------:|
| C0       |            1 |                   427.513 |         1       |                         64.29   |                        25.5109 |
| C0       |            4 |                  1284.14  |         3.00374 |                         21.4583 |                        36.4917 |
| C0       |            8 |                  1833.58  |         4.28894 |                         15.1943 |                        44.06   |
| A0       |            1 |                   422.881 |         1       |                         64.5115 |                        25.5798 |
| A0       |            4 |                  1242.08  |         2.93718 |                         22.1096 |                        36.1206 |
| A0       |            8 |                  1781.08  |         4.21177 |                         15.4192 |                        43.5715 |

## Quality decisions

```json
{
  "A0": {
    "baseline": "A0_B1",
    "candidate": "A0_B4",
    "composition_change": -0.03125,
    "ehull_change": -0.033763405967144416,
    "gates": {
      "composition_decline_le_1_of_64": false,
      "ehull_degradation_le_0_002": true,
      "generation_success_not_lower": true,
      "hit_0_02_decline_le_1_of_64": false,
      "initial_random_tape_match": true,
      "nus_decline_le_1_of_64": true,
      "relaxation_failure_not_increased": true,
      "rmsd_no_systematic_worsening": false,
      "stable_decline_le_1_of_64": true,
      "structure_validity_not_lower": true,
      "throughput_ge_2x": true
    },
    "hit_0_02_change": -0.421875,
    "nus_change": 0.203125,
    "quality_equivalent": false,
    "rmsd_bootstrap_95_ci": [
      -0.0204212513252325,
      0.10392900259802972
    ],
    "rmsd_mean_change_angstrom": 0.04012854885113369,
    "stable_change": 0.375,
    "structure_change": 0.0,
    "throughput_speedup": 2.9371761254901054
  },
  "C0": {
    "baseline": "C0_B1",
    "candidate": "C0_B4",
    "composition_change": -0.046875,
    "ehull_change": -0.03730578358445896,
    "gates": {
      "composition_decline_le_1_of_64": false,
      "ehull_degradation_le_0_002": true,
      "generation_success_not_lower": true,
      "hit_0_02_decline_le_1_of_64": false,
      "initial_random_tape_match": true,
      "nus_decline_le_1_of_64": true,
      "relaxation_failure_not_increased": true,
      "rmsd_no_systematic_worsening": false,
      "stable_decline_le_1_of_64": true,
      "structure_validity_not_lower": true,
      "throughput_ge_2x": true
    },
    "hit_0_02_change": -0.34375,
    "nus_change": 0.203125,
    "quality_equivalent": false,
    "rmsd_bootstrap_95_ci": [
      0.01432286617236533,
      0.11016115146622325
    ],
    "rmsd_mean_change_angstrom": 0.05545118231590334,
    "stable_change": 0.3125,
    "structure_change": 0.0,
    "throughput_speedup": 3.003737098803732
  }
}
```

## Recommended next action

Implement only the highest-coverage static periodic-graph bucket MVP, then run an 8-seed exactness and endpoint timing gate.

The Fast Gate did not start a static graph implementation or 256-seed formal validation.
