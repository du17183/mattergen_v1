# A0 + E3-G independent validation quality analysis

| method   |   pre_relax_max_force |   pre_relax_max_force_median |   convergence_rate |   relaxation_steps_mean |   relaxation_steps_median |      rmsd |    ehull |   ehull_median |   ehull_coverage |   stable |   metastable |      nus |    msun |    novel |   unique |   composition_validity |   structure_validity |   relaxation_failure_rate |   short_bond_rate |   abnormal_cell_rate |   relaxation_elapsed_mean |   ehull_all_available |
|:---------|----------------------:|-----------------------------:|-------------------:|------------------------:|--------------------------:|----------:|---------:|---------------:|-----------------:|---------:|-------------:|---------:|--------:|---------:|---------:|-----------------------:|---------------------:|--------------------------:|------------------:|---------------------:|--------------------------:|----------------------:|
| A0       |               0.26528 |                     0.166365 |                  1 |                 37.7656 |                      20.5 | 0.0775784 | 0.139963 |      0.0943341 |                1 | 0.546875 |     0.796875 | 0.328125 | 0.46875 | 0.671875 |        1 |                0.84375 |                    1 |                         0 |                 0 |                    0 |                   1.08621 |              0.139963 |
| A0+E3-G  |               0.21483 |                     0.147531 |                  1 |                 37.0625 |                      19   | 0.0762931 | 0.139963 |      0.0943417 |                1 | 0.546875 |     0.796875 | 0.328125 | 0.46875 | 0.671875 |        1 |                0.84375 |                    1 |                         0 |                 0 |                    0 |                   1.02463 |              0.139963 |

```json
{
  "abnormal_cell_not_increased": true,
  "atomic_numbers_unchanged": true,
  "cell_unchanged": true,
  "composition_drop_le_1_over_64": true,
  "ehull_degradation_le_0_002": true,
  "full_rejection_exact_fallback": true,
  "gate_off_exact_fallback": true,
  "generation_success_not_lower": true,
  "maximum_displacement_bounded": true,
  "minimum_distance_safe": true,
  "no_nan_inf": true,
  "novel_drop_le_1_over_64": true,
  "nus_drop_le_1_over_64": true,
  "relaxation_failure_not_increased": true,
  "rmsd_degradation_le_5_percent": true,
  "short_bond_not_increased": true,
  "stable_drop_le_1_over_64": true,
  "structure_validity_not_lower": true,
  "unique_not_lower": true
}
```
