# Frozen64 paired statistics

| metric                     | type       |   baseline_mean |   selected_mean |   mean_difference |   median_difference |   bootstrap_95_ci_low |   bootstrap_95_ci_high | test                                       |     p_value |   wins |   ties |   losses |
|:---------------------------|:-----------|----------------:|----------------:|------------------:|--------------------:|----------------------:|-----------------------:|:-------------------------------------------|------------:|-------:|-------:|---------:|
| energy_above_hull_per_atom | continuous |       0.167429  |       0.167429  |       2.11182e-07 |           0         |          -1.15305e-05 |            1.16346e-05 | Wilcoxon signed-rank                       | 0.526714    |     27 |     15 |       22 |
| rmsd_from_relaxation       | continuous |       0.0690161 |       0.0673763 |      -0.00163979  |          -0.0001518 |          -0.00253883  |           -0.000823941 | Wilcoxon signed-rank                       | 1.9845e-05  |     48 |      0 |       16 |
| pre_relax_max_force_ev_ang | continuous |       0.397573  |       0.264137  |      -0.133436    |          -0.0105043 |          -0.261324    |           -0.048601    | Wilcoxon signed-rank                       | 1.41963e-08 |     52 |      0 |       12 |
| steps                      | continuous |      61.75      |      61.1406    |      -0.609375    |           0         |          -2.46875     |            0.9375      | Wilcoxon signed-rank                       | 0.918953    |     13 |     38 |       13 |
| stable                     | binary     |       0.34375   |       0.34375   |       0           |           0         |           0           |            0           | McNemar exact / paired discordant binomial | 1           |      0 |     64 |        0 |
| novel_unique_stable        | binary     |       0.1875    |       0.1875    |       0           |           0         |           0           |            0           | McNemar exact / paired discordant binomial | 1           |      0 |     64 |        0 |
| comp_validity              | binary     |       0.796875  |       0.796875  |       0           |           0         |           0           |            0           | McNemar exact / paired discordant binomial | 1           |      0 |     64 |        0 |
| structure_validity         | binary     |       1         |       1         |       0           |           0         |           0           |            0           | McNemar exact / paired discordant binomial | 1           |      0 |     64 |        0 |
| novel                      | binary     |       0.734375  |       0.734375  |       0           |           0         |           0           |            0           | McNemar exact / paired discordant binomial | 1           |      0 |     64 |        0 |
| unique                     | binary     |       0.984375  |       0.984375  |       0           |           0         |           0           |            0           | McNemar exact / paired discordant binomial | 1           |      0 |     64 |        0 |
| converged                  | binary     |       0.984375  |       1         |       0.015625    |           0         |           0           |            0.046875    | McNemar exact / paired discordant binomial | 1           |      1 |     63 |        0 |

## Maximum-force robustness

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
