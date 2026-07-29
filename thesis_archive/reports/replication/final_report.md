# A0 + E3-G independent validation 64

- Final state: `A0_E3G_INDEPENDENT64_GO`
- Primary effect pass: `True`
- Quality safety pass: `True`
- Independent seeds: `50000-50063` (seed audit passed; no prior-range intersections)
- A0 generation: `64/64`
- A0+E3-G refinement: `64/64`
- MatterSim relaxation: `128/128`
- A0+E3-G derives from the exact same A0 structures.
- No training, retuning, 256-seed combination, DFT, or independent MLIP experiment was started.
- No leaked result or seed anonymization was used as independent evidence.

## Aggregate metrics

| method   |   pre_relax_max_force |   pre_relax_max_force_median |   convergence_rate |   relaxation_steps_mean |   relaxation_steps_median |      rmsd |    ehull |   ehull_median |   ehull_coverage |   stable |   metastable |      nus |    msun |    novel |   unique |   composition_validity |   structure_validity |   relaxation_failure_rate |   short_bond_rate |   abnormal_cell_rate |   relaxation_elapsed_mean |   ehull_all_available |
|:---------|----------------------:|-----------------------------:|-------------------:|------------------------:|--------------------------:|----------:|---------:|---------------:|-----------------:|---------:|-------------:|---------:|--------:|---------:|---------:|-----------------------:|---------------------:|--------------------------:|------------------:|---------------------:|--------------------------:|----------------------:|
| A0       |               0.26528 |                     0.166365 |                  1 |                 37.7656 |                      20.5 | 0.0775784 | 0.139963 |      0.0943341 |                1 | 0.546875 |     0.796875 | 0.328125 | 0.46875 | 0.671875 |        1 |                0.84375 |                    1 |                         0 |                 0 |                    0 |                   1.08621 |              0.139963 |
| A0+E3-G  |               0.21483 |                     0.147531 |                  1 |                 37.0625 |                      19   | 0.0762931 | 0.139963 |      0.0943417 |                1 | 0.546875 |     0.796875 | 0.328125 | 0.46875 | 0.671875 |        1 |                0.84375 |                    1 |                         0 |                 0 |                    0 |                   1.02463 |              0.139963 |

## Primary endpoint

|   baseline_mean |   selected_mean |   baseline_median |   selected_median |   mean_difference |   median_difference |   relative_change | bootstrap_95_ci                              |   wilcoxon_p_raw |   wins |   ties |   losses | leave_one_out_mean_difference_range          |   remove_most_favorable_sample_index |   remove_most_favorable_sample_seed |   remove_most_favorable_relative_change |   remove_most_unfavorable_sample_index |   remove_most_unfavorable_sample_seed |   remove_most_unfavorable_relative_change |   maximum_single_sample_absolute_contribution_rate | effect_reduction_ge_10_percent   | bootstrap_ci_upper_below_zero   | wilcoxon_p_below_0_05   | statistical_evidence_pass   | primary_effect_pass   |   raw_numeric_wins_1e_minus_12 |   raw_numeric_ties_1e_minus_12 |   raw_numeric_losses_1e_minus_12 |   counting_epsilon |   gate_on_wins |   gate_on_ties |   gate_on_losses |   gate_off_exact_structure_ties |   gate_off_max_abs_numeric_difference |
|----------------:|----------------:|------------------:|------------------:|------------------:|--------------------:|------------------:|:---------------------------------------------|-----------------:|-------:|-------:|---------:|:---------------------------------------------|-------------------------------------:|------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------:|------------------------------------------:|---------------------------------------------------:|:---------------------------------|:--------------------------------|:------------------------|:----------------------------|:----------------------|-------------------------------:|-------------------------------:|---------------------------------:|-------------------:|---------------:|---------------:|-----------------:|--------------------------------:|--------------------------------------:|
|         0.26528 |         0.21483 |          0.166365 |          0.147531 |        -0.0504499 |        -0.000773545 |         -0.190176 | [-0.10221274022731316, -0.01069625841941559] |      0.000587377 |     35 |     18 |       11 | [-0.05463224662409739, -0.03311136501247086] |                                   54 |                               50054 |                                 -0.1388 |                                     23 |                                 50023 |                                 -0.205418 |                                           0.255976 | True                             | True                            | True                    | True                        | True                  |                             46 |                              0 |                               18 |              1e-06 |             35 |              0 |               11 |                              18 |                            5.6915e-07 |

## Limitations

- Stability is evaluated with the MatterSim-5M surrogate.
- DFT and independent property-target verification were not run.
