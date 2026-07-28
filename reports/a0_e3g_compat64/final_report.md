# A0 + E3-G compatibility 64

- Final state: `A0_E3G_COMPATIBILITY_GO`
- Primary effect pass: `True`
- Quality safety pass: `True`
- A0 generation: `64/64`
- A0+E3-G refinement: `64/64`
- MatterSim relaxation: `128/128`
- A0+E3-G derives from the exact same A0 structures.
- No training, retuning, 256-seed combination, DFT, or independent MLIP experiment was started.

## Aggregate metrics

| method   |   pre_relax_max_force |   pre_relax_max_force_median |   convergence_rate |   relaxation_steps_mean |   relaxation_steps_median |      rmsd |    ehull |   ehull_median |   ehull_coverage |   stable |   metastable |      nus |     msun |   novel |   unique |   composition_validity |   structure_validity |   relaxation_failure_rate |   short_bond_rate |   abnormal_cell_rate |   relaxation_elapsed_mean |   ehull_all_available |
|:---------|----------------------:|-----------------------------:|-------------------:|------------------------:|--------------------------:|----------:|---------:|---------------:|-----------------:|---------:|-------------:|---------:|---------:|--------:|---------:|-----------------------:|---------------------:|--------------------------:|------------------:|---------------------:|--------------------------:|----------------------:|
| A0       |              0.217302 |                    0.102851  |                  1 |                 44.75   |                      23   | 0.0696955 | 0.122703 |       0.100741 |                1 |      0.5 |     0.796875 | 0.265625 | 0.453125 | 0.65625 |        1 |                0.84375 |                    1 |                         0 |                 0 |                    0 |                   1.31816 |              0.122703 |
| A0+E3-G  |              0.158416 |                    0.0912628 |                  1 |                 43.4688 |                      24.5 | 0.0683539 | 0.122693 |       0.10072  |                1 |      0.5 |     0.796875 | 0.265625 | 0.453125 | 0.65625 |        1 |                0.84375 |                    1 |                         0 |                 0 |                    0 |                   1.25492 |              0.122693 |

## Primary endpoint

|   baseline_mean |   selected_mean |   baseline_median |   selected_median |   mean_difference |   median_difference |   relative_change | bootstrap_95_ci                               |   wilcoxon_p_raw |   wins |   ties |   losses | leave_one_out_mean_difference_range           |   remove_most_favorable_sample_index |   remove_most_favorable_sample_seed |   remove_most_favorable_relative_change |   remove_most_unfavorable_sample_index |   remove_most_unfavorable_sample_seed |   remove_most_unfavorable_relative_change |   maximum_single_sample_absolute_contribution_rate | effect_reduction_ge_10_percent   | bootstrap_ci_upper_below_zero   | wilcoxon_p_below_0_05   | statistical_evidence_pass   | primary_effect_pass   |
|----------------:|----------------:|------------------:|------------------:|------------------:|--------------------:|------------------:|:----------------------------------------------|-----------------:|-------:|-------:|---------:|:----------------------------------------------|-------------------------------------:|------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------:|------------------------------------------:|---------------------------------------------------:|:---------------------------------|:--------------------------------|:------------------------|:----------------------------|:----------------------|
|        0.217302 |        0.158416 |          0.102851 |         0.0912628 |        -0.0588853 |         -0.00154037 |         -0.270984 | [-0.09234130839938041, -0.029753947559803366] |      7.73987e-05 |     45 |      0 |       19 | [-0.06184503803671121, -0.051108498526013436] |                                   53 |                               41053 |                               -0.248955 |                                     36 |                                 41036 |                                  -0.29099 |                                            0.13048 | True                             | True                            | True                    | True                        | True                  |

## Limitations

- Stability is evaluated with the MatterSim-5M surrogate.
- DFT and independent property-target verification were not run.
