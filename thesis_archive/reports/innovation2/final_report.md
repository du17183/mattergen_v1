# E3-PCR Formal 256

- Final state: `E3_G_FORMAL_CONFIRMED`
- Final method: `LEARNED_GATED_E3_PCR`
- E3-A primary pass: `True`
- E3-G primary pass: `True`
- E3-A quality pass: `True`
- E3-G quality pass: `True`
- Gate mechanism formally supported: `True`
- C0 was generated exactly once for each formal seed; E3-A and E3-G derive from the same C0 structures.
- MatterSim relaxations: `768/768`.
- Atomic numbers and cells were not modified.
- A0 compatibility and DFT were not started.

## Aggregate metrics

| method   |   pre_relax_max_force |   pre_relax_max_force_median |   convergence_rate |   relaxation_steps_mean |   relaxation_steps_median |      rmsd |    ehull |   ehull_median |   ehull_coverage |   stable |   metastable |      nus |     msun |   novel |   unique |   composition_validity |   structure_validity |   relaxation_failure_rate |   short_bond_rate |   abnormal_cell_rate |   relaxation_elapsed_mean |   ehull_all_available |   ehull_median_all_available |
|:---------|----------------------:|-----------------------------:|-------------------:|------------------------:|--------------------------:|----------:|---------:|---------------:|-----------------:|---------:|-------------:|---------:|---------:|--------:|---------:|-----------------------:|---------------------:|--------------------------:|------------------:|---------------------:|--------------------------:|----------------------:|-----------------------------:|
| C0       |              0.342964 |                     0.182745 |                  1 |                 40.082  |                        27 | 0.0493895 | 0.156136 |       0.111602 |                1 | 0.445312 |      0.78125 | 0.222656 | 0.507812 | 0.71875 | 0.984375 |               0.789062 |                    1 |                         0 |                 0 |                    0 |                   1.20802 |              0.156136 |                     0.111602 |
| E3-A     |              0.243956 |                     0.156065 |                  1 |                 40.2422 |                        27 | 0.0450575 | 0.156179 |       0.112915 |                1 | 0.445312 |      0.78125 | 0.222656 | 0.507812 | 0.71875 | 0.984375 |               0.789062 |                    1 |                         0 |                 0 |                    0 |                   1.19584 |              0.156179 |                     0.112915 |
| E3-G     |              0.263107 |                     0.169457 |                  1 |                 40      |                        27 | 0.0459374 | 0.156177 |       0.112904 |                1 | 0.445312 |      0.78125 | 0.222656 | 0.507812 | 0.71875 | 0.984375 |               0.789062 |                    1 |                         0 |                 0 |                    0 |                   1.19701 |              0.156177 |                     0.112904 |

## Primary endpoint

| arm   |   baseline_mean |   selected_mean |   baseline_median |   selected_median |   mean_difference |   median_difference |   relative_change | bootstrap_95_ci                               |   wilcoxon_p_raw |   wins |   ties |   losses | leave_one_out_mean_difference_range          |   remove_most_favorable_sample_index |   remove_most_favorable_sample_seed |   remove_most_favorable_relative_change |   remove_most_unfavorable_sample_index |   remove_most_unfavorable_sample_seed |   remove_most_unfavorable_relative_change |   maximum_single_sample_absolute_contribution_rate |   wilcoxon_p_holm | effect_reduction_ge_10_percent   | bootstrap_ci_upper_below_zero   | holm_p_below_0_05   | primary_effect_pass   |
|:------|----------------:|----------------:|------------------:|------------------:|------------------:|--------------------:|------------------:|:----------------------------------------------|-----------------:|-------:|-------:|---------:|:---------------------------------------------|-------------------------------------:|------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------:|------------------------------------------:|---------------------------------------------------:|------------------:|:---------------------------------|:--------------------------------|:--------------------|:----------------------|
| E3-A  |        0.342964 |        0.243956 |          0.182745 |          0.156065 |        -0.0990083 |         -0.0116177  |         -0.288684 | [-0.16489407698758815, -0.050201906937694905] |      1.31e-14    |    191 |      0 |       65 | [-0.10017259935570337, -0.07812874603841058] |                                  158 |                               40158 |                               -0.249921 |                                    130 |                                 40130 |                                 -0.291775 |                                           0.184602 |       2.62e-14    | True                             | True                            | True                | True                  |
| E3-G  |        0.342964 |        0.263107 |          0.182745 |          0.169457 |        -0.0798572 |         -8.6046e-07 |         -0.232844 | [-0.14496579962748402, -0.032452653777908895] |      4.18802e-10 |    163 |      0 |       93 | [-0.08062860842342655, -0.05890259553960769] |                                  158 |                               40158 |                               -0.18842  |                                    191 |                                 40191 |                                 -0.23443  |                                           0.236832 |       4.18802e-10 | True                             | True                            | True                | True                  |

## Mechanism decision

```json
{
  "GATE_MECHANISM_FORMAL_SUPPORTED": true,
  "always_gain": 0.09900830302618674,
  "condition_1_gain_coverage": false,
  "condition_2_reduce_harm": true,
  "condition_3_protect_low_force": true,
  "condition_4_smaller_intervention": false,
  "condition_5_better_quality_safety": false,
  "coverage_reduction": 0.3359375,
  "e3a_harm_rate": 0.25390625,
  "e3a_only_harm": 22,
  "e3a_refinement_rate": 1.0,
  "e3g_harm_rate": 0.18359375,
  "e3g_only_harm": 4,
  "e3g_quality_not_worse_gates": {
    "abnormal_cell": true,
    "composition": true,
    "ehull": true,
    "failure": true,
    "novel": true,
    "nus": true,
    "rmsd": true,
    "short_bond": true,
    "stable": true,
    "structure": true,
    "unique": true
  },
  "e3g_quality_not_worse_than_e3a": true,
  "e3g_refinement_rate": 0.6640625,
  "gain_retention": 0.8065711914977165,
  "gated_gain": 0.0798572449399984,
  "harm_mcnemar_exact_p": 0.0005335211753845215,
  "low_force_count": 128,
  "low_force_e3a_harm_rate": 0.296875,
  "low_force_e3g_harm_rate": 0.1796875,
  "low_force_threshold_max_force": 0.1815765301168724,
  "mean_displacement_reduction": 0.3088678399873124,
  "p95_displacement_reduction": 0.10287858466491695
}
```

## Limitations

- Stability is evaluated with the MatterSim-5M surrogate.
- No DFT or independent property-target verification was run.
