# RP-QTFG Phase 0 final report

## Decision

- `RP_QTFG_PHASE0_COMPLETED=True`
- `CHGNET_MAG_ORACLE_GO=True`
- `PHYSICS_DIRECTION_GO=True`
- `RP_QTFG_EIGHT_SEED_GO=False`
- `RP_QTFG_MVP_GO=False`
- `RP_QTFG_MVP_NO_GO=True` (early stop at Gate 1; 32-seed was not started)

## What worked

Gate 0A validated the CHGNet site-moment density candidate on all 8,716 finite held-out MP-20 structures: MAE 0.004015, Spearman 0.6417, and Top-100 target enrichment 22.67x.

Gate 0B showed that a small direct post-generation position+cell update had the intended independent MatterSim direction on 64 frozen A0 structures: energy improved for 78.12%, maximum force for 73.44%, and relaxation RMSD for 60.94%, without changing validity, composition, or mean E-hull materially.

Engineering safeguards also worked: disabled RP-QTFG is bitwise identical to A0, atomic numbers are never directly modified, unsafe proposals fall back to A0 without consuming RNG, and all 40 generation plus 40 relaxation tasks completed successfully.

## Why online RP-QTFG failed Gate 1

The closest diagnostic candidate, `G1_P75_S`, improved mean E-hull by 0.003353 eV/atom and kept Stable/Composition/Structure/NUS unchanged. However, mean relaxation RMSD worsened by 68.28%. Mean pre-relaxation maximum force improved by only 0.26%, with only 4/8 paired wins, so it is not a clear improvement. Median generation latency increased by 30.19%, marginally exceeding the 30% advisory limit.

The other candidates also failed: every candidate increased mean RMSD by 16.3%–293.3%; the medium position-only candidate lost 12.5 percentage points of composition validity and worsened E-hull by 0.0312 eV/atom; medium position+cell increased pre-relax force by 23.8%.

The offline and online results therefore reveal a mechanism mismatch: a tiny deterministic cleanup of a finished A0 structure is locally useful, but injecting repeated CHGNet corrections into the coupled diffusion trajectory changes later denoiser predictions and increases the final relaxation displacement.

## Test and safety status

- RP-QTFG tests: 25/25 passed.
- RP-QTFG + Adaptive CFG tests: 38/38 passed.
- Full repository: 179 passed; five failures are unchanged `main` compatibility failures (four missing-`dt` test calls and one removed PyTorch scheduler `verbose` argument).
- Determinism: Level 1 passed; disabled path bitwise identical; enabled repeated-seed structure and trace hashes identical.
- 32/64/256-seed experiments: not started.
- Other processes terminated: false. SIGKILL used: false.

## Detailed Gate 1 results

# RP-QTFG eight paired evaluation

- Seeds: 22000–22007 (8 per method).
- Initial-state pairing passed: `True`.
- Selected config: `None`.
- Gate decision: `False`.
- Evaluator: independent MatterSim-5M with TRI2024 compatibility; DFT verified: false.

## Method summary

| method   |   n |   generation_success |   generation_elapsed_median |   initial_energy_per_atom_mean |   initial_max_force_mean |   relaxation_rmsd_mean |   average_ehull |   stable_rate |   composition_validity |   structure_validity |   novel_rate |   unique_rate |   nus_rate |   force_convergence_rate |   relaxation_failure_rate |   severe_short_bond_count | atomic_numbers_modified   |   chgnet_forward_count |   chgnet_backward_count |   backtracking_mean |   fallback_rate |   conflict_rate |   clipping_rate |
|:---------|----:|---------------------:|----------------------------:|-------------------------------:|-------------------------:|-----------------------:|----------------:|--------------:|-----------------------:|---------------------:|-------------:|--------------:|-----------:|-------------------------:|--------------------------:|--------------------------:|:--------------------------|-----------------------:|------------------------:|--------------------:|----------------:|----------------:|----------------:|
| A0       |   8 |                    1 |                     74.9806 |                       -7.21367 |                 0.322783 |              0.0373281 |        0.121476 |         0.375 |                  0.875 |                    1 |          0.5 |             1 |       0    |                        1 |                         0 |                         0 | False                     |                      0 |                       0 |               0     |        0        |        0        |         0       |
| G1_P75_S |   8 |                    1 |                     97.6206 |                       -7.21101 |                 0.321934 |              0.0628155 |        0.118122 |         0.375 |                  0.875 |                    1 |          0.5 |             1 |       0    |                        1 |                         0 |                         0 | False                     |                   8044 |                    8044 |               2.75  |        0.296    |        0.29325  |         0.00275 |
| G1_P60_M |   8 |                    1 |                    111.587  |                       -7.23571 |                 0.320591 |              0.14682   |        0.15269  |         0.5   |                  0.75  |                    1 |          0.5 |             1 |       0.25 |                        1 |                         0 |                         0 | False                     |                  12211 |                   12211 |               0.875 |        0.315312 |        0.265937 |         0.09    |
| G2_P75_S |   8 |                    1 |                     99.1211 |                       -7.2135  |                 0.328435 |              0.0434131 |        0.121081 |         0.375 |                  0.875 |                    1 |          0.5 |             1 |       0    |                        1 |                         0 |                         0 | False                     |                   8002 |                    8002 |               0     |        0.24125  |        0.514375 |         0.21975 |
| G2_P60_M |   8 |                    1 |                    108.109  |                       -7.01437 |                 0.39966  |              0.0768072 |        0.117723 |         0.625 |                  0.875 |                    1 |          0.5 |             1 |       0.25 |                        1 |                         0 |                         0 | False                     |                  12057 |                   12057 |               0.125 |        0.206094 |        0.431016 |         0.13625 |

## Comparisons

| method   |   matterSim_energy_improvement_rate |   matterSim_force_improvement_rate |   relaxation_rmsd_improvement_rate |   any_primary_improvement_rate |   ehull_change_candidate_minus_a0 |   stable_change |   composition_change |   structure_change |   novel_change |   unique_change |   nus_change |   rmsd_relative_change |   pre_relax_max_force_relative_change |   latency_overhead | latency_risk_over_30_percent   | clear_improvement_direction   | clear_improvement_rule                                                                         | gate_go   |
|:---------|------------------------------------:|-----------------------------------:|-----------------------------------:|-------------------------------:|----------------------------------:|----------------:|---------------------:|-------------------:|---------------:|----------------:|-------------:|-----------------------:|--------------------------------------:|-------------------:|:-------------------------------|:------------------------------|:-----------------------------------------------------------------------------------------------|:----------|
| G1_P75_S |                                0.25 |                              0.5   |                              0.5   |                          0.625 |                      -0.00335305  |           0     |                0     |                  0 |              0 |               0 |         0    |               0.682795 |                           -0.00263042 |           0.301944 | True                           | False                         | mean RMSD or pre-relax max force improves at least 3 percent and at least 5 of 8 pairs improve | False     |
| G1_P60_M |                                0.5  |                              0.375 |                              0.375 |                          0.625 |                       0.0312144   |           0.125 |               -0.125 |                  0 |              0 |               0 |         0.25 |               2.93323  |                           -0.00679134 |           0.488207 | True                           | False                         | mean RMSD or pre-relax max force improves at least 3 percent and at least 5 of 8 pairs improve | False     |
| G2_P75_S |                                0.5  |                              0.375 |                              0.25  |                          0.625 |                      -0.000394328 |           0     |                0     |                  0 |              0 |               0 |         0    |               0.163015 |                            0.0175105  |           0.321955 | True                           | False                         | mean RMSD or pre-relax max force improves at least 3 percent and at least 5 of 8 pairs improve | False     |
| G2_P60_M |                                0.5  |                              0.5   |                              0.375 |                          0.625 |                      -0.00375282  |           0.25  |                0     |                  0 |              0 |               0 |         0.25 |               1.05763  |                            0.238171   |           0.441825 | True                           | False                         | mean RMSD or pre-relax max force improves at least 3 percent and at least 5 of 8 pairs improve | False     |
