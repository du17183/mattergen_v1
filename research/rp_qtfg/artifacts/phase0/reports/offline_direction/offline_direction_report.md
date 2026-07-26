# RP-QTFG Gate 0B offline physical-direction validation

- Structures: 64 frozen A0 outputs, seeds 20000–20063 (read-only).
- CHGNet 0.3.0 proposes trust-region position or weak position+cell updates.
- MatterSim-5M independently evaluates initial energy/force and full relaxations.
- `DFT_VERIFIED=False`; `MAGNETIC_PROPERTY_VERIFIED=False`.

## Decision

- `PHYSICS_DIRECTION_GO=True`
- Selected offline variant: `poscell_1`

## Method summary

| method    |   n |   initial_energy_per_atom_mean |   initial_max_force_mean |   relaxation_rmsd_mean |   average_ehull |   stable_rate |   composition_validity |   structure_validity |   convergence_rate |   severe_short_bond_count |
|:----------|----:|-------------------------------:|-------------------------:|-----------------------:|----------------:|--------------:|-----------------------:|---------------------:|-------------------:|--------------------------:|
| baseline  |  64 |                       -7.60462 |                 0.249489 |              0.0591831 |        0.134095 |       0.53125 |               0.859375 |                    1 |           0.984375 |                         0 |
| pos_1     |  64 |                       -7.60484 |                 0.211862 |              0.0591298 |        0.134072 |       0.53125 |               0.859375 |                    1 |           0.984375 |                         0 |
| pos_3     |  64 |                       -7.60501 |                 0.194458 |              0.0588082 |        0.134082 |       0.53125 |               0.859375 |                    1 |           0.984375 |                         0 |
| pos_5     |  64 |                       -7.60507 |                 0.189525 |              0.0584079 |        0.134104 |       0.53125 |               0.859375 |                    1 |           0.984375 |                         0 |
| poscell_1 |  64 |                       -7.60501 |                 0.210539 |              0.0588285 |        0.134085 |       0.53125 |               0.859375 |                    1 |           0.984375 |                         0 |
| poscell_3 |  64 |                       -7.60544 |                 0.192682 |              0.0579928 |        0.13409  |       0.53125 |               0.859375 |                    1 |           0.984375 |                         0 |

## Paired Gate comparisons

| method    |   matterSim_energy_improvement_rate |   matterSim_force_improvement_rate |   relaxation_rmsd_improvement_rate |   any_primary_improvement_rate |   ehull_change_candidate_minus_a0 |   structure_validity_change |   composition_validity_change | validity_safe   | variant_gate_go   |
|:----------|------------------------------------:|-----------------------------------:|-----------------------------------:|-------------------------------:|----------------------------------:|----------------------------:|------------------------------:|:----------------|:------------------|
| pos_1     |                            0.640625 |                           0.703125 |                           0.546875 |                       0.796875 |                      -2.30811e-05 |                           0 |                             0 | True            | True              |
| pos_3     |                            0.671875 |                           0.65625  |                           0.5625   |                       0.765625 |                      -1.24559e-05 |                           0 |                             0 | True            | True              |
| pos_5     |                            0.65625  |                           0.671875 |                           0.578125 |                       0.75     |                       8.57953e-06 |                           0 |                             0 | True            | True              |
| poscell_1 |                            0.78125  |                           0.734375 |                           0.609375 |                       0.921875 |                      -1.00553e-05 |                           0 |                             0 | True            | True              |
| poscell_3 |                            0.78125  |                           0.671875 |                           0.625    |                       0.875    |                      -5.29179e-06 |                           0 |                             0 | True            | True              |
