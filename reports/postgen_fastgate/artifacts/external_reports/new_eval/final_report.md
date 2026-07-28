# Q6 NS-SetRank 32-pool blind evaluation

- Final GO: `False`
- Safety gate: `False`
- Positive gate: `True`
- Generator: original C0 MatterGen; sampling trajectory unchanged
- Pool: 32 independent pools x 4 new C0 trajectories
- Selection: frozen CHGNet features + frozen three-member SetRank ensemble
- MatterSim was used only after candidate selection
- DFT verified: False

## Aggregate metrics

| method        |    ehull |      rmsd |   stable |     nus |   composition_validity |   structure_validity |   novel |   unique |   converged |   pre_relax_max_force_ev_ang |
|:--------------|---------:|----------:|---------:|--------:|-----------------------:|---------------------:|--------:|---------:|------------:|-----------------------------:|
| C0_FIRST      | 0.159748 | 0.0718947 |  0.28125 | 0.09375 |                0.75    |                    1 | 0.71875 |        1 |     0.96875 |                     0.287507 |
| Q6_NS_SETRANK | 0.126025 | 0.0338248 |  0.5625  | 0.1875  |                0.96875 |                    1 | 0.59375 |        1 |     1       |                     0.15642  |

## Changes

|      ehull |   rmsd_relative |   pre_relax_max_force_relative |   stable |     nus |   composition_validity |   structure_validity |   novel |   unique |   converged |
|-----------:|----------------:|-------------------------------:|---------:|--------:|-----------------------:|---------------------:|--------:|---------:|------------:|
| -0.0337225 |       -0.529522 |                      -0.455944 |  0.28125 | 0.09375 |                0.21875 |                    0 |  -0.125 |        0 |     0.03125 |
