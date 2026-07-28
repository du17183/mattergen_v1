# Q5 CQPS 32-pool blind evaluation

- Final GO: `False`
- Safety gate: `False`
- Positive gate: `True`
- CHGNet target hit is a validated surrogate, not DFT proof.
- New MatterSim labels were not used for selection.

| method   |    ehull |      rmsd |   stable |     nus |   composition_validity |   structure_validity |   novel |   unique |   converged |   pre_relax_max_force_ev_ang |
|:---------|---------:|----------:|---------:|--------:|-----------------------:|---------------------:|--------:|---------:|------------:|-----------------------------:|
| C0_FIRST | 0.159748 | 0.0718947 |  0.28125 | 0.09375 |                0.75    |                    1 | 0.71875 |        1 |     0.96875 |                     0.287507 |
| Q5_CQPS  | 0.128351 | 0.0345357 |  0.5     | 0.125   |                0.84375 |                    1 | 0.5625  |        1 |     0.96875 |                     0.181675 |

|      ehull |   rmsd_relative |   pre_relax_max_force_relative |   stable |     nus |   composition_validity |   structure_validity |    novel |   unique |   converged |   target_hit_0_02 |
|-----------:|----------------:|-------------------------------:|---------:|--------:|-----------------------:|---------------------:|---------:|---------:|------------:|------------------:|
| -0.0313966 |       -0.519635 |                      -0.368102 |  0.21875 | 0.03125 |                0.09375 |                    0 | -0.15625 |        0 |           0 |             0.375 |
