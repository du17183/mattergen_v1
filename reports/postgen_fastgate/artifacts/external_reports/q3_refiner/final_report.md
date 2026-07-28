# Q3 equivariant post-generation crystal refiner

- Final GO: `True`
- Safety gate: `True`
- Positive gate: `True`
- Original MatterGen sampling trajectory and backbone are unchanged.
- Atomic species and lattice are unchanged; position updates are equivariant force-vector steps under a learned invariant scalar gate.

| method    |    ehull |      rmsd |   stable |     nus |   composition_validity |   structure_validity |   novel |   unique |   converged |   pre_relax_max_force_ev_ang |
|:----------|---------:|----------:|---------:|--------:|-----------------------:|---------------------:|--------:|---------:|------------:|-----------------------------:|
| C0_FIRST  | 0.159748 | 0.0718947 |  0.28125 | 0.09375 |                   0.75 |                    1 | 0.71875 |        1 |     0.96875 |                     0.287507 |
| Q3_E3_PCR | 0.159185 | 0.0718872 |  0.28125 | 0.09375 |                   0.75 |                    1 | 0.71875 |        1 |     1       |                     0.228701 |

|        ehull |   rmsd_relative |   pre_relax_max_force_relative |   stable |   nus |   composition_validity |   structure_validity |   novel |   unique |   converged |
|-------------:|----------------:|-------------------------------:|---------:|------:|-----------------------:|---------------------:|--------:|---------:|------------:|
| -0.000562365 |    -0.000103435 |                      -0.204538 |        0 |     0 |                      0 |                    0 |       0 |        0 |     0.03125 |
